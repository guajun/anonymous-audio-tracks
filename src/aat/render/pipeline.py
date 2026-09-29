"""End-to-end deterministic sample rendering.

``render_sample`` turns a validated :class:`~aat.render.config.RenderConfig`
into a self-contained output directory:

``mix.wav`` / ``stems/<id>.wav`` / ``dry/<id>.wav``
    PCM16 exports.  ``stems`` are post-effect, ``dry`` are raw sampler outputs.
``sources.json`` / ``controls.json`` / ``manifest.json``
    Frozen-protocol documents (``rendered`` stage).
``render_report.json``
    Renderer-private evidence: latency, onsets, tail decay, stem-sum residual,
    clipping/non-finite counters and the resolved configuration.
``surge_probe.json``
    Only when a Surge XT plugin path was supplied; passed/failed plus reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aat.contracts.documents import SampleManifest
from aat.contracts.jsonio import dump_json

from . import analysis
from .artifacts import (
    CONTROLS_FILENAME,
    MANIFEST_FILENAME,
    REPORT_FILENAME,
    SOURCES_FILENAME,
    SURGE_REPORT_FILENAME,
    build_manifest,
    build_sources,
    dry_relative_path,
    mix_relative_path,
    sha256_file,
    sha256_json,
    stem_relative_path,
)
from .config import RenderConfig
from .errors import RenderValidationError
from .graph import render_graph, require_dawdreamer
from .probe_surge import SurgeProbeResult, probe_surge
from .timeline import build_controls, expand_score
from .version import REPORT_SCHEMA_VERSION
from .wavio import read_pcm16_wav, write_pcm16_wav

#: Ratio of final-window RMS to overall RMS above which a render is considered
#: truncated instead of naturally decayed.
TAIL_DECAY_RATIO_LIMIT = 0.05


@dataclass
class RenderResult:
    """Paths and documents produced by :func:`render_sample`."""

    out_dir: Path
    manifest: SampleManifest
    report: dict
    surge: SurgeProbeResult | None


def _stereo(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[0] == 1:
        array = np.vstack([array, array])
    return np.ascontiguousarray(array, dtype=np.float32)


def _buffer_stats(samples: np.ndarray, sample_rate: int) -> dict:
    return {
        "channels": int(samples.shape[0]),
        "samples": int(samples.shape[1]),
        "duration_seconds": float(samples.shape[1] / sample_rate),
        "peak_dbfs": float(analysis.peak_dbfs(samples)),
        "rms_dbfs": float(analysis.rms_dbfs(samples)),
        "clipped_samples": int(analysis.count_clipped(samples)),
        "non_finite_samples": int(analysis.count_non_finite(samples)),
        "silent": bool(analysis.is_empty(samples)),
    }


def _validate_audio(config: RenderConfig, rendered) -> None:
    problems: list[str] = []
    buffers = {"mix": rendered.mix}
    buffers.update({f"stem:{key}": value for key, value in rendered.stems.items()})
    buffers.update({f"dry:{key}": value for key, value in rendered.dry.items()})
    for name, buffer in buffers.items():
        non_finite = analysis.count_non_finite(buffer)
        if non_finite:
            problems.append(f"{name}: {non_finite} non-finite sample(s)")
        if analysis.is_empty(buffer):
            problems.append(f"{name}: silent/empty audio")
        clipped = analysis.count_clipped(buffer)
        if clipped and (name == "mix" or name.startswith("stem:")):
            problems.append(f"{name}: {clipped} clipped sample(s)")
    if problems:
        raise RenderValidationError("render produced invalid audio: " + "; ".join(problems))


def _required_tail_seconds(config: RenderConfig, score) -> float:
    """Latest absolute output time any note can reach (note-off + release)."""

    required = 0.0
    for event in score:
        source = config.sources[event.source_index]
        required = max(
            required,
            event.start_seconds + event.duration_seconds + source.amp.release_ms / 1000.0,
        )
    return required


def _check_tail(config: RenderConfig, score, mix: np.ndarray) -> dict:
    guard = min(0.05, config.tail_seconds / 4.0) if config.tail_seconds > 0.0 else 0.0
    total = mix.shape[1] / config.sample_rate
    required = _required_tail_seconds(config, score)
    margin = total - required
    ratio = analysis.tail_decay_ratio(mix, config.sample_rate, 0.1)
    evidence = {
        "required_seconds": float(required),
        "rendered_seconds": float(total),
        "tail_seconds": float(config.tail_seconds),
        "margin_seconds": float(margin),
        "final_window_rms_ratio": float(ratio),
        "decayed": bool(ratio <= TAIL_DECAY_RATIO_LIMIT),
    }
    if margin <= guard:
        raise RenderValidationError(
            f"tail is truncated: last release reaches {required:.3f}s but only "
            f"{total:.3f}s were rendered (increase render.tail_seconds)"
        )
    if ratio > TAIL_DECAY_RATIO_LIMIT:
        raise RenderValidationError(
            f"tail did not decay: final-window RMS ratio {ratio:.4f} > "
            f"{TAIL_DECAY_RATIO_LIMIT:.4f}; increase render.tail_seconds"
        )
    return evidence


def _onset_evidence(config: RenderConfig, score, rendered) -> list[dict]:
    per_source: dict[str, list] = {source_id: [] for source_id in config.source_ids}
    for event in score:
        per_source[event.source_id].append(event)
    rows: list[dict] = []
    for source in config.sources:
        events = per_source[source.source_id]
        first_control = min(event.start_seconds for event in events) if events else None
        stem = rendered.stems[source.source_id]
        active = analysis.first_active_sample(stem)
        onset_absolute = (
            None
            if active is None
            else config.track_start_seconds + active / float(config.sample_rate)
        )
        offset = (
            None
            if onset_absolute is None or first_control is None
            else float(onset_absolute - first_control)
        )
        latency_samples = rendered.effect_latency_samples.get(source.source_id)
        latency_seconds = rendered.effect_latency_seconds.get(source.source_id)
        rows.append(
            {
                "source_id": source.source_id,
                "dry_path": dry_relative_path(source.source_id),
                "stem_path": stem_relative_path(source.source_id),
                "effect_chain": [effect.type for effect in source.effects],
                "first_control_seconds": None if first_control is None else float(first_control),
                "stem_onset_seconds": onset_absolute,
                "onset_offset_seconds": offset,
                "effect_latency_samples": None if latency_samples is None else int(latency_samples),
                "effect_latency_seconds": latency_seconds,
                "stem_stats": _buffer_stats(stem, config.sample_rate),
                "dry_stats": _buffer_stats(rendered.dry[source.source_id], config.sample_rate),
            }
        )
    return rows


def _git_commit(start: Path) -> str | None:
    for parent in [start.resolve(), *start.resolve().parents]:
        dotgit = parent / ".git"
        if dotgit.is_file():
            text = dotgit.read_text(encoding="utf-8", errors="replace").strip()
            if not text.startswith("gitdir:"):
                return None
            target = Path(text.split(":", 1)[1].strip())
            if not target.is_absolute():
                target = (parent / target).resolve()
            commit = _read_head(target)
            if commit:
                return commit
        elif dotgit.is_dir():
            commit = _read_head(dotgit)
            if commit:
                return commit
    return None


def _common_git_dir(gitdir: Path) -> Path:
    common = gitdir / "commondir"
    if common.is_file():
        try:
            text = common.read_text(encoding="utf-8").strip()
        except OSError:
            return gitdir
        target = Path(text)
        return target if target.is_absolute() else (gitdir / target).resolve()
    return gitdir


def _read_head(gitdir: Path) -> str | None:
    try:
        head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if head.startswith("ref:"):
        reference = head.split(":", 1)[1].strip()
        for base in (gitdir, _common_git_dir(gitdir)):
            try:
                return (base / reference).read_text(encoding="utf-8").strip()[:12]
            except OSError:
                continue
            packed = base / "packed-refs"
            try:
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(reference) and not line.startswith("#"):
                        return line.split(" ", 1)[0][:12]
            except OSError:
                continue
        return None
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", head):
        return head[:12]
    return None


def render_sample(
    config: RenderConfig,
    out_dir: str | Path,
    *,
    config_source: str | Path | None = None,
    surge_plugin_path: str | None = None,
) -> RenderResult:
    """Render one sample and write every artifact into ``out_dir``."""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    require_dawdreamer()  # fail early with an actionable dependency message

    score = expand_score(config)
    rendered = render_graph(config, score)
    _validate_audio(config, rendered)

    mix = _stereo(rendered.mix)
    stems = {source_id: _stereo(buffer) for source_id, buffer in rendered.stems.items()}
    dry = {source_id: _stereo(buffer) for source_id, buffer in rendered.dry.items()}

    tail_evidence = _check_tail(config, score, mix)
    source_rows = _onset_evidence(config, score, rendered)

    write_pcm16_wav(target / mix_relative_path(), mix, config.sample_rate)
    for source in config.sources:
        write_pcm16_wav(
            target / stem_relative_path(source.source_id),
            stems[source.source_id],
            config.sample_rate,
        )
        write_pcm16_wav(
            target / dry_relative_path(source.source_id),
            dry[source.source_id],
            config.sample_rate,
        )

    # Read the exported files back so the additivity evidence refers to the
    # actual PCM artifacts, not to the in-memory float buffers.
    mix_pcm, mix_meta = read_pcm16_wav(target / mix_relative_path())
    stem_pcm = {}
    for source in config.sources:
        pcm, meta = read_pcm16_wav(target / stem_relative_path(source.source_id))
        if meta["sample_rate"] != config.sample_rate:
            raise RenderValidationError(
                f"stem {source.source_id}: sample rate {meta['sample_rate']} != {config.sample_rate}"
            )
        stem_pcm[source.source_id] = pcm
    if mix_meta["sample_rate"] != config.sample_rate:
        raise RenderValidationError(
            f"mix: sample rate {mix_meta['sample_rate']} != {config.sample_rate}"
        )
    stem_sum = analysis.check_stem_sum(
        [stem_pcm[source_id] for source_id in config.source_ids], mix_pcm
    )

    # Protocol documents.
    build_sources(config).save(target / SOURCES_FILENAME)
    build_controls(config, score).save(target / CONTROLS_FILENAME)
    digest_paths = {
        mix_relative_path(): sha256_file(target / mix_relative_path()),
        SOURCES_FILENAME: sha256_file(target / SOURCES_FILENAME),
        CONTROLS_FILENAME: sha256_file(target / CONTROLS_FILENAME),
    }
    for source in config.sources:
        relative = stem_relative_path(source.source_id)
        digest_paths[relative] = sha256_file(target / relative)

    onset_offsets = [
        row["onset_offset_seconds"]
        for row in source_rows
        if row["onset_offset_seconds"] is not None
    ]
    render_latency_seconds = max([0.0, *onset_offsets])
    manifest = build_manifest(
        config,
        digest_paths=digest_paths,
        renderer_version=rendered.renderer_version,
        render_latency_seconds=render_latency_seconds,
        tail_seconds=config.tail_seconds,
        notes=(
            "dry references: dry/<source_id>.wav; per-source latency and onset "
            "evidence: render_report.json"
        ),
    )
    manifest.save(target / MANIFEST_FILENAME)

    surge_result: SurgeProbeResult | None = None
    resolved_plugin_path = surge_plugin_path or config.surge.plugin_path
    if resolved_plugin_path:
        surge_result = probe_surge(
            resolved_plugin_path,
            sample_rate=config.sample_rate,
            block_size=config.block_size,
            duration_seconds=config.surge.probe_duration_seconds,
            note=config.surge.note,
            velocity=config.surge.velocity,
        )
        dump_json(target / SURGE_REPORT_FILENAME, surge_result.to_json_dict())

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "render_report",
        "sample_id": config.sample_id,
        "seed": config.seed,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(Path(__file__)),
        "renderer": rendered.renderer_version,
        "sample_rate": config.sample_rate,
        "block_size": config.block_size,
        "requested_total_seconds": float(config.total_seconds),
        "rendered_total_seconds": float(mix.shape[1] / config.sample_rate),
        "musical_duration_seconds": float(config.duration_seconds),
        "tail_seconds": float(config.tail_seconds),
        "track_start_seconds": float(config.track_start_seconds),
        "config_sha256": sha256_json(config.to_dict()),
        "config_basename": None if config_source is None else Path(config_source).name,
        "config_file_sha256": None if config_source is None else sha256_file(config_source),
        "artifacts": {
            "mix": mix_relative_path(),
            "stems": {source.source_id: stem_relative_path(source.source_id) for source in config.sources},
            "dry": {source.source_id: dry_relative_path(source.source_id) for source in config.sources},
            "sources": SOURCES_FILENAME,
            "controls": CONTROLS_FILENAME,
            "manifest": MANIFEST_FILENAME,
        },
        "mix_stats": _buffer_stats(mix, config.sample_rate),
        "stem_sum": stem_sum,
        "tail": tail_evidence,
        "sources": source_rows,
        "surge_probe": (
            {"status": "not_run", "reason": "no_surge_plugin_path_configured"}
            if surge_result is None
            else surge_result.to_json_dict()
        ),
        "config": config.to_dict(),
    }
    dump_json(target / REPORT_FILENAME, report)
    return RenderResult(out_dir=target, manifest=manifest, report=report, surge=surge_result)


__all__ = ["RenderResult", "TAIL_DECAY_RATIO_LIMIT", "render_sample"]
