#!/usr/bin/env python3
"""Issue #11 end-to-end demo pipeline: config/data or checkpoint -> E/P -> tracks -> viewer.

Three clearly separated modes (never mixed in reports or filenames):

``smoke``
    CPU fake-encoder engineering smoke on the synthetic protocol corpus made by
    ``aat.training.dataset.make_smoke_dataset``: dataset -> a few training
    steps -> checkpoint -> **reload from disk** -> E/P -> associations ->
    serialized ``prediction.json/npz`` + ``trajectory.json`` + a local viewer
    session + a report.  ``data_kind = mock``; this is not a real model result.

``synthetic``
    A real frozen-AuT training checkpoint (for example the accepted issue #8
    ``aut-short`` checkpoint) evaluated on held-out synthesized songs from a
    verified dataset index.  Runs the canonical issue #8 ``evaluate_split``
    protocol (activity/boundary/identity/count metrics plus all-inactive,
    all-active and no-identity baselines) and additionally serializes the
    per-song prediction/trajectory/viewer artifacts so the same numbers can be
    re-checked on disk.

``audio``
    The same checkpoint on one local audio file/segment (user music).  The
    audio and all decoded samples stay local; the source file digest, the exact
    segment origin/duration and the decoding command/version are recorded in
    the report.  There is no ground truth, so no accuracy metric is reported;
    the generated ``manual_inspection.md`` marks human listening explicitly as
    pending until a person performs it.

Common guarantees:

* original-track absolute time axis everywhere (no re-basing, no timing
  rescale, no beat fitting, no per-frame ground-truth remapping);
* the pinned 2 s independent-window encoder policy and the checkpoint's own
  attention/dtype/extraction settings are used for inference;
* every report records Git SHA, ``uv.lock`` digest, config/dataset digests, the
  random seeds, the encoder revision and the attention/window policy;
* output directories are archived (never deleted) unless ``--overwrite`` is
  passed explicitly.

All inputs are CLI flags; ``--config`` can supply the same values as a flat
TOML file (long option names as keys) so a run does not require hand-editing
JSON.  Example::

    uv run --no-sync python scripts/demo_pipeline.py smoke --out runs/demo/smoke --steps 3
    uv run --no-sync python scripts/demo_pipeline.py synthetic \
        --checkpoint runs/train/aut-short/checkpoint.pt \
        --index runs/train-index/index.json --data-root runs/train-data \
        --split test --out runs/demo/aut-test
    uv run --no-sync python scripts/demo_pipeline.py audio \
        --checkpoint runs/train/aut-short/checkpoint.pt \
        --audio /path/to/song.mp3 --start-seconds 30.0 --duration-seconds 8.0 \
        --hop-seconds 0.10 --out runs/demo/clip-a
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from aat.contracts import (  # noqa: E402
    ActivityData,
    AudioSpan,
    ContractError,
    RunProvenance,
)
from aat.contracts.jsonio import dumps_json  # noqa: E402
from aat.data import DatasetError, DatasetIndex  # noqa: E402
from aat.evaluation import EvaluationConfig, EvaluationError, evaluate_trajectory  # noqa: E402
from aat.labels import LabelError  # noqa: E402
from aat.labels.wav import read_wav  # noqa: E402
from aat.tracking import PredictionSequence, TrackingError, associate_sequence  # noqa: E402
from aat.training.checkpoint import (  # noqa: E402
    file_sha256,
    git_provenance,
    utc_now,
    uv_lock_sha256,
)
from aat.training.config import TrainConfig, default_smoke_config  # noqa: E402
from aat.training.dataset import (  # noqa: E402
    dataset_fingerprint,
    index_file_sha256,
    load_verified_index,
    make_smoke_dataset,
)
from aat.training.errors import TrainingError  # noqa: E402
from aat.training.evaluate import evaluate_split, tracking_config_from  # noqa: E402
from aat.training.inference import HeadInference, load_head_from_checkpoint  # noqa: E402
from aat.windowing import center_times  # noqa: E402

SCHEMA_REPORT = "aat-demo-pipeline-report/1"
SCHEMA_SESSION = "aat-demo-viewer-session/1"

MODE_SMOKE = "fake-encoder-smoke"
MODE_SYNTHETIC = "aut-frozen-synthetic"
MODE_SYNTHETIC_FAKE = "fake-encoder-synthetic"
MODE_AUDIO = "aut-frozen-audio-clip"
MODE_AUDIO_FAKE = "fake-encoder-audio-clip"

SMOKE_RESULT_KIND = "fake-encoder-smoke (engineering pipeline only, not a real model result)"
AUT_RESULT_KIND = "aut-frozen-checkpoint-inference (real frozen AuT features)"


# --------------------------------------------------------------------------- #
# small filesystem / provenance helpers
# --------------------------------------------------------------------------- #


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _write_json(path: Path, payload: Any) -> Path:
    return _write_text(path, dumps_json(payload) + "\n")


def _archive_if_needed(target: Path, *, overwrite: bool, kind: str) -> None:
    """Move an existing non-empty directory aside; never delete anything."""


    if not target.exists():
        return
    if not any(target.iterdir()):
        return
    if not overwrite:
        raise TrainingError(
            f"{kind} {target} already exists and is not empty; pass --overwrite to archive it "
            f"as {target.name}.bak-<timestamp> (nothing is deleted)"
        )
    stamp = re.sub(r"[^0-9A-Za-z]", "", datetime.now(timezone.utc).isoformat())
    backup = target.with_name(f"{target.name}.bak-{stamp}")
    os.replace(target, backup)
    print(f"archived existing {kind} as {backup}", file=sys.stderr)


def _artifact_index(root: Path) -> dict[str, str]:
    """sha256 of every regular file under ``root`` keyed by POSIX relative path."""

    index: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = file_sha256(path)
            if digest:
                index[path.relative_to(root).as_posix()] = digest
    return index


def _finalize_report(out: Path, report: dict[str, Any]) -> None:
    """Write ``artifacts.json`` plus the report that references it.

    The report file cannot be listed by its own completed digest, so it is
    excluded from the artifact index instead of recording a stale hash.
    """

    artifacts = _artifact_index(out)
    artifacts.pop("pipeline_report.json", None)
    artifacts.pop("artifacts.json", None)
    _write_json(out / "artifacts.json", artifacts)
    report["artifacts"] = {"path": "artifacts.json", "files": len(artifacts)}
    _write_json(out / "pipeline_report.json", report)


def _start_resources(device_str: str) -> dict[str, Any]:
    """Begin per-process resource accounting (CUDA peaks included when usable)."""

    info: dict[str, Any] = {
        "device": device_str,
        "cuda_available": False,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "measurement_scope": (
            "per-process torch CUDA allocator statistics from before encoder load "
            "through pipeline end; other processes are not observable"
        ),
        "_start": time.perf_counter(),
    }
    try:
        import torch  # noqa: PLC0415 - lazy so the CLI stays importable without the ml extra
    except ImportError:
        info["unavailable_reason"] = "torch is not installed"
        return info
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"] and device_str.startswith("cuda"):
        try:
            torch.cuda.init()
            index = torch.device(device_str).index or 0
            info["device_index"] = int(index)
            torch.cuda.reset_peak_memory_stats(index)
        except (RuntimeError, ValueError) as error:  # pragma: no cover - hardware dependent
            info["unavailable_reason"] = f"CUDA peak measurement unavailable: {error}"
    else:
        info["unavailable_reason"] = "CPU or non-CUDA device: torch allocator peak not applicable"
    return info


def _finish_resources(info: dict[str, Any]) -> dict[str, Any]:
    start = info.pop("_start", None)
    if start is not None:
        info["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    if info.get("peak_allocated_bytes") is None and str(info.get("device", "")).startswith("cuda") and info.get("cuda_available"):
        try:
            import torch  # noqa: PLC0415

            index = int(info.get("device_index", 0))
            info["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(index))
            info["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(index))
            info.pop("unavailable_reason", None)
        except (RuntimeError, ValueError) as error:  # pragma: no cover - hardware dependent
            info["unavailable_reason"] = f"CUDA peak measurement unavailable: {error}"
    return info


def _mode_result_kind(config: TrainConfig, *, real: str, fake_note: str) -> str:
    """Never let a fake-encoder run claim a real frozen-AuT result."""

    return real if config.encoder.mode == "aut" else fake_note


def _pinned_policy(config: TrainConfig) -> dict[str, Any]:
    """The pinned inference policy recorded in every report."""

    encoder = config.encoder
    return {
        "mode": encoder.mode,
        "window_seconds": encoder.window_seconds,
        "attention": encoder.attention,
        "dtype": encoder.dtype,
        "extraction": encoder.extraction,
        "batch_windows": encoder.batch_windows,
        "feature_dim": encoder.feature_dim,
    }


def _run_provenance(
    *,
    mode: str,
    config: TrainConfig,
    inference: HeadInference,
    git: Mapping[str, Any],
    run_id: str,
) -> RunProvenance:
    is_aut = config.encoder.mode == "aut"
    return RunProvenance(
        run_id=run_id,
        data_kind="model" if is_aut else "mock",
        model_id=(
            f"source-query-head:step={inference.step}:encoder={config.encoder.mode}"
            if inference.step is not None
            else f"source-query-head:encoder={config.encoder.mode}"
        ),
        git_commit=git.get("sha") if isinstance(git.get("sha"), str) else None,
        config_hash=config.config_sha256()[:16],
        created_at_utc=utc_now(),
        notes=(
            "real frozen AuT checkpoint inference"
            if is_aut
            else "CPU fake-encoder smoke; pipeline wiring only, not a real model result"
        ),
    )


# --------------------------------------------------------------------------- #
# prediction -> association -> serialized artifacts -> viewer session
# --------------------------------------------------------------------------- #


def _active_spans(
    center: Sequence[float], activity: Sequence[float], threshold: float
) -> list[list[float]]:
    """Contiguous absolute-second spans where ``activity >= threshold``."""

    spans: list[list[float]] = []
    start: float | None = None
    previous: float | None = None
    for time_value, probability in zip(center, activity):
        if probability >= threshold:
            if start is None:
                start = float(time_value)
            previous = float(time_value)
        elif start is not None:
            spans.append([start, float(previous)])
            start = None
    if start is not None:
        spans.append([start, float(previous)])
    return spans


def _manual_inspection(
    *,
    label: str,
    mode: str,
    audio_path: Path,
    audio_duration: float,
    track_start_seconds: float,
    trajectory_path: Path,
    threshold: float,
    tracks: Sequence[Mapping[str, Any]],
    window_seconds: float,
    hop_seconds: float,
    extra_notes: Sequence[str] = (),
) -> str:
    lines = [
        f"# Local manual inspection: {label}",
        "",
        f"- mode: `{mode}`",
        "- human listening status: **pending** (no human auditory judgement has been recorded yet).",
        "",
        "## What to load",
        "",
        "The viewer is local-only; it never uploads audio and makes no network requests.",
        "",
        "```sh",
        "uv run python -m http.server 8123 --directory viewer",
        "# open http://127.0.0.1:8123/ in a browser",
        "```",
        "",
        f"1. Load audio: `{audio_path.as_posix()}`",
        f"2. Load trajectory: `{trajectory_path.as_posix()}`",
        "",
        "## Time axis",
        "",
        f"- analyzed span: `{track_start_seconds:.3f}s` .. `{track_start_seconds + audio_duration:.3f}s` "
        "(original-track absolute seconds; the viewer shows both local and absolute time).",
        f"- pinned window: {window_seconds:g}s; center hop: {hop_seconds:g}s; activity threshold: {threshold:g}.",
        "- the viewer never rescales time or fits a beat grid; probability samples stay at their center times.",
        "",
        "## Anonymous tracks (objective view; not an instrument identity claim)",
        "",
        "| track_id | points | active points | first active (abs s) | last active (abs s) | max p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for track in tracks:
        lines.append(
            "| {track_id} | {points} | {active} | {first} | {last} | {maximum} |".format(**track)
        )
    lines.extend(
        [
            "",
            "## Checklist (fill in while listening; do not guess from the chart)",
            "",
            "- [ ] count: how many clearly separable layers do you hear in this segment? ____",
            "- [ ] false positives: are there tracks active with no audible layer? which spans? ____",
            "- [ ] misses: are there audible layers with no/too-late track? which spans? ____",
            "- [ ] identity: does any track visibly swap source (e.g. drums -> bass) without a gap? ____",
            "- [ ] boundaries: are onsets/offsets clearly early/late (record absolute seconds)? ____",
            "- [ ] limitations/ambiguous layers (sounds that cannot be told apart by ear): ____",
            "",
            "Record measurements on the original-track axis. If the segment is too short or too dense to judge, say so.",
            "",
            "## Machine-readable companion",
            "",
            "- `session.json` in the same directory links audio, trajectory and prediction files.",
            "- objective inference checks are in `../pipeline_report.json`; they are not listening results.",
        ]
    )
    for note in extra_notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def run_song_artifacts(
    entry: Any,
    *,
    data_root: Path,
    inference: HeadInference,
    config: TrainConfig,
    out_dir: Path,
    run_id: str,
    git: Mapping[str, Any],
    hop_seconds: float | None = None,
    reference: ActivityData | None = None,
) -> dict[str, Any]:
    """Predict one indexed song, associate tracks, and serialize every artifact."""

    directory = data_root / entry.path
    if reference is None:
        reference = ActivityData.load(directory)
    wav = read_wav(directory / str(entry.audio["mix_path"]))
    mono = wav.samples.mean(axis=1) if wav.channels > 1 else wav.samples[:, 0]
    centers = [float(value) for value in reference.center_times]
    effective_hop = (
        hop_seconds
        if hop_seconds is not None
        else (reference.hop_seconds if reference.hop_seconds else 0.02)
    )
    provenance = _run_provenance(
        mode=config.encoder.mode,
        config=config,
        inference=inference,
        git=git,
        run_id=run_id,
    )
    predictions = inference.predict_at_times(
        mono,
        centers,
        sample_rate=wav.sample_rate,
        origin_seconds=float(entry.track_start_seconds),
        sample_id=entry.sample_id,
        hop_seconds=effective_hop,
        provenance=provenance,
    )
    audio_seconds = wav.frames / wav.sample_rate
    trajectory = associate_sequence(
        PredictionSequence.from_prediction(predictions),
        AudioSpan(
            duration_seconds=float(audio_seconds),
            track_start_seconds=float(entry.track_start_seconds),
        ),
        provenance,
        config=tracking_config_from(config),
        sample_id=entry.sample_id,
    )

    song_dir = out_dir / entry.sample_id
    song_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = song_dir / "prediction"
    predictions.save(prediction_dir)
    trajectory_path = trajectory.save(song_dir / "trajectory.json")
    canonical = evaluate_trajectory(
        reference,
        trajectory,
        EvaluationConfig(
            activity_threshold=config.eval.threshold,
            time_tolerance_seconds=1e-6,
        ),
    )
    evaluation = canonical.to_dict()
    _write_json(song_dir / "evaluation.json", evaluation)

    threshold = config.eval.threshold
    track_rows: list[dict[str, Any]] = []
    for track in trajectory.tracks:
        values = list(track.activity)
        active = sum(1 for value in values if value >= threshold)
        spans = _active_spans(track.center_times, values, threshold)
        track_rows.append(
            {
                "track_id": track.track_id,
                "points": len(values),
                "active": active,
                "first": f"{spans[0][0]:.3f}" if spans else "-",
                "last": f"{spans[-1][1]:.3f}" if spans else "-",
                "maximum": f"{max(values):.4f}" if values else "-",
                "active_spans_absolute_seconds": spans,
            }
        )

    session = {
        "schema": SCHEMA_SESSION,
        "run_id": run_id,
        "sample_id": entry.sample_id,
        "mode": config.encoder.mode,
        "data_kind": provenance.data_kind,
        "warning": (
            "fake encoder smoke: this trajectory is engineering evidence only and must not be "
            "read as a real model result"
            if config.encoder.mode == "fake"
            else "real frozen AuT checkpoint inference"
        ),
        "audio": {
            "path": str((directory / str(entry.audio["mix_path"])).resolve()),
            "duration_seconds": float(audio_seconds),
            "track_start_seconds": float(entry.track_start_seconds),
            "sha256": file_sha256(directory / str(entry.audio["mix_path"])),
            "sample_rate": wav.sample_rate,
        },
        "trajectory": str(trajectory_path.resolve()),
        "prediction_dir": str(prediction_dir.resolve()),
        "evaluation": str((song_dir / "evaluation.json").resolve()),
        "tracks": track_rows,
        "objective_checks": {
            "centers_total": len(centers),
            "centers_valid": int(sum(1 for value in predictions.center_valid if value)),
            "centers_padded_edges": int(sum(1 for value in predictions.center_valid if not value)),
            "tracks": len(trajectory.tracks),
            "hop_seconds": effective_hop,
            "window_seconds": config.encoder.window_seconds,
            "activity_threshold": threshold,
        },
        "human_listening": {
            "status": "pending",
            "instructions": "edit manual_inspection.md after listening; keep the original timestamps",
        },
        "privacy": "audio stays local on this machine; the viewer uploads nothing",
    }
    _write_json(song_dir / "session.json", session)
    _write_text(
        song_dir / "manual_inspection.md",
        _manual_inspection(
            label=f"{entry.sample_id} ({config.encoder.mode})",
            mode=config.encoder.mode,
            audio_path=(directory / str(entry.audio["mix_path"])).resolve(),
            audio_duration=float(audio_seconds),
            track_start_seconds=float(entry.track_start_seconds),
            trajectory_path=trajectory_path.resolve(),
            threshold=threshold,
            tracks=track_rows,
            window_seconds=config.encoder.window_seconds,
            hop_seconds=effective_hop,
        ),
    )

    return {
        "sample_id": entry.sample_id,
        "split": entry.split,
        "song_dir": str(song_dir),
        "sample_rate": wav.sample_rate,
        "duration_seconds": float(audio_seconds),
        "track_start_seconds": float(entry.track_start_seconds),
        "centers_total": len(centers),
        "centers_valid": int(sum(1 for value in predictions.center_valid if value)),
        "tracks": len(trajectory.tracks),
        "slot_count": int(predictions.slots),
        "evaluation": evaluation,
        "active_spans_absolute_seconds": {
            row["track_id"]: row["active_spans_absolute_seconds"] for row in track_rows
        },
        "session": str(song_dir / "session.json"),
        "trajectory": str(trajectory_path),
        "prediction_dir": str(prediction_dir),
    }


# --------------------------------------------------------------------------- #
# viewer / node validation
# --------------------------------------------------------------------------- #


def validate_with_viewer(trajectory_path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Validate one trajectory with the real viewer protocol module (Node)."""

    node = shutil.which("node")
    if not node:
        return {"status": "skipped", "reason": "node executable not found"}
    protocol_path = repo_root / "viewer" / "js" / "protocol.js"
    if not protocol_path.is_file():
        return {"status": "skipped", "reason": f"missing {protocol_path}"}
    script = (
        "import { readFileSync } from 'node:fs';\n"
        "import { pathToFileURL } from 'node:url';\n"
        "const protocol = await import(pathToFileURL(process.argv[1]).href);\n"
        "const document = protocol.parseTrajectoryText(readFileSync(process.argv[2], 'utf8'));\n"
        "console.log(JSON.stringify({ schemaVersion: document.schemaVersion,"
        " kind: document.kind, tracks: document.tracks.length,"
        " dataKind: document.provenance.dataKind }));\n"
    )
    completed = subprocess.run(
        [node, "--input-type=module", "-e", script, str(protocol_path), str(trajectory_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip()[-2000:],
        }
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:  # pragma: no cover - defensive
        return {"status": "failed", "reason": f"unparsable node output: {error}"}
    return {"status": "ok", "viewer": payload, "protocol": str(protocol_path)}


# --------------------------------------------------------------------------- #
# smoke mode (CPU fake encoder, tiny)
# --------------------------------------------------------------------------- #


def cmd_smoke(args: argparse.Namespace) -> int:
    from aat.training.trainer import train_from_config

    out = Path(args.out).resolve()
    _archive_if_needed(out, overwrite=args.overwrite, kind="demo output")
    out.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else out / "dataset"
    dataset = make_smoke_dataset(dataset_root, seed=args.seed)
    data_root = Path(dataset["data_root"])
    index = DatasetIndex.load(dataset["index_path"], data_root=data_root)

    config = default_smoke_config(
        data_root=data_root,
        index_path=dataset["index_path"],
        out_dir=out / "train",
        steps=args.steps,
        seed=args.seed,
        eval_splits=("test", "val"),
    )
    print(
        "FAKE ENCODER SMOKE: dataset -> training -> checkpoint reload -> tracking; "
        "this is not a real AuT model result",
        file=sys.stderr,
    )
    resources = _start_resources(config.run.device)
    summary = train_from_config(
        config,
        out_dir=out / "train",
        log=lambda message: print(f"  {message}", file=sys.stderr),
    )
    inference = load_head_from_checkpoint(summary.checkpoint_path)

    entries = sorted(index.samples_for_split("test"), key=lambda item: item.sample_id)
    if not entries:
        entries = sorted(index.samples_for_split("val"), key=lambda item: item.sample_id)
    if not entries:
        raise TrainingError("smoke dataset has no held-out test/val song to serialize")
    song = run_song_artifacts(
        entries[0],
        data_root=data_root,
        inference=inference,
        config=config,
        out_dir=out / "songs",
        run_id=f"demo-smoke-{config.run.seed}",
        git=git_provenance(_REPO_ROOT),
    )

    viewer = validate_with_viewer(Path(song["trajectory"]), repo_root=_REPO_ROOT)
    report = {
        "schema": SCHEMA_REPORT,
        "mode": MODE_SMOKE,
        "result_kind": SMOKE_RESULT_KIND,
        "created_at_utc": utc_now(),
        "run_id": f"demo-smoke-{config.run.seed}",
        "provenance": {
            "git": git_provenance(_REPO_ROOT),
            "uv_lock_sha256": uv_lock_sha256(_REPO_ROOT),
            "config_sha256": config.config_sha256(),
            "config_identity_sha256": config.identity_sha256(),
            "dataset": {
                "index_path": str(Path(dataset["index_path"]).resolve()),
                "index_sha256": index_file_sha256(dataset["index_path"]),
                "fingerprint_digest": dataset_fingerprint(index)["digest"],
                "synthetic": True,
                "split_counts": dataset["split_counts"],
                "leak_free": dataset["leak_free"],
                "empty_splits": dataset["empty_splits"],
            },
            "checkpoint": {
                "path": str(Path(summary.checkpoint_path).resolve()),
                "sha256": file_sha256(summary.checkpoint_path),
                "step": summary.last_step,
                "config_sha256": summary.config_sha256,
            },
            "encoder": {
                "identity": inference.encoder.identity(),
                "provenance": inference.encoder.provenance(),
            },
            "pinned_policy": _pinned_policy(config),
            "seeds": {"run_seed": config.run.seed, "sampler_scheme": "seed-sequence-v1"},
        },
        "training": summary.to_dict(),
        "resources": _finish_resources(resources),
        "songs": [song],
        "viewer_validation": viewer,
        "failures_and_limits": [
            "fake encoder: pipeline wiring evidence only; no source-identity or activity-quality claim",
            "single synthetic held-out song; metrics here are not a model result",
        ],
    }
    _finalize_report(out, report)
    print(dumps_json({"status": "ok", "mode": MODE_SMOKE, "out": str(out), **{
        "checkpoint": report["provenance"]["checkpoint"]["path"],
        "trajectory": song["trajectory"],
        "session": song["session"],
        "viewer_validation": viewer.get("status"),
    }}))
    return 0


# --------------------------------------------------------------------------- #
# synthetic mode (real checkpoint, held-out synthesized songs)
# --------------------------------------------------------------------------- #


def _metrics_delta(left: Any, right: Any) -> float | None:
    """Max absolute numeric delta of two metric trees; ``None`` on shape/type mismatch.

    Used to compare the canonical ``evaluate_split`` metrics with the metrics
    recomputed from the saved trajectory.  BLAS backends may differ by a few
    ULPs when the same window lands in a different encoder chunk, so the check
    records the delta instead of relying on exact float equality.
    """

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return None
        worst = 0.0
        for key in left:
            delta = _metrics_delta(left[key], right[key])
            if delta is None:
                return None
            worst = max(worst, delta)
        return worst
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return None
        worst = 0.0
        for left_item, right_item in zip(left, right):
            delta = _metrics_delta(left_item, right_item)
            if delta is None:
                return None
            worst = max(worst, delta)
        return worst
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if left == right else None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else None


def _song_failure_notes(canonical: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    for song in canonical["songs"]:
        metrics = song["metrics"]
        notes.append(
            f"{song['sample_id']}: F1={metrics['f1']} P={metrics['precision']} "
            f"R={metrics['recall']} TP/FP/FN={metrics['true_positives']}/"
            f"{metrics['false_positives']}/{metrics['false_negatives']} "
            f"id_switch={metrics['id_switch_count']} "
            f"count_abs_error={metrics['source_count_abs_error']} "
            f"boundary_onset_mae={metrics['boundary_onset_mae']} "
            f"boundary_missed_segments={metrics['boundary_segments_missed']}"
        )
        for source_id, row in metrics["per_source"].items():
            if row["track_id"] is None:
                notes.append(
                    f"  {song['sample_id']}/{source_id}: no track mapped; "
                    f"{row['false_negatives']} active frames are false negatives"
                )
            elif row["f1"] is not None and row["f1"] < 0.5:
                notes.append(
                    f"  {song['sample_id']}/{source_id}->{row['track_id']}: low F1 "
                    f"{row['f1']:.3f} (FP={row['false_positives']}, FN={row['false_negatives']})"
                )
            if row["id_switch_count"]:
                notes.append(
                    f"  {song['sample_id']}/{source_id}: {row['id_switch_count']} ID switch(es) "
                    f"on unambiguous owner frames"
                )
        for track_id in metrics["unmapped_track_ids"]:
            notes.append(
                f"  {song['sample_id']}/{track_id}: predicted track never mapped to a source "
                "(over-prediction / source-count inflation)"
            )
    return notes


def cmd_synthetic(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    _archive_if_needed(out, overwrite=args.overwrite, kind="demo output")
    out.mkdir(parents=True, exist_ok=True)
    resources = _start_resources(args.device or "cuda:0")

    index_path = Path(args.index).resolve()
    index = load_verified_index(index_path, data_root=args.data_root)
    data_root = index.resolve_data_root(index_path=index_path, data_root=args.data_root)
    inference = load_head_from_checkpoint(
        args.checkpoint, model_dir=args.model_dir, device=args.device
    )
    config = inference.config
    resources["device"] = config.run.device
    git = git_provenance(_REPO_ROOT)

    entries = sorted(index.samples_for_split(args.split), key=lambda item: item.sample_id)
    if args.songs:
        requested = [value.strip() for value in args.songs.split(",") if value.strip()]
        by_id = {entry.sample_id: entry for entry in entries}
        missing = [value for value in requested if value not in by_id]
        if missing:
            raise TrainingError(f"--songs: not in split {args.split!r}: {missing}")
        deduped: list[Any] = []
        seen_ids: set[str] = set()
        for value in requested:
            if value not in seen_ids:
                seen_ids.add(value)
                deduped.append(by_id[value])
        entries = deduped  # fixed user order, no model selection
    if args.max_songs is not None:
        entries = entries[: int(args.max_songs)]
    if not entries:
        raise TrainingError(f"synthetic evaluation: split {args.split!r} has no songs")

    run_id = f"demo-aut-{args.split}-{config.run.seed}"
    print(
        f"REAL FROZEN AuT: checkpoint step {inference.step} on {len(entries)} held-out "
        f"{args.split} song(s): {[entry.sample_id for entry in entries]}",
        file=sys.stderr,
    )
    canonical = evaluate_split(
        index=index,
        data_root=data_root,
        split=args.split,
        inference=inference,
        config=config,
        max_songs=len(entries),
        git_commit=git.get("sha") if isinstance(git.get("sha"), str) else None,
    )

    artifact_songs: list[dict[str, Any]] = []
    for entry in entries:
        song = run_song_artifacts(
            entry,
            data_root=data_root,
            inference=inference,
            config=config,
            out_dir=out / "songs",
            run_id=run_id,
            git=git,
        )
        # Canonical per-song metrics from evaluate_split for this sample id.
        per_song = next(
            (row for row in canonical["songs"] if row["sample_id"] == entry.sample_id), None
        )
        song["canonical_metrics"] = per_song["metrics"] if per_song else None
        song["canonical_baselines"] = per_song["baselines"] if per_song else None
        delta = (
            _metrics_delta(per_song["metrics"], song["evaluation"])
            if per_song is not None
            else None
        )
        song["canonical_metrics_max_abs_delta"] = delta
        song["canonical_metrics_match"] = delta is not None and delta <= 1e-6
        artifact_songs.append(song)

    viewer = validate_with_viewer(
        Path(artifact_songs[0]["trajectory"]), repo_root=_REPO_ROOT
    )
    report = {
        "schema": SCHEMA_REPORT,
        "mode": MODE_SYNTHETIC if config.encoder.mode == "aut" else MODE_SYNTHETIC_FAKE,
        "result_kind": _mode_result_kind(
            config,
            real=AUT_RESULT_KIND,
            fake_note=SMOKE_RESULT_KIND,
        ),
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "provenance": {
            "git": git,
            "uv_lock_sha256": uv_lock_sha256(_REPO_ROOT),
            "config_sha256": config.config_sha256(),
            "dataset": {
                "index_path": str(index_path),
                "index_sha256": index_file_sha256(index_path),
                "fingerprint_digest": dataset_fingerprint(index)["digest"],
                "plan": dict(index.plan),
                "split_counts": dict(index.summary["split_counts"]),
                "leak_free": bool(index.summary["leak_free"]),
            },
            "checkpoint": {
                "path": str(Path(args.checkpoint).resolve()),
                "sha256": file_sha256(args.checkpoint),
                "step": inference.step,
                "config_sha256": inference.describe()["config_sha256"],
            },
            "encoder": {
                "identity": inference.encoder.identity(),
                "provenance": inference.encoder.provenance(),
            },
            "pinned_policy": _pinned_policy(config),
            "seeds": {"run_seed": config.run.seed},
        },
        "canonical": canonical,
        "resources": _finish_resources(resources),
        "songs": artifact_songs,
        "viewer_validation": viewer,
        "failures_and_limits": _song_failure_notes(canonical)
        + [
            "single short self-synthesized held-out split (2 songs max); metrics variance unknown",
            "boundary/identity diagnostics are limited by ambiguous ownership frames when sources overlap",
            f"canonical evaluate_split protocol threshold={config.eval.threshold} fixed; no tuning on this split",
        ],
    }
    _finalize_report(out, report)
    print(
        dumps_json(
            {
                "status": "ok",
                "mode": report["mode"],
                "out": str(out),
                "gpu": config.run.device,
                "songs": [song["sample_id"] for song in artifact_songs],
                "micro": canonical["micro"],
                "baselines_micro": canonical["baselines_micro"],
                "canonical_match": [song["canonical_metrics_match"] for song in artifact_songs],
                "viewer_validation": viewer.get("status"),
            }
        )
    )
    return 0


# --------------------------------------------------------------------------- #
# audio mode (real checkpoint, local user clip; no ground truth)
# --------------------------------------------------------------------------- #


def _tool_version(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(list(command), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    line = (completed.stdout or completed.stderr or "").splitlines()
    return line[0].strip() if line else None


def _probe_duration(ffprobe: str, source: Path) -> float:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise TrainingError(
            f"ffprobe failed for {source}: {completed.stderr.strip()[:500]}"
        )
    try:
        value = float(completed.stdout.strip())
    except ValueError as error:
        raise TrainingError(f"ffprobe returned an unparsable duration: {completed.stdout!r}") from error
    if not (value > 0.0):
        raise TrainingError(f"ffprobe duration is not positive: {value}")
    return value


def _decode_segment(
    *,
    source: Path,
    target: Path,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise TrainingError("ffmpeg is required to decode --audio; install it or provide a WAV")
    if not source.is_file():
        raise TrainingError(f"--audio does not exist: {source}")
    ffprobe = shutil.which("ffprobe")
    source_duration = _probe_duration(ffprobe, source) if ffprobe else None
    if source_duration is not None:
        end = start_seconds + duration_seconds
        if end > source_duration + 1e-3:
            raise TrainingError(
                f"--audio is too short: requested {start_seconds:.3f}s..{end:.3f}s but the "
                f"source duration is {source_duration:.3f}s; refusing to silently substitute"
            )
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if completed.returncode != 0:
        raise TrainingError(
            f"ffmpeg decode failed (exit {completed.returncode}): "
            f"{completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else ''}"
        )
    return {
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "source_duration_seconds": source_duration,
        "segment": {
            "start_seconds": start_seconds,
            "duration_seconds": duration_seconds,
            "end_seconds": start_seconds + duration_seconds,
            "origin_semantics": "original-track absolute time of audio[0]",
        },
        "decode_command": command,
        "ffmpeg_version": _tool_version([ffmpeg, "-version"]),
        "ffprobe_version": _tool_version([ffprobe, "-version"]) if ffprobe else None,
        "decoded_path": str(target),
        "decoded_sha256": file_sha256(target),
        "sample_rate": sample_rate,
    }


def cmd_audio(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    _archive_if_needed(out, overwrite=args.overwrite, kind="demo output")
    out.mkdir(parents=True, exist_ok=True)
    resources = _start_resources(args.device or "cuda:0")

    source = Path(args.audio).resolve()
    label = args.label or re.sub(r"[^0-9A-Za-z._-]+", "-", source.stem) or "clip"
    clip_dir = out / label
    clip_dir.mkdir(parents=True, exist_ok=True)

    inference = load_head_from_checkpoint(
        args.checkpoint, model_dir=args.model_dir, device=args.device
    )
    config = inference.config
    resources["device"] = config.run.device

    if args.duration_seconds is not None:
        duration = float(args.duration_seconds)
    else:
        ffprobe = shutil.which("ffprobe")
        total = _probe_duration(ffprobe, source) if ffprobe else None
        if total is None:
            raise TrainingError("--duration-seconds is required when ffprobe is unavailable")
        duration = total - float(args.start_seconds)
    if duration <= 0.0:
        raise TrainingError(f"--duration-seconds must be > 0, got {duration}")

    decoded = clip_dir / "audio-segment.wav"
    decode = _decode_segment(
        source=source,
        target=decoded,
        start_seconds=float(args.start_seconds),
        duration_seconds=duration,
        sample_rate=int(args.sample_rate),
    )
    wav = read_wav(decoded)
    mono = wav.samples.mean(axis=1) if wav.channels > 1 else wav.samples[:, 0]
    centers = [float(value) for value in center_times(
        wav.frames / wav.sample_rate,
        float(args.hop_seconds),
        origin_seconds=float(args.start_seconds),
    )]
    provenance = _run_provenance(
        mode=config.encoder.mode,
        config=config,
        inference=inference,
        git=git_provenance(_REPO_ROOT),
        run_id=f"demo-audio-{label}",
    )
    predictions = inference.predict_at_times(
        mono,
        centers,
        sample_rate=wav.sample_rate,
        origin_seconds=float(args.start_seconds),
        sample_id=label,
        hop_seconds=float(args.hop_seconds),
        provenance=provenance,
    )
    trajectory = associate_sequence(
        PredictionSequence.from_prediction(predictions),
        AudioSpan(
            duration_seconds=float(wav.frames / wav.sample_rate),
            track_start_seconds=float(args.start_seconds),
        ),
        provenance,
        config=tracking_config_from(config),
        sample_id=label,
    )
    prediction_dir = clip_dir / "prediction"
    predictions.save(prediction_dir)
    trajectory_path = trajectory.save(clip_dir / "trajectory.json")

    threshold = config.eval.threshold
    track_rows: list[dict[str, Any]] = []
    for track in trajectory.tracks:
        values = list(track.activity)
        active = sum(1 for value in values if value >= threshold)
        spans = _active_spans(track.center_times, values, threshold)
        track_rows.append(
            {
                "track_id": track.track_id,
                "points": len(values),
                "active": active,
                "first": f"{spans[0][0]:.3f}" if spans else "-",
                "last": f"{spans[-1][1]:.3f}" if spans else "-",
                "maximum": f"{max(values):.4f}" if values else "-",
                "active_spans_absolute_seconds": spans,
            }
        )
    session = {
        "schema": SCHEMA_SESSION,
        "run_id": f"demo-audio-{label}",
        "sample_id": label,
        "mode": config.encoder.mode,
        "data_kind": provenance.data_kind,
        "warning": (
            "fake encoder: engineering pipeline evidence only, not a real model result"
            if config.encoder.mode == "fake"
            else "real frozen AuT checkpoint inference; no ground truth for user music"
        ),
        "audio": {
            "path": str(decoded.resolve()),
            "source_path": decode["source_path"],
            "source_sha256": decode["source_sha256"],
            "segment": decode["segment"],
            "duration_seconds": float(wav.frames / wav.sample_rate),
            "track_start_seconds": float(args.start_seconds),
            "sha256": decode["decoded_sha256"],
            "sample_rate": wav.sample_rate,
        },
        "trajectory": str(trajectory_path.resolve()),
        "prediction_dir": str(prediction_dir.resolve()),
        "tracks": track_rows,
        "objective_checks": {
            "centers_total": len(centers),
            "centers_valid": int(sum(1 for value in predictions.center_valid if value)),
            "centers_padded_edges": int(sum(1 for value in predictions.center_valid if not value)),
            "tracks": len(trajectory.tracks),
            "hop_seconds": float(args.hop_seconds),
            "window_seconds": config.encoder.window_seconds,
            "activity_threshold": threshold,
        },
        "human_listening": {
            "status": "pending",
            "instructions": "edit manual_inspection.md after listening; keep the original timestamps",
        },
        "privacy": "audio stays local on this machine; the viewer uploads nothing",
    }
    _write_json(clip_dir / "session.json", session)
    _write_text(
        clip_dir / "manual_inspection.md",
        _manual_inspection(
            label=f"{label} (user music, {config.encoder.mode})",
            mode=config.encoder.mode,
            audio_path=decoded.resolve(),
            audio_duration=float(wav.frames / wav.sample_rate),
            track_start_seconds=float(args.start_seconds),
            trajectory_path=trajectory_path.resolve(),
            threshold=threshold,
            tracks=track_rows,
            window_seconds=config.encoder.window_seconds,
            hop_seconds=float(args.hop_seconds),
            extra_notes=(
                f"source file sha256: {decode['source_sha256']}",
                f"decoded segment sha256: {decode['decoded_sha256']}",
                "no ground truth exists for this clip; do not report an overall accuracy",
            ),
        ),
    )

    viewer = validate_with_viewer(trajectory_path, repo_root=_REPO_ROOT)
    report = {
        "schema": SCHEMA_REPORT,
        "mode": MODE_AUDIO if config.encoder.mode == "aut" else MODE_AUDIO_FAKE,
        "result_kind": _mode_result_kind(
            config,
            real=AUT_RESULT_KIND,
            fake_note=SMOKE_RESULT_KIND,
        ),
        "created_at_utc": utc_now(),
        "run_id": f"demo-audio-{label}",
        "provenance": {
            "git": git_provenance(_REPO_ROOT),
            "uv_lock_sha256": uv_lock_sha256(_REPO_ROOT),
            "config_sha256": config.config_sha256(),
            "checkpoint": {
                "path": str(Path(args.checkpoint).resolve()),
                "sha256": file_sha256(args.checkpoint),
                "step": inference.step,
                "config_sha256": inference.describe()["config_sha256"],
            },
            "encoder": {
                "identity": inference.encoder.identity(),
                "provenance": inference.encoder.provenance(),
            },
            "pinned_policy": _pinned_policy(config),
            "input_audio": decode,
        },
        "resources": _finish_resources(resources),
        "songs": [
            {
                "sample_id": label,
                "duration_seconds": float(wav.frames / wav.sample_rate),
                "track_start_seconds": float(args.start_seconds),
                "centers_total": len(centers),
                "centers_valid": int(sum(1 for value in predictions.center_valid if value)),
                "tracks": len(trajectory.tracks),
                "objective_checks": session["objective_checks"],
                "active_spans_absolute_seconds": {
                    row["track_id"]: row["active_spans_absolute_seconds"] for row in track_rows
                },
                "trajectory": str(trajectory_path),
                "session": str(clip_dir / "session.json"),
            }
        ],
        "viewer_validation": viewer,
        "failures_and_limits": [
            "user music has no ground truth: no precision/recall/F1/identity metric is reported",
            "human listening is pending; objective inference checks are not auditory judgements",
            f"coarser center hop {float(args.hop_seconds):g}s than the synthetic 0.02s grid "
            "(explicit budget choice, fixed before listening)",
            "fixed checkpoint thresholds/config; nothing is tuned on this clip",
        ],
    }
    _finalize_report(out, report)
    print(
        dumps_json(
            {
                "status": "ok",
                "mode": report["mode"],
                "out": str(out),
                "label": label,
                "segment": decode["segment"],
                "tracks": len(trajectory.tracks),
                "centers_valid": int(sum(1 for value in predictions.center_valid if value)),
                "human_listening": "pending",
                "viewer_validation": viewer.get("status"),
            }
        )
    )
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - argparse internals
        self.print_usage(sys.stderr)
        raise TrainingError(f"{self.prog}: {message}")


def _load_config_payload(path: str | None) -> dict[str, Any]:
    """Load a flat TOML overlay of long option names -> values (optional)."""

    if not path:
        return {}
    source = Path(path)
    if not source.is_file():
        raise TrainingError(f"--config {source}: file does not exist")
    with source.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, Mapping):
        raise TrainingError(f"--config {source}: expected a TOML table")
    return dict(payload)


def _subparser_dests(parser: argparse.ArgumentParser) -> set[str]:
    return {
        action.dest
        for action in parser._actions
        if action.dest not in ("help", "func") and action.dest != argparse.SUPPRESS
    }


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="demo_pipeline.py",
        description=(
            "End-to-end config/data-or-checkpoint -> E/P -> stable tracks -> local viewer "
            "artifacts for issue #11; fake CPU smoke and real frozen-AuT modes stay separated."
        ),
    )
    parser.add_argument("--config", default=None, help="flat TOML overlay for the subcommand")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="CPU fake-encoder engineering smoke")
    smoke.add_argument("--out", default=None, help="output root (required unless --config)")
    smoke.add_argument("--steps", type=int, default=3, help="tiny training budget (default 3)")
    smoke.add_argument("--seed", type=int, default=20260929)
    smoke.add_argument("--dataset-root", default=None, help="override the synthetic corpus root")
    smoke.add_argument("--overwrite", action="store_true", help="archive an existing output root")
    smoke.set_defaults(func=cmd_smoke)

    synthetic = subparsers.add_parser(
        "synthetic", help="real frozen AuT checkpoint on held-out synthesized songs"
    )
    synthetic.add_argument("--checkpoint", default=None, help="training checkpoint.pt (required)")
    synthetic.add_argument("--index", default=None, help="verified dataset index.json (required)")
    synthetic.add_argument("--data-root", default=None, help="dataset root override")
    synthetic.add_argument("--split", default="test", choices=("train", "val", "test"))
    synthetic.add_argument(
        "--songs",
        default=None,
        help="fixed comma-separated sample ids (default: all in split, sorted)",
    )
    synthetic.add_argument("--max-songs", type=int, default=None)
    synthetic.add_argument("--model-dir", default=None, help="AuT checkpoint directory override")
    synthetic.add_argument("--device", default=None, help="torch device (e.g. cuda:0)")
    synthetic.add_argument("--out", default=None, help="output root (required unless --config)")
    synthetic.add_argument("--overwrite", action="store_true")
    synthetic.set_defaults(func=cmd_synthetic)

    audio = subparsers.add_parser("audio", help="real frozen AuT checkpoint on a local clip")
    audio.add_argument("--audio", default=None, help="local audio file (required)")
    audio.add_argument("--checkpoint", default=None, help="training checkpoint.pt (required)")
    audio.add_argument("--out", default=None, help="output root (required unless --config)")
    audio.add_argument("--start-seconds", type=float, default=0.0)
    audio.add_argument("--duration-seconds", type=float, default=None)
    audio.add_argument("--hop-seconds", type=float, default=0.02)
    audio.add_argument("--sample-rate", type=int, default=44100)
    audio.add_argument("--label", default=None, help="output label (default: decoded from file name)")
    audio.add_argument("--model-dir", default=None, help="AuT checkpoint directory override")
    audio.add_argument("--device", default=None, help="torch device (e.g. cpu)")
    audio.add_argument("--overwrite", action="store_true")
    audio.set_defaults(func=cmd_audio)

    return parser


def _subcommand_parsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    subcommands = _subcommand_parsers(parser)
    # Apply the optional flat TOML overlay as subcommand defaults *before*
    # parsing, so explicit CLI flags always win over the config file.
    preparser = argparse.ArgumentParser(add_help=False)
    preparser.add_argument("--config", default=None)
    known_args, _ = preparser.parse_known_args(argv)
    try:
        payload = _load_config_payload(known_args.config)
        if payload:
            dests = {name: _subparser_dests(sub) for name, sub in subcommands.items()}
            union = set().union(*dests.values())
            unknown = sorted(set(payload) - union)
            if unknown:
                raise TrainingError(
                    f"--config {known_args.config}: unknown key(s) {unknown}; "
                    f"known: {sorted(union)}"
                )
            for name, sub in subcommands.items():
                applicable = {key: value for key, value in payload.items() if key in dests[name]}
                if applicable:
                    sub.set_defaults(**applicable)
        args = parser.parse_args(argv)
        if getattr(args, "command", None) == "smoke" and not args.out:
            raise TrainingError("smoke: --out is required (or set 'out' in --config)")
        if getattr(args, "command", None) == "synthetic":
            for name in ("checkpoint", "index", "out"):
                if not getattr(args, name):
                    raise TrainingError(f"synthetic: --{name.replace('_', '-')} is required")
        if getattr(args, "command", None) == "audio":
            for name in ("audio", "checkpoint", "out"):
                if not getattr(args, name):
                    raise TrainingError(f"audio: --{name.replace('_', '-')} is required")
        return int(args.func(args))
    except TrainingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (
        ContractError,
        DatasetError,
        EvaluationError,
        LabelError,
        TrackingError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
