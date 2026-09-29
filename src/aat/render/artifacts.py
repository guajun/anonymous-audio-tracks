"""Protocol documents and digests for one rendered sample.

The renderer writes the ``rendered`` stage only: ``sources.json``,
``controls.json`` and ``manifest.json`` are built through the frozen
``aat.contracts`` dataclasses, so a renderer bug cannot invent a parallel
schema.  ``activity.json``/``activity.npz`` belong to the labelling issue and
are deliberately not created here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from aat.contracts.documents import SampleManifest, SourceEntry, SourceRegistry
from aat.contracts.version import SCHEMA_VERSION

from .config import RenderConfig
from .version import RENDER_VERSION

MIX_FILENAME = "mix.wav"
STEMS_DIR = "stems"
DRY_DIR = "dry"
SOURCES_FILENAME = "sources.json"
CONTROLS_FILENAME = "controls.json"
MANIFEST_FILENAME = "manifest.json"
REPORT_FILENAME = "render_report.json"
SURGE_REPORT_FILENAME = "surge_probe.json"


def sha256_file(path: str | Path) -> str:
    """Lower-case sha256 hex digest of a file's bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(data: Any) -> str:
    """Stable JSON text used for digests (sorted keys, ASCII-safe)."""

    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_json(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def mix_relative_path() -> str:
    return MIX_FILENAME


def stem_relative_path(source_id: str) -> str:
    return f"{STEMS_DIR}/{source_id}.wav"


def dry_relative_path(source_id: str) -> str:
    return f"{DRY_DIR}/{source_id}.wav"


def build_sources(config: RenderConfig) -> SourceRegistry:
    """``sources.json`` with generation-time source identities (no semantics)."""

    entries = tuple(
        SourceEntry(
            source_id=source.source_id,
            index=source.index,
            renderer="dawdreamer-sampler",
            preset_ref=source.resolved_preset_ref,
            sample_ref=source.resolved_sample_ref,
            seed=source.sample.seed,
            notes=source.notes,
        )
        for source in config.sources
    )
    return SourceRegistry(sources=entries, sample_id=config.sample_id)


def build_manifest(
    config: RenderConfig,
    *,
    digest_paths: Mapping[str, str],
    renderer_version: str,
    render_latency_seconds: float,
    tail_seconds: float,
    notes: str | None = None,
) -> SampleManifest:
    """``manifest.json`` for the ``rendered`` stage."""

    return SampleManifest(
        sample_id=config.sample_id,
        stage="rendered",
        seed=config.seed,
        sample_rate=config.sample_rate,
        duration_seconds=config.total_seconds,
        track_start_seconds=config.track_start_seconds,
        groups={
            "composition": config.composition,
            "preset": config.presets,
            "sample_origin": config.sample_origins,
        },
        mix_path=mix_relative_path(),
        stem_paths={source.source_id: stem_relative_path(source.source_id) for source in config.sources},
        sources_path=SOURCES_FILENAME,
        controls_path=CONTROLS_FILENAME,
        versions={
            "renderer": renderer_version,
            "protocol": SCHEMA_VERSION,
            "aat_render": RENDER_VERSION,
        },
        content_sha256=dict(digest_paths),
        render_latency_seconds=render_latency_seconds,
        tail_seconds=tail_seconds,
        notes=notes,
    )


def required_digest_paths(source_ids: Sequence[str]) -> tuple[str, ...]:
    """Relative artifact paths that ``manifest.content_sha256`` must cover."""

    paths = [mix_relative_path(), SOURCES_FILENAME, CONTROLS_FILENAME]
    paths.extend(stem_relative_path(source_id) for source_id in source_ids)
    return tuple(paths)


__all__ = [
    "CONTROLS_FILENAME",
    "DRY_DIR",
    "MANIFEST_FILENAME",
    "MIX_FILENAME",
    "REPORT_FILENAME",
    "SOURCES_FILENAME",
    "STEMS_DIR",
    "SURGE_REPORT_FILENAME",
    "build_manifest",
    "build_sources",
    "canonical_json",
    "dry_relative_path",
    "mix_relative_path",
    "required_digest_paths",
    "sha256_file",
    "sha256_json",
    "stem_relative_path",
]
