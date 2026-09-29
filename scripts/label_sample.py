#!/usr/bin/env python3
"""Generate center-time activity labels for rendered stems (issue #4).

Reads a protocol sample directory (``manifest.json`` / ``sources.json`` /
``controls.json`` and the per-source WAV stems), verifies every recorded
SHA-256 digest, derives a configurable short-time energy label per source and
writes the ``activity.json`` + ``activity.npz`` pair plus a readable
``activity_summary.json``.  By default the manifest is then advanced to
``stage = labeled`` with both label digests recorded.

Activity is derived from the stem audio only.  ``controls.json`` is loaded and
validated as reference data but never changes the labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aat.contracts import (  # noqa: E402
    ActivityData,
    Controls,
    SampleManifest,
    SourceRegistry,
    dump_json,
)
from aat.labels import CONFIG_VERSION, LabelConfig, LabelError, label_stems, read_wav  # noqa: E402

SUMMARY_VERSION = "activity-summary-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Label rendered per-source stems with center-time activity and update "
            "the manifest to stage=labeled."
        )
    )
    parser.add_argument("sample_dir", type=Path, help="sample directory with manifest.json")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "LabelConfig file (.json or .toml). Flat LabelConfig keys are read; "
            "when a [task] table exists, task.window_seconds and task.hop_seconds "
            "map to center_window_seconds and hop_seconds."
        ),
    )
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--summary-name", default="activity_summary.json")
    parser.add_argument(
        "--duration-tolerance-seconds",
        type=float,
        default=0.01,
        help="allowed |stem duration - manifest duration| in seconds",
    )
    parser.add_argument(
        "--no-manifest-update",
        action="store_true",
        help="write label artifacts but leave manifest.json untouched",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_label_config(path: Path) -> LabelConfig:
    """Load a LabelConfig from a JSON or TOML mapping (unknown keys ignored)."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LabelError(f"{path}: cannot read config: {exc}") from exc
    if path.suffix.lower() == ".toml":
        import tomllib

        try:
            data: Any = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise LabelError(f"{path}: invalid TOML: {exc}") from exc
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LabelError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise LabelError(f"{path}: expected a mapping at the top level")

    known = set(LabelConfig.field_names())
    overrides: dict[str, Any] = {}
    task = data.get("task")
    if isinstance(task, Mapping):
        if "window_seconds" in task:
            overrides["center_window_seconds"] = task["window_seconds"]
        if "hop_seconds" in task:
            overrides["hop_seconds"] = task["hop_seconds"]
    labels = data.get("labels")
    if isinstance(labels, Mapping):
        for key, value in labels.items():
            if key in known:
                overrides[key] = value
    for key in known:
        if key in data:
            overrides[key] = data[key]
    return LabelConfig(**overrides)


def _verify_content_hashes(sample_dir: Path, manifest: SampleManifest) -> None:
    for relative in sorted(manifest.content_sha256):
        path = sample_dir / relative
        if not path.is_file():
            raise LabelError(
                f"manifest.content_sha256[{relative!r}]: file not found at {path}"
            )
        actual = sha256_file(path)
        expected = manifest.content_sha256[relative]
        if actual != expected:
            raise LabelError(
                f"manifest.content_sha256[{relative!r}]: expected {expected}, "
                f"got {actual}"
            )


def run_labeling(
    sample_dir: Path,
    *,
    config: LabelConfig | None = None,
    manifest_name: str = "manifest.json",
    summary_name: str = "activity_summary.json",
    update_manifest: bool = True,
    duration_tolerance_seconds: float = 0.01,
) -> dict[str, Any]:
    """Label one sample directory and return a report for printing."""

    config = config if config is not None else LabelConfig()
    if duration_tolerance_seconds < 0.0:
        raise LabelError("duration_tolerance_seconds: must be >= 0")

    manifest = SampleManifest.load(sample_dir / manifest_name)
    sources = SourceRegistry.load(sample_dir / manifest.sources_path)
    if sources.sample_id is not None and sources.sample_id != manifest.sample_id:
        raise LabelError(
            "sources.sample_id: "
            f"{sources.sample_id!r} does not match manifest sample_id "
            f"{manifest.sample_id!r}"
        )
    if set(manifest.stem_paths) != set(sources.source_ids):
        raise LabelError(
            "manifest.stem_paths keys must match sources.json source_ids; "
            f"manifest has {sorted(manifest.stem_paths)}, "
            f"sources has {list(sources.source_ids)}"
        )

    _verify_content_hashes(sample_dir, manifest)

    controls = Controls.load(sample_dir / manifest.controls_path)
    controls.validate_source_references(sources.source_ids)

    source_ids = sources.source_ids
    stem_samples: dict[str, Any] = {}
    for source_id in source_ids:
        wav = read_wav(sample_dir / manifest.stem_paths[source_id])
        if wav.sample_rate != manifest.sample_rate:
            raise LabelError(
                f"stem {source_id!r}: sample rate {wav.sample_rate} Hz does not match "
                f"manifest {manifest.sample_rate} Hz"
            )
        difference = abs(wav.duration_seconds - manifest.duration_seconds)
        if difference > duration_tolerance_seconds:
            raise LabelError(
                f"stem {source_id!r}: duration {wav.duration_seconds:.6f} s differs "
                f"from manifest {manifest.duration_seconds:.6f} s by more than "
                f"{duration_tolerance_seconds} s"
            )
        stem_samples[source_id] = wav.samples

    result = label_stems(
        stem_samples,
        source_ids=source_ids,
        sample_rate=manifest.sample_rate,
        duration_seconds=manifest.duration_seconds,
        track_start_seconds=manifest.track_start_seconds,
        config=config,
        sample_id=manifest.sample_id,
    )

    metadata_path, arrays_path = result.activity.save(sample_dir)
    # Re-read through the protocol validator so a bad write cannot pass silently.
    ActivityData.load(sample_dir)

    event_counts = Counter(event.event_type for event in controls.events)
    summary = {
        "summary_version": SUMMARY_VERSION,
        "generator": "aat.labels.label_stems",
        "sample_id": manifest.sample_id,
        "sample_rate": manifest.sample_rate,
        "duration_seconds": manifest.duration_seconds,
        "track_start_seconds": manifest.track_start_seconds,
        "hop_seconds": config.hop_seconds,
        "center_window_seconds": config.center_window_seconds,
        "energy_window_seconds": config.energy_window_seconds,
        "config_version": CONFIG_VERSION,
        "config_sha256": config.config_sha256(),
        "config": config.to_dict(),
        "center_count": int(result.activity.center_times.size),
        "valid_count": int(result.activity.valid.sum()),
        "source_order": list(source_ids),
        "control_events_reference": {
            "total": len(controls.events),
            "by_type": dict(sorted(event_counts.items())),
            "used_for_labels": False,
        },
        "sources": [item.to_dict() for item in result.summaries],
        "notes": [
            "Activity labels are a deterministic short-time acoustic proxy with "
            "explicit thresholds; they are not a claim about subjective audibility.",
            "controls.json is recorded for reference only and never changes the labels.",
        ],
    }
    dump_json(sample_dir / summary_name, summary)

    manifest_updated = False
    if update_manifest:
        payload = manifest.to_json_dict()
        payload["stage"] = "labeled"
        payload["activity_metadata_path"] = metadata_path.name
        payload["activity_arrays_path"] = arrays_path.name
        payload["content_sha256"][metadata_path.name] = sha256_file(metadata_path)
        payload["content_sha256"][arrays_path.name] = sha256_file(arrays_path)
        SampleManifest.from_json_dict(payload).save(sample_dir / manifest_name)
        manifest_updated = True

    return {
        "sample_dir": sample_dir,
        "source_count": len(source_ids),
        "center_count": int(result.activity.center_times.size),
        "valid_count": int(result.activity.valid.sum()),
        "metadata_name": metadata_path.name,
        "arrays_name": arrays_path.name,
        "summary_name": summary_name,
        "manifest_updated": manifest_updated,
        "summaries": result.summaries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_label_config(args.config) if args.config is not None else None
        report = run_labeling(
            args.sample_dir,
            config=config,
            manifest_name=args.manifest_name,
            summary_name=args.summary_name,
            update_manifest=not args.no_manifest_update,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
        )
    except LabelError as exc:
        print(f"label_sample: error: {exc}", file=sys.stderr)
        return 2

    print(
        f"labeled {report['source_count']} stem(s) in {report['sample_dir']}"
    )
    print(
        f"  center times:      {report['center_count']} "
        f"({report['valid_count']} valid)"
    )
    print(f"  activity metadata: {report['metadata_name']}")
    print(f"  activity arrays:   {report['arrays_name']}")
    print(f"  summary:           {report['summary_name']}")
    print(
        "  manifest:          "
        + (
            "updated to stage=labeled"
            if report["manifest_updated"]
            else "unchanged (--no-manifest-update)"
        )
    )
    for item in report["summaries"]:
        print(
            f"  {item.source_id}: active_fraction={item.active_fraction:.3f} "
            f"segments={item.segments} silent={item.silent} "
            f"low_snr={item.low_snr} ambiguous={item.ambiguous}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
