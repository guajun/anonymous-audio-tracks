#!/usr/bin/env python
"""Generate center-time activity labels for one rendered sample directory.

Usage (from the repository root, inside the uv environment)::

    uv run python scripts/label_sample.py path/to/sample_dir

The directory must contain ``manifest.json`` and ``sources.json``; stems are read
from the manifest's ``stem_paths``.  By default the manifest is rewritten in the
``labeled`` stage with both activity paths and their SHA-256 digests.  Use
``--no-manifest-update`` to leave it untouched, or ``--config overrides.json``
with a partial :class:`~aat.labels.LabelConfig` mapping.

This script reads audio files, not credentials, and writes only inside the
sample directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
from pathlib import Path
from typing import Sequence

from aat.contracts import (
    ContractError,
    SampleManifest,
    SourceRegistry,
    dump_json,
    load_json,
)
from aat.labels import LabelConfig, LabelError, WavError, label_stems, read_wav

SUMMARY_FILENAME = "activity_summary.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="label_sample.py",
        description="Label center-time activity of every stem in one rendered sample.",
    )
    parser.add_argument(
        "sample_dir", type=Path, help="directory with manifest.json/sources.json/stems"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest path (default: SAMPLE_DIR/manifest.json)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON file with LabelConfig overrides (partial mapping); "
        "all thresholds/windows are recorded in the output",
    )
    parser.add_argument(
        "--no-manifest-update",
        action="store_true",
        help="do not rewrite manifest.json into the labeled stage",
    )
    return parser.parse_args(argv)


def _load_config(path: Path | None) -> LabelConfig:
    if path is None:
        return LabelConfig()
    data = load_json(path)
    return LabelConfig.from_dict(data, allow_partial=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    sample_dir = args.sample_dir
    if not sample_dir.is_dir():
        return _fail(f"sample_dir is not a directory: {sample_dir}")
    manifest_path = args.manifest if args.manifest is not None else sample_dir / "manifest.json"
    if not manifest_path.is_file():
        return _fail(f"manifest not found: {manifest_path}")

    try:
        manifest = SampleManifest.load(manifest_path)
        sources = SourceRegistry.load(sample_dir / manifest.sources_path)
    except ContractError as exc:
        return _fail(f"invalid renderer document: {exc}")
    except OSError as exc:
        return _fail(f"cannot read renderer document: {exc}")

    if tuple(manifest.stem_paths.keys()) != sources.source_ids:
        return _fail(
            "manifest.stem_paths keys must match sources.json source_ids in order: "
            f"{sorted(manifest.stem_paths)} vs {list(sources.source_ids)}"
        )

    try:
        config = _load_config(args.config)
    except (ContractError, LabelError, OSError) as exc:
        return _fail(f"invalid config {args.config}: {exc}")

    stems: dict[str, object] = {}
    for source_id in sources.source_ids:
        stem_path = sample_dir / manifest.stem_paths[source_id]
        try:
            audio, sample_rate = read_wav(stem_path)
        except WavError as exc:
            return _fail(str(exc))
        if sample_rate != manifest.sample_rate:
            return _fail(
                f"{stem_path}: sample rate {sample_rate} Hz does not match "
                f"manifest.sample_rate {manifest.sample_rate} Hz; resampling is not "
                "part of the labeler"
            )
        stems[source_id] = audio

    try:
        result = label_stems(
            stems,
            manifest.sample_rate,
            duration_seconds=manifest.duration_seconds,
            origin_seconds=manifest.track_start_seconds,
            config=config,
            sample_id=manifest.sample_id,
        )
    except LabelError as exc:
        return _fail(f"labeling failed: {exc}")

    metadata_path, arrays_path = result.activity.save(sample_dir)
    summary_path = dump_json(sample_dir / SUMMARY_FILENAME, result.summary)

    if not args.no_manifest_update:
        content = dict(manifest.content_sha256)
        content[metadata_path.name] = _sha256_file(metadata_path)
        content[arrays_path.name] = _sha256_file(arrays_path)
        versions = dict(manifest.versions)
        versions["labeler"] = config.labeler_version
        updated = dataclasses.replace(
            manifest,
            stage="labeled",
            activity_metadata_path=metadata_path.name,
            activity_arrays_path=arrays_path.name,
            content_sha256=content,
            versions=versions,
        )
        updated.save(manifest_path)

    print(f"labeled {len(stems)} stem(s) in {sample_dir}")
    print(f"  activity metadata: {metadata_path.name}")
    print(f"  activity arrays:   {arrays_path.name}")
    print(f"  summary:           {summary_path.name}")
    for source in result.summary["sources"]:
        print(
            f"  {source['source_id']}: active_fraction={source['active_fraction']:.3f} "
            f"segments={source['n_segments']} silent={source['silent']} "
            f"low_snr={source['low_snr']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
