#!/usr/bin/env python3
"""Build, verify, summarize and sample a dataset index (issue #6).

The CLI scans a data root for labeled protocol samples (``manifest.json`` with
``stage = "labeled"``), validates manifests/sources/controls/activity plus every
recorded sha256, groups samples by composition/preset/sample_origin and assigns
whole groups to train/val/test with an explicit seed.  The resulting JSON index
stores data-root-relative sample paths, artifact digests, the split seed and the
actual (not requested) split ratios, including any oversized connected component
that could not fill a split.

Subcommands::

    build   --data-root DATA --out INDEX [--seed N] [--ratios 0.8,0.1,0.1] ...
    verify  --index INDEX [--data-root DATA]
    summary --index INDEX [--data-root DATA] [--samples]
    batch   --index INDEX [--data-root DATA] --split train [--items N] ...

Exit codes: ``0`` success, ``1`` verification problems, ``2`` invalid input or
unbuildable dataset.  Generated indexes and batch dumps belong in ignored
directories (``runs/``); no absolute data path is written into the index.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aat.contracts import DEFAULT_SLOTS, DEFAULT_WINDOW_SECONDS  # noqa: E402
from aat.data import (  # noqa: E402
    DEFAULT_DURATION_TOLERANCE_SECONDS,
    DatasetError,
    DatasetIndex,
    build_dataset_index,
    sample_batch,
    verify_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Index labeled protocol samples, split by shared composition/preset/"
            "sample_origin groups and sample same-song multi-center batches."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="scan, validate, split and write an index")
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True, help="output index JSON path")
    build.add_argument("--seed", type=int, default=20260929, help="explicit split seed")
    build.add_argument(
        "--ratios",
        default="0.8,0.1,0.1",
        help="train,val,test ratios (normalized); defaults to 0.8,0.1,0.1",
    )
    build.add_argument("--slots", type=int, default=DEFAULT_SLOTS, help="capacity K")
    build.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="model center window; must match the labeled center window",
    )
    build.add_argument(
        "--duration-tolerance-seconds",
        type=float,
        default=DEFAULT_DURATION_TOLERANCE_SECONDS,
        help="allowed |mix duration - manifest duration| in seconds",
    )
    build.add_argument(
        "--no-verify-digests",
        action="store_true",
        help="skip re-hashing artifact digests (not recommended)",
    )

    verify = subparsers.add_parser("verify", help="re-check an index against its data root")
    verify.add_argument("--index", type=Path, required=True)
    verify.add_argument("--data-root", type=Path, default=None)
    verify.add_argument(
        "--duration-tolerance-seconds",
        type=float,
        default=DEFAULT_DURATION_TOLERANCE_SECONDS,
    )

    summary = subparsers.add_parser("summary", help="print the index plan and split summary")
    summary.add_argument("--index", type=Path, required=True)
    summary.add_argument("--data-root", type=Path, default=None)
    summary.add_argument(
        "--samples", action="store_true", help="also print one row per sample"
    )

    batch = subparsers.add_parser("batch", help="sample and print one multi-center batch")
    batch.add_argument("--index", type=Path, required=True)
    batch.add_argument("--data-root", type=Path, default=None)
    batch.add_argument("--split", default="train", help="train/val/test")
    batch.add_argument("--seed", type=int, default=None, help="defaults to the index seed")
    batch.add_argument("--items", type=int, default=None, help="song count (default: all)")
    batch.add_argument("--centers", type=int, default=4, help="centers per song")
    batch.add_argument(
        "--min-gap",
        type=int,
        default=0,
        help="minimum center distance in grid steps (2 forbids adjacent windows)",
    )
    batch.add_argument(
        "--include-invalid",
        action="store_true",
        help="also sample center rows whose activity.valid is False",
    )
    batch.add_argument("--window-seconds", type=float, default=None)
    batch.add_argument("--slots", type=int, default=None)
    batch.add_argument(
        "--no-verify-digests",
        action="store_true",
        help=(
            "skip re-hashing the consumed mix/activity files against the index "
            "digests (unverified consumption; the default verifies)"
        ),
    )
    batch.add_argument(
        "--skip-unusable",
        action="store_true",
        help="skip songs without usable centers instead of failing",
    )
    return parser


def _parse_ratios(text: str) -> list[float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise DatasetError(
            f"--ratios: expected three comma-separated values (train,val,test), got {text!r}"
        )
    values: list[float] = []
    for name, part in zip(("train", "val", "test"), parts):
        try:
            values.append(float(part))
        except ValueError as exc:
            raise DatasetError(f"--ratios.{name}: invalid number {part!r}") from exc
    return values


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def _load_index(args: argparse.Namespace) -> DatasetIndex:
    return DatasetIndex.load(args.index, data_root=args.data_root, verify_files=False)


def _resolve_root(index: DatasetIndex, args: argparse.Namespace) -> Path:
    return index.resolve_data_root(index_path=args.index, data_root=args.data_root)


def cmd_build(args: argparse.Namespace) -> int:
    index = build_dataset_index(
        args.data_root,
        seed=args.seed,
        ratios=_parse_ratios(args.ratios),
        slots=args.slots,
        window_seconds=args.window_seconds,
        duration_tolerance_seconds=args.duration_tolerance_seconds,
        verify_digests=not args.no_verify_digests,
        index_dir=args.out.parent,
    )
    index.save(args.out)
    summary = index.summary
    _print_json(
        {
            "index": args.out.as_posix(),
            "plan": index.plan,
            "sample_count": summary["sample_count"],
            "split_counts": summary["split_counts"],
            "split_ratios_requested": summary["split_ratios_requested"],
            "split_ratios_actual": summary["split_ratios_actual"],
            "empty_splits": summary["empty_splits"],
            "oversized_components": summary["oversized_components"],
            "leak_free": summary["leak_free"],
            "cross_split_assets": summary["cross_split_assets"],
            "warnings": summary["warnings"],
        }
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    index = _load_index(args)
    root = _resolve_root(index, args)
    verification = verify_dataset(
        index, root, duration_tolerance_seconds=args.duration_tolerance_seconds
    )
    if not verification.ok:
        for error in verification.errors:
            print(f"verify: error: {error}", file=sys.stderr)
        print(
            f"verify: failed: {len(verification.errors)} problem(s) over "
            f"{verification.checked_samples} sample(s) and {verification.checked_files} file(s)",
            file=sys.stderr,
        )
        return 1
    print(
        f"verify: ok: {verification.checked_samples} sample(s), "
        f"{verification.checked_files} digest-checked file(s)"
    )
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    index = _load_index(args)
    payload: dict[str, Any] = {
        "index_version": index.index_version,
        "plan": index.plan,
        "layout": index.layout,
        "summary": index.summary,
        "components": list(index.components),
    }
    if args.samples:
        payload["samples"] = [
            {
                "sample_id": entry.sample_id,
                "path": entry.path,
                "split": entry.split,
                "source_ids": list(entry.source_ids),
                "usable": entry.usable,
                "warnings": list(entry.warnings),
                "center_count": entry.labels["center_count"],
                "valid_count": entry.labels["valid_count"],
                "duration_seconds": entry.duration_seconds,
                "track_start_seconds": entry.track_start_seconds,
            }
            for entry in index.samples
        ]
    _print_json(payload)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    index = _load_index(args)
    root = _resolve_root(index, args)
    seed = index.plan["seed"] if args.seed is None else args.seed
    batch = sample_batch(
        index,
        root,
        seed=seed,
        split=args.split,
        items=args.items,
        centers_per_item=args.centers,
        min_center_gap=args.min_gap,
        valid_only=not args.include_invalid,
        on_unusable="skip" if args.skip_unusable else "error",
        window_seconds=args.window_seconds,
        slots=args.slots,
        verify_digests=not args.no_verify_digests,
    )
    _print_json(batch.to_json_dict())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "build": cmd_build,
        "verify": cmd_verify,
        "summary": cmd_summary,
        "batch": cmd_batch,
    }
    try:
        return handlers[args.command](args)
    except DatasetError as exc:
        print(f"build_dataset_index: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
