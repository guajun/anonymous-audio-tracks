#!/usr/bin/env python3
"""Associate a ``prediction.json``/``prediction.npz`` pair into ``trajectory.json``.

The prediction protocol does not record the analyzed audio span, so
``--duration-seconds`` (and ``--track-start-seconds`` for a clip) must be given
explicitly; times stay on the original-track axis.  Provenance is required and
defaults to ``mock`` so controlled prediction sequences stay distinguishable
from real model runs.

Example::

    uv run --no-sync python scripts/track.py \
        --prediction outputs/run-01 --output outputs/run-01/trajectory.json \
        --duration-seconds 120.0 --run-id run-01 --data-kind mock
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aat.contracts import AudioSpan, ContractError, PredictionData, RunProvenance
from aat.tracking import TrackingConfig, TrackingError, track_prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gated one-to-one cross-window association of a prediction document "
            "into a protocol trajectory.json"
        )
    )
    parser.add_argument(
        "--prediction",
        required=True,
        type=Path,
        help="directory containing prediction.json and prediction.npz",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="trajectory.json output path"
    )
    parser.add_argument(
        "--duration-seconds",
        required=True,
        type=float,
        help="duration of the analyzed audio in original-track seconds",
    )
    parser.add_argument(
        "--track-start-seconds",
        type=float,
        default=0.0,
        help="absolute original-track time of the first analyzed sample (default 0)",
    )
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--data-kind",
        choices=("model", "annotation", "mock"),
        default="mock",
        help="provenance kind; mock by default because no model is trained yet",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--config-hash", default=None)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--created-at-utc", default=None)
    parser.add_argument("--activity-threshold", type=float, default=None)
    parser.add_argument("--match-threshold", type=float, default=None)
    parser.add_argument("--retention-seconds", type=float, default=None)
    parser.add_argument("--prototype-alpha", type=float, default=None)
    parser.add_argument("--birth-threshold", type=float, default=None)
    parser.add_argument("--max-exact-slots", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_names = (
        "activity_threshold",
        "match_threshold",
        "retention_seconds",
        "prototype_alpha",
        "birth_threshold",
        "max_exact_slots",
    )
    config_kwargs = {
        name: getattr(args, name) for name in config_names if getattr(args, name) is not None
    }

    try:
        prediction = PredictionData.load(args.prediction)
        config = TrackingConfig(**config_kwargs)
        provenance = RunProvenance(
            run_id=args.run_id,
            data_kind=args.data_kind,
            model_id=args.model_id,
            config_hash=args.config_hash,
            git_commit=args.git_commit,
            created_at_utc=args.created_at_utc,
        )
        trajectory = track_prediction(
            prediction,
            AudioSpan(
                duration_seconds=args.duration_seconds,
                track_start_seconds=args.track_start_seconds,
            ),
            provenance,
            config=config,
            sample_id=args.sample_id or prediction.sample_id,
        )
        trajectory.save(args.output)
    except (ContractError, TrackingError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    point_count = sum(len(track.center_times) for track in trajectory.tracks)
    print(
        f"tracks={len(trajectory.tracks)} points={point_count} "
        f"slots={trajectory.slots} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
