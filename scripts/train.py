#!/usr/bin/env python3
"""Issue #8 training entry points: data prep, training, evaluation, CPU smoke.

```sh
# 1) Real DawDreamer training data (render -> label -> index; render extra)
uv run --no-sync python scripts/train.py prepare-data \
    --plan configs/train/dataset_local.toml \
    --out runs/train-data --index-out runs/train-index/index.json

# 2) CPU fake-encoder smoke (no weights, no renderer; proves the pipeline only)
uv run --no-sync python scripts/train.py smoke --out runs/train/smoke-0001

# 3) Training (fake or real frozen AuT, selected by the config)
uv run --no-sync python scripts/train.py train --config configs/train/fake_local.toml
uv run --no-sync python scripts/train.py train --config configs/train/aut_short.toml

# 4) Resume / extend, then evaluate a checkpoint on a held-out split
uv run --no-sync python scripts/train.py train --config configs/train/fake_local.toml \
    --resume runs/train/fake-local --steps 120
uv run --no-sync python scripts/train.py eval --config configs/train/fake_local.toml \
    --checkpoint runs/train/fake-local/checkpoint.pt --split val
```

Exit codes: 0 success (training with effective supervision), 1 training
completed without any effective activity supervision, 2 config/plan error,
3 render/label/data error, 4 dataset leakage/capacity error.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from aat.contracts.jsonio import dumps_json  # noqa: E402
from aat.data import DatasetError, build_dataset_index  # noqa: E402
from aat.labels import LabelError  # noqa: E402
from aat.render.errors import RenderError  # noqa: E402
from aat.training.config import default_smoke_config  # noqa: E402
from aat.training.dataset import (  # noqa: E402
    dataset_fingerprint,
    load_dataset_plan,
    load_verified_index,
    make_smoke_dataset,
)
from aat.training.errors import TrainingError  # noqa: E402


def _print_json(payload: Any) -> None:
    print(dumps_json(payload))


def _archive_if_needed(target: Path, *, overwrite: bool, kind: str) -> None:
    """Explicit, safe overwrite: move an existing directory aside, never delete."""

    if not target.exists():
        return
    if not any(target.iterdir()):
        return
    if not overwrite:
        raise TrainingError(
            f"{kind} {target} already exists and is not empty; pass --overwrite to archive it "
            f"as {target.name}.bak-<timestamp> and regenerate (nothing is deleted)"
        )
    stamp = re.sub(r"[^0-9A-Za-z]", "", datetime.now(timezone.utc).isoformat())
    backup = target.with_name(f"{target.name}.bak-{stamp}")
    os.replace(target, backup)
    print(f"archived existing {kind} as {backup}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# prepare-data
# --------------------------------------------------------------------------- #


def cmd_prepare_data(args: argparse.Namespace) -> int:
    plan = load_dataset_plan(args.plan)
    data_root = Path(args.out)
    index_out = Path(args.index_out) if args.index_out else data_root / "index.json"
    _archive_if_needed(data_root, overwrite=args.overwrite, kind="data root")
    data_root.mkdir(parents=True, exist_ok=True)

    import label_sample  # sibling script: reusable manifest-updating labeler
    from aat.render.pipeline import render_sample

    rendered: list[dict[str, Any]] = []
    for song in plan.songs:
        out_dir = data_root / song.sample_id
        result = render_sample(song, out_dir, config_source=args.plan)
        report = label_sample.run_labeling(out_dir)
        rendered.append(
            {
                "sample_id": song.sample_id,
                "composition": song.composition,
                "sources": len(song.source_ids),
                "presets": list(song.presets),
                "sample_origins": list(song.sample_origins),
                "stem_sum_max_abs_error_lsb": result.report["stem_sum"]["max_abs_error_lsb"],
                "center_count": report["center_count"],
                "valid_count": report["valid_count"],
            }
        )
        print(
            f"rendered+labeled {song.sample_id}: {len(song.source_ids)} source(s), "
            f"{report['valid_count']}/{report['center_count']} valid centers",
            file=sys.stderr,
        )

    index = build_dataset_index(
        data_root,
        seed=plan.seed,
        ratios=plan.ratios,
        slots=plan.slots,
        window_seconds=plan.window_seconds,
        verify_digests=not args.no_verify_digests,
        index_dir=index_out.parent,
    )
    index.save(index_out)
    summary = index.summary
    provenance = {
        "samples": [
            {
                "sample_id": song.sample_id,
                "composition": song.composition,
                "presets": list(song.presets),
                "sample_origins": list(song.sample_origins),
            }
            for song in plan.songs
        ],
        "synthetic_smoke": False,
        "renderer": "DawDreamer",
    }
    payload = {
        "status": "ok",
        "plan": plan.name,
        "data_root": str(data_root),
        "index": str(index_out),
        "sample_count": summary["sample_count"],
        "split_counts": summary["split_counts"],
        "split_ratios_requested": summary["split_ratios_requested"],
        "split_ratios_actual": summary["split_ratios_actual"],
        "empty_splits": summary["empty_splits"],
        "oversized_components": summary["oversized_components"],
        "cross_split_assets": summary["cross_split_assets"],
        "leak_free": summary["leak_free"],
        "warnings": summary["warnings"],
        "fingerprint": dataset_fingerprint(index),
        "provenance": provenance,
    }
    _print_json(payload)
    if not summary["leak_free"] or any(
        int(value) != 0 for value in summary["cross_split_assets"].values()
    ):
        print("error: dataset leakage detected; refusing to declare success", file=sys.stderr)
        return 4
    return 0


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #


def _load_train_config(args: argparse.Namespace):
    from aat.training.config import TrainConfig

    config = TrainConfig.from_toml(args.config)
    return config.with_overrides(
        out_dir=args.out,
        model_dir=args.model_dir,
        steps=args.steps,
        device=args.device,
    )


def cmd_train(args: argparse.Namespace) -> int:
    from aat.training.trainer import train_from_config

    config = _load_train_config(args)
    summary = train_from_config(
        config,
        out_dir=args.out,
        resume_from=args.resume,
        overwrite=args.overwrite,
        log=lambda message: print(message, file=sys.stderr),
    )
    _print_json(summary.to_dict())
    if summary.status != "completed":
        print(
            "error: run finished without effective activity supervision; not a successful "
            "training result",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from aat.training.config import TrainConfig
    from aat.training.evaluate import evaluate_split
    from aat.training.inference import load_head_from_checkpoint

    config = TrainConfig.from_toml(args.config).with_overrides(
        model_dir=args.model_dir, device=args.device
    )
    checkpoint = Path(args.checkpoint)
    inference = load_head_from_checkpoint(
        checkpoint,
        model_dir=args.model_dir,
        device=args.device,
    )
    index = load_verified_index(config.data.index, data_root=config.data.data_root)
    data_root = index.resolve_data_root(
        index_path=config.data.index, data_root=config.data.data_root
    )
    report = evaluate_split(
        index=index,
        data_root=data_root,
        split=args.split,
        inference=inference,
        config=config,
        max_songs=args.max_songs or config.eval.max_songs,
    )
    out = Path(args.out) if args.out else checkpoint.parent / f"eval_{args.split}.json"
    out.write_text(dumps_json(report) + "\n", encoding="utf-8", newline="\n")
    _print_json(
        {
            "status": "ok",
            "run": str(checkpoint.parent),
            "checkpoint": str(checkpoint),
            "eval": str(out),
            "split": report["split"],
            "micro": report["micro"],
            "baselines_micro": report["baselines_micro"],
            "undefined_reasons": report["undefined_reasons"],
            "effective": report["effective"],
            "protocol": report["protocol"],
        }
    )
    return 0


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #


def cmd_smoke(args: argparse.Namespace) -> int:
    from aat.training.trainer import train_from_config

    out = Path(args.out)
    _archive_if_needed(out, overwrite=args.overwrite, kind="smoke output")
    dataset_root = Path(args.dataset_root) if args.dataset_root else out / "dataset"
    if args.dataset_root:
        _archive_if_needed(dataset_root, overwrite=args.overwrite, kind="smoke dataset")
    dataset = make_smoke_dataset(dataset_root, seed=args.seed)
    config = default_smoke_config(
        data_root=dataset["data_root"],
        index_path=dataset["index_path"],
        out_dir=out / "run",
        steps=args.steps,
        seed=args.seed,
    )
    print(
        "FAKE ENCODER SMOKE: fake features prove the engineering pipeline only; "
        "this is not a real AuT model result",
        file=sys.stderr,
    )
    summary = train_from_config(
        config,
        out_dir=out / "run",
        log=lambda message: print(message, file=sys.stderr),
    )
    _print_json(
        {
            "status": summary.status,
            "result_kind": summary.result_kind,
            "run_dir": summary.run_dir,
            "checkpoint": summary.checkpoint_path,
            "train_log": summary.train_log_path,
            "eval_paths": dict(summary.eval_paths),
            "steps_completed": summary.steps_completed,
            "totals": dict(summary.totals),
            "dataset": dataset,
        }
    )
    if summary.status != "completed":
        print(
            "error: smoke run reported no effective activity supervision",
            file=sys.stderr,
        )
        return 1
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=(
            "Prepare real DawDreamer training data, train the frozen-AuT E/P head, "
            "evaluate held-out splits and run the CPU fake-encoder smoke (issue #8)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="render, label and index a dataset plan")
    prepare.add_argument("--plan", required=True, help="dataset plan TOML")
    prepare.add_argument("--out", required=True, help="data root for rendered samples")
    prepare.add_argument(
        "--index-out", default=None, help="index JSON path (default: <out>/index.json)"
    )
    prepare.add_argument("--overwrite", action="store_true", help="archive and regenerate")
    prepare.add_argument(
        "--no-verify-digests",
        action="store_true",
        help="skip the immediate re-hash during index build (explicit opt-out)",
    )
    prepare.set_defaults(func=cmd_prepare_data)

    train = subparsers.add_parser("train", help="train or resume a run")
    train.add_argument("--config", required=True, help="training config TOML")
    train.add_argument("--out", default=None, help="run directory (overrides config)")
    train.add_argument("--model-dir", default=None, help="AuT checkpoint directory (aut mode)")
    train.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda:0")
    train.add_argument("--steps", type=int, default=None, help="override the step budget")
    train.add_argument("--resume", default=None, help="run directory or checkpoint.pt")
    train.add_argument("--overwrite", action="store_true", help="archive an existing run first")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("eval", help="evaluate a checkpoint on a held-out split")
    evaluate.add_argument("--config", required=True, help="training config TOML")
    evaluate.add_argument("--checkpoint", required=True, help="checkpoint.pt path")
    evaluate.add_argument("--split", default="val", choices=("train", "val", "test"))
    evaluate.add_argument("--out", default=None, help="report JSON path")
    evaluate.add_argument("--model-dir", default=None, help="AuT checkpoint directory (aut mode)")
    evaluate.add_argument("--device", default=None, help="torch device")
    evaluate.add_argument("--max-songs", type=int, default=None, help="cap evaluated songs")
    evaluate.set_defaults(func=cmd_eval)

    smoke = subparsers.add_parser(
        "smoke", help="CPU fake-encoder end-to-end smoke (no weights, no renderer)"
    )
    smoke.add_argument("--out", required=True, help="smoke output root")
    smoke.add_argument("--dataset-root", default=None, help="override the dataset directory")
    smoke.add_argument("--steps", type=int, default=40, help="training steps (default 40)")
    smoke.add_argument("--seed", type=int, default=20260929)
    smoke.add_argument("--overwrite", action="store_true")
    smoke.set_defaults(func=cmd_smoke)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except TrainingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except DatasetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 4
    except (RenderError, LabelError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
