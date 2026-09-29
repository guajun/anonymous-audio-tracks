#!/usr/bin/env python3
"""Render one deterministic DawDreamer sample or probe Surge XT.

Reproducible smoke render (from the repository root, with the render extra):

```sh
uv run python scripts/render_sample.py render \
    --config configs/render/smoke.toml \
    --out runs/render/smoke-0001
```

Surge XT probe (plugin location never committed; pass it at runtime):

```sh
uv run python scripts/render_sample.py probe-surge \
    --plugin-path "C:/Program Files/Common Files/VST3/Surge XT.vst3" \
    --out runs/render/surge-probe
```

Exit codes: 0 success, 2 configuration error, 3 render/dependency error,
4 explicitly requested Surge probe failed.  Audio stays in the ignored
``runs/`` directory; only JSON summaries are printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path


def _ensure_package_importable() -> None:
    try:
        import aat  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


_ensure_package_importable()

from aat.contracts.jsonio import dumps_json  # noqa: E402
from aat.render.config import config_from_dict, load_config  # noqa: E402
from aat.render.errors import (  # noqa: E402
    RenderConfigError,
    RenderDependencyError,
    RenderError,
)
from aat.render.pipeline import render_sample  # noqa: E402
from aat.render.probe_surge import probe_surge  # noqa: E402


def _load_config_with_seed_override(config_path: str, seed: int | None):
    if seed is None:
        return load_config(config_path)
    with open(config_path, "rb") as handle:
        data = tomllib.load(handle)
    render = data.setdefault("render", {})
    render["seed"] = seed
    return config_from_dict(data, origin=Path(config_path).name)


def _cmd_render(args: argparse.Namespace) -> int:
    config = _load_config_with_seed_override(args.config, args.seed)
    result = render_sample(
        config,
        args.out,
        config_source=args.config,
        surge_plugin_path=args.surge_plugin_path,
    )
    summary = {
        "status": "ok",
        "sample_id": config.sample_id,
        "seed": config.seed,
        "out": args.out,
        "mix": "mix.wav",
        "stems": list(config.source_ids),
        "rendered_total_seconds": result.report["rendered_total_seconds"],
        "stem_sum_max_abs_error_lsb": result.report["stem_sum"]["max_abs_error_lsb"],
        "stem_sum_tolerance_lsb": result.report["stem_sum"]["tolerance_lsb"],
        "tail_margin_seconds": result.report["tail"]["margin_seconds"],
        "surge_probe": result.report["surge_probe"]["status"],
    }
    print(dumps_json(summary))
    if result.surge is not None and not result.surge.passed:
        print(
            f"error: Surge XT probe failed: {result.surge.reason} "
            f"({result.surge.detail})",
            file=sys.stderr,
        )
        return 4
    return 0


def _cmd_probe_surge(args: argparse.Namespace) -> int:
    result = probe_surge(
        args.plugin_path,
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        duration_seconds=args.duration_seconds,
        note=args.note,
        velocity=args.velocity,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "surge_probe.json").write_text(
        dumps_json(result.to_json_dict()) + "\n", encoding="utf-8", newline="\n"
    )
    print(dumps_json(result.to_json_dict()))
    if not result.passed:
        print(f"error: Surge XT probe failed: {result.reason} ({result.detail})", file=sys.stderr)
        return 4
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render_sample.py",
        description="Render one deterministic sample with DawDreamer or probe Surge XT.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="render a configured sample")
    render.add_argument("--config", required=True, help="render config TOML path")
    render.add_argument("--out", required=True, help="output directory (typically runs/...)")
    render.add_argument("--seed", type=int, default=None, help="override render.seed")
    render.add_argument(
        "--surge-plugin-path",
        default=None,
        help="optional Surge XT VST3 path; when given the probe result is included",
    )
    render.set_defaults(func=_cmd_render)

    probe = subparsers.add_parser("probe-surge", help="probe a Surge XT VST3")
    probe.add_argument("--plugin-path", required=True, help="Surge XT VST3 path or bundle")
    probe.add_argument("--out", required=True, help="output directory for surge_probe.json")
    probe.add_argument("--sample-rate", type=int, default=44100)
    probe.add_argument("--block-size", type=int, default=512)
    probe.add_argument("--duration-seconds", type=float, default=1.0)
    probe.add_argument("--note", type=int, default=60)
    probe.add_argument("--velocity", type=int, default=100)
    probe.set_defaults(func=_cmd_probe_surge)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RenderConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RenderDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except RenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
