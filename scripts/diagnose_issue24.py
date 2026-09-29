#!/usr/bin/env python3
"""Deterministic Phase A diagnostics for issue #24 (design/diagnosis only).

Usage::

    uv sync --locked --extra ml
    uv run --no-sync python scripts/diagnose_issue24.py report --out /tmp/issue24.json
    uv run --no-sync python scripts/diagnose_issue24.py validate-plan

The ``report`` command never reads private, external or real audio, AuT
weights, checkpoints or a trained model.  The sampler section generates and
reads a tiny temporary synthetic smoke WAV corpus (``make_smoke_dataset``)
that the process deletes before exiting.  All other fixtures are synthetic and
created in memory, and every section is clearly labelled.

``validate-plan`` only checks the Phase B design checklist; it is not a run
authorization.

Both commands work in a base environment without torch: sections that need
``aat.losses`` report ``status = "unavailable"`` instead of failing.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_PLAN = REPO_ROOT / "configs" / "research" / "issue24_phase_b_plan.toml"


def _environment() -> dict[str, str | bool | None]:
    import subprocess

    import numpy

    git_sha: str | None = None
    git_dirty: bool | None = None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        git_sha = sha or None
        git_dirty = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
    }


def command_report(args: argparse.Namespace) -> int:
    from aat.diagnostics.issue24 import build_issue24_report

    corpus_root: str | None = None
    with tempfile.TemporaryDirectory(prefix="aat-issue24-") as temporary:
        if not args.skip_corpus:
            corpus_root = str(Path(temporary) / "corpus")
        report = build_issue24_report(
            include_matching=not args.skip_matching,
            include_supervision=not args.skip_torch,
            corpus_root=corpus_root,
            sampling_steps=args.sampling_steps,
        )
    report["environment"] = _environment()
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {target}")
    else:
        print(payload)
    return 0


def command_validate_plan(args: argparse.Namespace) -> int:
    from aat.diagnostics.issue24 import load_phase_b_plan

    result = load_phase_b_plan(args.plan)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    if not result["valid"]:
        print("plan validation failed", file=sys.stderr)
        return 1
    print(
        f"plan {result['plan']['id']} is a valid description "
        f"({len(result['experiments'])} experiments, not executable)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report = subparsers.add_parser(
        "report", help="run deterministic synthetic diagnostics and print JSON"
    )
    report.add_argument("--out", default=None, help="write JSON here instead of stdout")
    report.add_argument(
        "--skip-torch",
        action="store_true",
        help="skip the identity-supervision section even when torch is available",
    )
    report.add_argument(
        "--skip-matching",
        action="store_true",
        help="skip the exact-matching section (requires the ml extra)",
    )
    report.add_argument(
        "--skip-corpus",
        action="store_true",
        help="skip the smoke-corpus sampling-coverage section (no files written)",
    )
    report.add_argument(
        "--sampling-steps",
        type=int,
        default=10,
        help="simulated training steps for the sampling-coverage section",
    )
    report.set_defaults(func=command_report)

    validate = subparsers.add_parser(
        "validate-plan", help="validate the Phase B design checklist (no run)"
    )
    validate.add_argument("--plan", default=str(DEFAULT_PLAN), help="path to the plan TOML")
    validate.set_defaults(func=command_validate_plan)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
