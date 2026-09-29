"""CLI behaviour that does not need DawDreamer (config failures, help)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "render_sample.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_help_exits_zero() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "render" in result.stdout


def test_missing_config_exits_two(tmp_path) -> None:
    result = _run("render", "--config", str(tmp_path / "missing.toml"), "--out", str(tmp_path))
    assert result.returncode == 2
    assert "file not found" in result.stderr


def test_invalid_config_exits_two(tmp_path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        """
[render]
sample_id = "bad"
composition = "bad"
seed = 1
sample_rate = 8000
block_size = 64
bpm = 100.0
duration_seconds = 0.5
tail_seconds = 0.5
[[sources]]
id = "s01"
""",
        encoding="utf-8",
    )
    result = _run("render", "--config", str(bad), "--out", str(tmp_path))
    assert result.returncode == 2
    assert "duration_seconds" in result.stderr


def test_probe_surge_requires_plugin_path() -> None:
    result = _run("probe-surge", "--out", "unused")
    assert result.returncode == 2  # argparse missing required argument


@pytest.mark.parametrize("args", [("probe-surge", "--help"), ("render", "--help")])
def test_subcommand_help(args) -> None:
    assert _run(*args).returncode == 0
