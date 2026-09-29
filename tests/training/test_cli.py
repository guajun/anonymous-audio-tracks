"""End-to-end CLI smoke: the same entry point CI and the docs use."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.ml

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_smoke_runs_cpu_fake_end_to_end(tmp_path: Path):
    out = tmp_path / "smoke-cli"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "smoke",
            "--out",
            str(out),
            "--steps",
            "3",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert "fake-encoder-smoke" in payload["result_kind"]
    assert Path(payload["checkpoint"]).is_file()
    assert Path(payload["train_log"]).is_file()
    assert "FAKE ENCODER SMOKE" in result.stderr
    assert payload["dataset"]["leak_free"] is True


def test_cli_train_rejects_missing_config(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "train",
            "--config",
            str(tmp_path / "missing.toml"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 2
    assert "does not exist" in result.stderr
