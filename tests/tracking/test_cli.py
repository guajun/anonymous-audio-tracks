"""CLI end-to-end test for ``scripts/track.py``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from aat.contracts import PredictionData, Trajectory

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_prediction(directory: Path) -> None:
    embeddings = np.zeros((3, 2, 128), dtype=np.float32)
    embeddings[:, 0, 0] = 1.0
    embeddings[:, 1, 1] = 1.0
    prediction = PredictionData(
        center_times=np.array([0.0, 0.02, 0.04]),
        embeddings=embeddings,
        activity=np.full((3, 2), 0.9, dtype=np.float32),
        slot_valid=np.ones((3, 2), dtype=bool),
        center_valid=np.ones(3, dtype=bool),
        slots=2,
        hop_seconds=0.02,
        sample_id="cli-sample",
    )
    prediction.save(directory)


def _run_cli(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "track.py"), *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_track_cli_exports_valid_trajectory(tmp_path):
    prediction_dir = tmp_path / "prediction"
    _write_prediction(prediction_dir)
    output = tmp_path / "trajectory.json"

    completed = _run_cli(
        "--prediction",
        str(prediction_dir),
        "--output",
        str(output),
        "--duration-seconds",
        "1.0",
        "--run-id",
        "cli-test",
        "--data-kind",
        "mock",
    )

    assert completed.returncode == 0, completed.stderr
    assert "tracks=2" in completed.stdout
    trajectory = Trajectory.load(output)
    assert trajectory.provenance.run_id == "cli-test"
    assert trajectory.provenance.data_kind == "mock"
    assert trajectory.sample_id == "cli-sample"
    assert len(trajectory.tracks) == 2


def test_track_cli_reports_missing_prediction_as_error(tmp_path):
    completed = _run_cli(
        "--prediction",
        str(tmp_path / "missing"),
        "--output",
        str(tmp_path / "trajectory.json"),
        "--duration-seconds",
        "1.0",
        "--run-id",
        "cli-test",
    )

    assert completed.returncode == 1
    assert "error:" in completed.stderr
