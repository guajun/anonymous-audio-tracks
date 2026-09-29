"""Issue #11 end-to-end CPU smoke: fixture -> training -> reload -> tracks -> viewer.

The whole chain runs through the production CLI (``scripts/demo_pipeline.py``)
on the clearly identified synthetic smoke corpus (``make_smoke_dataset``; no
DawDreamer, no weights).  Marked ``integration`` **and** ``ml``:

* the base job excludes both markers;
* the render-only job excludes ``ml`` (so no torch import is required);
* the ml job (``-m ml``) runs it with CPU-only torch.

The optional ``pytest.importorskip("torch")`` executes before any ML import so
collection stays valid in the base/render environments.  Assertions target
observable wiring/provenance/edge-mask behaviour, not copies of implementation
logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: F841 - ML imports follow below

from aat.contracts import Trajectory  # noqa: E402
from aat.contracts.arrays import PredictionData  # noqa: E402
from aat.data import DatasetIndex  # noqa: E402
from aat.training.dataset import dataset_fingerprint  # noqa: E402
from aat.training.inference import load_head_from_checkpoint  # noqa: E402
from aat.windowing import extract_windows_at_times  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.ml]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO = REPO_ROOT / "scripts" / "demo_pipeline.py"


def _run_demo(*args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "4")
    return subprocess.run(
        [sys.executable, str(DEMO), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("issue11-smoke") / "run"
    completed = _run_demo("smoke", "--out", str(out), "--steps", "3")
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    report = json.loads((out / "pipeline_report.json").read_text(encoding="utf-8"))
    return {"out": out, "report": report, "completed": completed}


def test_smoke_serializes_viewer_ready_artifacts_with_provenance(smoke: dict) -> None:
    out: Path = smoke["out"]
    report: dict = smoke["report"]

    assert report["schema"] == "aat-demo-pipeline-report/1"
    assert report["mode"] == "fake-encoder-smoke"
    assert "fake" in report["result_kind"] and "not a real model result" in report["result_kind"]

    provenance = report["provenance"]
    # Git SHA / lock digest / config digest / dataset digest must be real.
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True
    ).stdout.strip()
    assert provenance["git"]["sha"] == head
    assert provenance["uv_lock_sha256"] == _sha256(REPO_ROOT / "uv.lock")
    assert len(provenance["config_sha256"]) == 64
    assert provenance["pinned_policy"] == {
        "mode": "fake",
        "window_seconds": 2.0,
        "attention": "block_diagonal",
        "dtype": "float32",
        "extraction": "independent_windows",
        "batch_windows": 8,
        "feature_dim": 16,
    }
    assert report["resources"]["device"] == "cpu"
    assert isinstance(report["resources"]["elapsed_seconds"], (int, float))

    checkpoint = Path(provenance["checkpoint"]["path"])
    assert checkpoint.is_file()
    assert provenance["checkpoint"]["sha256"] == _sha256(checkpoint)
    assert provenance["checkpoint"]["step"] == 2

    dataset = provenance["dataset"]
    index = DatasetIndex.load(dataset["index_path"])
    assert dataset["index_sha256"] == _sha256(Path(dataset["index_path"]))
    assert dataset["fingerprint_digest"] == dataset_fingerprint(index)["digest"]
    assert dataset["synthetic"] is True and dataset["leak_free"] is True

    song = report["songs"][0]
    trajectory = Trajectory.load(song["trajectory"])
    assert trajectory.provenance.data_kind == "mock"
    assert trajectory.slots == 8
    assert len(trajectory.tracks) >= 1
    assert len({track.track_id for track in trajectory.tracks}) == len(trajectory.tracks)

    predictions = PredictionData.load(song["prediction_dir"])
    assert predictions.center_times.shape[0] == song["centers_total"]
    # Padded edge windows are canonical invalid slots (not fake supervised ones).
    assert predictions.center_valid.sum() == song["centers_valid"]
    assert int(predictions.center_valid.sum()) < int(predictions.center_times.shape[0])
    invalid_rows = ~np.asarray(predictions.center_valid)
    assert np.all(np.asarray(predictions.activity)[invalid_rows] == 0.0)
    assert np.all(np.asarray(predictions.embeddings)[invalid_rows] == 0.0)
    assert not np.any(np.asarray(predictions.slot_valid)[invalid_rows])

    session = json.loads((out / "songs" / song["sample_id"] / "session.json").read_text("utf-8"))
    assert session["schema"] == "aat-demo-viewer-session/1"
    assert session["data_kind"] == "mock"
    assert session["human_listening"]["status"] == "pending"
    assert session["audio"]["track_start_seconds"] == song["track_start_seconds"]
    manual = (out / "songs" / song["sample_id"] / "manual_inspection.md").read_text("utf-8")
    assert "human listening status: **pending**" in manual

    # Artifact index lists consumable files with real digests.
    artifacts = json.loads((out / "artifacts.json").read_text("utf-8"))
    assert artifacts[str(Path(song["trajectory"]).relative_to(out).as_posix())] == _sha256(
        Path(song["trajectory"])
    )

    # Viewer protocol contract (the CLI already ran it when Node is present).
    assert report["viewer_validation"]["status"] in {"ok", "skipped"}
    if report["viewer_validation"]["status"] == "ok":
        viewer = report["viewer_validation"]["viewer"]
        assert viewer["schemaVersion"] == "0.1.0"
        assert viewer["kind"] == "trajectory"
        assert viewer["tracks"] == len(trajectory.tracks)
        assert viewer["dataKind"] == "mock"


def test_checkpoint_reload_is_deterministic_and_keeps_absolute_origin(smoke: dict) -> None:
    checkpoint = Path(smoke["report"]["provenance"]["checkpoint"]["path"])
    first = load_head_from_checkpoint(checkpoint)
    second = load_head_from_checkpoint(checkpoint)
    assert first.config.encoder.window_seconds == 2.0
    assert first.config.encoder.attention == "block_diagonal"

    sample_rate = 16000
    samples = np.sin(
        2.0 * np.pi * 220.0 * np.arange(4 * sample_rate, dtype=np.float64) / sample_rate
    ).astype(np.float32)
    centers = np.array([12.0, 12.5, 13.0, 15.0, 15.5, 16.0], dtype=np.float64)
    a = first.predict_at_times(samples, centers, sample_rate=sample_rate, origin_seconds=12.0)
    b = second.predict_at_times(samples, centers, sample_rate=sample_rate, origin_seconds=12.0)
    assert np.array_equal(a.center_times, centers)  # absolute original-track times
    for name in ("embeddings", "activity", "slot_valid", "center_valid"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name

    # 4 s of audio at origin 12.0 s: only centers 13.0 and 15.0 have a full
    # 2 s centered window inside the span; the padded ones must be invalid.
    assert a.center_valid.tolist() == [False, False, True, True, False, False]
    invalid = ~np.asarray(a.center_valid)
    assert np.all(a.activity[invalid] == 0.0)
    assert np.all(a.embeddings[invalid] == 0.0)
    assert not np.any(a.slot_valid[invalid])
    valid_rows = np.asarray(a.slot_valid)
    norms = np.linalg.norm(a.embeddings, axis=2)
    assert np.allclose(norms[valid_rows], 1.0, atol=1e-3)
    assert np.all(norms[~valid_rows] == 0.0)


def test_callback_matches_predict_at_times_only_with_audio_duration(smoke: dict) -> None:
    checkpoint = Path(smoke["report"]["provenance"]["checkpoint"]["path"])
    inference = load_head_from_checkpoint(checkpoint)
    sample_rate = 16000
    samples = np.sin(
        2.0 * np.pi * 330.0 * np.arange(4 * sample_rate, dtype=np.float64) / sample_rate
    ).astype(np.float32)
    centers = np.array([12.0, 12.5, 13.0, 15.0, 15.5, 16.0], dtype=np.float64)

    reference = inference.predict_at_times(
        samples, centers, sample_rate=sample_rate, origin_seconds=12.0
    )
    windows, _valid = extract_windows_at_times(
        samples, centers, sample_rate, 2.0, origin_seconds=12.0
    )
    bound = inference.predictor_for_centers(
        centers, sample_rate=sample_rate, origin_seconds=12.0, audio_duration_seconds=4.0
    )(windows)
    assert np.array_equal(bound.embeddings, reference.embeddings)
    assert np.array_equal(bound.activity, reference.activity)
    assert np.array_equal(bound.slot_valid, reference.slot_valid)
    assert np.array_equal(np.asarray(bound.slot_valid).any(axis=1), reference.center_valid)

    # Without the duration the end edge is unknowable: the documented
    # limitation must be observable, not silently identical to the safe path.
    loose = inference.predictor_for_centers(
        centers, sample_rate=sample_rate, origin_seconds=12.0
    )(windows)
    loose_valid = np.asarray(loose.slot_valid).any(axis=1)
    assert not np.array_equal(loose_valid, reference.center_valid)
    assert loose_valid[:3].tolist() == reference.center_valid[:3].tolist()
    assert loose_valid[3:].sum() > reference.center_valid[3:].sum()


def test_config_overlay_runs_synthetic_path_and_overwrite_is_guarded(
    smoke: dict, tmp_path: Path
) -> None:
    out = tmp_path / "configured-synthetic"
    config_path = tmp_path / "demo.toml"
    config_path.write_text(
        "\n".join(
            [
                f'out = "{out.as_posix()}"',
                f'checkpoint = "{Path(smoke["report"]["provenance"]["checkpoint"]["path"]).as_posix()}"',
                f'index = "{Path(smoke["report"]["provenance"]["dataset"]["index_path"]).as_posix()}"',
                'split = "test"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    completed = _run_demo("--config", str(config_path), "synthetic")
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    report = json.loads((out / "pipeline_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "fake-encoder-synthetic"
    assert "fake" in report["result_kind"]
    canonical = report["canonical"]
    assert set(canonical["baselines_micro"]) == {"all_inactive", "all_active", "no_identity"}
    assert canonical["protocol"]["fake_vs_real"].startswith("fake-encoder")
    song = report["songs"][0]
    assert song["canonical_metrics_match"] is True
    assert song["canonical_metrics"] == song["evaluation"]
    assert report["viewer_validation"]["status"] in {"ok", "skipped"}

    repeated = _run_demo("--config", str(config_path), "synthetic")
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
