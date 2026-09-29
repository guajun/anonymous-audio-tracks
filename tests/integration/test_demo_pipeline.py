"""Issue #11 end-to-end CPU smoke: fixture -> training -> reload -> tracks -> viewer.

The whole chain runs through the production CLI (``scripts/demo_pipeline.py``)
on the clearly identified synthetic smoke corpus (``make_smoke_dataset``; no
DawDreamer, no weights).  Marked ``integration`` **and** ``ml``:

* the base job excludes both markers;
* the render-only job excludes ``ml`` (so no torch import is required);
* the ml job (``-m ml``) runs it with CPU-only torch.

The optional ``pytest.importorskip("torch")`` executes before any ML import so
collection stays valid in the base/render environments.  Assertions target
observable wiring/provenance/edge-mask/selection/containment behaviour, not
copies of implementation logic.  Real encoder runs stay out of this file.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: F841 - ML imports follow below

from aat.contracts import Trajectory  # noqa: E402
from aat.contracts.arrays import PredictionData  # noqa: E402
from aat.data import DatasetIndex  # noqa: E402
from aat.labels.wav import read_wav  # noqa: E402
from aat.render.wavio import write_pcm16_wav  # noqa: E402
from aat.training.dataset import dataset_fingerprint  # noqa: E402
from aat.training.errors import TrainingError  # noqa: E402
from aat.training.inference import load_head_from_checkpoint  # noqa: E402
from aat.windowing import extract_windows_at_times  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.ml]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO = REPO_ROOT / "scripts" / "demo_pipeline.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import demo_pipeline  # noqa: E402


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


def _synthetic_args(
    smoke: dict, out: Path, *, split: str, songs: str | None = None
) -> list[str]:
    provenance = smoke["report"]["provenance"]
    arguments = [
        "synthetic",
        "--checkpoint",
        provenance["checkpoint"]["path"],
        "--index",
        provenance["dataset"]["index_path"],
        "--split",
        split,
        "--out",
        str(out),
        "--device",
        "cpu",
    ]
    if songs is not None:
        arguments += ["--songs", songs]
    return arguments


def _write_tone(path: Path, *, seconds: float, sample_rate: int, frequency: float = 220.0):
    samples = 0.2 * np.sin(
        2.0 * np.pi * frequency * np.arange(int(seconds * sample_rate)) / sample_rate
    )
    write_pcm16_wav(path, samples.astype(np.float32), sample_rate)
    return path


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
    assert "--bind 127.0.0.1" in manual

    # Artifact index lists consumable files with real digests.
    artifacts = json.loads((out / "artifacts.json").read_text("utf-8"))
    assert artifacts[str(Path(song["trajectory"]).relative_to(out).as_posix())] == _sha256(
        Path(song["trajectory"])
    )

    # Viewer protocol contract (the CLI already ran it when Node is present).
    viewer = report["viewer_validation"]
    assert viewer["status"] in {"ok", "skipped"}
    assert list(viewer["per_song"]) == [song["sample_id"]]
    if viewer["status"] == "ok":
        payload = viewer["per_song"][song["sample_id"]]["viewer"]
        assert payload["schemaVersion"] == "0.1.0"
        assert payload["kind"] == "trajectory"
        assert payload["tracks"] == len(trajectory.tracks)
        assert payload["dataKind"] == "mock"
    assert report["checks"] == {"status": "ok", "failures": []}


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
    assert song["canonical_metrics_max_abs_delta"] is not None
    assert song["canonical_metrics_max_abs_delta"] <= 1e-6
    assert set(song["canonical_metrics"]) == set(song["evaluation"])
    assert report["viewer_validation"]["status"] in {"ok", "skipped"}
    assert list(report["viewer_validation"]["per_song"]) == [song["sample_id"]]
    assert report["checks"]["status"] == "ok"

    repeated = _run_demo("--config", str(config_path), "synthetic")
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr

    # Explicit CLI flags must override the TOML defaults (and the overlay may
    # not widen the selected evaluation set).
    override_out = tmp_path / "configured-override"
    override_config = tmp_path / "demo-override.toml"
    override_config.write_text(
        "\n".join(
            [
                f'out = "{override_out.as_posix()}"',
                f'checkpoint = "{Path(smoke["report"]["provenance"]["checkpoint"]["path"]).as_posix()}"',
                f'index = "{Path(smoke["report"]["provenance"]["dataset"]["index_path"]).as_posix()}"',
                'split = "test"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    overridden = _run_demo(
        "--config",
        str(override_config),
        "synthetic",
        "--split",
        "train",
        "--songs",
        "smoke-train-01",
    )
    assert overridden.returncode == 0, overridden.stderr
    override_report = json.loads((override_out / "pipeline_report.json").read_text("utf-8"))
    assert override_report["provenance"]["dataset"]["selection"]["split"] == "train"
    assert [song["sample_id"] for song in override_report["songs"]] == ["smoke-train-01"]


def test_selection_filters_canonical_evaluation_to_requested_songs(
    smoke: dict, tmp_path: Path, capsys
) -> None:
    one_out = tmp_path / "selected-one"
    assert demo_pipeline.main(_synthetic_args(smoke, one_out, split="train", songs="smoke-train-02")) == 0
    stderr = capsys.readouterr().err
    assert "FAKE ENCODER" in stderr and "REAL FROZEN" not in stderr
    one = json.loads((one_out / "pipeline_report.json").read_text(encoding="utf-8"))
    assert [row["sample_id"] for row in one["canonical"]["songs"]] == ["smoke-train-02"]
    assert [song["sample_id"] for song in one["songs"]] == ["smoke-train-02"]
    selection = one["provenance"]["dataset"]["selection"]
    assert selection["mode"] == "explicit-song-list"
    assert selection["requested_songs"] == ["smoke-train-02"]
    assert selection["evaluated_songs"] == ["smoke-train-02"]
    assert selection["canonical_songs"] == ["smoke-train-02"]
    assert one["canonical"]["effective"]["songs"] == 1

    both_out = tmp_path / "selected-both-reversed"
    assert (
        demo_pipeline.main(
            _synthetic_args(
                smoke, both_out, split="train", songs="smoke-train-02,smoke-train-01"
            )
        )
        == 0
    )
    both = json.loads((both_out / "pipeline_report.json").read_text(encoding="utf-8"))
    assert [row["sample_id"] for row in both["canonical"]["songs"]] == [
        "smoke-train-01",
        "smoke-train-02",
    ]
    assert [song["sample_id"] for song in both["songs"]] == [
        "smoke-train-02",
        "smoke-train-01",
    ]  # fixed requested order, no re-selection
    micro = both["canonical"]["micro"]
    for key in ("true_positives", "false_positives", "false_negatives"):
        assert micro[key] == sum(song["canonical_metrics"][key] for song in both["songs"]), key
    baseline = both["canonical"]["baselines_micro"]["all_inactive"]
    assert baseline["false_negatives"] == sum(
        song["canonical_baselines"]["all_inactive"]["false_negatives"] for song in both["songs"]
    )
    assert all(song["canonical_metrics_match"] for song in both["songs"])
    assert both["checks"]["status"] == "ok"

    one_micro = one["canonical"]["micro"]
    assert (
        one_micro["true_positives"],
        one_micro["false_positives"],
        one_micro["false_negatives"],
    ) != (
        micro["true_positives"],
        micro["false_positives"],
        micro["false_negatives"],
    ), "selecting one song must not silently evaluate the whole split"


def test_out_ancestor_of_inputs_rejected_without_touching_them(
    smoke: dict, tmp_path: Path, capsys
) -> None:
    with pytest.raises(TrainingError):
        demo_pipeline._check_output_containment(
            tmp_path,
            [("checkpoint", tmp_path / "inputs" / "run" / "checkpoint.pt")],
            command="test",
        )
    inputs = tmp_path / "inputs"
    shutil.copytree(smoke["out"] / "dataset", inputs / "dataset")
    shutil.copy2(
        smoke["report"]["provenance"]["checkpoint"]["path"], inputs / "checkpoint.pt"
    )
    sentinel = inputs / "sentinel.txt"
    sentinel.write_text("do-not-touch", encoding="utf-8")
    index_path = inputs / "dataset" / "index.json"

    rc = demo_pipeline.main(
        [
            "synthetic",
            "--checkpoint",
            str(inputs / "checkpoint.pt"),
            "--index",
            str(index_path),
            "--split",
            "test",
            "--out",
            str(inputs),
            "--overwrite",
            "--device",
            "cpu",
        ]
    )
    assert rc == 2
    assert "--out" in capsys.readouterr().err
    assert sentinel.read_text(encoding="utf-8") == "do-not-touch"
    assert (inputs / "checkpoint.pt").is_file()
    assert index_path.is_file()
    # No archive happened and nothing was moved.
    assert not list(tmp_path.glob("inputs.bak-*"))

    audio_out = inputs  # audio source also lives under the same rejected root
    wav = inputs / "clip.wav"
    _write_tone(wav, seconds=3.0, sample_rate=16000)
    wav_hash = _sha256(wav)
    rc = demo_pipeline.main(
        [
            "audio",
            "--audio",
            str(wav),
            "--checkpoint",
            str(inputs / "checkpoint.pt"),
            "--model-dir",
            str(inputs / "dataset"),
            "--out",
            str(audio_out),
            "--start-seconds",
            "1.0",
            "--duration-seconds",
            "1.0",
            "--device",
            "cpu",
        ]
    )
    assert rc == 2
    assert sentinel.read_text(encoding="utf-8") == "do-not-touch"
    assert _sha256(wav) == wav_hash  # source untouched


def test_label_traversal_and_reserved_names_rejected_before_decode(
    smoke: dict, tmp_path: Path, capsys
) -> None:
    for bad in ("..", "../escape", "a/b", "a\\b", "C:abs", "NUL", "clip.", "  "):
        with pytest.raises(TrainingError):
            demo_pipeline.validate_label(bad)

    src = tmp_path / "clip.wav"
    _write_tone(src, seconds=3.0, sample_rate=16000)
    source_before = _sha256(src)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "audio-segment.wav"
    sentinel.write_text("must-stay", encoding="utf-8")
    out = tmp_path / "run"
    for bad in ("../outside", "a/b", "NUL"):
        rc = demo_pipeline.main(
            [
                "audio",
                "--audio",
                str(src),
                "--checkpoint",
                smoke["report"]["provenance"]["checkpoint"]["path"],
                "--out",
                str(out),
                "--start-seconds",
                "1.0",
                "--duration-seconds",
                "1.0",
                "--label",
                bad,
                "--device",
                "cpu",
            ]
        )
        assert rc == 2
        assert "--label" in capsys.readouterr().err
    assert sentinel.read_text(encoding="utf-8") == "must-stay"
    assert _sha256(src) == source_before
    assert not out.exists()


def test_config_and_cli_negative_values_rejected_before_mutation(
    smoke: dict, tmp_path: Path, capsys
) -> None:
    wav = _write_tone(tmp_path / "clip.wav", seconds=3.0, sample_rate=16000)
    checkpoint = smoke["report"]["provenance"]["checkpoint"]["path"]
    index = smoke["report"]["provenance"]["dataset"]["index_path"]

    toml_cases = [
        ("smoke", {"steps": 3, "overwrite": "false"}, "--overwrite must be a boolean"),
        ("smoke", {"steps": 0}, "--steps must be an integer >= 1"),
        ("synthetic", {"split": "bogus"}, "--split must be one of train/val/test"),
        ("synthetic", {"max_songs": -1}, "--max-songs must be an integer >= 1"),
        ("synthetic", {"checkpoint": 123}, "--checkpoint must be a non-empty string"),
        ("synthetic", {"steps": 3}, "do not apply to command"),
        ("audio", {"hop_seconds": 0}, "--hop-seconds"),
        ("audio", {"start_seconds": -1}, "--start-seconds must be >= 0"),
        ("audio", {"sample_rate": 0}, "--sample-rate must be an integer >= 1"),
    ]
    for position, (command, overrides, expected) in enumerate(toml_cases):
        out = tmp_path / f"toml-negative-{position}"
        bases = {
            "smoke": {},
            "synthetic": {
                "checkpoint": Path(checkpoint).as_posix(),
                "index": Path(index).as_posix(),
                "split": "test",
            },
            "audio": {
                "audio": wav.as_posix(),
                "checkpoint": Path(checkpoint).as_posix(),
            },
        }
        payload = {"out": out.as_posix(), **bases[command], **overrides}
        config_path = tmp_path / f"negative-{position}.toml"
        config_path.write_text(
            "\n".join(f"{key} = {json.dumps(value)}" for key, value in payload.items()),
            encoding="utf-8",
        )
        rc = demo_pipeline.main(["--config", str(config_path), command])
        assert rc == 2, (command, overrides)
        assert expected in capsys.readouterr().err, (command, overrides)
        assert not out.exists(), "config validation must run before any filesystem mutation"

    cli_cases = [
        (
            ["smoke", "--out", str(tmp_path / "cli-smoke"), "--steps", "0"],
            "--steps must be an integer >= 1",
        ),
        (
            [
                "synthetic", "--checkpoint", checkpoint, "--index", index,
                "--out", str(tmp_path / "cli-synthetic"), "--max-songs", "-1",
                "--device", "cpu",
            ],
            "--max-songs must be an integer >= 1",
        ),
        (
            [
                "synthetic", "--checkpoint", checkpoint, "--index", index,
                "--out", str(tmp_path / "cli-split"), "--split", "bogus",
            ],
            "--split",
        ),
        (
            [
                "audio", "--audio", str(wav), "--checkpoint", checkpoint,
                "--out", str(tmp_path / "cli-audio-start"), "--start-seconds", "-1",
                "--device", "cpu",
            ],
            "--start-seconds must be >= 0",
        ),
        (
            [
                "audio", "--audio", str(wav), "--checkpoint", checkpoint,
                "--out", str(tmp_path / "cli-audio-hop"), "--hop-seconds", "0",
                "--device", "cpu",
            ],
            "--hop-seconds",
        ),
        (
            [
                "audio", "--audio", str(wav), "--checkpoint", checkpoint,
                "--out", str(tmp_path / "cli-audio-rate"), "--sample-rate", "0",
                "--device", "cpu",
            ],
            "--sample-rate must be an integer >= 1",
        ),
    ]
    for arguments, expected in cli_cases:
        rc = demo_pipeline.main(arguments)
        assert rc == 2, arguments
        assert expected in capsys.readouterr().err, arguments


def test_completion_status_fails_on_second_viewer_and_canonical_mismatch(
    smoke: dict, tmp_path: Path, monkeypatch, capsys
) -> None:
    real_validate = demo_pipeline.validate_with_viewer

    def failing_second(path, *, repo_root):
        if "smoke-train-02" in Path(path).parts:
            return {"status": "failed", "reason": "injected viewer failure"}
        return real_validate(path, repo_root=repo_root)

    monkeypatch.setattr(demo_pipeline, "validate_with_viewer", failing_second)
    viewer_out = tmp_path / "viewer-failure"
    rc = demo_pipeline.main(
        _synthetic_args(
            smoke, viewer_out, split="train", songs="smoke-train-01,smoke-train-02"
        )
    )
    assert rc == 1
    report = json.loads((viewer_out / "pipeline_report.json").read_text(encoding="utf-8"))
    assert report["checks"]["status"] == "failed"
    assert any("smoke-train-02" in failure for failure in report["checks"]["failures"])
    assert set(report["viewer_validation"]["per_song"]) == {"smoke-train-01", "smoke-train-02"}
    assert "failed checks" in capsys.readouterr().err

    monkeypatch.setattr(demo_pipeline, "validate_with_viewer", real_validate)
    real_run = demo_pipeline.run_song_artifacts
    artifact_calls = {"count": 0}

    def mismatching_second_artifact(*args, **kwargs):
        song = real_run(*args, **kwargs)
        artifact_calls["count"] += 1
        if artifact_calls["count"] == 2:
            # Break the second saved artifact's metrics on purpose: the real
            # cross-check must detect the inconsistency and fail the run.
            song["evaluation"]["true_positives"] += 5
        return song

    monkeypatch.setattr(demo_pipeline, "run_song_artifacts", mismatching_second_artifact)
    metrics_out = tmp_path / "metrics-failure"
    rc = demo_pipeline.main(
        _synthetic_args(
            smoke, metrics_out, split="train", songs="smoke-train-01,smoke-train-02"
        )
    )
    assert rc == 1
    report = json.loads((metrics_out / "pipeline_report.json").read_text(encoding="utf-8"))
    assert report["checks"]["status"] == "failed"
    assert any("canonical_metrics_match=false" in failure for failure in report["checks"]["failures"])
    assert [song["canonical_metrics_match"] for song in report["songs"]] == [True, False]


def test_audio_cli_with_fake_checkpoint_serializes_local_artifacts(
    smoke: dict, tmp_path: Path
) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the audio mode integration test")

    source = _write_tone(tmp_path / "source.wav", seconds=8.0, sample_rate=16000)
    source_hash = _sha256(source)
    out = tmp_path / "clip-run"
    rc = demo_pipeline.main(
        [
            "audio",
            "--audio",
            str(source),
            "--checkpoint",
            smoke["report"]["provenance"]["checkpoint"]["path"],
            "--out",
            str(out),
            "--start-seconds",
            "1.0",
            "--duration-seconds",
            "3.0",
            "--hop-seconds",
            "0.5",
            "--sample-rate",
            "16000",
            "--label",
            "clip-a",
            "--device",
            "cpu",
        ]
    )
    assert rc == 0

    report = json.loads((out / "pipeline_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "fake-encoder-audio-clip"
    assert report["checks"]["status"] == "ok"
    decode = report["provenance"]["input_audio"]
    assert decode["segment"] == {
        "start_seconds": 1.0,
        "duration_seconds": 3.0,
        "end_seconds": 4.0,
        "origin_semantics": "original-track absolute time of audio[0]",
    }
    assert decode["source_sha256"] == source_hash
    decoded = Path(decode["decoded_path"])
    assert decoded.is_file() and decode["decoded_sha256"] == _sha256(decoded)
    check = decode["duration_check"]
    assert check["ok"] is True
    assert check["requested_samples"] == 3 * 16000
    assert check["decoded_samples"] == 3 * 16000
    assert abs(check["delta_seconds"]) <= check["tolerance_seconds"]
    assert _sha256(source) == source_hash  # source never modified

    # No ground truth -> no accuracy fields, only objective checks.
    song = report["songs"][0]
    assert "evaluation" not in song and "metrics" not in song

    trajectory = Trajectory.load(song["trajectory"])
    assert trajectory.audio.track_start_seconds == 1.0
    assert trajectory.audio.duration_seconds == 3.0
    for track in trajectory.tracks:
        assert all(1.0 <= value <= 4.0 for value in track.center_times)
    predictions = PredictionData.load(
        str(Path(song["trajectory"]).parent / "prediction")
    )
    # Centers 1.0..4.0 at H=0.5; only the three fully-inside windows are valid.
    assert predictions.center_times.tolist() == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    assert predictions.center_valid.tolist() == [False, False, True, True, True, False, False]

    session = json.loads(Path(song["session"]).read_text(encoding="utf-8"))
    assert session["audio"]["track_start_seconds"] == 1.0
    assert session["audio"]["duration_check"]["ok"] is True
    manual = Path(song["session"]).with_name("manual_inspection.md").read_text(encoding="utf-8")
    assert "--bind 127.0.0.1" in manual

    viewer = report["viewer_validation"]
    assert viewer["status"] in {"ok", "skipped"}
    assert list(viewer["per_song"]) == ["clip-a"]


def test_truncated_decode_is_rejected_by_duration_check(
    smoke: dict, tmp_path: Path, monkeypatch, capsys
) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the audio decode test")

    ok = demo_pipeline._decoded_duration_check(8.0, 8 * 44100, 44100)
    assert ok["ok"] is True
    short = demo_pipeline._decoded_duration_check(8.0, 8 * 44100 - 2000, 44100)
    assert short["ok"] is False and short["delta_seconds"] < 0

    real_decode = demo_pipeline._decode_segment

    def truncating_decode(*, source, target, start_seconds, duration_seconds, sample_rate):
        decoded = real_decode(
            source=source,
            target=target,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
        )
        wav = read_wav(target)
        keep = max(1, wav.frames - int(0.05 * sample_rate))
        write_pcm16_wav(target, wav.samples[:keep, 0], sample_rate)
        return decoded

    monkeypatch.setattr(demo_pipeline, "_decode_segment", truncating_decode)
    source = _write_tone(tmp_path / "source.wav", seconds=4.0, sample_rate=16000)
    source_hash = _sha256(source)
    out = tmp_path / "truncated-run"
    rc = demo_pipeline.main(
        [
            "audio",
            "--audio",
            str(source),
            "--checkpoint",
            smoke["report"]["provenance"]["checkpoint"]["path"],
            "--out",
            str(out),
            "--start-seconds",
            "0.0",
            "--duration-seconds",
            "2.0",
            "--sample-rate",
            "16000",
            "--label",
            "clip-short",
            "--device",
            "cpu",
        ]
    )
    assert rc == 2
    stderr = capsys.readouterr().err
    assert "decoded segment" in stderr and "tolerance" in stderr
    assert _sha256(source) == source_hash  # source untouched
