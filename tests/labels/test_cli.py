"""End-to-end CLI tests on a synthetic sample directory (stdlib WAV writer)."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

from aat.contracts import ActivityData, SampleManifest, SourceEntry, SourceRegistry, dump_json, load_json

from . import signals

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "label_sample.py"
RATE = 16000
DURATION = 4.0
ZERO_HASH = "0" * 64


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = RATE) -> None:
    array = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    channels = 1 if array.ndim == 1 else array.shape[1]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(array.tobytes())


def _build_sample_dir(tmp_path: Path, *, stem_rate: int = RATE) -> Path:
    """s01: 1 s tone from absolute 1.0 s. s02: digital silence."""

    sample_dir = tmp_path / "sample"
    (sample_dir / "stems").mkdir(parents=True)
    active = signals.silence(DURATION)
    signals.add_at(active, signals.tone(1.0, freq=440.0), 1.0)
    quiet = signals.silence(DURATION)
    _write_wav(sample_dir / "stems" / "s01.wav", active, stem_rate)
    _write_wav(sample_dir / "stems" / "s02.wav", quiet, stem_rate)
    _write_wav(sample_dir / "mix.wav", active + quiet, stem_rate)

    SourceRegistry(
        sources=(
            SourceEntry(source_id="s01", index=0, renderer="test-synth"),
            SourceEntry(source_id="s02", index=1, renderer="test-synth"),
        ),
        sample_id="synth-cli",
    ).save(sample_dir / "sources.json")
    dump_json(
        sample_dir / "controls.json",
        {
            "schema_version": "0.1.0",
            "kind": "controls",
            "sample_id": "synth-cli",
            "events": [
                {"time_seconds": 0.1, "source_id": "s02", "event_type": "note_on", "data": {}},
                {"time_seconds": 0.5, "source_id": "s01", "event_type": "note_on", "data": {}},
                {"time_seconds": 1.5, "source_id": "s01", "event_type": "note_off", "data": {}},
            ],
        },
    )

    paths = ("mix.wav", "stems/s01.wav", "stems/s02.wav", "sources.json", "controls.json")
    SampleManifest(
        sample_id="synth-cli",
        stage="rendered",
        seed=7,
        sample_rate=RATE,
        duration_seconds=DURATION,
        track_start_seconds=0.0,
        groups={"composition": "comp-cli", "preset": [], "sample_origin": []},
        mix_path="mix.wav",
        stem_paths={"s01": "stems/s01.wav", "s02": "stems/s02.wav"},
        sources_path="sources.json",
        controls_path="controls.json",
        versions={"renderer": "test"},
        content_sha256={path: ZERO_HASH for path in paths},
        tail_seconds=0.2,
    ).save(sample_dir / "manifest.json")
    return sample_dir


def _run(sample_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(sample_dir), *extra],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _value_at(data: ActivityData, seconds: float, column: int) -> float:
    index = int(np.argmin(np.abs(data.center_times - seconds)))
    return float(data.activity[index, column])


def test_cli_labels_sample_and_rewrites_manifest_to_labeled(tmp_path):
    sample_dir = _build_sample_dir(tmp_path)
    completed = _run(sample_dir)
    assert completed.returncode == 0, completed.stderr
    assert "labeled 2 stem(s)" in completed.stdout

    data = ActivityData.load(sample_dir)
    assert data.source_ids == ("s01", "s02")
    # s01 is audible 1.0-2.0 s: a control note_on at 0.5 s and note_off at 1.5 s
    # must not change the acoustic labels.
    assert _value_at(data, 1.4, 0) > 0.9
    assert _value_at(data, 1.6, 0) > 0.9
    assert _value_at(data, 0.5, 0) == 0.0
    # s02 is digital silence despite its control note_on.
    assert float(data.activity[:, 1].max()) == 0.0

    manifest = SampleManifest.load(sample_dir / "manifest.json")
    assert manifest.stage == "labeled"
    assert manifest.activity_metadata_path == "activity.json"
    assert manifest.activity_arrays_path == "activity.npz"
    assert manifest.versions["labeler"]
    for name in ("activity.json", "activity.npz"):
        digest = hashlib.sha256((sample_dir / name).read_bytes()).hexdigest()
        assert manifest.content_sha256[name] == digest
    assert manifest.content_sha256["mix.wav"] == ZERO_HASH  # renderer hashes preserved

    summary = load_json(sample_dir / "activity_summary.json")
    assert summary["kind"] == "activity_summary"
    assert summary["source_ids"] == ["s01", "s02"]
    silent = [source for source in summary["sources"] if source["source_id"] == "s02"][0]
    assert silent["silent"] is True
    assert any("control events" in warning for warning in summary["warnings"])


def test_cli_no_manifest_update_keeps_rendered_stage(tmp_path):
    sample_dir = _build_sample_dir(tmp_path)
    completed = _run(sample_dir, "--no-manifest-update")
    assert completed.returncode == 0, completed.stderr

    manifest = SampleManifest.load(sample_dir / "manifest.json")
    assert manifest.stage == "rendered"
    assert manifest.activity_metadata_path is None
    assert (sample_dir / "activity.npz").is_file()
    assert (sample_dir / "activity_summary.json").is_file()


def test_cli_config_override_is_recorded_and_applied(tmp_path):
    sample_dir = _build_sample_dir(tmp_path)
    config_path = tmp_path / "overrides.json"
    dump_json(config_path, {"probability_mode": "binary", "abs_on_db": -50.0})

    completed = _run(sample_dir, "--config", str(config_path))
    assert completed.returncode == 0, completed.stderr

    data = ActivityData.load(sample_dir)
    values = np.unique(data.activity)
    assert set(np.round(values, 6).tolist()) <= {0.0, 1.0}
    assert data.label_params["config"]["abs_on_db"] == -50.0
    summary = load_json(sample_dir / "activity_summary.json")
    assert summary["probability_mode"] == "binary"


def test_cli_rejects_sample_rate_mismatch(tmp_path):
    sample_dir = _build_sample_dir(tmp_path, stem_rate=8000)
    completed = _run(sample_dir)
    assert completed.returncode == 2
    assert "sample rate" in completed.stderr


def test_cli_rejects_missing_manifest(tmp_path):
    completed = _run(tmp_path)
    assert completed.returncode == 2
    assert "manifest" in completed.stderr


def test_cli_rejects_unknown_config_field(tmp_path):
    sample_dir = _build_sample_dir(tmp_path)
    config_path = tmp_path / "overrides.json"
    dump_json(config_path, {"not_a_field": 1})
    completed = _run(sample_dir, "--config", str(config_path))
    assert completed.returncode == 2
    assert "unknown" in completed.stderr
