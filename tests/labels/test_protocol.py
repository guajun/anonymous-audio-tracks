"""Protocol validation and reproducible config snapshot tests."""

from __future__ import annotations

import numpy as np

from aat.contracts import ActivityData, load_json
from aat.labels import LabelConfig, label_stems

from . import signals

RATE = 16000


def build_result():
    tone = np.zeros(RATE * 2)
    signals.add_tone(tone, RATE, 0.5, 1.0, amplitude=0.5)
    return label_stems(
        {"s01": tone, "s02": np.zeros(RATE * 2)},
        source_ids=("s01", "s02"),
        sample_rate=RATE,
        duration_seconds=2.0,
        track_start_seconds=0.0,
        sample_id="synth-0001",
    )


def test_saved_activity_roundtrips_protocol_validation(tmp_path):
    result = build_result()
    result.activity.save(tmp_path)
    loaded = ActivityData.load(tmp_path)
    assert loaded.source_ids == ("s01", "s02")
    assert loaded.sample_rate == RATE
    assert loaded.activity.dtype == np.float32
    assert loaded.center_times.dtype == np.float64
    assert loaded.valid.dtype == np.bool_
    assert set(np.unique(loaded.activity)).issubset({0.0, 1.0})


def test_metadata_arrays_follow_the_frozen_protocol(tmp_path):
    result = build_result()
    result.activity.save(tmp_path)
    metadata = load_json(tmp_path / "activity.json")
    assert metadata["schema_version"] == "0.1.0"
    assert metadata["kind"] == "activity"
    assert metadata["arrays_path"] == "activity.npz"
    assert metadata["arrays"]["center_times"]["dtype"] == "float64"
    assert metadata["arrays"]["center_times"]["origin"] == "original_track_start"
    assert metadata["arrays"]["center_times"]["unit"] == "seconds"
    assert metadata["arrays"]["activity"]["dtype"] == "float32"
    assert metadata["arrays"]["activity"]["unit"] == "probability"
    assert metadata["arrays"]["valid"]["dtype"] == "bool"


def test_npz_keys_are_exactly_protocol_keys(tmp_path):
    result = build_result()
    result.activity.save(tmp_path)
    with np.load(tmp_path / "activity.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"center_times", "activity", "valid"}


def test_label_params_rebuild_the_config(tmp_path):
    result = build_result()
    result.activity.save(tmp_path)
    metadata = load_json(tmp_path / "activity.json")
    rebuilt = LabelConfig.from_label_params(metadata["label_params"])
    assert rebuilt == LabelConfig()
    assert rebuilt.config_sha256() == metadata["label_params"]["sha256"]
    assert rebuilt.config_sha256() == LabelConfig().config_sha256()
