"""Protocol integration: ActivityData validation, valid mask and reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from aat.contracts import ActivityData
from aat.labels import LabelConfig, label_stems

from . import signals
from .support import label_one


def _two_source_result():
    active = signals.pulse_buffer(3.0, 1.0)
    quiet = signals.silence(3.0)
    return label_stems(
        {"s02": active, "s01": quiet},
        signals.SAMPLE_RATE,
        duration_seconds=3.0,
        sample_id="synth-test",
    )


def test_activity_saves_loads_and_keeps_source_column_order(tmp_path):
    result = _two_source_result()
    metadata_path, arrays_path = result.activity.save(tmp_path)
    assert metadata_path.name == "activity.json"
    assert arrays_path.name == "activity.npz"

    loaded = ActivityData.load(tmp_path)
    assert loaded.source_ids == ("s02", "s01")
    assert loaded.sample_id == "synth-test"
    assert loaded.sample_rate == signals.SAMPLE_RATE
    assert loaded.hop_seconds == signals.HOP_SECONDS
    np.testing.assert_array_equal(loaded.center_times, result.activity.center_times)
    np.testing.assert_array_equal(loaded.activity, result.activity.activity)
    np.testing.assert_array_equal(loaded.valid, result.activity.valid)
    assert loaded.center_times.dtype == np.float64
    assert loaded.activity.dtype == np.float32
    assert loaded.valid.dtype == np.bool_

    # The pulse stem is column 0 (s02); the silent stem is column 1 (s01).
    assert float(loaded.activity[:, 0].max()) > 0.9
    assert float(loaded.activity[:, 1].max()) == 0.0


def test_valid_mask_marks_centers_whose_model_window_leaves_the_audio():
    result = label_one(signals.pulse_buffer(3.0, 1.0), 3.0)
    valid = result.activity.valid
    times = result.activity.center_times

    assert times[0] == 0.0 and times[-1] == pytest.approx(3.0)
    assert not bool(valid[0])
    assert not bool(valid[-1])
    assert int(np.count_nonzero(valid)) == 51  # exactly [1.0, 2.0] at 20 ms hop
    assert bool(valid[int(np.argmin(np.abs(times - 1.0)))])
    assert bool(valid[int(np.argmin(np.abs(times - 2.0)))])


def test_label_params_rebuild_the_config_and_fingerprint(tmp_path):
    config = LabelConfig(
        abs_on_db=-50.0,
        hysteresis_db=2.0,
        release_seconds=0.1,
        probability_mode="binary",
    )
    result = label_one(signals.pulse_buffer(3.0, 1.0), 3.0, config=config)
    result.activity.save(tmp_path)

    loaded = ActivityData.load(tmp_path)
    params = loaded.label_params
    assert params["labeler_version"] == config.labeler_version
    assert params["origin_seconds"] == 0.0
    assert params["duration_seconds"] == 3.0
    rebuilt = LabelConfig.from_dict(params["config"])
    assert rebuilt == config
    assert params["config_sha256"] == config.fingerprint() == rebuilt.fingerprint()


def test_same_config_same_fingerprint_across_runs():
    signal = signals.pulse_buffer(2.0, 0.5)
    first = label_one(signal, 2.0)
    second = label_one(signal, 2.0)
    assert first.activity.label_params == second.activity.label_params
    np.testing.assert_array_equal(first.activity.activity, second.activity.activity)


def test_probabilities_are_finite_and_bounded():
    signal = signals.pulse_buffer(3.0, 1.0)
    result = label_one(signal, 3.0)
    probability = result.activity.activity
    assert np.all(np.isfinite(probability))
    assert float(probability.min()) >= 0.0
    assert float(probability.max()) <= 1.0


def test_empty_center_times_produce_zero_rows():
    signal = signals.pulse_buffer(2.0, 1.0)
    result = label_stems(
        {"s01": signal},
        signals.SAMPLE_RATE,
        duration_seconds=2.0,
        center_times=np.empty(0, dtype=np.float64),
    )
    assert result.activity.center_times.shape == (0,)
    assert result.activity.activity.shape == (0, 1)
    assert result.activity.valid.shape == (0,)


def test_no_sources_is_a_valid_empty_column_document(tmp_path):
    result = label_stems({}, signals.SAMPLE_RATE, duration_seconds=1.0)
    assert result.activity.source_ids == ()
    assert result.activity.activity.shape == (51, 0)
    assert result.summary["sources"] == []

    result.activity.save(tmp_path)
    loaded = ActivityData.load(tmp_path)
    assert loaded.activity.shape == (51, 0)
    assert loaded.source_ids == ()
