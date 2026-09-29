"""Time-axis behaviour: shifting, track start offsets and sample-rate invariance."""

from __future__ import annotations

import numpy as np
import pytest

from aat.labels import LabelConfig, label_stems

from . import signals
from .support import activity_at, label_one


def test_shifting_the_waveform_shifts_labels_by_the_same_amount():
    base = signals.pulse_buffer(5.0, 1.0)
    shifted = signals.pulse_buffer(5.0, 2.0)
    base_result = label_one(base, 5.0)
    shifted_result = label_one(shifted, 5.0)

    offset = int(round(1.0 / signals.HOP_SECONDS))
    base_activity = base_result.activity.activity[:, 0]
    shifted_activity = shifted_result.activity.activity[:, 0]
    np.testing.assert_allclose(
        shifted_activity[offset:],
        base_activity[: base_activity.size - offset],
        atol=1e-9,
    )


def test_track_start_offsets_keep_absolute_times_and_labels():
    signal = signals.pulse_buffer(3.0, 1.0)
    at_zero = label_one(signal, 3.0, origin_seconds=0.0)
    at_seven = label_one(signal, 3.0, origin_seconds=7.0)

    assert at_seven.activity.center_times[0] == pytest.approx(7.0)
    assert at_seven.activity.center_times[-1] == pytest.approx(10.0)
    assert at_seven.activity.center_times.dtype == np.float64
    np.testing.assert_allclose(
        at_seven.activity.activity, at_zero.activity.activity, atol=1e-9
    )
    # The pulse is at absolute 8.0 s in the offset clip.
    index = int(np.argmin(np.abs(at_seven.activity.center_times - 8.0)))
    assert float(at_seven.activity.activity[index, 0]) > 0.9


def _centroid(result) -> float:
    probability = result.activity.activity[:, 0]
    times = result.activity.center_times
    return float(np.sum(times * probability) / np.sum(probability))


@pytest.mark.parametrize("sample_rate", [16000, 22050, 44100])
def test_sample_rate_changes_do_not_shift_labels(sample_rate):
    reference_rate = 16000
    signal = signals.pulse_buffer(3.0, 1.0, sample_rate=sample_rate)
    reference = signals.pulse_buffer(3.0, 1.0, sample_rate=reference_rate)
    result = label_one(signal, 3.0, sample_rate=sample_rate)
    baseline = label_one(reference, 3.0, sample_rate=reference_rate)

    np.testing.assert_allclose(
        result.activity.activity[:, 0],
        baseline.activity.activity[:, 0],
        atol=0.05,
    )
    assert abs(_centroid(result) - _centroid(baseline)) < 0.005
    assert np.argmax(result.activity.activity[:, 0]) == np.argmax(
        baseline.activity.activity[:, 0]
    )


def test_center_beyond_the_last_envelope_frame_is_inactive():
    # frame hop 0.04 s -> last envelope frame at 2.96 s for a 2.99 s stem.
    config = LabelConfig(frame_seconds=0.04, frame_hop_seconds=0.04)
    signal = signals.pulse_buffer(2.99, 0.5, pulse_seconds=2.49, freq=440.0, amplitude=0.5)
    centers = np.array([2.94, 2.96, 2.98], dtype=np.float64)
    result = label_stems(
        {"s01": signal},
        signals.SAMPLE_RATE,
        duration_seconds=2.99,
        center_times=centers,
        config=config,
    )
    probability = result.activity.activity[:, 0]
    assert probability[0] > 0.9
    assert probability[1] > 0.9
    assert probability[2] == 0.0


def test_center_grid_matches_shared_windowing_helper():
    from aat.windowing import center_times

    result = label_one(signals.pulse_buffer(3.0, 1.0), 3.0)
    expected = center_times(3.0, signals.HOP_SECONDS, origin_seconds=0.0)
    np.testing.assert_array_equal(result.activity.center_times, expected)
    assert activity_at(result, 1.0) > 0.9
