"""Absolute-time axis, center-window validity and rate invariance tests."""

from __future__ import annotations

import numpy as np
import pytest

from aat.labels import LabelConfig, label_stems

from . import signals

RATE = 16000


def run(samples, rate, duration, *, config=None, track_start=0.0):
    return label_stems(
        {"s01": samples},
        source_ids=("s01",),
        sample_rate=rate,
        duration_seconds=duration,
        track_start_seconds=track_start,
        config=config,
    )


def onset_and_end(rate):
    buffer = np.zeros(round(2.5 * rate))
    signals.add_tone(buffer, rate, 1.0, 1.3, amplitude=0.5)
    result = run(buffer, rate, 2.5)
    active = result.activity.activity[:, 0] > 0.5
    times = result.activity.center_times[active]
    assert times.size > 0
    return float(times[0]), float(times[-1]), active


def test_sample_rate_does_not_shift_labels_systematically():
    onsets = {}
    ends = {}
    masks = {}
    for rate in (16000, 22050, 44100):
        onset, end, mask = onset_and_end(rate)
        onsets[rate] = onset
        ends[rate] = end
        masks[rate] = mask
    for rate in (16000, 22050):
        assert abs(onsets[rate] - onsets[44100]) <= 0.041
        assert abs(ends[rate] - ends[44100]) <= 0.041
        mismatch = np.flatnonzero(masks[rate] != masks[44100])
        assert mismatch.size <= 4


def test_valid_mask_uses_model_center_window_only():
    buffer = np.zeros(RATE * 2)
    signals.add_tone(buffer, RATE, 1.0, 1.1, amplitude=0.5)
    result = run(buffer, RATE, 2.0)  # default center window is 2.0 s
    times = result.activity.center_times
    valid = result.activity.valid
    assert int(valid.sum()) == 1
    only = int(np.flatnonzero(valid)[0])
    assert times[only] == pytest.approx(1.0)


def test_invalid_centers_can_still_carry_energy_labels():
    buffer = np.zeros(RATE)
    signals.add_tone(buffer, RATE, 0.0, 0.2, amplitude=0.5)
    result = run(buffer, RATE, 1.0)  # a 2 s model window never fits into 1 s audio
    assert not result.activity.valid.any()
    assert result.activity.activity.any()


def test_energy_window_is_separate_from_center_window():
    buffer = np.zeros(RATE)
    signals.add_tone(buffer, RATE, 0.0, 0.2, amplitude=0.5)
    fine = LabelConfig()
    coarse = LabelConfig.from_dict(
        {**fine.to_dict(), "energy_window_seconds": 0.2}
    )
    first = run(buffer, RATE, 1.0, config=fine)
    second = run(buffer, RATE, 1.0, config=coarse)
    np.testing.assert_array_equal(first.activity.valid, second.activity.valid)
    assert first.config.energy_window_seconds == 0.05
    assert second.config.energy_window_seconds == 0.2


def test_hop_controls_row_grid():
    buffer = np.zeros(RATE * 2)
    config = LabelConfig.from_dict({"hop_seconds": 0.04})
    result = run(buffer, RATE, 2.0, config=config)
    assert result.activity.center_times.size == 51
    assert result.activity.hop_seconds == 0.04
