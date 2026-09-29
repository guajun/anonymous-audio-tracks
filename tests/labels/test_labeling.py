"""Center-time labeling behavior on programmatic waveforms."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aat.labels import LabelConfig, LabelError, label_stems

from . import signals

RATE = 16000
HOP = LabelConfig().hop_seconds


def label_one(samples, *, config=None, duration=None, track_start=0.0):
    config = config or LabelConfig()
    if duration is None:
        duration = np.asarray(samples).shape[0] / RATE
    return label_stems(
        {"s01": samples},
        source_ids=("s01",),
        sample_rate=RATE,
        duration_seconds=duration,
        track_start_seconds=track_start,
        config=config,
    )


def active_mask(result, column=0):
    return result.activity.activity[:, column] > 0.5


def at(result, seconds, column=0):
    index = int(round(seconds / result.config.hop_seconds))
    return bool(active_mask(result, column)[index])


def test_short_pulse_does_not_paint_the_whole_model_window():
    buffer = np.zeros(RATE * 4)
    signals.add_tone(buffer, RATE, 0.2, 0.5, amplitude=0.5)
    result = label_one(buffer)
    assert at(result, 0.3)
    assert not at(result, 1.5)  # 1.2 s after the pulse: a 2 s OR would be True
    assert not at(result, 0.0)
    active_times = result.activity.center_times[active_mask(result)]
    assert active_times[-1] < 0.75


def test_sustained_tone_is_active_throughout():
    buffer = np.zeros(RATE * 4)
    signals.add_tone(buffer, RATE, 0.5, 3.5, amplitude=0.25)
    result = label_one(buffer)
    assert not at(result, 0.1)
    assert at(result, 1.0)
    assert at(result, 2.0)
    assert at(result, 3.0)


def test_silence_is_zero_and_flagged():
    result = label_one(np.zeros(RATE), duration=1.0)
    assert not result.activity.activity.any()
    summary = result.summaries[0]
    assert summary.silent
    assert not summary.low_snr
    assert summary.segments == 0


def test_repeated_notes_form_separate_segments():
    buffer = np.zeros(RATE * 4)
    signals.add_tone(buffer, RATE, 0.2, 0.4, amplitude=0.5)
    signals.add_tone(buffer, RATE, 1.0, 1.2, amplitude=0.5)
    result = label_one(buffer)
    assert result.summaries[0].segments == 2
    assert not at(result, 0.6)


def test_slow_attack_note_on_is_not_a_forced_onset():
    buffer = np.zeros(RATE * 3)
    signals.add_ramp_tone(buffer, RATE, 0.5, 1.0, 2.5, peak_amplitude=1.0)
    config = replace(LabelConfig(), absolute_threshold_dbfs=-30.0)
    result = label_one(buffer, config=config, duration=3.0)
    assert not at(result, 0.5)
    first_active = result.activity.center_times[active_mask(result)][0]
    assert first_active > 0.53


def test_tail_after_note_off_can_stay_active():
    buffer = np.zeros(RATE * 3)
    signals.add_decaying_tone(
        buffer, RATE, 0.5, 1.0, 2.8, peak_amplitude=1.0, tau_seconds=0.2
    )
    result = label_one(buffer, duration=3.0)
    assert at(result, 0.9)  # sustained before note-off
    assert at(result, 1.5)  # 0.5 s after note-off
    assert at(result, 2.0)  # tail still above threshold
    assert not at(result, 2.75)


def test_shift_moves_labels_by_the_same_amount():
    base = np.zeros(RATE * 4)
    signals.add_tone(base, RATE, 1.0, 1.3, amplitude=0.5)
    shifted = np.zeros(RATE * 4)
    signals.add_tone(shifted, RATE, 2.0, 2.3, amplitude=0.5)
    first = label_one(base)
    second = label_one(shifted)
    step = int(round(1.0 / HOP))
    np.testing.assert_array_equal(
        second.activity.activity[step:], first.activity.activity[:-step]
    )
    np.testing.assert_array_equal(
        second.activity.activity[:step],
        np.zeros((step, 1), dtype=np.float32),
    )


def test_track_start_offsets_all_times():
    buffer = np.zeros(RATE * 4)
    signals.add_tone(buffer, RATE, 1.0, 1.3, amplitude=0.5)
    at_zero = label_one(buffer, track_start=0.0)
    at_twelve = label_one(buffer, track_start=12.0)
    assert at_twelve.activity.center_times[0] == 12.0
    np.testing.assert_array_equal(
        at_zero.activity.activity, at_twelve.activity.activity
    )
    np.testing.assert_array_equal(at_zero.activity.valid, at_twelve.activity.valid)
    assert at_twelve.summaries[0].first_active_seconds == pytest.approx(
        at_zero.summaries[0].first_active_seconds + 12.0
    )


def test_source_order_controls_columns():
    silence = np.zeros(RATE)
    tone = np.zeros(RATE)
    signals.add_tone(tone, RATE, 0.1, 0.9, amplitude=0.5)
    result = label_stems(
        {"sB": silence, "sA": tone},
        source_ids=("sB", "sA"),
        sample_rate=RATE,
        duration_seconds=1.0,
        track_start_seconds=0.0,
    )
    assert result.activity.source_ids == ("sB", "sA")
    assert not result.activity.activity[:, 0].any()
    assert result.activity.activity[:, 1].any()


def test_low_snr_flag_is_reported():
    buffer = np.zeros(RATE * 3)
    signals.add_white_noise(buffer, 0.02, seed=3)
    signals.add_tone(buffer, RATE, 0.5, 2.5, freq_hz=220.0, amplitude=0.04)
    result = label_one(buffer, duration=3.0)
    summary = result.summaries[0]
    assert summary.low_snr
    assert summary.snr_db < 12.0
    assert not summary.silent


def test_ambiguity_flag_for_signal_near_threshold():
    config = replace(LabelConfig(), absolute_threshold_dbfs=-50.0)
    buffer = np.zeros(RATE * 3)
    t = np.arange(buffer.size, dtype=np.float64) / RATE
    modulation = 1.0 + 0.6 * np.sin(2.0 * np.pi * 0.25 * t)
    buffer += 0.0045 * np.sin(2.0 * np.pi * 220.0 * t) * modulation
    result = label_one(buffer, config=config, duration=3.0)
    summary = result.summaries[0]
    assert summary.ambiguous
    assert summary.ambiguous_fraction > 0.2


def test_stem_validation_errors():
    tone = np.zeros(RATE)
    with pytest.raises(LabelError):
        label_stems(
            {"s01": tone},
            source_ids=("s01", "s02"),
            sample_rate=RATE,
            duration_seconds=1.0,
            track_start_seconds=0.0,
        )
    with pytest.raises(LabelError):
        label_stems(
            {"s01": tone[:8000], "s02": tone},
            source_ids=("s01", "s02"),
            sample_rate=RATE,
            duration_seconds=1.0,
            track_start_seconds=0.0,
        )
    with pytest.raises(LabelError):
        label_stems(
            {"s01": np.empty(0)},
            source_ids=("s01",),
            sample_rate=RATE,
            duration_seconds=1.0,
            track_start_seconds=0.0,
        )
    with pytest.raises(LabelError):
        label_stems(
            {"s01": tone},
            source_ids=("s01",),
            sample_rate=RATE,
            duration_seconds=0.0,
            track_start_seconds=0.0,
        )


def test_no_sources_yields_empty_columns():
    result = label_stems(
        {},
        source_ids=(),
        sample_rate=RATE,
        duration_seconds=1.0,
        track_start_seconds=0.0,
    )
    assert result.activity.activity.shape == (51, 0)
    assert result.summaries == ()
