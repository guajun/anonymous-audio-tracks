"""Short-time envelope and threshold tests on synthetic signals."""

from __future__ import annotations

import numpy as np
import pytest

from aat.labels import (
    SILENCE_DBFS,
    LabelConfig,
    LabelError,
    compute_thresholds,
    envelope_db,
    envelope_db_at_samples,
    estimate_noise_floor_db,
    mean_square_at_samples,
    mean_square_envelope,
    rms_envelope,
)


def sine(rate: int, seconds: float, *, amplitude: float = 0.5, freq: float = 440.0):
    t = np.arange(round(rate * seconds), dtype=np.float64) / rate
    return amplitude * np.sin(2.0 * np.pi * freq * t)


def test_constant_tone_level_is_rms_dbfs():
    rate = 16000
    tone = sine(rate, 1.0)
    levels = envelope_db(tone, rate, 0.05)
    assert levels.shape == tone.shape
    assert levels[8000] == pytest.approx(-9.0309, abs=0.1)


def test_silence_is_floored():
    levels = envelope_db(np.zeros(1600), 16000, 0.05)
    assert np.all(levels <= SILENCE_DBFS + 1e-6)
    assert np.all(levels >= SILENCE_DBFS - 1e-6)


def test_edges_average_available_samples():
    rate = 1000
    constant = np.full(1000, 0.5)
    levels = envelope_db(constant, rate, 0.05)
    assert levels[0] == pytest.approx(-6.0206, abs=1e-6)
    assert levels[-1] == pytest.approx(-6.0206, abs=1e-6)


def test_stereo_channels_use_power_average():
    rate = 1000
    tone = sine(rate, 2.0)
    stereo = np.stack([tone, np.zeros_like(tone)], axis=1)
    levels = envelope_db(stereo, rate, 0.05)
    assert levels[1000] == pytest.approx(-12.0412, abs=0.1)


def test_mean_square_and_rms_are_consistent():
    rate = 8000
    tone = sine(rate, 0.5)
    rms = rms_envelope(tone, rate, 0.05)
    np.testing.assert_allclose(
        rms, np.sqrt(mean_square_envelope(tone, rate, 0.05))
    )


def test_noise_floor_percentile_and_empty():
    values = np.full(100, SILENCE_DBFS)
    values[5:] = -60.0
    assert estimate_noise_floor_db(values, 10.0) == pytest.approx(-60.0)
    assert estimate_noise_floor_db(np.empty(0), 10.0) == SILENCE_DBFS


def test_thresholds_combine_absolute_relative_and_noise_floor():
    config = LabelConfig()

    silence_with_pulse = np.full(2000, SILENCE_DBFS)
    silence_with_pulse[100:120] = -10.0
    thresholds = compute_thresholds(silence_with_pulse, config)
    assert thresholds.on_dbfs == pytest.approx(config.absolute_threshold_dbfs)

    continuous = np.full(2000, -9.0)
    thresholds = compute_thresholds(continuous, config)
    assert thresholds.on_dbfs == pytest.approx(
        -9.0 - config.peak_relative_threshold_db
    )

    noise_bed = np.full(2000, -70.0)
    noise_bed[1500:1600] = -30.0
    thresholds = compute_thresholds(noise_bed, config)
    assert thresholds.on_dbfs == pytest.approx(config.absolute_threshold_dbfs)
    assert thresholds.off_dbfs == pytest.approx(
        thresholds.on_dbfs - config.hysteresis_db
    )


def test_nan_audio_is_rejected():
    with pytest.raises(LabelError):
        envelope_db(np.array([0.0, np.nan]), 16000, 0.05)


def test_mean_square_at_samples_uses_the_window_at_that_index():
    audio = np.ones(100, dtype=np.float64)
    indices = np.array([0, 50, 94, 99, 104, 105, 106])
    values = mean_square_at_samples(audio, 1000, 0.01, indices)  # 10-sample window
    assert values[0] == pytest.approx(1.0)  # [0, 5) at the start
    assert values[1] == pytest.approx(1.0)  # [45, 55)
    assert values[3] == pytest.approx(1.0)  # [94, 100) fully inside
    assert values[4] == pytest.approx(1.0)  # [99, 100): one real sample
    assert values[5] == 0.0  # past the end: not a repetition of sample 99
    assert values[6] == 0.0


def test_envelope_db_at_samples_is_floored_past_the_end():
    audio = np.ones(50, dtype=np.float64)
    levels = envelope_db_at_samples(audio, 1000, 0.01, np.array([49, 54, 55]))
    assert levels[0] == pytest.approx(0.0)
    assert levels[1] == pytest.approx(0.0)
    assert levels[2] == SILENCE_DBFS


def test_mean_square_at_samples_matches_full_envelope_in_range():
    rate = 8000
    tone = sine(rate, 0.5)
    envelope = mean_square_envelope(tone, rate, 0.05)
    indices = np.arange(tone.size, dtype=np.int64)
    np.testing.assert_allclose(
        mean_square_at_samples(tone, rate, 0.05, indices), envelope
    )
