"""Short-time energy envelope behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from aat.labels import LabelConfig, compute_energy_envelope

from . import signals


def _envelope(audio, duration, *, config=None, origin=0.0, sample_rate=signals.SAMPLE_RATE):
    return compute_energy_envelope(
        audio,
        sample_rate,
        duration_seconds=duration,
        origin_seconds=origin,
        config=config or LabelConfig(),
    )


def test_full_scale_sine_reads_minus_3_dbfs():
    config = LabelConfig()
    envelope = _envelope(signals.tone(0.5, amplitude=1.0), 0.5, config=config)
    interior = envelope.db[envelope.frame_valid]
    assert interior.size > 10
    np.testing.assert_allclose(interior, -3.0103, atol=0.1)


def test_digital_silence_hits_the_numeric_floor():
    config = LabelConfig(floor_db=-100.0)
    envelope = _envelope(signals.silence(0.5), 0.5, config=config)
    assert np.allclose(envelope.db, -100.0)


def test_frame_grid_is_absolute_and_regular():
    config = LabelConfig(frame_hop_seconds=0.01)
    envelope = _envelope(signals.tone(0.5), 0.5, config=config, origin=2.0)
    assert envelope.frame_times.dtype == np.float64
    assert envelope.frame_times[0] == pytest.approx(2.0)
    np.testing.assert_allclose(np.diff(envelope.frame_times), 0.01, atol=1e-12)


def test_zero_padded_edge_frames_are_flagged_invalid():
    config = LabelConfig(frame_seconds=0.02)
    envelope = _envelope(signals.tone(1.0), 1.0, config=config)
    assert not envelope.frame_valid[0]
    assert not envelope.frame_valid[-1]
    assert envelope.frame_valid[envelope.frame_valid.size // 2]


def test_noise_floor_follows_the_quiet_region():
    audio = np.concatenate(
        [
            signals.tone(0.5, freq=440.0, amplitude=0.001),
            signals.tone(0.5, freq=440.0, amplitude=0.5),
        ]
    )
    config = LabelConfig(noise_floor_percentile=10.0)
    envelope = _envelope(audio, 1.0, config=config)
    # A -63 dBFS sine floor is well below the loud half, so the low percentile
    # must land there instead of on the tone.
    assert -70.0 < envelope.noise_floor_db < -55.0


def test_explicit_noise_floor_overrides_the_estimate():
    config = LabelConfig(noise_floor_db=-72.0)
    envelope = _envelope(signals.tone(0.5), 0.5, config=config)
    assert envelope.noise_floor_db == -72.0


def test_smoothing_keeps_frame_count_and_does_not_shift_time():
    config = LabelConfig(smoothing_seconds=0.05, frame_hop_seconds=0.01)
    unsmoothed = _envelope(signals.pulse_buffer(1.0, 0.5), 1.0)
    smoothed = _envelope(signals.pulse_buffer(1.0, 0.5), 1.0, config=config)
    assert smoothed.db.shape == unsmoothed.db.shape
    np.testing.assert_array_equal(smoothed.frame_times, unsmoothed.frame_times)
    # A 50 ms centered average cannot raise the pre-pulse silence.
    quiet = smoothed.frame_times < 0.45
    assert np.all(smoothed.db[quiet] <= config.floor_db + 1e-9)


def test_stereo_power_averaging_does_not_cancel_anti_phase():
    mono = signals.tone(0.5, amplitude=0.5)
    stereo = np.stack([mono, -mono], axis=1)
    config = LabelConfig()
    mono_env = _envelope(mono, 0.5, config=config)
    stereo_env = _envelope(stereo, 0.5, config=config)
    np.testing.assert_allclose(stereo_env.db, mono_env.db, atol=1e-9)
