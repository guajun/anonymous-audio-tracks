"""Center-time windowing, sample/time conversion and validity-mask tests."""

from __future__ import annotations

import numpy as np
import pytest

from aat.contracts.errors import WindowError, WindowRangeError
from aat.windowing import (
    center_times,
    centered_window_bounds,
    extract_centered_windows,
    extract_windows_at_times,
    samples_to_seconds,
    seconds_to_samples,
    uniform_times,
    window_sample_count,
)

RAMP_16K = np.arange(16000, dtype=np.float64)


@pytest.mark.parametrize(
    ("sample_rate", "hop_samples", "window_samples"),
    [(16000, 320, 32000), (44100, 882, 88200)],
)
def test_time_conversion_at_16k_and_44k1(sample_rate, hop_samples, window_samples):
    assert seconds_to_samples(0.02, sample_rate) == hop_samples
    assert window_sample_count(2.0, sample_rate) == window_samples
    assert samples_to_seconds(hop_samples, sample_rate) == pytest.approx(0.02)
    assert samples_to_seconds(window_samples, sample_rate) == pytest.approx(2.0)


def test_seconds_to_samples_rounds_half_up():
    assert seconds_to_samples(0.5 / 16000.0, 16000) == 1
    assert seconds_to_samples(0.0001, 16000) == 2


def test_conversion_rejects_bad_arguments():
    with pytest.raises(WindowError):
        seconds_to_samples(0.02, 0)
    with pytest.raises(WindowError):
        samples_to_seconds(1.5, 16000)
    with pytest.raises(WindowError):
        window_sample_count(0.0, 16000)
    with pytest.raises(WindowError):
        seconds_to_samples(float("nan"), 16000)


def test_center_times_start_at_track_start_and_include_the_end():
    np.testing.assert_allclose(center_times(1.0, 0.25), [0.0, 0.25, 0.5, 0.75, 1.0])


def test_center_times_handles_non_integer_hop_spans():
    np.testing.assert_allclose(center_times(1.0, 0.3), [0.0, 0.3, 0.6, 0.9])
    np.testing.assert_allclose(center_times(0.5, 0.3), [0.0, 0.3])
    np.testing.assert_allclose(center_times(0.3, 0.3), [0.0, 0.3])


def test_center_times_keeps_absolute_origin_for_clips():
    np.testing.assert_allclose(
        center_times(0.1, 0.05, origin_seconds=12.0), [12.0, 12.05, 12.1]
    )


def test_center_times_empty_track_has_no_centers():
    assert center_times(0.0, 0.02).shape == (0,)
    assert center_times(0.0, 0.02, origin_seconds=5.0).shape == (0,)


def test_center_times_rejects_invalid_arguments():
    with pytest.raises(WindowError):
        center_times(1.0, 0.0)
    with pytest.raises(WindowError):
        center_times(1.0, -0.1)
    with pytest.raises(WindowError):
        center_times(1.0, float("nan"))
    with pytest.raises(WindowError):
        center_times(-0.1, 0.02)
    with pytest.raises(WindowError):
        center_times(1.0, 0.02, origin_seconds=-1.0)


def test_uniform_times_matches_grid():
    np.testing.assert_allclose(uniform_times(0.5, 0.1, 4), [0.5, 0.6, 0.7, 0.8])
    assert uniform_times(0.0, 0.1, 0).shape == (0,)


def test_first_window_is_left_zero_padded_with_mask():
    windows, valid = extract_windows_at_times(RAMP_16K, [0.0], 16000, 0.1)
    assert windows.shape == (1, 1600)
    assert not valid[0, :800].any()
    assert valid[0, 800:].all()
    np.testing.assert_allclose(windows[0, :800], 0.0)
    np.testing.assert_allclose(windows[0, 800:], RAMP_16K[:800])


def test_last_window_is_right_zero_padded_with_mask():
    windows, valid = extract_windows_at_times(RAMP_16K, [1.0], 16000, 0.1)
    assert valid[0, :800].all()
    assert not valid[0, 800:].any()
    np.testing.assert_allclose(windows[0, :800], RAMP_16K[15200:])
    np.testing.assert_allclose(windows[0, 800:], 0.0)


def test_interior_window_matches_exact_audio_slice():
    windows, valid = extract_windows_at_times(RAMP_16K, [0.5], 16000, 0.1)
    assert valid.all()
    np.testing.assert_allclose(windows[0], RAMP_16K[7200:8800])


def test_non_integer_span_end_window_padding():
    audio = np.arange(800, dtype=np.float64)  # 0.05 s at 16 kHz
    times = center_times(0.05, 0.02)
    np.testing.assert_allclose(times, [0.0, 0.02, 0.04])
    windows, valid = extract_windows_at_times(audio, times, 16000, 0.02)
    assert windows.shape == (3, 320)
    assert valid[0].sum() == 160  # left half is padding at t = 0
    assert valid[1].all()
    assert valid[2].all()  # center 0.04 -> [480, 800) is fully inside
    np.testing.assert_allclose(windows[2], audio[480:800])


def test_origin_offset_keeps_clip_times_absolute():
    windows, valid = extract_windows_at_times(
        RAMP_16K, [12.5], 16000, 0.1, origin_seconds=12.0
    )
    assert valid.all()
    np.testing.assert_allclose(windows[0], RAMP_16K[7200:8800])


def test_centers_outside_audio_are_rejected():
    with pytest.raises(WindowRangeError):
        extract_windows_at_times(RAMP_16K, [1.001], 16000, 0.1)
    with pytest.raises(WindowRangeError):
        extract_centered_windows(np.zeros(16), [17], 4)
    with pytest.raises(WindowError):
        extract_windows_at_times(RAMP_16K, [-0.001], 16000, 0.1)
    with pytest.raises(WindowError):
        extract_centered_windows(np.zeros(16), [-1], 4)


def test_non_increasing_or_non_finite_centers_are_rejected():
    with pytest.raises(WindowError):
        extract_windows_at_times(RAMP_16K, [0.1, 0.1], 16000, 0.1)
    with pytest.raises(WindowError):
        extract_windows_at_times(RAMP_16K, [0.2, 0.1], 16000, 0.1)
    with pytest.raises(WindowError):
        extract_windows_at_times(RAMP_16K, [float("nan")], 16000, 0.1)
    with pytest.raises(WindowError):
        extract_centered_windows(np.zeros(16), [2, 2], 4)
    with pytest.raises(WindowError):
        extract_centered_windows(np.zeros(16), [1.5], 4)


def test_empty_audio_yields_empty_window_batch():
    windows, valid = extract_windows_at_times(np.zeros(0), [], 16000, 0.1)
    assert windows.shape == (0, 1600)
    assert valid.shape == (0, 1600)
    windows, valid = extract_centered_windows(np.zeros(0), np.empty(0, dtype=np.int64), 16)
    assert windows.shape == (0, 16)
    assert valid.shape == (0, 16)


def test_explicit_center_on_empty_track_is_all_padding():
    windows, valid = extract_centered_windows(np.zeros(0), [0], 16)
    assert windows.shape == (1, 16)
    assert not valid.any()
    np.testing.assert_allclose(windows, 0.0)


def test_odd_window_puts_extra_sample_on_the_right():
    assert centered_window_bounds(5, 5) == (3, 8)
    assert centered_window_bounds(0, 4) == (-2, 2)


def test_44k1_window_extraction_shapes_and_slice():
    audio = np.arange(44100, dtype=np.float64)
    windows, valid = extract_windows_at_times(audio, [0.05], 44100, 0.1)
    assert windows.shape == (1, 4410)
    assert valid.all()
    np.testing.assert_allclose(windows[0], audio[0:4410])


def test_audio_must_be_one_dimensional():
    with pytest.raises(WindowError):
        extract_windows_at_times(np.zeros((2, 16)), [0.0], 16000, 0.1)
    with pytest.raises(WindowError):
        extract_centered_windows(np.zeros((2, 16)), [8], 4)
