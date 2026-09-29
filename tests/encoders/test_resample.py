"""16 kHz resampling for the AuT input (pure NumPy, no torch)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aat.encoders import (
    AUT_SAMPLE_RATE,
    EncoderInputError,
    prepare_audio,
    real_region_after_resample,
    resample_audio,
    resample_kernel_half_width,
)


def _tone(frequency: float, seconds: float, rate: int, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(round(seconds * rate))) / rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _peak_amplitude(signal: np.ndarray, rate: int, frequency: float) -> float:
    window = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(signal * window)) * 2.0 / np.sum(window)
    bin_index = int(round(frequency * signal.size / rate))
    return float(spectrum[bin_index])


def test_identity_when_rate_already_matches():
    signal = _tone(440.0, 0.1, AUT_SAMPLE_RATE)
    out = resample_audio(signal, AUT_SAMPLE_RATE)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, signal)


def test_output_length_matches_round_half_up_convention():
    # 1 s at 44.1 kHz -> 16000 samples.
    assert resample_audio(np.zeros(44100, dtype=np.float32), 44100).shape[0] == 16000
    # 1000 samples at 22.05 kHz -> floor(1000 * 16000/22050 + 0.5) = 726.
    assert resample_audio(np.zeros(1000, dtype=np.float32), 22050).shape[0] == math.floor(
        1000 * AUT_SAMPLE_RATE / 22050 + 0.5
    )
    assert resample_audio(np.zeros(0, dtype=np.float32), 44100).shape == (0,)


def test_dc_gain_is_preserved():
    signal = np.full(44100, 0.25, dtype=np.float32)
    out = resample_audio(signal, 44100)
    assert np.allclose(out[100:-100], 0.25, atol=1e-3)


def test_audible_tone_survives_44100_to_16000():
    signal = _tone(440.0, 0.5, 44100)
    out = resample_audio(signal, 44100)
    # Ignore filter transients at both edges.
    amplitude = _peak_amplitude(out[2400:-2400], AUT_SAMPLE_RATE, 440.0)
    assert amplitude == pytest.approx(0.5, rel=0.02)


def test_above_nyquist_tone_is_rejected_instead_of_aliased():
    # 10 kHz at 44.1 kHz cannot be represented at 16 kHz (Nyquist 8 kHz);
    # a naive decimator would fold it to 6 kHz.  The windowed-sinc filter
    # must attenuate it well below -40 dB instead.
    signal = _tone(10000.0, 0.5, 44100, amplitude=0.5)
    out = resample_audio(signal, 44100)
    interior = out[2400:-2400]
    alias = _peak_amplitude(interior, AUT_SAMPLE_RATE, 6000.0)
    assert alias < 0.5 * 10 ** (-40 / 20)


def test_prepare_audio_validates_shape_and_finiteness():
    with pytest.raises(EncoderInputError):
        prepare_audio(np.zeros((2, 100), dtype=np.float32), AUT_SAMPLE_RATE)
    with pytest.raises(EncoderInputError):
        prepare_audio(np.array([0.0, np.nan], dtype=np.float32), AUT_SAMPLE_RATE)
    with pytest.raises(EncoderInputError):
        resample_audio(np.zeros(10, dtype=np.float32), 0)
    with pytest.raises(EncoderInputError):
        resample_audio(np.zeros(10, dtype=np.float32), 44100, target_rate=0)
    with pytest.raises(EncoderInputError):
        resample_audio(np.zeros(10, dtype=np.float32), 44100, taps=0)


def test_prepare_audio_resamples_44k1_to_16k():
    signal = _tone(1000.0, 0.25, 44100)
    out = prepare_audio(signal, 44100)
    assert out.dtype == np.float32
    assert out.shape[0] == math.floor(0.25 * AUT_SAMPLE_RATE + 0.5)
    assert np.all(np.isfinite(out))


def test_resample_kernel_half_width_matches_filter_support():
    assert resample_kernel_half_width(16000, 16000) == 0
    assert resample_kernel_half_width(44100, 16000) == 45  # ceil(16 * 44100/16000)
    assert resample_kernel_half_width(22050, 16000) == 23


def test_real_region_same_rate_is_half_open():
    valid = np.zeros(4000, dtype=bool)
    valid[500:1001] = True
    assert real_region_after_resample(valid, 16000, 16000, 4000) == (500, 1001)
    endpoint = np.zeros(4000, dtype=bool)
    endpoint[:1319] = True
    assert real_region_after_resample(endpoint, 16000, 16000, 4000) == (0, 1319)
    assert real_region_after_resample(np.zeros(10, dtype=bool), 16000, 16000, 10) is None


def test_real_region_44k_non_integral_boundary_uses_kernel_margin():
    valid = np.zeros(44100, dtype=bool)
    valid[1001:2000] = True  # both borders interior to the buffer
    region = real_region_after_resample(valid, 44100, 16000, 16000)
    assert region == (380, 710)


def test_real_region_rejects_non_boolean_or_non_contiguous():
    with pytest.raises(EncoderInputError, match="contiguous"):
        real_region_after_resample(np.array([1, 0, 1, 1], dtype=bool), 16000, 16000, 4)
    with pytest.raises(EncoderInputError, match="boolean"):
        real_region_after_resample(np.ones(4, dtype=np.int8), 16000, 16000, 4)
