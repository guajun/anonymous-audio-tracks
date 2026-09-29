"""PCM WAV I/O and the audio-measurement helpers used for render evidence."""

from __future__ import annotations

import numpy as np
import pytest

from aat.render import analysis
from aat.render.errors import StemSumError
from aat.render.wavio import (
    LSB,
    dequantize_int16,
    quantize_int16,
    read_pcm16_wav,
    write_pcm16_wav,
)


def test_quantize_round_trip_within_half_lsb() -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(-1.0, 1.0, size=(2, 997)).astype(np.float32)
    quantized = quantize_int16(values)
    restored = dequantize_int16(quantized)
    assert np.max(np.abs(restored - values)) <= 0.5 * LSB + 1e-9


def test_quantize_clamps_extremes() -> None:
    values = np.array([[1.5, -1.5, 1.0, -1.0]], dtype=np.float32)
    quantized = quantize_int16(values)
    assert quantized[0, 0] == 32767
    assert quantized[0, 1] == -32768
    assert quantized[0, 2] == 32767
    assert quantized[0, 3] == -32768


def test_write_read_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(1)
    values = rng.uniform(-0.9, 0.9, size=(2, 1234)).astype(np.float32)
    path = tmp_path / "stem.wav"
    write_pcm16_wav(path, values, 44100)
    restored, metadata = read_pcm16_wav(path)
    assert metadata == {"channels": 2, "sample_rate": 44100, "frames": 1234}
    assert restored.dtype == np.int16
    assert restored.tobytes() == quantize_int16(values).tobytes()


def test_mono_write_is_mono(tmp_path) -> None:
    path = tmp_path / "mono.wav"
    write_pcm16_wav(path, np.zeros(64, dtype=np.float32), 8000)
    restored, metadata = read_pcm16_wav(path)
    assert metadata["channels"] == 1
    assert restored.shape == (1, 64)


def test_detection_helpers() -> None:
    signal = np.zeros((2, 100), dtype=np.float32)
    signal[:, 10:20] = 0.5
    signal[0, 50] = 1.5
    assert analysis.count_clipped(signal) == 1
    assert analysis.count_non_finite(signal) == 0
    assert analysis.first_active_sample(signal) == 10
    assert analysis.last_active_sample(signal) == 50
    assert analysis.is_empty(np.zeros((2, 10), dtype=np.float32))
    assert not analysis.is_empty(signal)
    assert analysis.dbfs(0.0) == analysis.DBFS_FLOOR

    corrupt = signal.copy()
    corrupt[1, 60] = np.nan
    assert analysis.count_non_finite(corrupt) == 1


def test_stem_sum_tolerance_and_check() -> None:
    assert analysis.stem_sum_tolerance_lsb(2) == 2.5
    assert analysis.stem_sum_tolerance_lsb(4) == 3.5
    stems = [np.full((1, 8), 5, dtype=np.int16), np.full((1, 8), 7, dtype=np.int16)]
    mix = np.full((1, 8), 12, dtype=np.int16)
    evidence = analysis.check_stem_sum(stems, mix)
    assert evidence["max_abs_error_lsb"] == 0.0
    with pytest.raises(StemSumError):
        analysis.check_stem_sum(stems, np.full((1, 8), 30, dtype=np.int16))
    with pytest.raises(StemSumError):
        analysis.stem_sum_error_lsb([np.zeros((1, 2), dtype=np.int16)], np.zeros((1, 3), dtype=np.int16))


def test_tail_decay_ratio() -> None:
    sample_rate = 1000
    decaying = np.zeros((1, 1000), dtype=np.float32)
    decaying[0, :100] = 1.0
    assert analysis.tail_decay_ratio(decaying, sample_rate, 0.1) == 0.0
    sustained = np.ones((1, 1000), dtype=np.float32)
    assert analysis.tail_decay_ratio(sustained, sample_rate, 0.1) == pytest.approx(1.0)
