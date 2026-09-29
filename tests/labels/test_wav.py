"""Dependency-free WAV decoding tests (stdlib-written and hand-built files)."""

from __future__ import annotations

import struct
import wave

import numpy as np
import pytest

from aat.labels import WavError, decode_wav_bytes, read_wav

RATE = 16000


def _write_pcm16(path, samples: np.ndarray, sample_rate: int = RATE) -> None:
    array = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    channels = 1 if array.ndim == 1 else array.shape[1]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(array.tobytes())


def _write_u8(path, samples: np.ndarray, sample_rate: int = RATE) -> None:
    array = np.round(np.clip(samples, -1.0, 1.0) * 127.5 + 128.0).astype(np.uint8)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(sample_rate)
        handle.writeframes(array.tobytes())


def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    padding = b"\x00" if len(payload) % 2 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + padding


def _wav_bytes(
    *,
    format_tag: int,
    channels: int,
    sample_rate: int,
    bits: int,
    payload: bytes,
    extensible_subformat: int | None = None,
) -> bytes:
    block_align = channels * bits // 8
    if extensible_subformat is None:
        fmt = struct.pack(
            "<HHIIHH", format_tag, channels, sample_rate, sample_rate * block_align, block_align, bits
        )
    else:
        fmt = struct.pack(
            "<HHIIHH", 0xFFFE, channels, sample_rate, sample_rate * block_align, block_align, bits
        )
        fmt += struct.pack("<H", 22)
        fmt += struct.pack("<H", bits)
        fmt += struct.pack("<I", 0)
        fmt += struct.pack("<H", extensible_subformat)
        fmt += bytes.fromhex("0000000000001000800000aa00389b71")
    body = b"WAVE" + _chunk(b"fmt ", fmt) + _chunk(b"data", payload)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_pcm16_mono_roundtrip(tmp_path):
    samples = 0.5 * np.sin(2.0 * np.pi * 440.0 * np.arange(RATE) / RATE)
    path = tmp_path / "mono.wav"
    _write_pcm16(path, samples)

    decoded, sample_rate = read_wav(path)
    assert sample_rate == RATE
    assert decoded.shape == (RATE, 1)
    np.testing.assert_allclose(decoded[:, 0], samples, atol=2.0 / 32768.0)


def test_pcm16_stereo_keeps_channels(tmp_path):
    left = 0.5 * np.sin(2.0 * np.pi * 220.0 * np.arange(RATE) / RATE)
    right = -0.25 * np.sin(2.0 * np.pi * 660.0 * np.arange(RATE) / RATE)
    path = tmp_path / "stereo.wav"
    _write_pcm16(path, np.stack([left, right], axis=1))

    decoded, sample_rate = read_wav(path)
    assert sample_rate == RATE
    assert decoded.shape == (RATE, 2)
    np.testing.assert_allclose(decoded[:, 0], left, atol=2.0 / 32768.0)
    np.testing.assert_allclose(decoded[:, 1], right, atol=2.0 / 32768.0)


def test_unsigned_8bit_is_centred(tmp_path):
    samples = 0.5 * np.sin(2.0 * np.pi * 440.0 * np.arange(RATE) / RATE)
    path = tmp_path / "u8.wav"
    _write_u8(path, samples)

    decoded, _ = read_wav(path)
    np.testing.assert_allclose(decoded[:, 0], samples, atol=1.5 / 128.0)


def test_float32_wav_is_decoded_exactly():
    values = np.array([-0.5, 0.0, 0.25, 0.75], dtype="<f4")
    data = _wav_bytes(format_tag=3, channels=1, sample_rate=44100, bits=32, payload=values.tobytes())

    decoded, sample_rate = decode_wav_bytes(data)
    assert sample_rate == 44100
    np.testing.assert_array_equal(decoded[:, 0], values.astype(np.float64))


def test_24bit_pcm_is_sign_extended():
    integers = np.array([-(1 << 23), -1, 0, 1, (1 << 23) - 1], dtype=np.int64)
    octets = bytearray()
    for value in integers:
        packed = int(value) & 0xFFFFFF
        octets += bytes((packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF))
    data = _wav_bytes(format_tag=1, channels=1, sample_rate=48000, bits=24, payload=bytes(octets))

    decoded, sample_rate = decode_wav_bytes(data)
    assert sample_rate == 48000
    np.testing.assert_allclose(decoded[:, 0], integers / float(1 << 23), atol=0.0)


def test_wave_format_extensible_pcm_is_supported():
    values = np.array([0.0, 16384, -16384], dtype="<i2")
    data = _wav_bytes(
        format_tag=1,
        channels=1,
        sample_rate=16000,
        bits=16,
        payload=values.tobytes(),
        extensible_subformat=1,
    )

    decoded, sample_rate = decode_wav_bytes(data)
    assert sample_rate == 16000
    np.testing.assert_allclose(decoded[:, 0], values.astype(np.float64) / 32768.0)


def test_compressed_formats_are_rejected():
    data = _wav_bytes(format_tag=2, channels=1, sample_rate=16000, bits=4, payload=b"\x00\x00")
    with pytest.raises(WavError, match="unsupported"):
        decode_wav_bytes(data)


def test_non_riff_payload_is_rejected():
    with pytest.raises(WavError, match="RIFF"):
        decode_wav_bytes(b"this is not a wave file at all")


def test_non_finite_float_samples_are_rejected():
    values = np.array([0.0, np.nan], dtype="<f4")
    data = _wav_bytes(format_tag=3, channels=1, sample_rate=16000, bits=32, payload=values.tobytes())
    with pytest.raises(WavError, match="non-finite"):
        decode_wav_bytes(data)


def test_read_wav_reports_missing_file_with_path(tmp_path):
    missing = tmp_path / "missing.wav"
    with pytest.raises(WavError, match="missing.wav"):
        read_wav(missing)
