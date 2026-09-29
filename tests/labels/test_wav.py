"""RIFF/WAVE decoding tests built from raw byte payloads."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from aat.labels import LabelError, read_wav


def wav_bytes(
    fmt_tag: int,
    channels: int,
    rate: int,
    bits: int,
    data: bytes,
    *,
    extensible: bool = False,
    extra_chunks: bytes = b"",
    data_size: int | None = None,
) -> bytes:
    block_align = channels * bits // 8
    byte_rate = rate * block_align
    if extensible:
        subformat = struct.pack("<H", fmt_tag) + b"\x00" * 14
        fmt_payload = struct.pack(
            "<HHIIHH", 0xFFFE, channels, rate, byte_rate, block_align, bits
        ) + struct.pack("<HHI", 22, bits, 0) + subformat
    else:
        fmt_payload = struct.pack(
            "<HHIIHH", fmt_tag, channels, rate, byte_rate, block_align, bits
        )
    fmt_chunk = (
        b"fmt "
        + struct.pack("<I", len(fmt_payload))
        + fmt_payload
        + (b"\x00" if len(fmt_payload) % 2 else b"")
    )
    size = len(data) if data_size is None else data_size
    data_chunk = (
        b"data"
        + struct.pack("<I", size)
        + data
        + (b"\x00" if len(data) % 2 else b"")
    )
    body = b"WAVE" + extra_chunks + fmt_chunk + data_chunk
    return b"RIFF" + struct.pack("<I", len(body)) + body


def write(tmp_path, blob: bytes):
    path = tmp_path / "test.wav"
    path.write_bytes(blob)
    return path


def test_pcm16_decodes_full_scale(tmp_path):
    values = np.array([-32768, -1, 0, 1, 32767], dtype="<i2")
    wav = read_wav(write(tmp_path, wav_bytes(1, 1, 8000, 16, values.tobytes())))
    assert (wav.sample_rate, wav.bits_per_sample, wav.encoding) == (8000, 16, "pcm")
    np.testing.assert_allclose(
        wav.samples[:, 0], values.astype(np.float64) / 32768.0, atol=1e-12
    )


def test_pcm8_is_unsigned_offset(tmp_path):
    wav = read_wav(write(tmp_path, wav_bytes(1, 1, 8000, 8, bytes([0, 128, 255]))))
    np.testing.assert_allclose(wav.samples[:, 0], [-1.0, 0.0, 127.0 / 128.0])


def test_pcm24_sign_extension(tmp_path):
    raw = b"".join(
        int(value & 0xFFFFFF).to_bytes(3, "little")
        for value in [0, 1, -1, 8388607, -8388608]
    )
    wav = read_wav(write(tmp_path, wav_bytes(1, 1, 8000, 24, raw)))
    expected = np.array([0, 1, -1, 8388607, -8388608], dtype=np.float64) / 8388608.0
    np.testing.assert_allclose(wav.samples[:, 0], expected)


def test_pcm32_and_float_formats(tmp_path):
    values32 = np.array([-2147483648, 0, 2147483647], dtype="<i4")
    wav = read_wav(write(tmp_path, wav_bytes(1, 1, 8000, 32, values32.tobytes())))
    np.testing.assert_allclose(
        wav.samples[:, 0], values32.astype(np.float64) / 2147483648.0
    )

    single = np.array([0.5, -0.25], dtype="<f4")
    wav = read_wav(write(tmp_path, wav_bytes(3, 1, 8000, 32, single.tobytes())))
    assert wav.encoding == "float"
    np.testing.assert_allclose(wav.samples[:, 0], [0.5, -0.25])

    double = np.array([0.125, -0.5], dtype="<f8")
    wav = read_wav(write(tmp_path, wav_bytes(3, 1, 8000, 64, double.tobytes())))
    np.testing.assert_allclose(wav.samples[:, 0], [0.125, -0.5])


def test_extensible_pcm16(tmp_path):
    values = np.array([100, -100], dtype="<i2")
    wav = read_wav(
        write(
            tmp_path,
            wav_bytes(1, 1, 16000, 16, values.tobytes(), extensible=True),
        )
    )
    np.testing.assert_allclose(
        wav.samples[:, 0], values.astype(np.float64) / 32768.0
    )


def test_unknown_chunks_with_odd_padding_are_skipped(tmp_path):
    extra = b"LIST" + struct.pack("<I", 3) + b"abc" + b"\x00"
    values = np.array([1000], dtype="<i2")
    wav = read_wav(
        write(tmp_path, wav_bytes(1, 1, 8000, 16, values.tobytes(), extra_chunks=extra))
    )
    assert wav.frames == 1


def test_stereo_shape(tmp_path):
    interleaved = np.array([[100, -100], [200, -200]], dtype="<i2")
    wav = read_wav(write(tmp_path, wav_bytes(1, 2, 8000, 16, interleaved.tobytes())))
    assert wav.channels == 2
    assert wav.samples.shape == (2, 2)


def test_compressed_format_is_rejected(tmp_path):
    with pytest.raises(LabelError, match="unsupported"):
        read_wav(write(tmp_path, wav_bytes(6, 1, 8000, 8, bytes([0, 1]))))


def test_truncated_chunk_is_rejected(tmp_path):
    with pytest.raises(LabelError, match="truncated"):
        read_wav(
            write(tmp_path, wav_bytes(1, 1, 8000, 16, b"\x00\x00", data_size=1024))
        )


def test_non_riff_and_partial_frames_are_rejected(tmp_path):
    with pytest.raises(LabelError, match="RIFF"):
        read_wav(write(tmp_path, b"not a wav file at all"))
    with pytest.raises(LabelError, match="whole number"):
        read_wav(write(tmp_path, wav_bytes(1, 1, 8000, 16, b"\x01\x02\x03")))


def test_non_finite_float_samples_are_rejected(tmp_path):
    raw = np.array([np.nan], dtype="<f4").tobytes()
    with pytest.raises(LabelError, match="NaN"):
        read_wav(write(tmp_path, wav_bytes(3, 1, 8000, 32, raw)))
