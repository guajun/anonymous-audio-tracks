"""Dependency-free RIFF/WAVE reader for rendered stems.

The labeler must run with numpy + pytest only, so this module decodes the WAV
files the renderer writes without ``soundfile``/``scipy``.  Supported payloads
are uncompressed PCM (8/16/24/32-bit, including ``WAVE_FORMAT_EXTENSIBLE``) and
IEEE float (32/64-bit).  Compressed formats are rejected explicitly instead of
being mis-decoded.

Multichannel files are returned as ``(samples, channels)`` float64 in
``[-1, 1]`` for integer PCM.  Samples are *not* clipped, and non-finite float
samples are rejected.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import WavError

_FORMAT_PCM = 1
_FORMAT_FLOAT = 3
_FORMAT_EXTENSIBLE = 0xFFFE
_PCM_BITS = (8, 16, 24, 32)
_FLOAT_BITS = (32, 64)


@dataclass(frozen=True)
class WavFormat:
    format_tag: int
    channels: int
    sample_rate: int
    bits_per_sample: int
    block_align: int


def _parse_fmt(payload: bytes) -> WavFormat:
    if len(payload) < 16:
        raise WavError(f"fmt chunk: expected at least 16 bytes, got {len(payload)}")
    tag, channels, sample_rate, _byte_rate, block_align, bits = struct.unpack_from(
        "<HHIIHH", payload, 0
    )
    if tag == _FORMAT_EXTENSIBLE:
        if len(payload) < 40:
            raise WavError("fmt chunk: WAVE_FORMAT_EXTENSIBLE requires a 40-byte format")
        cb_size = struct.unpack_from("<H", payload, 16)[0]
        if cb_size < 22:
            raise WavError(f"fmt chunk: extensible cbSize {cb_size} is too small")
        tag = struct.unpack_from("<H", payload, 24)[0]
    if tag not in (_FORMAT_PCM, _FORMAT_FLOAT):
        raise WavError(
            f"unsupported WAV format tag {tag}; only uncompressed PCM (1) and "
            f"IEEE float (3) are supported"
        )
    if channels < 1:
        raise WavError(f"fmt chunk: channels must be >= 1, got {channels}")
    if sample_rate < 1:
        raise WavError(f"fmt chunk: sample_rate must be >= 1 Hz, got {sample_rate}")
    if tag == _FORMAT_PCM and bits not in _PCM_BITS:
        raise WavError(f"unsupported PCM bit depth {bits}")
    if tag == _FORMAT_FLOAT and bits not in _FLOAT_BITS:
        raise WavError(f"unsupported float bit depth {bits}")
    expected_align = channels * (bits // 8)
    if block_align not in (0, expected_align):
        raise WavError(
            f"fmt chunk: block_align {block_align} does not match "
            f"{channels} channel(s) x {bits} bit"
        )
    return WavFormat(
        format_tag=tag,
        channels=channels,
        sample_rate=sample_rate,
        bits_per_sample=bits,
        block_align=expected_align,
    )


def _decode_payload(raw: bytes, fmt: WavFormat) -> np.ndarray:
    channels = fmt.channels
    if fmt.format_tag == _FORMAT_PCM:
        if fmt.bits_per_sample == 8:
            frame_bytes = channels
            raw = raw[: len(raw) - len(raw) % frame_bytes]
            values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, channels)
            samples = (values.astype(np.float64) - 128.0) / 128.0
        elif fmt.bits_per_sample == 16:
            frame_bytes = 2 * channels
            raw = raw[: len(raw) - len(raw) % frame_bytes]
            values = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
            samples = values.astype(np.float64) / 32768.0
        elif fmt.bits_per_sample == 24:
            frame_bytes = 3 * channels
            raw = raw[: len(raw) - len(raw) % frame_bytes]
            octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            packed = (
                octets[:, 0].astype(np.int32)
                | (octets[:, 1].astype(np.int32) << 8)
                | (octets[:, 2].astype(np.int32) << 16)
            )
            packed = np.where(packed >= (1 << 23), packed - (1 << 24), packed)
            samples = packed.astype(np.float64).reshape(-1, channels) / float(1 << 23)
        else:  # 32-bit PCM
            frame_bytes = 4 * channels
            raw = raw[: len(raw) - len(raw) % frame_bytes]
            values = np.frombuffer(raw, dtype="<i4").reshape(-1, channels)
            samples = values.astype(np.float64) / float(1 << 31)
    else:
        dtype = "<f4" if fmt.bits_per_sample == 32 else "<f8"
        itemsize = fmt.bits_per_sample // 8
        frame_bytes = itemsize * channels
        raw = raw[: len(raw) - len(raw) % frame_bytes]
        samples = np.frombuffer(raw, dtype=dtype).reshape(-1, channels).astype(np.float64)

    if not np.all(np.isfinite(samples)):
        raise WavError("WAV payload contains non-finite samples")
    return samples


def decode_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode RIFF/WAVE bytes into ``(samples, channels)`` float64 and a rate."""

    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavError("not a RIFF/WAVE file")
    declared_size = int.from_bytes(data[4:8], "little")

    fmt: WavFormat | None = None
    data_chunks: list[bytes] = []
    offset = 12
    limit = min(len(data), 8 + declared_size) if declared_size else len(data)
    while offset + 8 <= limit:
        chunk_id = data[offset : offset + 4]
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        payload_start = offset + 8
        payload_end = min(payload_start + size, len(data))
        payload = data[payload_start:payload_end]
        if chunk_id == b"fmt ":
            fmt = _parse_fmt(payload)
        elif chunk_id == b"data" and payload:
            data_chunks.append(payload)
        if payload_end < payload_start + size:
            break  # truncated final chunk
        offset = payload_end + (size & 1)
    if fmt is None:
        raise WavError("missing fmt chunk")
    if not data_chunks:
        raise WavError("missing data chunk")
    return _decode_payload(b"".join(data_chunks), fmt), fmt.sample_rate


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read one WAV file; returns ``(samples, channels)`` float64 and rate."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise WavError(f"cannot read {source}: {exc}") from exc
    try:
        return decode_wav_bytes(payload)
    except WavError as exc:
        raise WavError(f"{source}: {exc}") from exc


__all__ = ["WavFormat", "decode_wav_bytes", "read_wav"]
