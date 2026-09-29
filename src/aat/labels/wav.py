"""Dependency-free RIFF/WAVE reader for PCM (and IEEE float) stems.

Only the standard library and numpy are used.  Compressed WAV encodings are
rejected instead of guessed: a stem that cannot be decoded exactly must not
silently produce wrong activity labels.  Samples are returned as float64 in
``[-1, 1]`` for PCM and verbatim for float formats.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import LabelError

_RIFF = b"RIFF"
_WAVE = b"WAVE"
_FORMAT_PCM = 1
_FORMAT_FLOAT = 3
_FORMAT_EXTENSIBLE = 0xFFFE
_SUPPORTED_BITS = {
    _FORMAT_PCM: (8, 16, 24, 32),
    _FORMAT_FLOAT: (32, 64),
}
_EXTENSIBLE_SUBFORMAT_OFFSET = 24
_EXTENSIBLE_MIN_SIZE = 40


@dataclass(frozen=True)
class WavFile:
    """Decoded WAV payload with protocol-relevant metadata."""

    samples: np.ndarray  # float64, shape (frames, channels)
    sample_rate: int
    bits_per_sample: int
    encoding: str  # "pcm" or "float"

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


def _parse_fmt(payload: bytes, path: Any) -> tuple[int, int, int, int]:
    if len(payload) < 16:
        raise LabelError(f"{path}: malformed 'fmt ' chunk ({len(payload)} bytes)")
    tag, channels, sample_rate, _byte_rate, block_align, bits = struct.unpack_from(
        "<HHIIHH", payload, 0
    )
    if tag == _FORMAT_EXTENSIBLE:
        if len(payload) < _EXTENSIBLE_MIN_SIZE:
            raise LabelError(f"{path}: truncated WAVE_FORMAT_EXTENSIBLE 'fmt ' chunk")
        tag = struct.unpack_from(
            "<H", payload, _EXTENSIBLE_SUBFORMAT_OFFSET
        )[0]
    if tag not in _SUPPORTED_BITS:
        raise LabelError(
            f"{path}: unsupported WAV encoding (format tag {tag}); only PCM and "
            "IEEE float are supported"
        )
    if channels < 1:
        raise LabelError(f"{path}: invalid channel count {channels}")
    if sample_rate < 1:
        raise LabelError(f"{path}: invalid sample rate {sample_rate}")
    if bits not in _SUPPORTED_BITS[tag]:
        raise LabelError(
            f"{path}: unsupported {bits}-bit payload for format tag {tag}"
        )
    frame_bytes = channels * bits // 8
    if block_align not in (0, frame_bytes):
        raise LabelError(
            f"{path}: block align {block_align} does not match {channels} channel(s) "
            f"of {bits}-bit samples"
        )
    return tag, channels, sample_rate, bits


def _decode(tag: int, bits: int, payload: bytes, path: Any) -> np.ndarray:
    if tag == _FORMAT_PCM:
        if bits == 8:
            values = (np.frombuffer(payload, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        elif bits == 16:
            values = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
        elif bits == 24:
            raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
            gathered = (
                raw[:, 0].astype(np.int32)
                | (raw[:, 1].astype(np.int32) << 8)
                | (raw[:, 2].astype(np.int32) << 16)
            )
            signed = np.where(gathered & 0x800000, gathered - (1 << 24), gathered)
            values = signed.astype(np.float64) / float(1 << 23)
        else:  # 32
            values = np.frombuffer(payload, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        if bits == 32:
            values = np.frombuffer(payload, dtype="<f4").astype(np.float64)
        else:  # 64
            values = np.frombuffer(payload, dtype="<f8").astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise LabelError(f"{path}: contains NaN or infinite samples")
    return values


def read_wav(path: str | Path) -> WavFile:
    """Read a PCM/IEEE-float WAV file into float64 samples."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise LabelError(f"{source}: cannot read WAV file: {exc}") from exc
    if len(raw) < 12 or raw[:4] != _RIFF or raw[8:12] != _WAVE:
        raise LabelError(f"{source}: not a RIFF/WAVE file")

    fmt: tuple[int, int, int, int] | None = None
    data_parts: list[bytes] = []
    offset = 12
    while offset + 8 <= len(raw):
        chunk_id = raw[offset : offset + 4]
        size = struct.unpack_from("<I", raw, offset + 4)[0]
        payload_start = offset + 8
        if size == 0xFFFFFFFF:  # streaming data chunk: extend to end of file
            size = len(raw) - payload_start
        if payload_start + size > len(raw):
            raise LabelError(
                f"{source}: truncated chunk {chunk_id!r} (declared {size} bytes)"
            )
        if chunk_id == b"fmt ":
            fmt = _parse_fmt(raw[payload_start : payload_start + size], source)
        elif chunk_id == b"data":
            data_parts.append(raw[payload_start : payload_start + size])
        offset = payload_start + size + (size & 1)
    if offset != len(raw) and offset + 8 > len(raw):
        # A trailing partial chunk header is tolerated only when it is padding.
        if offset < len(raw) and raw[offset:].strip(b"\x00"):
            raise LabelError(f"{source}: malformed trailing bytes")
    if fmt is None:
        raise LabelError(f"{source}: missing 'fmt ' chunk")
    if not data_parts:
        raise LabelError(f"{source}: missing 'data' chunk")

    tag, channels, sample_rate, bits = fmt
    frame_bytes = channels * bits // 8
    payload = b"".join(data_parts)
    if len(payload) % frame_bytes != 0:
        raise LabelError(
            f"{source}: data chunk is not a whole number of {frame_bytes}-byte frames"
        )
    values = _decode(tag, bits, payload, source)
    samples = values.reshape(-1, channels)
    return WavFile(
        samples=samples,
        sample_rate=sample_rate,
        bits_per_sample=bits,
        encoding="pcm" if tag == _FORMAT_PCM else "float",
    )
