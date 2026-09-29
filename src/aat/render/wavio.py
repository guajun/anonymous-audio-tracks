"""Minimal PCM WAV I/O built on the standard library's :mod:`wave`.

The renderer exports 16-bit PCM WAV so downstream tooling (and the tests) can
read audio without NumPy, SciPy, soundfile or any other extra dependency.  A
single shared quantizer keeps the stem-sum tolerance well defined: every sample
is rounded half away from zero and clamped to the signed 16-bit range, so one
rounded value differs from its float source by at most half an LSB.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from .errors import RenderValidationError

#: One 16-bit least significant bit in normalized float units.
LSB = 1.0 / 32768.0


def quantize_int16(samples: Any) -> np.ndarray:
    """Round float audio in ``[-1, 1]`` to int16, half away from zero."""

    values = np.asarray(samples, dtype=np.float64)
    clipped = np.clip(values, -1.0, 1.0)
    scaled = clipped * 32768.0
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    return np.clip(rounded, -32768.0, 32767.0).astype(np.int16)


def dequantize_int16(samples: Any) -> np.ndarray:
    """Convert int16 audio back to float32 in ``[-1, 1]``."""

    return (np.asarray(samples, dtype=np.float32) / 32768.0).astype(np.float32)


def _as_channels_first(samples: Any, path: str) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise RenderValidationError(
            f"{path}: expected a 1-D or 2-D array, got shape {tuple(array.shape)}"
        )
    if array.shape[0] not in (1, 2):
        raise RenderValidationError(f"{path}: expected 1 or 2 channels, got {array.shape[0]}")
    return np.ascontiguousarray(array)


def write_pcm16_wav(path: str | Path, samples: Any, sample_rate: int) -> Path:
    """Write ``(channels, samples)`` float audio as a 16-bit PCM WAV file."""

    array = _as_channels_first(samples, "wav")
    if not isinstance(sample_rate, int) or sample_rate < 1:
        raise RenderValidationError(f"wav sample_rate: expected a positive integer, got {sample_rate!r}")
    interleaved = quantize_int16(array).T.reshape(-1).astype("<i2", copy=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(array.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(interleaved.tobytes())
    return target


def read_pcm16_wav(path: str | Path) -> tuple[np.ndarray, dict[str, int]]:
    """Read a 16-bit PCM WAV file as ``(channels, samples)`` int16 + metadata."""

    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            params = handle.getparams()
            frames = handle.readframes(params.nframes)
    except wave.Error as exc:
        raise RenderValidationError(f"{source.name}: not a readable WAV file: {exc}") from exc
    if params.sampwidth != 2:
        raise RenderValidationError(
            f"{source.name}: expected 16-bit PCM, got sample width {params.sampwidth}"
        )
    data = np.frombuffer(frames, dtype="<i2")
    if params.nchannels > 0:
        data = data.reshape(-1, params.nchannels).T
    else:
        data = data.reshape(0, 0)
    metadata = {
        "channels": int(params.nchannels),
        "sample_rate": int(params.framerate),
        "frames": int(params.nframes),
    }
    return np.ascontiguousarray(data), metadata


__all__ = [
    "LSB",
    "dequantize_int16",
    "quantize_int16",
    "read_pcm16_wav",
    "write_pcm16_wav",
]
