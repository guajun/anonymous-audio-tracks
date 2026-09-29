"""Center-time windowing on the original-track time axis.

Conventions
-----------
* All times are seconds of the original track: ``t = 0`` is the start of the
  original composition.  ``origin_seconds`` is the absolute time of
  ``audio[0]``; a full-track array uses ``0.0``.  Clip-relative times are never
  produced.
* A center window of ``window_samples`` samples centered at sample ``c`` covers
  ``[c - window_samples // 2, c - window_samples // 2 + window_samples)``.  Even
  window lengths split equally; for odd lengths the extra sample is on the right.
* Whenever a window leaves the audio, the missing samples are zero-filled.  The
  returned boolean mask is ``True`` exactly where real audio samples were used,
  so ``mask.all()`` identifies windows fully inside the audio.
* Centers are validated: finite, non-negative, strictly increasing and inside
  ``[origin_seconds, origin_seconds + duration_seconds]``.  Anything else raises
  :class:`~aat.contracts.errors.WindowRangeError` or
  :class:`~aat.contracts.errors.WindowError` instead of silently clamping.

All helpers accept small synthetic arrays and never load real audio.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contracts.errors import WindowError, WindowRangeError


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convert seconds to a sample index, rounding half up.

    >>> seconds_to_samples(0.02, 16000)
    320
    >>> seconds_to_samples(0.02, 44100)
    882
    """

    rate = _check_sample_rate(sample_rate)
    value = _check_finite(seconds, "seconds")
    return math.floor(value * rate + 0.5)


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    """Convert a sample index to seconds on the same time axis."""

    rate = _check_sample_rate(sample_rate)
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)):
        raise WindowError(f"samples: expected an integer, got {type(samples).__name__}")
    return float(samples) / rate


def window_sample_count(window_seconds: float, sample_rate: int) -> int:
    """Window length in samples; a window shorter than one sample is an error."""

    count = seconds_to_samples(window_seconds, sample_rate)
    if count < 1:
        raise WindowError(
            f"window_seconds: {window_seconds!r} is shorter than one sample at "
            f"{sample_rate} Hz"
        )
    return count


def uniform_times(origin_seconds: float, hop_seconds: float, count: int) -> np.ndarray:
    """Absolute times ``origin + i * hop`` for ``i in range(count)`` (float64)."""

    origin = _check_finite(origin_seconds, "origin_seconds")
    hop = _check_positive(hop_seconds, "hop_seconds")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise WindowError(f"count: expected a non-negative integer, got {count!r}")
    if count == 0:
        return np.empty(0, dtype=np.float64)
    return origin + hop * np.arange(count, dtype=np.float64)


def center_times(
    duration_seconds: float,
    hop_seconds: float,
    *,
    origin_seconds: float = 0.0,
) -> np.ndarray:
    """Grid of window center times covering ``[origin, origin + duration]``.

    The grid starts exactly at ``origin_seconds`` and includes every
    ``origin + i * hop`` that does not exceed the end of the audio.  The end is
    therefore not necessarily hit (non-integer hop spans) but the last center is
    always at or before it.  A zero-length audio track has no center times.
    """

    duration = _check_finite(duration_seconds, "duration_seconds")
    hop = _check_positive(hop_seconds, "hop_seconds")
    origin = _check_finite(origin_seconds, "origin_seconds")
    if duration < 0.0:
        raise WindowError(f"duration_seconds: must be >= 0, got {duration}")
    if origin < 0.0:
        raise WindowError(f"origin_seconds: must be >= 0, got {origin}")
    if duration == 0.0:
        return np.empty(0, dtype=np.float64)

    end = origin + duration
    tolerance = 1e-9 * max(1.0, duration)
    count = math.floor(duration / hop + 1e-9) + 1
    times = origin + hop * np.arange(count, dtype=np.float64)
    while count > 1 and times[-1] > end + tolerance:
        count -= 1
        times = times[:count]
    return np.ascontiguousarray(times)


def centered_window_bounds(center_samples: int, window_samples: int) -> tuple[int, int]:
    """Half-open ``[start, stop)`` sample bounds of one centered window."""

    if isinstance(center_samples, bool) or not isinstance(center_samples, (int, np.integer)):
        raise WindowError(f"center_samples: expected an integer, got {center_samples!r}")
    width = _check_window_samples(window_samples)
    start = int(center_samples) - width // 2
    return start, start + width


def extract_centered_windows(
    audio: Any,
    center_samples: Any,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract zero-padded centered windows from a 1-D audio array.

    Returns ``(windows, valid)`` with shapes ``(N, window_samples)`` and
    ``(N, window_samples)``; ``valid`` is ``True`` where ``audio`` contributed a
    real sample.  Centers must be strictly increasing and lie in
    ``[0, len(audio)]``.
    """

    samples = np.asarray(audio)
    if samples.ndim != 1:
        raise WindowError(f"audio: expected a 1-D array, got shape {tuple(samples.shape)}")
    width = _check_window_samples(window_samples)
    centers = _center_sample_array(center_samples)
    length = samples.shape[0]

    if centers.size and (centers[0] < 0 or centers[-1] > length):
        raise WindowRangeError(
            f"center_samples: centers must lie in [0, {length}] for audio of "
            f"length {length}, got range [{centers[0]}, {centers[-1]}]"
        )

    output_dtype = np.result_type(samples.dtype, np.float32)
    windows = np.zeros((centers.size, width), dtype=output_dtype)
    valid = np.zeros((centers.size, width), dtype=bool)
    for row, center in enumerate(centers):
        start, stop = centered_window_bounds(int(center), width)
        source_start = max(start, 0)
        source_stop = min(stop, length)
        if source_stop > source_start:
            offset = source_start - start
            span = source_stop - source_start
            windows[row, offset : offset + span] = samples[source_start:source_stop]
            valid[row, offset : offset + span] = True
    return windows, valid


def extract_windows_at_times(
    audio: Any,
    center_seconds: Any,
    sample_rate: int,
    window_seconds: float,
    *,
    origin_seconds: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute-time wrapper around :func:`extract_centered_windows`.

    ``center_seconds`` are absolute original-track times; ``audio[0]`` is at
    ``origin_seconds``.  ``window_seconds`` is converted to samples with
    :func:`window_sample_count`.  Centers must be finite, non-negative, strictly
    increasing and inside ``[origin_seconds, origin_seconds + duration]``.
    """

    samples = np.asarray(audio)
    if samples.ndim != 1:
        raise WindowError(f"audio: expected a 1-D array, got shape {tuple(samples.shape)}")
    rate = _check_sample_rate(sample_rate)
    width = window_sample_count(window_seconds, rate)
    origin = _check_finite(origin_seconds, "origin_seconds")
    if origin < 0.0:
        raise WindowError(f"origin_seconds: must be >= 0, got {origin}")

    times = _time_array(center_seconds)
    duration = samples.shape[0] / rate
    end = origin + duration
    tolerance = 1e-9 * max(1.0, duration)
    if times.size and (times[0] < origin - tolerance or times[-1] > end + tolerance):
        raise WindowRangeError(
            f"center_seconds: centers must lie in [{origin}, {end}] for audio of "
            f"duration {duration} at {rate} Hz, got range [{times[0]}, {times[-1]}]"
        )

    local_samples = np.floor((times - origin) * rate + 0.5)
    local_samples = np.clip(local_samples, 0.0, float(samples.shape[0])).astype(np.int64)
    return extract_centered_windows(samples, local_samples, width)


def _check_sample_rate(sample_rate: Any) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise WindowError(f"sample_rate: expected an integer, got {type(sample_rate).__name__}")
    rate = int(sample_rate)
    if rate < 1:
        raise WindowError(f"sample_rate: must be >= 1 Hz, got {rate}")
    return rate


def _check_window_samples(window_samples: Any) -> int:
    if isinstance(window_samples, bool) or not isinstance(window_samples, (int, np.integer)):
        raise WindowError(
            f"window_samples: expected an integer, got {type(window_samples).__name__}"
        )
    width = int(window_samples)
    if width < 1:
        raise WindowError(f"window_samples: must be >= 1, got {width}")
    return width


def _check_finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise WindowError(f"{path}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise WindowError(f"{path}: must be finite, got {value!r}")
    return number


def _check_positive(value: Any, path: str) -> float:
    number = _check_finite(value, path)
    if number <= 0.0:
        raise WindowError(f"{path}: must be > 0, got {number}")
    return number


def _time_array(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise WindowError(f"center_seconds: expected a 1-D array, got shape {tuple(array.shape)}")
    if array.size == 0:
        return np.ascontiguousarray(array.astype(np.float64))
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise WindowError(f"center_seconds: expected a numeric array, got dtype {array.dtype}")
    array = array.astype(np.float64)
    if not np.all(np.isfinite(array)):
        raise WindowError("center_seconds: contains NaN or infinite values")
    if np.any(array < 0.0):
        raise WindowError("center_seconds: times must be >= 0")
    if array.size > 1 and not np.all(np.diff(array) > 0):
        raise WindowError("center_seconds: values must be strictly increasing")
    return np.ascontiguousarray(array)


def _center_sample_array(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise WindowError(f"center_samples: expected a 1-D array, got shape {tuple(array.shape)}")
    if array.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise WindowError("center_samples: contains NaN or infinite values")
        if not np.all(np.mod(array, 1) == 0):
            raise WindowError("center_samples: expected integral sample indices")
        array = array.astype(np.int64)
    elif not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise WindowError(f"center_samples: expected an integer array, got dtype {array.dtype}")
    array = array.astype(np.int64)
    if array.size > 1 and not np.all(np.diff(array) > 0):
        raise WindowError("center_samples: values must be strictly increasing")
    return np.ascontiguousarray(array)


__all__ = [
    "WindowError",
    "WindowRangeError",
    "center_times",
    "centered_window_bounds",
    "extract_centered_windows",
    "extract_windows_at_times",
    "samples_to_seconds",
    "seconds_to_samples",
    "uniform_times",
    "window_sample_count",
]
