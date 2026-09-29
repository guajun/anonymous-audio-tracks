"""Audio checks and measurements for rendered artifacts.

Nothing here depends on DawDreamer, so the detection and tolerance logic is
unit-tested on synthetic arrays and reused for both the real renderer and the
mock harnesses.  Every measurement that ends up in ``render_report.json`` is
computed by these functions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import StemSumError
from .wavio import LSB

#: Absolute floor for dBFS reporting (silence is reported as this value instead
#: of ``-inf`` so the value stays JSON-serialisable).
DBFS_FLOOR = -200.0

#: Samples are treated as clipped at (or beyond) full scale.
CLIP_THRESHOLD = 1.0

#: Relative threshold used when detecting the first audible sample of a stem.
ONSET_RELATIVE_THRESHOLD = 0.05
ONSET_ABSOLUTE_THRESHOLD = 1e-5


def dbfs(value: float) -> float:
    """Convert a linear amplitude to dBFS, floored at :data:`DBFS_FLOOR`."""

    if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
        return DBFS_FLOOR
    return float(20.0 * math.log10(float(value)))


def peak(x: Any) -> float:
    array = np.asarray(x, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def rms(x: Any) -> float:
    array = np.asarray(x, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(array * array)))


def peak_dbfs(x: Any) -> float:
    return dbfs(peak(x))


def rms_dbfs(x: Any) -> float:
    return dbfs(rms(x))


def count_non_finite(x: Any) -> int:
    array = np.asarray(x)
    return int(np.count_nonzero(~np.isfinite(array)))


def count_clipped(x: Any) -> int:
    array = np.asarray(x, dtype=np.float64)
    if array.size == 0:
        return 0
    return int(np.count_nonzero(np.abs(array) >= CLIP_THRESHOLD))


def is_empty(x: Any) -> bool:
    array = np.asarray(x)
    return array.size == 0 or bool(np.all(array == 0))


def first_active_sample(x: Any) -> int | None:
    """Index of the first sample above the onset threshold, or ``None``.

    The threshold is the larger of a fixed absolute floor and a fraction of the
    signal peak, so a quiet but clean stem is not mistaken for silence.
    """

    array = np.asarray(x, dtype=np.float64)
    if array.size == 0:
        return None
    envelope = np.max(np.abs(array), axis=0) if array.ndim == 2 else np.abs(array)
    threshold = max(ONSET_ABSOLUTE_THRESHOLD, ONSET_RELATIVE_THRESHOLD * float(envelope.max()))
    indices = np.flatnonzero(envelope >= threshold)
    return int(indices[0]) if indices.size else None


def last_active_sample(x: Any) -> int | None:
    """Index of the last sample above the onset threshold, or ``None``."""

    array = np.asarray(x, dtype=np.float64)
    if array.size == 0:
        return None
    envelope = np.max(np.abs(array), axis=0) if array.ndim == 2 else np.abs(array)
    threshold = max(ONSET_ABSOLUTE_THRESHOLD, ONSET_RELATIVE_THRESHOLD * float(envelope.max()))
    indices = np.flatnonzero(envelope >= threshold)
    return int(indices[-1]) if indices.size else None


def tail_decay_ratio(x: Any, sample_rate: int, window_seconds: float = 0.1) -> float:
    """RMS of the final window divided by the overall RMS (0.0 for silence)."""

    array = np.asarray(x, dtype=np.float64)
    if array.size == 0 or sample_rate < 1:
        return 0.0
    window = max(1, int(round(window_seconds * sample_rate)))
    tail = array[..., -window:]
    overall = rms(array)
    if overall <= 0.0:
        return 0.0
    return float(rms(tail) / overall)


def stem_sum_tolerance_lsb(num_stems: int, extra_stems: int = 0) -> float:
    """Upper bound for ``sum(stems) - mix`` in int16 LSBs.

    Each of the ``num_stems`` stems and the mix is rounded to the nearest int16
    (half away from zero), so the worst-case error is half an LSB per rounded
    value plus a one-LSB guard for accumulated float rounding in the render
    graph.  ``extra_stems`` lets callers account for additional exported
    artifacts that participate in the same check (currently unused; kept
    explicit in the report).
    """

    if num_stems < 1:
        raise StemSumError(f"stem_sum_tolerance_lsb: need at least one stem, got {num_stems}")
    return 0.5 * (num_stems + 1 + extra_stems) + 1.0


def stem_sum_error_lsb(stems: list[Any], mix: Any) -> float:
    """Maximum absolute per-sample error of ``sum(stems) - mix`` in LSBs."""

    if not stems:
        raise StemSumError("stem_sum_error_lsb: no stems supplied")
    stacked = [np.asarray(stem, dtype=np.int64) for stem in stems]
    shape = stacked[0].shape
    for index, stem in enumerate(stacked):
        if stem.shape != shape:
            raise StemSumError(
                f"stem_sum_error_lsb: stem {index} shape {stem.shape} != {shape}"
            )
    mix_array = np.asarray(mix, dtype=np.int64)
    if mix_array.shape != shape:
        raise StemSumError(f"stem_sum_error_lsb: mix shape {mix_array.shape} != {shape}")
    total = np.sum(stacked, axis=0)
    return float(np.max(np.abs(total - mix_array)))


def check_stem_sum(stems: list[Any], mix: Any, num_stems: int | None = None) -> dict[str, float]:
    """Validate the stem/mix additivity invariant and return its evidence."""

    count = len(stems) if num_stems is None else int(num_stems)
    tolerance = stem_sum_tolerance_lsb(count)
    error = stem_sum_error_lsb(stems, mix)
    if error > tolerance:
        raise StemSumError(
            f"exported stems do not sum to the exported mix: max error {error:.3f} LSB "
            f"> tolerance {tolerance:.3f} LSB ({count} stems)"
        )
    return {
        "num_stems": float(count),
        "tolerance_lsb": float(tolerance),
        "max_abs_error_lsb": float(error),
        "max_abs_error_float": float(error * LSB),
    }


__all__ = [
    "CLIP_THRESHOLD",
    "DBFS_FLOOR",
    "ONSET_ABSOLUTE_THRESHOLD",
    "ONSET_RELATIVE_THRESHOLD",
    "check_stem_sum",
    "count_clipped",
    "count_non_finite",
    "dbfs",
    "first_active_sample",
    "is_empty",
    "last_active_sample",
    "peak",
    "peak_dbfs",
    "rms",
    "rms_dbfs",
    "stem_sum_error_lsb",
    "stem_sum_tolerance_lsb",
    "tail_decay_ratio",
]
