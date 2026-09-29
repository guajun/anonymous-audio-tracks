"""Short-time acoustic energy envelope and threshold estimation.

The envelope is a per-sample mean-square value over a short rectangular window
(``config.energy_window_seconds``).  It is intentionally *not* the model center
window used by the protocol ``valid`` mask: the energy window only shapes the
acoustic activity proxy.

All levels are dBFS relative to full scale ``1.0`` of the decoded stem.  The
envelope is floored at :data:`SILENCE_DBFS` so digital silence stays finite and
JSON-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..windowing import seconds_to_samples
from .config import LabelConfig
from .errors import LabelError

#: -240 dBFS: far below a 16-bit LSB; keeps silence finite for JSON summaries.
SILENCE_DBFS = -240.0
_POWER_FLOOR = 10.0 ** (SILENCE_DBFS / 10.0)


def _check_sample_rate(sample_rate: Any) -> int:
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, np.integer))
        or int(sample_rate) < 1
    ):
        raise LabelError(f"sample_rate: expected an integer >= 1, got {sample_rate!r}")
    return int(sample_rate)


def _window_samples(window_seconds: Any, sample_rate: int) -> int:
    try:
        width = seconds_to_samples(float(window_seconds), sample_rate)
    except (TypeError, ValueError) as exc:
        raise LabelError(f"energy_window_seconds: {exc}") from exc
    if width < 1:
        raise LabelError(
            f"energy_window_seconds: {window_seconds!r} is shorter than one sample "
            f"at {sample_rate} Hz"
        )
    return width


def mean_square_envelope(
    audio: Any,
    sample_rate: int,
    window_seconds: float,
) -> np.ndarray:
    """Per-sample mean square over a centered rectangular window.

    ``audio`` may be mono ``(N,)`` or multi-channel ``(N, C)``; channels are
    combined with an equal-power average (mean of squared samples).  Window
    edges average only the samples that exist, so a constant tone keeps its
    level at the boundary instead of fading into zero padding.
    """

    rate = _check_sample_rate(sample_rate)
    array = np.asarray(audio, dtype=np.float64)
    if array.ndim == 1:
        power = array * array
    elif array.ndim == 2:
        if array.shape[1] < 1:
            raise LabelError("audio: expected at least one channel")
        power = np.mean(array * array, axis=1)
    else:
        raise LabelError(
            f"audio: expected 1-D or 2-D samples, got shape {tuple(array.shape)}"
        )
    if not np.all(np.isfinite(power)):
        raise LabelError("audio: contains NaN or infinite samples")

    width = _window_samples(window_seconds, rate)
    frames = power.shape[0]
    if frames == 0:
        return np.empty(0, dtype=np.float64)
    if width == 1:
        return power

    half = width // 2
    starts = np.arange(frames, dtype=np.int64) - half
    stops = starts + width
    clipped_starts = np.clip(starts, 0, frames)
    clipped_stops = np.clip(stops, 0, frames)
    totals = np.concatenate(([0.0], np.cumsum(power)))
    sums = totals[clipped_stops] - totals[clipped_starts]
    counts = np.maximum(clipped_stops - clipped_starts, 1)
    return np.maximum(sums / counts, 0.0)


def rms_envelope(audio: Any, sample_rate: int, window_seconds: float) -> np.ndarray:
    """Linear RMS envelope (per-sample, same length as the audio frames)."""

    return np.sqrt(mean_square_envelope(audio, sample_rate, window_seconds))


def envelope_db(audio: Any, sample_rate: int, window_seconds: float) -> np.ndarray:
    """RMS envelope in dBFS, floored at :data:`SILENCE_DBFS`."""

    mean_square = mean_square_envelope(audio, sample_rate, window_seconds)
    if mean_square.size == 0:
        return np.empty(0, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(mean_square, _POWER_FLOOR))


def estimate_noise_floor_db(envelope_values: Any, percentile: float) -> float:
    """Percentile of the envelope used as the local noise-floor estimate."""

    values = np.asarray(envelope_values, dtype=np.float64)
    if values.size == 0:
        return SILENCE_DBFS
    return float(np.percentile(values, float(percentile)))


@dataclass(frozen=True)
class LevelThresholds:
    """Resolved per-stem thresholds (all dBFS)."""

    peak_dbfs: float
    noise_floor_dbfs: float
    relative_to_peak_dbfs: float
    noise_floor_gate_dbfs: float
    on_dbfs: float
    off_dbfs: float

    def to_dict(self) -> dict[str, float]:
        return {
            "peak_dbfs": self.peak_dbfs,
            "noise_floor_dbfs": self.noise_floor_dbfs,
            "relative_to_peak_dbfs": self.relative_to_peak_dbfs,
            "noise_floor_gate_dbfs": self.noise_floor_gate_dbfs,
            "on_dbfs": self.on_dbfs,
            "off_dbfs": self.off_dbfs,
        }


def compute_thresholds(envelope_values: Any, config: LabelConfig) -> LevelThresholds:
    """Resolve the on/off thresholds from an envelope and the label config."""

    values = np.asarray(envelope_values, dtype=np.float64)
    if values.ndim != 1:
        raise LabelError("envelope: expected a 1-D array")
    peak = float(values.max()) if values.size else SILENCE_DBFS
    noise_floor = estimate_noise_floor_db(values, config.noise_floor_percentile)
    relative_to_peak = peak - config.peak_relative_threshold_db
    noise_floor_gate = noise_floor + config.noise_floor_margin_db
    adaptive = min(noise_floor_gate, relative_to_peak)
    on_dbfs = max(config.absolute_threshold_dbfs, adaptive)
    return LevelThresholds(
        peak_dbfs=peak,
        noise_floor_dbfs=noise_floor,
        relative_to_peak_dbfs=relative_to_peak,
        noise_floor_gate_dbfs=noise_floor_gate,
        on_dbfs=on_dbfs,
        off_dbfs=on_dbfs - config.hysteresis_db,
    )
