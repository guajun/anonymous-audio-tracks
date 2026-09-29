"""Short-time RMS energy envelope for one rendered stem.

The envelope is computed on its own short-frame grid (``frame_seconds`` /
``frame_hop_seconds`` from :class:`~aat.labels.config.LabelConfig`) with the
shared :mod:`aat.windowing` helpers, then converted to dBFS.  It is deliberately
much shorter than the model window: label values at a center time describe the
energy *around that center*, not "active anywhere in the 2-second model window".

Stereo/multichannel stems are handled by averaging per-channel frame power, so
anti-phase channels do not cancel each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aat.windowing import center_times as _center_times
from aat.windowing import extract_windows_at_times

from .config import LabelConfig
from .errors import LabelAudioError

#: Power floor added before taking the logarithm so digital silence maps to a
#: finite value that is then clamped to ``floor_db``.
_SILENCE_POWER = 1e-30


@dataclass(frozen=True)
class EnergyEnvelope:
    """Short-time energy of one stem on an absolute-time frame grid."""

    frame_times: np.ndarray  # (F,) float64, absolute original-track seconds
    db: np.ndarray  # (F,) float64, clamped to [floor_db, ...]
    frame_valid: np.ndarray  # (F,) bool, frame fully inside real audio
    noise_floor_db: float
    peak_db: float
    audio_duration_seconds: float


def audio_duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    samples = _as_samples(audio)
    return samples.shape[0] / _check_sample_rate(sample_rate)


def _check_sample_rate(sample_rate: int) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise LabelAudioError(
            f"sample_rate: expected an integer, got {type(sample_rate).__name__}"
        )
    rate = int(sample_rate)
    if rate < 1:
        raise LabelAudioError(f"sample_rate: must be >= 1 Hz, got {rate}")
    return rate


def _as_samples(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio)
    if samples.ndim not in (1, 2):
        raise LabelAudioError(
            f"stem audio: expected shape (samples,) or (samples, channels), "
            f"got {tuple(samples.shape)}"
        )
    if samples.ndim == 2 and samples.shape[1] < 1:
        raise LabelAudioError("stem audio: expected at least one channel")
    if not np.issubdtype(samples.dtype, np.number) or np.issubdtype(
        samples.dtype, np.complexfloating
    ):
        raise LabelAudioError(f"stem audio: expected a real numeric array, got {samples.dtype}")
    values = np.asarray(samples, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise LabelAudioError("stem audio: contains NaN or infinite samples")
    return values


def _channel_matrix(audio: np.ndarray) -> np.ndarray:
    """Return ``(samples, channels)`` float64; 1-D input becomes one channel."""

    samples = _as_samples(audio)
    if samples.ndim == 1:
        return samples[:, None]
    return samples


def _smooth_db(db: np.ndarray, smoothing_seconds: float, frame_hop_seconds: float) -> np.ndarray:
    width = int(round(smoothing_seconds / frame_hop_seconds))
    if width <= 1 or db.size == 0:
        return db
    if width % 2 == 0:
        width += 1  # keep a symmetric kernel with no half-frame time shift
    kernel = np.ones(width, dtype=np.float64)
    numerator = np.convolve(db, kernel, mode="same")
    denominator = np.convolve(np.ones_like(db), kernel, mode="same")
    return numerator / denominator


def compute_energy_envelope(
    audio: np.ndarray,
    sample_rate: int,
    *,
    duration_seconds: float,
    origin_seconds: float,
    config: LabelConfig,
) -> EnergyEnvelope:
    """Compute the dB envelope of one stem over ``[origin, origin+duration]``.

    ``duration_seconds`` is the authoritative manifest duration; it must not
    exceed the real audio length, otherwise the missing tail would be silently
    treated as silence.
    """

    rate = _check_sample_rate(sample_rate)
    channels = _channel_matrix(audio)
    n_samples, n_channels = channels.shape
    if n_samples == 0:
        raise LabelAudioError("stem audio: empty array")
    if not np.isfinite(origin_seconds) or origin_seconds < 0.0:
        raise LabelAudioError(f"origin_seconds: must be finite and >= 0, got {origin_seconds!r}")
    real_duration = n_samples / rate
    if duration_seconds > real_duration + 1e-6:
        raise LabelAudioError(
            f"duration_seconds: {duration_seconds} exceeds the stem audio length "
            f"{real_duration} at {rate} Hz"
        )
    # Manifest durations can be a hair longer than the decoded audio; clamping
    # keeps the shared windowing helper's range check from firing on rounding.
    effective_duration = min(float(duration_seconds), real_duration)

    frame_times = _center_times(
        effective_duration, config.frame_hop_seconds, origin_seconds=float(origin_seconds)
    )
    if frame_times.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return EnergyEnvelope(
            frame_times=empty,
            db=empty,
            frame_valid=np.empty(0, dtype=bool),
            noise_floor_db=_noise_floor_db(empty, np.empty(0, dtype=bool), config),
            peak_db=float(config.floor_db),
            audio_duration_seconds=real_duration,
        )

    total_power = np.zeros(frame_times.size, dtype=np.float64)
    frame_valid = np.ones(frame_times.size, dtype=bool)
    for channel in range(n_channels):
        windows, mask = extract_windows_at_times(
            channels[:, channel],
            frame_times,
            rate,
            config.frame_seconds,
            origin_seconds=float(origin_seconds),
        )
        total_power += np.mean(np.square(windows.astype(np.float64)), axis=1)
        frame_valid &= mask.all(axis=1)
    power = total_power / n_channels

    reference = config.full_scale**2
    db = 10.0 * np.log10(power / reference + _SILENCE_POWER)
    db = np.maximum(db, config.floor_db)
    db = _smooth_db(db, config.smoothing_seconds, config.frame_hop_seconds)

    return EnergyEnvelope(
        frame_times=frame_times,
        db=db,
        frame_valid=frame_valid,
        noise_floor_db=_noise_floor_db(db, frame_valid, config),
        peak_db=float(db.max()),
        audio_duration_seconds=real_duration,
    )


def _noise_floor_db(db: np.ndarray, frame_valid: np.ndarray, config: LabelConfig) -> float:
    if config.noise_floor_db is not None:
        return float(config.noise_floor_db)
    if db.size == 0:
        return float(config.floor_db)
    # Prefer frames fully inside real audio; zero-padded edge frames would bias
    # the estimate toward the numeric floor.
    pool = db[frame_valid] if bool(frame_valid.any()) else db
    return float(np.percentile(pool, config.noise_floor_percentile))


__all__ = ["EnergyEnvelope", "audio_duration_seconds", "compute_energy_envelope"]
