"""Center-time activation from a stem energy envelope.

The pipeline is short-window and center-local:

1. a dB envelope is computed on a short frame grid (:mod:`aat.labels.energy`);
2. the envelope is interpolated at the requested center times;
3. an absolute/relative threshold pair and a hysteresis state machine with a
   configurable release/tail produce the active state at each center;
4. the probability is either the gated ramp between the thresholds (``soft``)
   or the state itself (``binary``).

MIDI/parameter control events never enter this module.  A note-on is not forced
to be an acoustic onset and a note-off does not end activity; both come from the
rendered audio only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LabelConfig
from .energy import audio_duration_seconds, compute_energy_envelope
from .errors import LabelAudioError


@dataclass(frozen=True)
class StemActivation:
    """Per-center activation of one stem plus the diagnostics behind it."""

    activity: np.ndarray  # (T,) float64 in [0, 1]
    active_state: np.ndarray  # (T,) bool, hysteresis state
    db: np.ndarray  # (T,) float64, envelope dB at each center
    noise_floor_db: float
    on_db: float
    off_db: float
    peak_db: float
    ambiguous_fraction: float
    audio_duration_seconds: float
    frame_count: int


def resolve_thresholds(noise_floor_db: float, config: LabelConfig) -> tuple[float, float]:
    """Return ``(on_db, off_db)`` from absolute/relative parts and hysteresis."""

    on_db = max(float(config.abs_on_db), float(noise_floor_db) + float(config.rel_on_db))
    off_db = on_db - float(config.hysteresis_db)
    return on_db, off_db


def hysteresis_state(
    times: np.ndarray,
    db: np.ndarray,
    on_db: float,
    off_db: float,
    release_seconds: float,
) -> np.ndarray:
    """Boolean activation with hysteresis and a release/tail hold.

    * crossing ``db >= on_db`` activates immediately;
    * while active, dropping to ``db <= off_db`` starts a release timer, and the
      state falls after the level stayed at or below ``off_db`` for
      ``release_seconds`` -- this keeps short dropouts and note tails active;
    * rising back above ``off_db`` cancels the release timer without requiring
      ``on_db`` again (classic hysteresis).
    """

    state = np.zeros(int(times.size), dtype=bool)
    active = False
    below_start: float | None = None
    for index in range(int(times.size)):
        value = float(db[index])
        if value >= on_db:
            active = True
            below_start = None
        elif value <= off_db:
            if active:
                if below_start is None:
                    below_start = float(times[index])
                if float(times[index]) - below_start >= release_seconds:
                    active = False
                    below_start = None
        else:
            below_start = None
        state[index] = active
    return state


def probability_curve(
    db: np.ndarray,
    state: np.ndarray,
    on_db: float,
    off_db: float,
    mode: str,
) -> np.ndarray:
    """Map dB + hysteresis state to probabilities in ``[0, 1]``."""

    if mode == "binary":
        return state.astype(np.float64)
    if on_db > off_db:
        level = np.clip((db - off_db) / (on_db - off_db), 0.0, 1.0)
    else:  # hysteresis disabled: a hard threshold is the only definition
        level = (db >= on_db).astype(np.float64)
    return np.clip(level * state.astype(np.float64), 0.0, 1.0)


def label_stem(
    audio: np.ndarray,
    sample_rate: int,
    center_times: np.ndarray,
    *,
    origin_seconds: float = 0.0,
    config: LabelConfig | None = None,
    duration_seconds: float | None = None,
) -> StemActivation:
    """Label one stem at ``center_times`` (absolute original-track seconds).

    ``center_times`` must be finite, non-negative and strictly increasing.
    ``duration_seconds`` overrides the envelope range (the manifest duration);
    by default the real audio duration is used.
    """

    config = config or LabelConfig()
    times = np.asarray(center_times, dtype=np.float64)
    if times.ndim != 1:
        raise LabelAudioError(f"center_times: expected a 1-D array, got {tuple(times.shape)}")
    if not np.all(np.isfinite(times)):
        raise LabelAudioError("center_times: contains NaN or infinite values")
    if np.any(times < 0.0):
        raise LabelAudioError("center_times: values must be >= 0")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise LabelAudioError("center_times: values must be strictly increasing")

    real_duration = audio_duration_seconds(audio, sample_rate)
    if duration_seconds is None:
        span = float(times[-1] - origin_seconds) if times.size else 0.0
        duration_seconds = max(real_duration, span)
    envelope = compute_energy_envelope(
        audio,
        sample_rate,
        duration_seconds=float(duration_seconds),
        origin_seconds=float(origin_seconds),
        config=config,
    )

    if times.size == 0 or envelope.frame_times.size == 0:
        noise_floor = envelope.noise_floor_db
        on_db, off_db = resolve_thresholds(noise_floor, config)
        empty = np.empty(times.size, dtype=np.float64)
        return StemActivation(
            activity=empty,
            active_state=np.empty(times.size, dtype=bool),
            db=empty.copy(),
            noise_floor_db=noise_floor,
            on_db=on_db,
            off_db=off_db,
            peak_db=envelope.peak_db,
            ambiguous_fraction=0.0,
            audio_duration_seconds=envelope.audio_duration_seconds,
            frame_count=int(envelope.frame_times.size),
        )

    db = np.interp(times, envelope.frame_times, envelope.db)
    outside = (times < envelope.frame_times[0] - 1e-9) | (times > envelope.frame_times[-1] + 1e-9)
    if bool(outside.any()):
        # No analysis frame covers this center: treat it as below the numeric
        # floor instead of holding the nearest frame's level.
        db = db.copy()
        db[outside] = config.floor_db

    on_db, off_db = resolve_thresholds(envelope.noise_floor_db, config)
    state = hysteresis_state(times, db, on_db, off_db, config.release_seconds)
    activity = probability_curve(db, state, on_db, off_db, config.probability_mode)
    if on_db > off_db:
        ambiguous = float(np.mean((db > off_db) & (db < on_db)))
    else:
        ambiguous = 0.0

    return StemActivation(
        activity=activity,
        active_state=state,
        db=db,
        noise_floor_db=envelope.noise_floor_db,
        on_db=on_db,
        off_db=off_db,
        peak_db=envelope.peak_db,
        ambiguous_fraction=ambiguous,
        audio_duration_seconds=envelope.audio_duration_seconds,
        frame_count=int(envelope.frame_times.size),
    )


__all__ = [
    "StemActivation",
    "hysteresis_state",
    "label_stem",
    "probability_curve",
    "resolve_thresholds",
]
