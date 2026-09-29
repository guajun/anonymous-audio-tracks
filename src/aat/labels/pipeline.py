"""Center-time activity labeling for rendered stems.

``label_stems`` maps each per-source stem to the protocol center grid: the
local energy envelope is computed on the audio samples, the state machine is
evaluated per sample, and the result is sampled at the center times.  The
protocol ``valid`` mask is derived from the *model* center window recorded in
the config, never from the local energy window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import ActivityData
from ..windowing import center_times, seconds_to_samples
from .activity import active_segments, label_envelope
from .config import LabelConfig
from .energy import SILENCE_DBFS, compute_thresholds, envelope_db
from .errors import LabelError

_VALID_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SourceSummary:
    """Readable per-source summary; never used for protocol decisions."""

    source_id: str
    active_fraction: float
    valid_fraction: float
    active_seconds: float
    segments: int
    first_active_seconds: float | None
    last_active_seconds: float | None
    peak_dbfs: float
    noise_floor_dbfs: float
    snr_db: float
    on_threshold_dbfs: float
    off_threshold_dbfs: float
    silent: bool
    low_snr: bool
    ambiguous_fraction: float
    ambiguous: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (round(value, 6) if isinstance(value, float) else value)
            for name, value in asdict(self).items()
        }


@dataclass(frozen=True)
class LabelResult:
    """Activity document plus readable summaries and the config snapshot."""

    activity: ActivityData
    summaries: tuple[SourceSummary, ...]
    config: LabelConfig
    label_params: Mapping[str, Any]


def _check_sample_rate(sample_rate: Any) -> int:
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, np.integer))
        or int(sample_rate) < 1
    ):
        raise LabelError(f"sample_rate: expected an integer >= 1, got {sample_rate!r}")
    return int(sample_rate)


def label_stems(
    stems: Mapping[str, Any],
    *,
    source_ids: Sequence[str],
    sample_rate: int,
    duration_seconds: float,
    track_start_seconds: float,
    config: LabelConfig | None = None,
    sample_id: str | None = None,
) -> LabelResult:
    """Label every stem at the protocol center grid.

    ``source_ids`` defines the column order and must match the ``stems`` keys
    exactly.  All stems must share the same frame count; the CLI additionally
    checks each stem against the manifest sample rate and duration.
    """

    config = config if config is not None else LabelConfig()
    if not isinstance(config, LabelConfig):
        raise LabelError("config: expected a LabelConfig")
    rate = _check_sample_rate(sample_rate)
    ids = tuple(source_ids)
    if len(set(ids)) != len(ids):
        raise LabelError("source_ids: values must be unique")
    if set(stems) != set(ids):
        raise LabelError(
            "stems: keys must match source_ids exactly "
            f"(stems={sorted(stems)}, source_ids={list(ids)})"
        )
    if not isinstance(duration_seconds, (int, float)) or isinstance(
        duration_seconds, bool
    ):
        raise LabelError("duration_seconds: expected a number")
    duration = float(duration_seconds)
    if not np.isfinite(duration) or duration <= 0.0:
        raise LabelError(f"duration_seconds: must be > 0, got {duration_seconds!r}")
    if (
        not isinstance(track_start_seconds, (int, float))
        or isinstance(track_start_seconds, bool)
        or not np.isfinite(float(track_start_seconds))
        or float(track_start_seconds) < 0.0
    ):
        raise LabelError(
            f"track_start_seconds: must be a finite value >= 0, got {track_start_seconds!r}"
        )
    track_start = float(track_start_seconds)

    frames: int | None = None
    for source_id in ids:
        array = np.asarray(stems[source_id])
        if array.ndim not in (1, 2):
            raise LabelError(
                f"stem {source_id!r}: expected 1-D or 2-D samples, "
                f"got shape {tuple(array.shape)}"
            )
        if array.shape[0] == 0:
            raise LabelError(f"stem {source_id!r}: empty audio")
        if frames is None:
            frames = int(array.shape[0])
        elif int(array.shape[0]) != frames:
            raise LabelError(
                "stems: all stems must share the same frame count "
                f"({source_id!r} has {array.shape[0]}, expected {frames})"
            )
    if frames is None:
        frames = 0

    centers = center_times(duration, config.hop_seconds, origin_seconds=track_start)
    half_window = config.center_window_seconds / 2.0
    tolerance = _VALID_TOLERANCE * max(1.0, duration)
    end = track_start + duration
    valid = (centers - half_window >= track_start - tolerance) & (
        centers + half_window <= end + tolerance
    )

    local_seconds = centers - track_start
    if frames > 0:
        center_samples = np.floor(local_seconds * rate + 0.5).astype(np.int64)
        np.clip(center_samples, 0, frames - 1, out=center_samples)
    else:
        center_samples = np.empty(0, dtype=np.int64)

    hold_samples = seconds_to_samples(config.release_hold_seconds, rate)

    activity = np.zeros((centers.size, len(ids)), dtype=np.float32)
    summaries: list[SourceSummary] = []
    for column, source_id in enumerate(ids):
        levels = envelope_db(stems[source_id], rate, config.energy_window_seconds)
        thresholds = compute_thresholds(levels, config)
        labels = label_envelope(
            levels, thresholds.on_dbfs, thresholds.off_dbfs, hold_samples
        )
        center_levels = levels[center_samples]
        center_active = labels.active[center_samples]
        activity[:, column] = center_active.astype(np.float32)
        summaries.append(
            _build_summary(
                source_id,
                labels.active,
                thresholds,
                center_levels,
                center_active,
                valid,
                rate,
                track_start,
                config,
            )
        )

    data = ActivityData(
        center_times=centers,
        activity=activity,
        valid=valid,
        source_ids=ids,
        sample_rate=rate,
        hop_seconds=config.hop_seconds,
        sample_id=sample_id,
        label_params=config.to_label_params(),
    )
    return LabelResult(
        activity=data,
        summaries=tuple(summaries),
        config=config,
        label_params=config.to_label_params(),
    )


def _build_summary(
    source_id: str,
    active: np.ndarray,
    thresholds: Any,
    center_levels: np.ndarray,
    center_active: np.ndarray,
    valid: np.ndarray,
    sample_rate: int,
    track_start_seconds: float,
    config: LabelConfig,
) -> SourceSummary:
    rows = int(center_levels.size)
    active_fraction = float(center_active.mean()) if rows else 0.0
    valid_fraction = float(valid.mean()) if valid.size else 0.0
    runs = active_segments(active)
    if runs:
        first_active = track_start_seconds + runs[0][0] / sample_rate
        last_active = track_start_seconds + (runs[-1][1] - 1) / sample_rate
    else:
        first_active = None
        last_active = None
    silent = not bool(active.any())
    snr_db = thresholds.peak_dbfs - thresholds.noise_floor_dbfs
    low_snr = (not silent) and snr_db < config.low_snr_threshold_db
    if rows:
        near_threshold = (
            np.abs(center_levels - thresholds.on_dbfs) <= config.ambiguity_margin_db
        )
        ambiguous_fraction = float(near_threshold.mean())
    else:
        ambiguous_fraction = 0.0
    return SourceSummary(
        source_id=source_id,
        active_fraction=active_fraction,
        valid_fraction=valid_fraction,
        active_seconds=float(active.sum()) / sample_rate,
        segments=len(runs),
        first_active_seconds=first_active,
        last_active_seconds=last_active,
        peak_dbfs=thresholds.peak_dbfs,
        noise_floor_dbfs=thresholds.noise_floor_dbfs,
        snr_db=snr_db,
        on_threshold_dbfs=thresholds.on_dbfs,
        off_threshold_dbfs=thresholds.off_dbfs,
        silent=silent,
        low_snr=low_snr,
        ambiguous_fraction=ambiguous_fraction,
        ambiguous=ambiguous_fraction >= config.ambiguity_fraction_threshold,
    )
