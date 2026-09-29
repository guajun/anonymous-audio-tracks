"""Sample-level labeling: renderer stems in, ``ActivityData`` plus summary out.

One label run consumes the per-source stem arrays in ``sources.json`` order and
produces:

* an ``aat.contracts.ActivityData`` object (validated protocol 0.1.0 document)
  with center-local probabilities;
* a human-readable ``activity_summary.json`` payload that records the full
  config, its fingerprint, per-source statistics and explicit silence / low-SNR
  / ambiguity flags.

Control events are deliberately not an input to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from aat.contracts import ActivityData
from aat.windowing import center_times as _center_times

from .activity import StemActivation, label_stem
from .config import LabelConfig
from .errors import LabelAudioError


@dataclass(frozen=True)
class LabelResult:
    """Output of one sample-level labeling pass."""

    activity: ActivityData
    summary: dict[str, Any]
    diagnostics: dict[str, StemActivation]


def label_stems(
    stems: Mapping[str, Any],
    sample_rate: int,
    *,
    duration_seconds: float,
    origin_seconds: float = 0.0,
    center_times: Any | None = None,
    config: LabelConfig | None = None,
    sample_id: str | None = None,
) -> LabelResult:
    """Label every stem of one rendered sample.

    ``stems`` maps ``source_id`` to a 1-D or ``(samples, channels)`` array in
    ``sources.json`` column order.  ``duration_seconds`` / ``origin_seconds``
    come from ``manifest.json`` (``track_start_seconds``); center times are
    absolute original-track seconds and are never re-based to the clip.
    """

    config = config or LabelConfig()
    for position, source_id in enumerate(stems):
        if not isinstance(source_id, str) or not source_id:
            raise LabelAudioError(f"stems key at position {position}: expected a non-empty string")
    if len(set(stems)) != len(stems):
        raise LabelAudioError("stems: duplicate source_id keys")

    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise LabelAudioError(f"duration_seconds: must be finite and > 0, got {duration_seconds!r}")
    if not np.isfinite(origin_seconds) or origin_seconds < 0.0:
        raise LabelAudioError(f"origin_seconds: must be finite and >= 0, got {origin_seconds!r}")

    if center_times is None:
        times = _center_times(
            float(duration_seconds), config.hop_seconds, origin_seconds=float(origin_seconds)
        )
    else:
        times = np.asarray(center_times, dtype=np.float64)
        if times.ndim != 1:
            raise LabelAudioError(
                f"center_times: expected a 1-D array, got {tuple(times.shape)}"
            )
        if not np.all(np.isfinite(times)):
            raise LabelAudioError("center_times: contains NaN or infinite values")
        if np.any(times < 0.0):
            raise LabelAudioError("center_times: values must be >= 0")
        if times.size > 1 and not np.all(np.diff(times) > 0.0):
            raise LabelAudioError("center_times: values must be strictly increasing")
        tolerance = 1e-9 * max(1.0, float(duration_seconds))
        if times.size and (
            times[0] < origin_seconds - tolerance
            or times[-1] > origin_seconds + duration_seconds + tolerance
        ):
            raise LabelAudioError(
                f"center_times: must lie in [{origin_seconds}, "
                f"{origin_seconds + duration_seconds}], got "
                f"[{times[0]}, {times[-1]}]"
            )

    half_window = config.model_window_seconds / 2.0
    tolerance = 1e-9 * max(1.0, float(duration_seconds))
    valid = (times >= origin_seconds + half_window - tolerance) & (
        times <= origin_seconds + duration_seconds - half_window + tolerance
    )

    diagnostics: dict[str, StemActivation] = {}
    columns: list[np.ndarray] = []
    for source_id, audio in stems.items():
        activation = label_stem(
            audio,
            sample_rate,
            times,
            origin_seconds=float(origin_seconds),
            config=config,
            duration_seconds=float(duration_seconds),
        )
        diagnostics[source_id] = activation
        columns.append(activation.activity)
    if columns:
        stacked = np.column_stack(columns)
    else:
        stacked = np.zeros((times.size, 0), dtype=np.float64)
    activity = np.clip(stacked, 0.0, 1.0).astype(np.float32)

    label_params = {
        "labeler_version": config.labeler_version,
        "config_sha256": config.fingerprint(),
        "config": config.to_dict(),
        "origin_seconds": float(origin_seconds),
        "duration_seconds": float(duration_seconds),
        "n_centers": int(times.size),
    }
    data = ActivityData(
        center_times=times,
        activity=activity,
        valid=valid,
        source_ids=tuple(stems.keys()),
        sample_rate=sample_rate,
        hop_seconds=config.hop_seconds,
        sample_id=sample_id,
        label_params=label_params,
    )
    summary = build_summary(
        diagnostics=diagnostics,
        times=times,
        valid=valid,
        config=config,
        sample_rate=sample_rate,
        duration_seconds=float(duration_seconds),
        origin_seconds=float(origin_seconds),
        sample_id=sample_id,
    )
    return LabelResult(activity=data, summary=summary, diagnostics=diagnostics)


def _count_segments(active: np.ndarray) -> int:
    if active.size == 0:
        return 0
    padded = np.concatenate(([False], active.astype(bool), [False]))
    return int(np.count_nonzero(padded[1:] & ~padded[:-1]))


def _analysis_mask(valid: np.ndarray) -> np.ndarray:
    """Valid model-window centers; fall back to all centers when none fit."""

    if valid.size and bool(valid.any()):
        return valid
    return np.ones(valid.shape, dtype=bool)


def build_summary(
    *,
    diagnostics: Mapping[str, StemActivation],
    times: np.ndarray,
    valid: np.ndarray,
    config: LabelConfig,
    sample_rate: int,
    duration_seconds: float,
    origin_seconds: float,
    sample_id: str | None,
) -> dict[str, Any]:
    """Readable per-source statistics and explicit ambiguity flags."""

    mask = _analysis_mask(valid)
    sources: list[dict[str, Any]] = []
    warnings: list[str] = [
        (
            "activity values are reproducible thresholded short-time energy proxies "
            "of the rendered stem; they are not subjective audibility truth"
        ),
        "MIDI/parameter control events were not used as activity truth",
    ]
    if valid.size and not bool(valid.any()):
        warnings.append(
            "no center fits the full model_window_seconds inside the rendered audio; "
            "statistics fall back to all centers and the valid mask is all False"
        )

    for source_id, activation in diagnostics.items():
        probability = np.asarray(activation.activity, dtype=np.float64)
        db = np.asarray(activation.db, dtype=np.float64)
        analyzed_probability = probability[mask] if probability.size else probability
        analyzed_db = db[mask] if db.size else db
        active = analyzed_probability >= config.summary_active_probability
        silent = not bool(active.any())
        low_snr = (not silent) and (
            activation.peak_db - activation.noise_floor_db < config.min_snr_db
        )
        if activation.on_db > activation.off_db and analyzed_db.size:
            ambiguous_fraction = float(
                np.mean((analyzed_db > activation.off_db) & (analyzed_db < activation.on_db))
            )
        else:
            ambiguous_fraction = 0.0
        source: dict[str, Any] = {
            "source_id": source_id,
            "noise_floor_db": float(activation.noise_floor_db),
            "peak_db": float(activation.peak_db),
            "on_db": float(activation.on_db),
            "off_db": float(activation.off_db),
            "active_fraction": float(active.mean()) if active.size else 0.0,
            "active_centers": int(np.count_nonzero(active)),
            "n_segments": _count_segments(active),
            "max_activity": float(analyzed_probability.max()) if analyzed_probability.size else 0.0,
            "mean_activity": (
                float(analyzed_probability.mean()) if analyzed_probability.size else 0.0
            ),
            "ambiguous_fraction": ambiguous_fraction,
            "analyzed_centers": int(analyzed_probability.size),
            "silent": bool(silent),
            "low_snr": bool(low_snr),
        }
        if analyzed_db.size:
            source["mean_db"] = float(analyzed_db.mean())
        sources.append(source)
        if silent:
            warnings.append(
                f"{source_id}: no center reached the activation threshold; labelled "
                "inactive (absence of a detectable level, not proof of inaudibility)"
            )
        if low_snr:
            warnings.append(
                f"{source_id}: peak {activation.peak_db:.1f} dB is less than "
                f"{config.min_snr_db} dB above the estimated noise floor "
                f"{activation.noise_floor_db:.1f} dB; threshold decisions are ambiguous"
            )

    return {
        "kind": "activity_summary",
        "labeler_version": config.labeler_version,
        "config_sha256": config.fingerprint(),
        "config": config.to_dict(),
        "sample_id": sample_id,
        "sample_rate": int(sample_rate),
        "origin_seconds": float(origin_seconds),
        "duration_seconds": float(duration_seconds),
        "hop_seconds": config.hop_seconds,
        "model_window_seconds": config.model_window_seconds,
        "frame_seconds": config.frame_seconds,
        "frame_hop_seconds": config.frame_hop_seconds,
        "probability_mode": config.probability_mode,
        "n_centers": int(times.size),
        "n_valid_centers": int(np.count_nonzero(valid)),
        "source_ids": list(diagnostics.keys()),
        "sources": sources,
        "warnings": warnings,
    }


__all__ = ["LabelResult", "build_summary", "label_stems"]
