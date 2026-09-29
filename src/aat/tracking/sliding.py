"""Sliding-window model callback entry and full-chain helper.

A predictor callback maps a batch of centered windows ``(N, W)`` to the core
protocol outputs ``E[N, K, 128]`` and ``P[N, K]`` (optionally with an explicit
``slot_valid`` mask).  The runner normalises valid embeddings to unit L2 norm
and canonicalises invalid slots to all-zero embeddings with ``P = 0`` before the
strict protocol objects are built, so a fake callback can exercise the whole
pipeline without a trained model.

All times stay on the original-track axis: ``track_start_seconds`` is the
absolute time of ``audio[0]`` and is never re-based to the clip.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from aat.contracts.arrays import PredictionData
from aat.contracts.documents import AudioSpan, RunProvenance, Trajectory
from aat.contracts.version import DEFAULT_EMBEDDING_DIM
from aat.windowing import center_times, extract_windows_at_times

from .config import TrackingConfig
from .errors import TrackingError
from .sequence import PredictionSequence
from .tracker import associate_sequence

#: Below this L2 norm an embedding counts as absent rather than a valid vector.
_ZERO_NORM_EPSILON = 1e-8

PredictorCallback = Callable[[np.ndarray], Any]


@dataclass(frozen=True)
class WindowPrediction:
    """Callback output for a batch of ``N`` windows.

    ``embeddings`` is ``(N, K, 128)`` and ``activity`` is ``(N, K)``.
    ``slot_valid`` is optional; when omitted, non-zero embeddings are valid and
    zero-norm embeddings are invalid.
    """

    embeddings: np.ndarray
    activity: np.ndarray
    slot_valid: np.ndarray | None = None


def predict_windows(
    windows: Any,
    predictor: PredictorCallback,
    *,
    slots: int,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> WindowPrediction:
    """Run one predictor batch and canonicalise its output for the protocol."""

    window_array = np.asarray(windows)
    if window_array.ndim != 2:
        raise TrackingError(
            f"windows: expected a 2-D (N, W) array, got shape {tuple(window_array.shape)}"
        )
    if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
        raise TrackingError(f"slots: expected an integer >= 1, got {slots!r}")
    if embedding_dim != DEFAULT_EMBEDDING_DIM:
        raise TrackingError(
            f"embedding_dim: protocol 0.1.0 fixes the dimension at "
            f"{DEFAULT_EMBEDDING_DIM}; got {embedding_dim}"
        )
    batches = int(window_array.shape[0])
    if batches == 0:
        return WindowPrediction(
            embeddings=np.zeros((0, slots, embedding_dim), dtype=np.float32),
            activity=np.zeros((0, slots), dtype=np.float32),
            slot_valid=np.zeros((0, slots), dtype=bool),
        )

    output = predictor(window_array.astype(np.float32, copy=False))
    embeddings, activity, slot_valid = _unpack_output(output, batches, slots, embedding_dim)
    norms = np.linalg.norm(embeddings, axis=2)
    zero_norm = norms < _ZERO_NORM_EPSILON
    if slot_valid is None:
        valid = ~zero_norm
    else:
        if np.any(slot_valid & zero_norm):
            raise TrackingError(
                "predictor: marked a zero-norm embedding as a valid slot; a valid "
                "candidate must carry an identity vector"
            )
        valid = slot_valid

    embeddings = np.array(embeddings, dtype=np.float32, copy=True)
    activity = np.array(activity, dtype=np.float32, copy=True)
    embeddings[~valid] = 0.0
    activity[~valid] = 0.0
    if np.any(valid):
        embeddings[valid] = embeddings[valid] / norms[valid][:, None]
    return WindowPrediction(embeddings=embeddings, activity=activity, slot_valid=valid)


def build_prediction_data(
    audio: Any,
    center_seconds: Any,
    *,
    sample_rate: int,
    window_seconds: float,
    predictor: PredictorCallback,
    slots: int,
    origin_seconds: float = 0.0,
    hop_seconds: float | None = None,
    sample_id: str | None = None,
    provenance: RunProvenance | None = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> PredictionData:
    """Extract centered windows at absolute ``center_seconds`` and predict E/P."""

    windows, valid = extract_windows_at_times(
        audio,
        center_seconds,
        sample_rate,
        window_seconds,
        origin_seconds=origin_seconds,
    )
    batch = predict_windows(
        windows, predictor, slots=slots, embedding_dim=embedding_dim
    )
    center_valid = valid.all(axis=1) if valid.size else np.zeros(0, dtype=bool)
    return PredictionData(
        center_times=np.asarray(center_seconds, dtype=np.float64),
        embeddings=batch.embeddings,
        activity=batch.activity,
        slot_valid=batch.slot_valid,
        center_valid=center_valid,
        slots=slots,
        embedding_dim=embedding_dim,
        hop_seconds=hop_seconds,
        sample_id=sample_id,
        provenance=provenance,
    )


def track_audio(
    audio: Any,
    *,
    sample_rate: int,
    predictor: PredictorCallback,
    window_seconds: float,
    hop_seconds: float,
    slots: int,
    provenance: RunProvenance,
    config: TrackingConfig | None = None,
    track_start_seconds: float = 0.0,
    sample_id: str | None = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> tuple[PredictionData, Trajectory]:
    """Full chain: sliding windows -> fake/model callback -> trajectories."""

    samples = np.asarray(audio)
    if samples.ndim != 1:
        raise TrackingError(
            f"audio: expected a 1-D array, got shape {tuple(samples.shape)}"
        )
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise TrackingError(f"sample_rate: expected an integer, got {type(sample_rate).__name__}")
    if int(sample_rate) < 1:
        raise TrackingError(f"sample_rate: must be >= 1 Hz, got {sample_rate!r}")
    duration_seconds = float(samples.shape[0]) / float(sample_rate)
    centers = center_times(
        duration_seconds, hop_seconds, origin_seconds=track_start_seconds
    )
    predictions = build_prediction_data(
        samples,
        centers,
        sample_rate=int(sample_rate),
        window_seconds=window_seconds,
        predictor=predictor,
        slots=slots,
        origin_seconds=track_start_seconds,
        hop_seconds=hop_seconds,
        sample_id=sample_id,
        provenance=provenance,
        embedding_dim=embedding_dim,
    )
    trajectory = associate_sequence(
        PredictionSequence.from_prediction(predictions),
        AudioSpan(
            duration_seconds=duration_seconds,
            track_start_seconds=float(track_start_seconds),
        ),
        provenance,
        config=config,
        sample_id=sample_id,
    )
    return predictions, trajectory


def _unpack_output(
    output: Any,
    batches: int,
    slots: int,
    embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if isinstance(output, WindowPrediction):
        embeddings, activity, slot_valid = (
            output.embeddings,
            output.activity,
            output.slot_valid,
        )
    elif isinstance(output, (tuple, list)) and len(output) in (2, 3):
        embeddings, activity = output[0], output[1]
        slot_valid = output[2] if len(output) == 3 else None
    else:
        raise TrackingError(
            "predictor: expected a WindowPrediction or a (embeddings, activity[, "
            f"slot_valid]) tuple, got {type(output).__name__}"
        )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    activity = np.asarray(activity, dtype=np.float32)
    expected_embeddings = (batches, slots, embedding_dim)
    if embeddings.shape != expected_embeddings:
        raise TrackingError(
            f"predictor.embeddings: expected shape {list(expected_embeddings)}, "
            f"got {list(embeddings.shape)}"
        )
    if activity.shape != (batches, slots):
        raise TrackingError(
            f"predictor.activity: expected shape [{batches}, {slots}], "
            f"got {list(activity.shape)}"
        )
    if not np.all(np.isfinite(embeddings)):
        raise TrackingError("predictor.embeddings: contains NaN or infinite values")
    if not np.all(np.isfinite(activity)):
        raise TrackingError("predictor.activity: contains NaN or infinite values")
    if np.any(activity < 0.0) or np.any(activity > 1.0):
        raise TrackingError("predictor.activity: probabilities must lie in [0, 1]")

    valid_mask: np.ndarray | None = None
    if slot_valid is not None:
        valid_mask = np.asarray(slot_valid)
        if valid_mask.dtype != np.bool_:
            raise TrackingError(
                f"predictor.slot_valid: expected a boolean array, got dtype {valid_mask.dtype}"
            )
        if valid_mask.shape != (batches, slots):
            raise TrackingError(
                f"predictor.slot_valid: expected shape [{batches}, {slots}], "
                f"got {list(valid_mask.shape)}"
            )
    return embeddings, activity, valid_mask


__all__ = [
    "PredictorCallback",
    "WindowPrediction",
    "build_prediction_data",
    "predict_windows",
    "track_audio",
]
