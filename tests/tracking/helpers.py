"""Synthetic, controlled builders for tracking/evaluation tests.

Everything is explicit unit vectors and activity values; no real audio, model
or dataset is loaded.  Sliding-window tests use a deterministic energy-based
fake callback on synthesized samples.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from aat.contracts import ActivityData, AudioSpan, RunProvenance, Track, Trajectory
from aat.tracking import PredictionSequence

EMBEDDING_DIM = 128


def basis(index: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


def blend(index_a: int, index_b: int, cosine: float, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Unit vector whose cosine similarity to ``basis(index_a)`` is ``cosine``."""

    vector = np.zeros(dim, dtype=np.float64)
    vector[index_a] = cosine
    vector[index_b] = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def window(
    k: int, entries: dict[int, tuple[np.ndarray, float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(embeddings, activity, slot_valid)`` arrays for one window."""

    embeddings = np.zeros((k, EMBEDDING_DIM), dtype=np.float32)
    activity = np.zeros(k, dtype=np.float32)
    slot_valid = np.zeros(k, dtype=bool)
    for slot, (vector, probability) in entries.items():
        embeddings[slot] = vector
        activity[slot] = probability
        slot_valid[slot] = True
    return embeddings, activity, slot_valid


def sequence(
    times: Sequence[float],
    windows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    center_valid: Sequence[bool] | None = None,
) -> PredictionSequence:
    """Stack per-window arrays into one :class:`PredictionSequence`."""

    embeddings = np.stack([item[0] for item in windows]).astype(np.float32)
    activity = np.stack([item[1] for item in windows]).astype(np.float32)
    slot_valid = np.stack([item[2] for item in windows])
    return PredictionSequence(
        center_times=np.asarray(times, dtype=np.float64),
        embeddings=embeddings,
        activity=activity,
        slot_valid=slot_valid,
        center_valid=None if center_valid is None else np.asarray(center_valid, dtype=bool),
    )


def provenance(run_id: str = "test-run") -> RunProvenance:
    return RunProvenance(run_id=run_id, data_kind="mock")


def activity(
    times: Sequence[float],
    columns: Sequence[Sequence[float]],
    source_ids: Sequence[str],
    *,
    valid: Sequence[bool] | None = None,
    sample_rate: int = 16000,
) -> ActivityData:
    """``columns`` is one activity series per source; result is ``(T, S)``."""

    array = np.asarray([list(column) for column in columns], dtype=np.float32)
    if array.size == 0:
        array = np.zeros((len(times), 0), dtype=np.float32)
    else:
        array = array.T.copy()
    return ActivityData(
        center_times=np.asarray(times, dtype=np.float64),
        activity=array,
        valid=np.ones(len(times), dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
        source_ids=tuple(source_ids),
        sample_rate=sample_rate,
        hop_seconds=0.02,
    )


def trajectory(
    tracks: Sequence[dict],
    *,
    duration_seconds: float = 1.0,
    track_start_seconds: float = 0.0,
    slots: int = 8,
    run_id: str = "test-run",
) -> Trajectory:
    objects = []
    for spec in tracks:
        objects.append(
            Track(
                track_id=spec["track_id"],
                center_times=tuple(float(value) for value in spec["times"]),
                activity=tuple(float(value) for value in spec["activity"]),
                confidence=(
                    tuple(float(value) for value in spec["confidence"])
                    if "confidence" in spec
                    else None
                ),
                slot_indices=(
                    tuple(spec["slot_indices"]) if "slot_indices" in spec else None
                ),
            )
        )
    return Trajectory(
        audio=AudioSpan(
            duration_seconds=duration_seconds,
            track_start_seconds=track_start_seconds,
        ),
        provenance=provenance(run_id),
        tracks=tuple(objects),
        slots=slots,
    )
