"""Validated per-window prediction sequence consumed by the tracker.

This is a read-only view over the same arrays as
:class:`aat.contracts.arrays.PredictionData`; it re-checks the protocol
invariants that matter for association (shapes, unit-norm valid candidates,
canonical inactive slots) so the tracker can rely on them without importing the
whole contract machinery into its inner loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aat.contracts.arrays import PredictionData
from aat.contracts.version import DEFAULT_EMBEDDING_DIM, UNIT_NORM_TOLERANCE

from .errors import TrackingError


def _as_array(value: Any, dtype: Any, ndim: int, path: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise TrackingError(
            f"{path}: expected a {ndim}-D array, got shape {tuple(array.shape)}"
        )
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class PredictionSequence:
    """Per-window ``E[N, K, 128]`` / ``P[N, K]`` predictions plus masks."""

    center_times: np.ndarray
    embeddings: np.ndarray
    activity: np.ndarray
    slot_valid: np.ndarray
    center_valid: np.ndarray | None = None
    hop_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "center_times", _as_array(self.center_times, np.float64, 1, "center_times")
        )
        object.__setattr__(
            self, "embeddings", _as_array(self.embeddings, np.float32, 3, "embeddings")
        )
        object.__setattr__(
            self, "activity", _as_array(self.activity, np.float32, 2, "activity")
        )
        object.__setattr__(
            self, "slot_valid", _as_array(self.slot_valid, np.bool_, 2, "slot_valid")
        )
        if self.center_valid is None:
            object.__setattr__(
                self, "center_valid", np.ones(self.center_times.shape[0], dtype=bool)
            )
        else:
            object.__setattr__(
                self,
                "center_valid",
                _as_array(self.center_valid, np.bool_, 1, "center_valid"),
            )
        self.validate()

    @classmethod
    def from_prediction(cls, prediction: PredictionData) -> "PredictionSequence":
        """Build a sequence from a validated ``prediction.json``/``.npz`` pair."""

        if not isinstance(prediction, PredictionData):
            raise TrackingError("prediction: expected a PredictionData document")
        return cls(
            center_times=prediction.center_times,
            embeddings=prediction.embeddings,
            activity=prediction.activity,
            slot_valid=prediction.slot_valid,
            center_valid=prediction.center_valid,
            hop_seconds=prediction.hop_seconds,
        )

    @property
    def n_windows(self) -> int:
        return int(self.center_times.shape[0])

    @property
    def slots(self) -> int:
        return int(self.embeddings.shape[1])

    def validate(self) -> None:
        times = self.center_times
        if times.size and (not np.all(np.isfinite(times)) or np.any(times < 0.0)):
            raise TrackingError("center_times: must be finite and >= 0")
        if times.size > 1 and not np.all(np.diff(times) > 0):
            raise TrackingError("center_times: must be strictly increasing")

        windows = times.shape[0]
        embeddings = self.embeddings
        if embeddings.shape[0] != windows:
            raise TrackingError(
                f"embeddings: leading length {embeddings.shape[0]} must match "
                f"center_times length {windows}"
            )
        slots = embeddings.shape[1]
        if slots < 1:
            raise TrackingError("embeddings: at least one slot is required")
        if embeddings.shape[2] != DEFAULT_EMBEDDING_DIM:
            raise TrackingError(
                f"embeddings: protocol 0.1.0 fixes the embedding dimension at "
                f"{DEFAULT_EMBEDDING_DIM}; got {embeddings.shape[2]}"
            )
        if self.activity.shape != (windows, slots):
            raise TrackingError(
                f"activity: expected shape [{windows}, {slots}], "
                f"got {list(self.activity.shape)}"
            )
        if self.slot_valid.shape != (windows, slots):
            raise TrackingError(
                f"slot_valid: expected shape [{windows}, {slots}], "
                f"got {list(self.slot_valid.shape)}"
            )
        if self.center_valid.shape != (windows,):
            raise TrackingError(
                f"center_valid: expected shape [{windows}], "
                f"got {list(self.center_valid.shape)}"
            )
        if not np.all(np.isfinite(embeddings)):
            raise TrackingError("embeddings: contains NaN or infinite values")
        if not np.all(np.isfinite(self.activity)):
            raise TrackingError("activity: contains NaN or infinite values")
        if np.any(self.activity < 0.0) or np.any(self.activity > 1.0):
            raise TrackingError("activity: probabilities must lie in [0, 1]")
        if self.hop_seconds is not None:
            if isinstance(self.hop_seconds, bool) or not isinstance(
                self.hop_seconds, (int, float)
            ):
                raise TrackingError("hop_seconds: expected a number")
            if not np.isfinite(self.hop_seconds) or self.hop_seconds <= 0.0:
                raise TrackingError("hop_seconds: must be finite and > 0")

        inactive = ~self.slot_valid
        if np.any(self.activity[inactive] != 0.0):
            raise TrackingError("activity: inactive slots must have probability 0")
        if np.any(embeddings[inactive] != 0.0):
            raise TrackingError("embeddings: inactive slots must be canonical zeros")
        if np.any(self.slot_valid):
            valid_embeddings = embeddings[self.slot_valid]
            norms = np.linalg.norm(valid_embeddings, axis=1)
            if np.any(np.abs(norms - 1.0) > UNIT_NORM_TOLERANCE):
                worst = float(np.max(np.abs(norms - 1.0)))
                raise TrackingError(
                    "embeddings: valid candidates must be unit vectors "
                    f"(tolerance {UNIT_NORM_TOLERANCE}); worst deviation {worst:.6g}"
                )


__all__ = ["PredictionSequence"]
