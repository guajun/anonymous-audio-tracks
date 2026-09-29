"""Feature containers shared by the real AuT path and the fake test double.

``AutFeatures`` carries the protocol-relevant arrays (frame times, feature
matrix, validity mask) plus the token grid that produced them.  It can be
projected onto the shared :class:`aat.contracts.FeatureData` document; that
projection enforces the v0.1.0 feature contract (absolute non-negative,
strictly increasing times, finite float32 features, boolean mask of the same
length).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from aat.contracts.arrays import FeatureData

from .errors import EncoderError, EncoderInputError
from .grid import (
    AUT_NOMINAL_TOKEN_STEP_SECONDS,
    AUT_SAMPLE_RATE,
    TokenGrid,
)

#: Prefix of ``feature_name`` values produced by this package.
AUT_FEATURE_NAME_PREFIX = "aut.qwen3-omni-moe.audio_tower"


def feature_name_for_layer(layer: str) -> str:
    """Stable ``feature.feature_name`` for a layer selection (``final`` or ``layers.N``)."""

    if layer == "final":
        return f"{AUT_FEATURE_NAME_PREFIX}.final"
    return f"{AUT_FEATURE_NAME_PREFIX}.{layer}+head"


def _as_features_array(value: Any, *, ndim: int = 2) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim:
        raise EncoderError(f"features: expected a {ndim}-D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise EncoderError("features: contains NaN/Inf")
    return array


def _as_bool_mask(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise EncoderError(f"{name}: expected a boolean array, got dtype {array.dtype}")
    if array.shape != (length,):
        raise EncoderError(f"{name}: shape {array.shape} != expected {(length,)}")
    return array


@dataclass(frozen=True)
class AutFeatures:
    """Token features of one encoded buffer on the original-track time axis."""

    features: np.ndarray
    valid: np.ndarray
    frame_times: np.ndarray
    grid: TokenGrid
    buffer_origin_seconds: float = 0.0
    sample_rate: int = AUT_SAMPLE_RATE
    layer: str = "final"
    model_id: str = ""
    revision: str = ""
    preprocessing: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        features = _as_features_array(self.features)
        object.__setattr__(self, "features", features)
        times = np.asarray(self.frame_times, dtype=np.float64)
        if times.ndim != 1:
            raise EncoderError(f"frame_times: expected a 1-D array, got shape {times.shape}")
        if not np.all(np.isfinite(times)):
            raise EncoderError("frame_times: contains NaN/Inf")
        if times.shape[0] != self.grid.token_count:
            raise EncoderError(
                f"frame_times: length {times.shape[0]} != grid token count {self.grid.token_count}"
            )
        if times.shape[0] > 1 and not bool(np.all(np.diff(times) > 0.0)):
            raise EncoderError("frame_times: must be strictly increasing")
        object.__setattr__(self, "frame_times", times)
        object.__setattr__(self, "valid", _as_bool_mask(self.valid, times.shape[0], "valid"))
        if features.shape[0] != times.shape[0]:
            raise EncoderError(
                f"features: leading length {features.shape[0]} != frame_times length {times.shape[0]}"
            )
        if self.sample_rate < 1:
            raise EncoderError(f"sample_rate: must be >= 1, got {self.sample_rate}")

    @classmethod
    def build(
        cls,
        features: Any,
        valid: Any,
        grid: TokenGrid,
        *,
        buffer_origin_seconds: float,
        layer: str = "final",
        model_id: str = "",
        revision: str = "",
        preprocessing: Mapping[str, Any] | None = None,
        sample_rate: int = AUT_SAMPLE_RATE,
    ) -> "AutFeatures":
        frame_times = grid.frame_times(buffer_origin_seconds)
        return cls(
            features=features,
            valid=valid,
            frame_times=frame_times,
            grid=grid,
            buffer_origin_seconds=float(buffer_origin_seconds),
            sample_rate=int(sample_rate),
            layer=layer,
            model_id=model_id,
            revision=revision,
            preprocessing=dict(preprocessing or {}),
        )

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def token_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def hop_seconds(self) -> float:
        """Nominal token step (80 ms); ``frame_times`` remains authoritative."""

        return AUT_NOMINAL_TOKEN_STEP_SECONDS

    def to_feature_data(
        self,
        *,
        sample_id: str | None = None,
    ) -> FeatureData:
        """Project onto the shared v0.1.0 ``feature.json``/``feature.npz`` contract.

        The protocol requires non-negative, strictly increasing absolute frame
        times.  Window features that start before the track origin contain
        negative frame times; callers must drop those rows first (see
        :meth:`drop_before_track_origin`).
        """

        if self.token_count and float(np.min(self.frame_times)) < 0.0:
            raise EncoderInputError(
                "frame_times contain negative absolute times (window starts before the "
                "track origin); drop those rows with drop_before_track_origin() before "
                "building a FeatureData document"
            )
        metadata = {
            "feature_name": feature_name_for_layer(self.layer),
            "sample_rate": self.sample_rate,
            "frame_origin_seconds": float(self.buffer_origin_seconds),
            "hop_seconds": self.hop_seconds,
            "backend": "aut",
            "preprocessing": dict(self.preprocessing),
            "sample_id": sample_id,
        }
        return FeatureData(
            frame_times=self.frame_times,
            features=self.features,
            valid=self.valid,
            **metadata,
        )

    def subset(self, keep: Any) -> "AutFeatures":
        """Keep a boolean row subset (e.g. tokens inside a window).

        The derived grid is a valid row subset of the buffer grid; the formula
        parity check only applies to the full grid.
        """

        mask = np.asarray(keep)
        if mask.dtype != np.bool_ or mask.shape != (self.token_count,):
            raise EncoderError(
                f"keep: expected bool mask of shape {(self.token_count,)}, got {mask.dtype} {mask.shape}"
            )
        if bool(np.all(mask)):
            return self
        new_grid = TokenGrid(
            mel_len=self.grid.mel_len,
            chunk_index=self.grid.chunk_index[mask],
            token_in_chunk=self.grid.token_in_chunk[mask],
            mel_centers=self.grid.mel_centers[mask],
            field_start=self.grid.field_start[mask],
            field_stop=self.grid.field_stop[mask],
        )
        return AutFeatures.build(
            features=self.features[mask],
            valid=self.valid[mask],
            grid=new_grid,
            buffer_origin_seconds=self.buffer_origin_seconds,
            layer=self.layer,
            model_id=self.model_id,
            revision=self.revision,
            preprocessing=self.preprocessing,
            sample_rate=self.sample_rate,
        )

    def drop_before_track_origin(self) -> "AutFeatures":
        """Remove rows with negative absolute times (edge windows only)."""

        return self.subset(self.frame_times >= 0.0)


@dataclass(frozen=True)
class AutWindowBatch:
    """Independent fixed-size windows encoded in one batched forward pass.

    ``features`` is ``(N, T, D)``, ``valid`` is ``(N, T)`` and ``frame_times``
    is ``(N, T)`` (absolute original-track seconds; may be negative for windows
    that start before the track origin).  All windows share one mel length and
    therefore one :class:`TokenGrid`; each window is its own encoder sample and
    the AuT attention blocks never cross sample boundaries, so this is
    equivalent to running the windows one by one (verified in the real probe).
    """

    features: np.ndarray
    valid: np.ndarray
    frame_times: np.ndarray
    window_start_seconds: np.ndarray
    grid: TokenGrid
    layer: str = "final"
    model_id: str = ""
    revision: str = ""

    def __post_init__(self) -> None:
        features = _as_features_array(self.features, ndim=3)
        if features.ndim != 3:
            raise EncoderError(f"features: expected (N, T, D), got {features.shape}")
        n, t, _ = features.shape
        object.__setattr__(self, "features", features)
        if t != self.grid.token_count:
            raise EncoderError(
                f"features: token axis {t} != grid token count {self.grid.token_count}"
            )
        valid = np.asarray(self.valid)
        if valid.dtype != np.bool_ or valid.shape != (n, t):
            raise EncoderError(f"valid: expected bool {(n, t)}, got {valid.dtype} {valid.shape}")
        times = np.asarray(self.frame_times, dtype=np.float64)
        if times.shape != (n, t) or not np.all(np.isfinite(times)):
            raise EncoderError(f"frame_times: expected finite float64 {(n, t)}, got {times.shape}")
        starts = np.asarray(self.window_start_seconds, dtype=np.float64)
        if starts.shape != (n,) or not np.all(np.isfinite(starts)):
            raise EncoderError(f"window_start_seconds: expected finite float64 {(n,)}, got {starts.shape}")
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "frame_times", times)
        object.__setattr__(self, "window_start_seconds", starts)

    @property
    def window_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.features.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[2])

    def window(self, index: int) -> AutFeatures:
        """One window as :class:`AutFeatures` (grid shared, times shifted)."""

        return AutFeatures.build(
            features=self.features[index],
            valid=self.valid[index],
            grid=self.grid,
            buffer_origin_seconds=float(self.window_start_seconds[index]),
            layer=self.layer,
            model_id=self.model_id,
            revision=self.revision,
            sample_rate=AUT_SAMPLE_RATE,
        )

def compare_aligned_tokens(
    reference: AutFeatures,
    candidate: AutFeatures,
    *,
    atol_seconds: float = 1e-9,
) -> dict[str, Any]:
    """Compare tokens of two extractions at identical absolute frame times.

    Returns summary statistics; the caller decides what counts as equal.  Rows
    of ``candidate`` whose time has no counterpart in ``reference`` are
    reported as ``unmatched`` instead of being silently dropped.
    """

    if reference.feature_dim != candidate.feature_dim:
        raise EncoderError(
            f"feature dims differ: {reference.feature_dim} vs {candidate.feature_dim}"
        )
    ref_times = reference.frame_times
    cand_times = candidate.frame_times
    matched_candidate = np.zeros(cand_times.shape[0], dtype=bool)
    diffs: list[float] = []
    cosines: list[float] = []
    per_token_max_abs: list[float | None] = []
    per_token_cosine: list[float | None] = []
    for i, time in enumerate(cand_times):
        distance = np.abs(ref_times - time)
        j = int(np.argmin(distance))
        if distance[j] > atol_seconds:
            per_token_max_abs.append(None)
            per_token_cosine.append(None)
            continue
        matched_candidate[i] = True
        a = reference.features[j].astype(np.float64)
        b = candidate.features[i].astype(np.float64)
        token_diff = float(np.max(np.abs(a - b)))
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        token_cosine = float(np.dot(a, b) / denominator) if denominator > 0.0 else float("nan")
        diffs.append(token_diff)
        cosines.append(token_cosine)
        per_token_max_abs.append(token_diff)
        per_token_cosine.append(token_cosine)
    return {
        "reference_tokens": reference.token_count,
        "candidate_tokens": candidate.token_count,
        "matched_tokens": int(matched_candidate.sum()),
        "unmatched_candidate_tokens": int((~matched_candidate).sum()),
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
        "min_cosine": min(cosines) if cosines else None,
        "mean_cosine": float(np.mean(cosines)) if cosines else None,
        "candidate_frame_times": [float(t) for t in cand_times],
        "per_token_max_abs_diff": per_token_max_abs,
        "per_token_cosine": per_token_cosine,
    }
