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
    therefore one :class:`TokenGrid`; each window is its own encoder sample.

    With the default block-diagonal attention mask a batched forward is
    equivalent to per-window forwards only **numerically**: measured fp32
    maximum difference is 2.6e-4 and bf16 reaches 0.039 because GEMM/kernel
    shapes differ with batch composition.  It is not a bitwise guarantee, and
    without the mask (``masked_attention=False``) batching leaks context across
    samples.  Use a fixed batch policy for reproducible features.
    """

    features: np.ndarray
    valid: np.ndarray
    frame_times: np.ndarray
    window_start_seconds: np.ndarray
    grid: TokenGrid
    layer: str = "final"
    model_id: str = ""
    revision: str = ""
    preprocessing: Mapping[str, Any] = field(default_factory=dict)

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
        if not isinstance(self.preprocessing, Mapping):
            raise EncoderError("preprocessing: expected a mapping")
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
        """One window as :class:`AutFeatures` (grid and provenance preserved)."""

        return AutFeatures.build(
            features=self.features[index],
            valid=self.valid[index],
            grid=self.grid,
            buffer_origin_seconds=float(self.window_start_seconds[index]),
            layer=self.layer,
            model_id=self.model_id,
            revision=self.revision,
            preprocessing=self.preprocessing,
            sample_rate=AUT_SAMPLE_RATE,
        )

def compare_aligned_tokens(
    reference: AutFeatures,
    candidate: AutFeatures,
    *,
    atol_seconds: float = 1e-9,
    valid_only: bool = False,
) -> dict[str, Any]:
    """Compare two extractions at identical absolute frame times.

    Matching is by absolute frame time (nearest within ``atol_seconds``), never
    by array index.  Metrics are computed on every matched element:

    - ``elementwise_mae``: mean of ``|a - b|`` over all matched elements;
    - ``mean_token_mae`` / ``max_token_mae``: mean/max over tokens of each
      token's elementwise MAE;
    - ``mean_token_max_abs``: mean over tokens of each token's max |diff|
      (kept explicitly under this name; it is *not* an MAE);
    - ``max_abs_diff``: global maximum elementwise |diff|;
    - per-token cosine similarity.

    Valid-token handling: a matched pair is invalid when either side's ``valid``
    mask is ``False``.  With ``valid_only=False`` (default) invalid pairs are
    still compared and counted in ``invalid_matched_tokens``; with
    ``valid_only=True`` they are excluded from all statistics and counted in
    ``masked_out_tokens``.  An empty reference (or candidate) is handled
    gracefully: no matched tokens and ``None`` statistics, never an exception.
    """

    if reference.feature_dim != candidate.feature_dim:
        raise EncoderError(
            f"feature dims differ: {reference.feature_dim} vs {candidate.feature_dim}"
        )
    ref_times = reference.frame_times
    cand_times = candidate.frame_times
    per_token_mae: list[float | None] = []
    per_token_max_abs: list[float | None] = []
    per_token_cosine: list[float | None] = []
    matched = 0
    invalid_matched = 0
    masked_out = 0
    elementwise_sum = 0.0
    elementwise_count = 0
    global_max: float | None = None

    for i, time in enumerate(cand_times):
        if ref_times.size == 0:
            per_token_mae.append(None)
            per_token_max_abs.append(None)
            per_token_cosine.append(None)
            continue
        distance = np.abs(ref_times - time)
        j = int(np.argmin(distance))
        if distance[j] > atol_seconds:
            per_token_mae.append(None)
            per_token_max_abs.append(None)
            per_token_cosine.append(None)
            continue
        ref_valid = bool(reference.valid[j])
        cand_valid = bool(candidate.valid[i])
        if valid_only and not (ref_valid and cand_valid):
            masked_out += 1
            per_token_mae.append(None)
            per_token_max_abs.append(None)
            per_token_cosine.append(None)
            continue
        matched += 1
        if not (ref_valid and cand_valid):
            invalid_matched += 1
        a = reference.features[j].astype(np.float64)
        b = candidate.features[i].astype(np.float64)
        diff = np.abs(a - b)
        token_mae = float(diff.mean()) if diff.size else 0.0
        token_max_abs = float(diff.max()) if diff.size else 0.0
        global_max = token_max_abs if global_max is None else max(global_max, token_max_abs)
        elementwise_sum += float(diff.sum())
        elementwise_count += int(diff.size)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        token_cosine = float(np.dot(a, b) / denominator) if denominator > 0.0 else float("nan")
        per_token_mae.append(token_mae)
        per_token_max_abs.append(token_max_abs)
        per_token_cosine.append(token_cosine)

    maes = [value for value in per_token_mae if value is not None]
    cosine_values = [value for value in per_token_cosine if value is not None]
    total_candidate = int(cand_times.shape[0])
    return {
        "reference_tokens": reference.token_count,
        "candidate_tokens": candidate.token_count,
        "matched_tokens": matched,
        "unmatched_candidate_tokens": total_candidate - matched - masked_out,
        "masked_out_tokens": masked_out,
        "invalid_matched_tokens": invalid_matched,
        "valid_only": bool(valid_only),
        "elementwise_mae": (elementwise_sum / elementwise_count) if elementwise_count else None,
        "mean_token_mae": float(np.mean(maes)) if maes else None,
        "max_token_mae": max(maes) if maes else None,
        "mean_token_max_abs": (
            float(np.mean([value for value in per_token_max_abs if value is not None]))
            if matched
            else None
        ),
        "max_abs_diff": global_max,
        "min_cosine": min(cosine_values) if cosine_values else None,
        "mean_cosine": float(np.mean(cosine_values)) if cosine_values else None,
        "candidate_frame_times": [float(t) for t in cand_times],
        "per_token_mae": per_token_mae,
        "per_token_max_abs_diff": per_token_max_abs,
        "per_token_cosine": per_token_cosine,
    }
