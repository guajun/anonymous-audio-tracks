"""Masked multi-source query head for the frozen E/P prediction protocol.

The head reads per-window feature frames with a validity mask and optional frame
times, and emits K candidate identities (``E``, unit L2, fixed dimension 128)
plus center-time activity logits (``P``).  Windows share one set of learnable
queries, so the same source should be representable by the same query across
windows; stable ``track_id`` association stays a downstream concern.

``slot_valid`` is uniformly ``True`` here: K is a capacity, empty capacity is
handled by the training loss (``aat.losses``), not by hiding slots.  A valid
candidate keeps a unit identity vector even while ``P = 0`` (short-silence
identity memory), as required by protocol 0.1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from aat.contracts.version import DEFAULT_EMBEDDING_DIM, DEFAULT_SLOTS

__all__ = [
    "HeadConfig",
    "HeadOutput",
    "MultiSourceHead",
    "SinusoidalTimeEncoding",
    "head_output_to_prediction_arrays",
]

#: Finite stand-in for -inf in the attention mask: softmax over an all-masked
#: row stays finite and the row is re-zeroed after the mask multiply.
_MASKED_SCORE = -1e9
_NORM_EPS = 1e-8


@dataclass(frozen=True)
class HeadConfig:
    """Static head configuration.

    ``embedding_dim`` is fixed by the protocol; only ``slots`` (K) and internal
    width are configurable.
    """

    feature_dim: int
    slots: int = DEFAULT_SLOTS
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    d_model: int = 64
    shortest_time_period_seconds: float = 0.02
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1, got {self.feature_dim}")
        if self.slots < 1:
            raise ValueError(f"slots must be >= 1, got {self.slots}")
        if self.embedding_dim != DEFAULT_EMBEDDING_DIM:
            raise ValueError(
                "protocol 0.1.0 fixes embedding_dim at "
                f"{DEFAULT_EMBEDDING_DIM}; got {self.embedding_dim}"
            )
        if self.d_model < 2:
            raise ValueError(f"d_model must be >= 2, got {self.d_model}")
        if self.shortest_time_period_seconds <= 0.0:
            raise ValueError("shortest_time_period_seconds must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class SinusoidalTimeEncoding(nn.Module):
    """Parameter-free sinusoidal encoding of absolute frame times in seconds.

    Frequency ``i`` uses period ``shortest_period_seconds * 2**i`` so the
    fastest channel resolves the nominal hop while the slowest spans minutes.
    """

    def __init__(self, d_model: int, *, shortest_period_seconds: float = 0.02) -> None:
        super().__init__()
        if d_model < 2:
            raise ValueError(f"d_model must be >= 2, got {d_model}")
        if shortest_period_seconds <= 0.0:
            raise ValueError("shortest_period_seconds must be > 0")
        half = d_model // 2
        frequencies = 2.0 * math.pi / shortest_period_seconds
        frequencies = frequencies * (2.0 ** -torch.arange(half, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies)
        self._output_dim = 2 * half
        self._padded_dim = d_model - self._output_dim

    def forward(self, times: Tensor) -> Tensor:
        if times.dim() < 1:
            raise ValueError("frame times must have at least one dimension")
        angles = times.to(torch.float32).unsqueeze(-1) * self.frequencies
        encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        if self._padded_dim:
            encoded = F.pad(encoded, (0, self._padded_dim))
        return encoded


@dataclass(frozen=True)
class HeadOutput:
    """Per-window head output.

    Shapes are ``embeddings [B, K, 128]``, ``activity_logits [B, K]``,
    ``activity [B, K]`` and ``slot_valid [B, K]``.
    """

    embeddings: Tensor
    activity_logits: Tensor
    activity: Tensor
    slot_valid: Tensor


class MultiSourceHead(nn.Module):
    """K learnable queries cross-attending over masked feature frames."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        self.time_encoding = SinusoidalTimeEncoding(
            config.d_model,
            shortest_period_seconds=config.shortest_time_period_seconds,
        )
        self.feature_projection = nn.Linear(config.feature_dim, config.d_model)
        self.query = nn.Parameter(torch.empty(config.slots, config.d_model))
        nn.init.xavier_uniform_(self.query)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.identity_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.embedding_dim),
        )
        self.activity_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(
        self,
        features: Tensor,
        feature_valid: Tensor | None = None,
        frame_times: Tensor | None = None,
    ) -> HeadOutput:
        """Run the head on ``[B, T, F]`` features.

        ``feature_valid [B, T]`` (default all-valid) masks zero-padded frames.
        ``frame_times [B, T]`` is the authoritative frame time on the original
        absolute axis; it only feeds the fixed time encoding.
        """

        if features.dim() != 3:
            raise ValueError(f"features must be [B, T, F], got {tuple(features.shape)}")
        batch, frames, feature_dim = features.shape
        if feature_dim != self.config.feature_dim:
            raise ValueError(
                f"features last dimension {feature_dim} does not match "
                f"feature_dim {self.config.feature_dim}"
            )
        if feature_valid is None:
            valid = torch.ones(batch, frames, dtype=torch.bool, device=features.device)
        else:
            if feature_valid.shape != (batch, frames):
                raise ValueError(
                    f"feature_valid must be [{batch}, {frames}], "
                    f"got {tuple(feature_valid.shape)}"
                )
            valid = feature_valid.to(torch.bool)

        if batch == 0:
            return self._empty_output(features.device, features.dtype)

        hidden = self.feature_projection(features)
        if frame_times is not None:
            if frame_times.shape != (batch, frames):
                raise ValueError(
                    f"frame_times must be [{batch}, {frames}], "
                    f"got {tuple(frame_times.shape)}"
                )
            hidden = hidden + self.time_encoding(frame_times).to(hidden.dtype)

        keys = self.key(hidden)
        values = self.value(hidden)
        queries = self.query.unsqueeze(0).expand(batch, -1, -1).to(hidden.dtype)
        scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(self.config.d_model)
        scores = torch.where(
            valid[:, None, :],
            scores,
            torch.full_like(scores, _MASKED_SCORE),
        )
        attention = torch.softmax(scores, dim=-1)
        attention = attention * valid[:, None, :].to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
        hidden = torch.matmul(attention, values)
        hidden = self.dropout(self.ffn(hidden))

        raw_identity = self.identity_head(hidden)
        fallback = F.normalize(self.identity_head(self.query), dim=-1, eps=_NORM_EPS)
        norms = raw_identity.norm(dim=-1, keepdim=True)
        embeddings = torch.where(
            norms > _NORM_EPS,
            raw_identity / norms.clamp_min(_NORM_EPS),
            fallback.unsqueeze(0).to(raw_identity.dtype),
        )

        activity_logits = self.activity_head(hidden).squeeze(-1)
        activity = torch.sigmoid(activity_logits)
        slot_valid = torch.ones_like(activity_logits, dtype=torch.bool)
        return HeadOutput(
            embeddings=embeddings,
            activity_logits=activity_logits,
            activity=activity,
            slot_valid=slot_valid,
        )

    def _empty_output(self, device: torch.device, dtype: torch.dtype) -> HeadOutput:
        shape = (0, self.config.slots, self.config.embedding_dim)
        return HeadOutput(
            embeddings=torch.zeros(shape, device=device, dtype=dtype),
            activity_logits=torch.zeros((0, self.config.slots), device=device, dtype=dtype),
            activity=torch.zeros((0, self.config.slots), device=device, dtype=dtype),
            slot_valid=torch.zeros((0, self.config.slots), device=device, dtype=torch.bool),
        )


def head_output_to_prediction_arrays(
    output: HeadOutput,
    center_times: Any,
    center_valid: Any | None = None,
    *,
    hop_seconds: float | None = None,
    sample_id: str | None = None,
) -> dict[str, Any]:
    """Convert a :class:`HeadOutput` into ``PredictionData`` keyword arrays.

    Invalid slots are canonicalized to zero vectors and zero probability before
    the numpy conversion, matching the protocol semantics.
    """

    embeddings = output.embeddings.detach().to(torch.float32).cpu().numpy().copy()
    activity = output.activity.detach().to(torch.float32).cpu().numpy().copy()
    slot_valid = output.slot_valid.detach().cpu().numpy().astype(bool)
    embeddings[~slot_valid] = 0.0
    activity[~slot_valid] = 0.0

    times = np.asarray(center_times, dtype=np.float64).reshape(-1)
    if center_valid is None:
        valid_centers = np.ones(times.shape[0], dtype=bool)
    else:
        valid_centers = np.asarray(center_valid, dtype=bool).reshape(-1)
        if valid_centers.shape != times.shape:
            raise ValueError("center_valid must have the same length as center_times")

    arrays: dict[str, Any] = {
        "center_times": times,
        "embeddings": embeddings.astype(np.float32, copy=False),
        "activity": activity.astype(np.float32, copy=False),
        "slot_valid": slot_valid,
        "center_valid": valid_centers,
    }
    if hop_seconds is not None:
        arrays["hop_seconds"] = float(hop_seconds)
    if sample_id is not None:
        arrays["sample_id"] = sample_id
    return arrays
