"""K learnable source queries over masked, time-stamped frame features.

The head implements the ``E[K, 128]`` / center-activity ``P[K]`` contract from
``docs/SCHEMAS.md`` (see ``docs/MODEL_HEAD.md`` for the formulas):

* ``identity`` attention pools frame *content* only, so the identity vector
  stays stable across window centers (and across short silence);
* ``center`` attention sees time-stamped keys plus a learnable per-query time
  bias, so it can focus on the center frame while ``P`` is trained to be the
  activity exactly at the window center.

Only fake/feature tensors are required: no AuT weights, audio or training loop
live here.  ``slot_valid`` marks usable candidates; valid slots always carry a
finite unit-norm embedding even when ``P`` is ~0 (identity memory), invalid
slots are canonical all-zero ``E``/``P``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from aat.contracts.version import DEFAULT_EMBEDDING_DIM, DEFAULT_SLOTS

__all__ = [
    "HeadOutput",
    "SourceQueryHead",
    "head_output_to_prediction_data",
]


def _unit_normalize(raw: Tensor, eps: float = 1e-6) -> Tensor:
    """L2-normalize ``raw`` and guarantee a finite unit vector.

    A network could in principle emit an exactly zero vector (all weights
    cancel); then ``raw / eps`` is still zero.  Such degenerate rows fall back
    to a fixed canonical direction so the protocol requirement "valid slot
    embedding is a unit vector" cannot be broken by numerical accident.
    """

    norm = raw.norm(dim=-1, keepdim=True)
    unit = raw / norm.clamp_min(eps)
    degenerate = norm < eps
    if bool(degenerate.any()):
        fallback = torch.zeros_like(raw)
        fallback[..., 0] = 1.0
        unit = torch.where(degenerate, fallback, unit)
    return unit


class _FourierTimeEncoder(nn.Module):
    """Sine/cosine time features on ``base_hz * 2**j`` frequencies."""

    def __init__(self, out_dim: int, num_frequencies: int, base_hz: float) -> None:
        super().__init__()
        if out_dim < 1 or num_frequencies < 1 or base_hz <= 0:
            raise ValueError("time encoder needs out_dim/frequencies >= 1 and base_hz > 0")
        frequencies = base_hz * torch.pow(
            torch.full((num_frequencies,), 2.0, dtype=torch.float32),
            torch.arange(num_frequencies, dtype=torch.float32),
        )
        self.register_buffer("angular_frequencies", frequencies * (2.0 * torch.pi), persistent=True)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_frequencies, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, delta_seconds: Tensor) -> Tensor:
        angles = delta_seconds.unsqueeze(-1) * self.angular_frequencies
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(encoded)


class _MaskedAttention(nn.Module):
    """Multi-head attention whose softmax runs over valid keys only.

    Padded keys use values that are already zeroed by the caller, and fully
    masked rows degrade to a uniform average over zero vectors.  That keeps the
    output finite (no all ``-inf`` softmax NaNs) on empty source or fully padded
    batches; the caller then canonicalizes those rows to all-zero ``E``/``P``.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        valid: Tensor,
        bias: Tensor | None = None,
    ) -> Tensor:
        batch, query_len, _ = query.shape
        keys = self._split_heads(self.k_proj(key))
        values = self._split_heads(self.v_proj(value))
        queries = self._split_heads(self.q_proj(query))

        scores = torch.matmul(queries, keys.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if bias is not None:
            scores = scores + bias.unsqueeze(1)
        if valid is not None:
            scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        weights = self.dropout(weights)
        pooled = torch.matmul(weights, values)
        pooled = pooled.transpose(1, 2).reshape(batch, query_len, self.d_model)
        return self.out_proj(pooled)


@dataclass
class HeadOutput:
    """Output of :class:`SourceQueryHead` for one batch of center windows.

    Shapes are ``embeddings [B, K, 128]``, ``activity_logits [B, K]``,
    ``slot_valid [B, K]`` and ``center_valid [B]``.  ``embeddings`` is already
    canonical: unit rows for valid slots, all-zero rows for invalid slots.
    """

    embeddings: Tensor
    activity_logits: Tensor
    slot_valid: Tensor
    center_valid: Tensor

    @property
    def slots(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[2])

    def activity_probabilities(self) -> Tensor:
        """``P = sigmoid(logit)`` with invalid slots forced to exactly 0."""

        probabilities = torch.sigmoid(self.activity_logits)
        return probabilities * self.slot_valid.to(dtype=probabilities.dtype)

    def to_prediction_data(
        self,
        center_times: Sequence[float] | np.ndarray,
        *,
        hop_seconds: float | None = None,
        sample_id: str | None = None,
        provenance: Any | None = None,
    ):
        """Convert to the shared ``PredictionData`` contract object.

        ``center_times`` must be the strictly increasing original-track times of
        the ``B`` windows in this output.  The conversion re-checks the
        canonical zero/unit rules through ``PredictionData`` validation.
        """

        from aat.contracts import PredictionData

        times = np.asarray(center_times, dtype=np.float64).reshape(-1)
        if times.shape[0] != self.embeddings.shape[0]:
            raise ValueError(
                f"center_times has {times.shape[0]} entries but head output has "
                f"{self.embeddings.shape[0]} windows"
            )
        embeddings = self.embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
        activity = self.activity_probabilities().detach().cpu().numpy().astype(np.float32, copy=False)
        return PredictionData(
            center_times=times,
            embeddings=embeddings,
            activity=activity,
            slot_valid=self.slot_valid.detach().cpu().numpy().astype(bool, copy=False),
            center_valid=self.center_valid.detach().cpu().numpy().astype(bool, copy=False),
            slots=self.slots,
            embedding_dim=self.embedding_dim,
            hop_seconds=hop_seconds,
            sample_id=sample_id,
            provenance=provenance,
        )


class SourceQueryHead(nn.Module):
    """K learnable queries reading masked, time-stamped frame features.

    Parameters
    ----------
    feature_dim:
        Input feature dimension ``F`` (continues to be free per the feature
        contract; it is not the 128-D embedding dimension).
    slots:
        Number of queries/embedding slots ``K`` (default 8, configurable).
    embedding_dim:
        Fixed at 128 by protocol 0.1.0; other values are rejected.
    d_model, num_heads, dropout, time_frequencies, base_frequency_hz,
    ffn_multiplier:
        Small, interpretable transformer head hyper-parameters.
    """

    def __init__(
        self,
        feature_dim: int,
        slots: int = DEFAULT_SLOTS,
        *,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
        time_frequencies: int = 6,
        base_frequency_hz: float = 1.0,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        if slots < 1:
            raise ValueError("slots must be >= 1")
        if embedding_dim != DEFAULT_EMBEDDING_DIM:
            raise ValueError(
                f"protocol 0.1.0 fixes embedding_dim at {DEFAULT_EMBEDDING_DIM}; "
                f"got {embedding_dim}"
            )
        if d_model < 1 or num_heads < 1 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if ffn_multiplier < 1:
            raise ValueError("ffn_multiplier must be >= 1")

        self.feature_dim = feature_dim
        self.slots = slots
        self.embedding_dim = embedding_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.time_encoder = _FourierTimeEncoder(d_model, time_frequencies, base_frequency_hz)
        self.queries = nn.Parameter(torch.empty(slots, d_model))

        self.identity_attention = _MaskedAttention(d_model, num_heads, dropout)
        self.center_attention = _MaskedAttention(d_model, num_heads, dropout)
        # Per-query bilinear time bias: the query decides how sharply it looks
        # at the center frame instead of averaging the whole window.
        self.center_time_bias = nn.Parameter(torch.empty(slots, d_model))

        hidden = d_model * ffn_multiplier
        self.identity_ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model)
        )
        self.center_ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model)
        )
        self.identity_norm1 = nn.LayerNorm(d_model)
        self.identity_norm2 = nn.LayerNorm(d_model)
        self.center_norm1 = nn.LayerNorm(d_model)
        self.center_norm2 = nn.LayerNorm(d_model)

        self.embed_head = nn.Linear(d_model, embedding_dim)
        self.activity_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        nn.init.normal_(self.center_time_bias, mean=0.0, std=0.02)

    @staticmethod
    def _as_batch_times(
        times: Tensor | Sequence[float] | None,
        batch: int,
        length: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        name: str,
    ) -> Tensor:
        if times is None:
            return torch.zeros(batch, length, dtype=dtype, device=device)
        tensor = times if isinstance(times, Tensor) else torch.as_tensor(times)
        tensor = tensor.to(dtype=dtype, device=device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).expand(batch, -1)
        if tensor.dim() != 2 or tensor.shape[0] != batch or tensor.shape[1] != length:
            raise ValueError(
                f"{name}: expected shape [{batch}, {length}] (or [{length}]), "
                f"got {list(tensor.shape)}"
            )
        return tensor

    def forward(
        self,
        features: Tensor,
        frame_times: Tensor | Sequence[float] | None = None,
        frame_valid: Tensor | None = None,
        center_times: Tensor | Sequence[float] | None = None,
        center_valid: Tensor | None = None,
    ) -> HeadOutput:
        if not isinstance(features, Tensor):
            features = torch.as_tensor(features)
        if features.dim() != 3:
            raise ValueError(f"features must be [B, T, F]; got {list(features.shape)}")
        if features.shape[2] != self.feature_dim:
            raise ValueError(
                f"features last dim {features.shape[2]} does not match feature_dim {self.feature_dim}"
            )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features contain NaN/Inf; refusing to produce outputs")

        batch, length, _ = features.shape
        device = features.device
        dtype = features.dtype

        if frame_valid is None:
            frame_valid = torch.ones(batch, length, dtype=torch.bool, device=device)
        else:
            if not isinstance(frame_valid, Tensor) or frame_valid.dtype != torch.bool:
                raise ValueError("frame_valid must be a bool tensor")
            frame_valid = frame_valid.to(device=device)
            if frame_valid.shape != (batch, length):
                raise ValueError(
                    f"frame_valid: expected shape [{batch}, {length}], got {list(frame_valid.shape)}"
                )

        frame_times_t = self._as_batch_times(
            frame_times, batch, length, dtype=dtype, device=device, name="frame_times"
        )
        if not bool(torch.isfinite(frame_times_t).all()):
            raise ValueError("frame_times contain NaN/Inf")
        if center_times is None:
            center_times_t = torch.zeros(batch, dtype=dtype, device=device)
        else:
            center_times_t = torch.as_tensor(center_times, device=device, dtype=dtype).reshape(-1)
            if center_times_t.shape[0] != batch:
                raise ValueError(
                    f"center_times: expected {batch} entries, got {center_times_t.shape[0]}"
                )
            if not bool(torch.isfinite(center_times_t).all()):
                raise ValueError("center_times contain NaN/Inf")

        data_present = frame_valid.any(dim=1) if length > 0 else torch.zeros(batch, dtype=torch.bool, device=device)
        if center_valid is None:
            center_valid_t = data_present
        else:
            if not isinstance(center_valid, Tensor) or center_valid.dtype != torch.bool:
                raise ValueError("center_valid must be a bool tensor")
            center_valid_t = center_valid.to(device=device).reshape(-1)
            if center_valid_t.shape[0] != batch:
                raise ValueError(
                    f"center_valid: expected {batch} entries, got {center_valid_t.shape[0]}"
                )
        slot_valid = center_valid_t[:, None] & data_present[:, None]
        slot_valid = slot_valid.expand(batch, self.slots)

        selected = frame_valid.unsqueeze(-1)
        content = self.input_proj(features) * selected
        delta = frame_times_t - center_times_t[:, None]
        time_embedding = self.time_encoder(delta) * selected
        center_keys = (content + time_embedding) * selected

        query = self.queries.unsqueeze(0).expand(batch, -1, -1)

        identity_pooled = self.identity_attention(query, content, content, frame_valid)
        identity_repr = self.identity_norm1(query + identity_pooled)
        identity_repr = self.identity_norm2(identity_repr + self.identity_ffn(identity_repr))

        time_bias = torch.einsum("btd,kd->bkt", time_embedding, self.center_time_bias)
        center_pooled = self.center_attention(query, center_keys, content, frame_valid, bias=time_bias)
        center_repr = self.center_norm1(query + center_pooled)
        center_repr = self.center_norm2(center_repr + self.center_ffn(center_repr))

        embeddings = _unit_normalize(self.embed_head(identity_repr))
        embeddings = embeddings * slot_valid.unsqueeze(-1)
        logits = self.activity_head(torch.cat([center_repr, query], dim=-1)).squeeze(-1)
        return HeadOutput(
            embeddings=embeddings,
            activity_logits=logits,
            slot_valid=slot_valid,
            center_valid=center_valid_t,
        )


def head_output_to_prediction_data(
    output: HeadOutput,
    center_times: Sequence[float] | np.ndarray,
    **kwargs: Any,
):
    """Module-level alias of :meth:`HeadOutput.to_prediction_data`."""

    return output.to_prediction_data(center_times, **kwargs)
