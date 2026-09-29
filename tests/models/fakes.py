"""Synthetic-only fixtures for head and loss tests.

Nothing here touches audio, checkpoints, datasets or the real AuT backbone.
Features are tiny hand-made mixtures of per-source signature vectors with
distinct binary activity patterns, so a small head can in principle learn both
"which source is active" and "which identity is stable across windows".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from aat.losses import GroupTargets
from aat.models import HeadConfig, MultiSourceHead


@dataclass
class SyntheticGroup:
    """One same-composition training group plus its source signatures."""

    features: torch.Tensor
    feature_valid: torch.Tensor
    frame_times: torch.Tensor
    activity: np.ndarray
    signatures: np.ndarray
    targets: GroupTargets


def activity_patterns(n_windows: int, n_sources: int) -> np.ndarray:
    """Distinct binary on/off patterns per source (period doubling codes)."""

    activity = np.zeros((n_windows, n_sources), dtype=np.float32)
    indexes = np.arange(n_windows)
    for source in range(n_sources):
        block = 2**source
        activity[:, source] = ((indexes // block) % 2 == 0).astype(np.float32)
    return activity


def synthetic_group(
    *,
    seed: int = 0,
    n_windows: int = 8,
    n_sources: int = 3,
    feature_dim: int = 8,
    frames: int = 2,
    noise: float = 0.03,
) -> SyntheticGroup:
    rng = np.random.default_rng(seed)
    activity = activity_patterns(n_windows, n_sources)
    signatures = rng.normal(size=(n_sources, feature_dim)).astype(np.float32)
    signatures /= np.linalg.norm(signatures, axis=1, keepdims=True)
    features = np.zeros((n_windows, frames, feature_dim), dtype=np.float32)
    for window in range(n_windows):
        mixture = sum(
            activity[window, source] * signatures[source] for source in range(n_sources)
        )
        features[window] = mixture + noise * rng.normal(size=(frames, feature_dim))
    feature_valid = np.ones((n_windows, frames), dtype=bool)
    offsets = np.arange(frames, dtype=np.float64) * 0.08
    frame_times = 0.5 * np.arange(n_windows, dtype=np.float64)[:, None] + offsets[None, :]
    return SyntheticGroup(
        features=torch.from_numpy(features),
        feature_valid=torch.from_numpy(feature_valid),
        frame_times=torch.from_numpy(frame_times),
        activity=activity,
        signatures=signatures,
        targets=GroupTargets(activity=torch.from_numpy(activity)),
    )


def make_head(
    *,
    feature_dim: int = 8,
    slots: int = 3,
    d_model: int = 16,
    seed: int = 0,
) -> MultiSourceHead:
    torch.manual_seed(seed)
    return MultiSourceHead(
        HeadConfig(feature_dim=feature_dim, slots=slots, d_model=d_model)
    )


def random_unit_embeddings(
    *, n_windows: int, slots: int, dim: int, seed: int = 0
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(n_windows, slots, dim, generator=generator)
    return torch.nn.functional.normalize(raw, dim=-1)


def orthogonal_embeddings(*, n_windows: int, slots: int, seed: int = 0) -> torch.Tensor:
    """One orthogonal-ish unit direction per slot, repeated across windows."""

    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(slots, 32, generator=generator)
    base = torch.nn.functional.normalize(raw, dim=-1)
    return base.unsqueeze(0).expand(n_windows, -1, -1).clone()
