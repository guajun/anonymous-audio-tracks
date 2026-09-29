"""Synthetic feature/target batches for head and loss tests (issue #7).

No AuT, no audio, no real training data: each source is a distinct DCT-like
"pitch" signature, center windows are non-adjacent in time, and local frame
activity deliberately includes

* neighbors active while the center frame is silent (``boundary``),
* the center frame active while neighbors are silent (``isolated``),
* sustained activity,

so tests can check that ``P`` learns the *center* rather than "anywhere in the
window".  Compositions differ in their signature set so cross-composition
negatives are genuinely different sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from torch import Tensor


def make_signatures(num_sources: int, feature_dim: int, *, pitch_start: int = 1) -> Tensor:
    """Near-orthonormal DCT-like signatures with distinct "pitches"."""

    if num_sources < 1 or feature_dim < 1:
        raise ValueError("num_sources and feature_dim must be >= 1")
    positions = torch.arange(feature_dim, dtype=torch.float32)
    vectors = []
    for source in range(num_sources):
        pitch = pitch_start + source
        vector = torch.cos(math.pi * (positions + 0.5) * pitch / feature_dim)
        vector = vector - vector.mean()
        vectors.append(vector / vector.norm())
    return torch.stack(vectors)


@dataclass
class SyntheticFeatureBatch:
    """One fake training batch with group/window structure kept explicit."""

    features: Tensor  # [G, N, T, F]
    frame_times: Tensor  # [G, N, T]
    frame_valid: Tensor  # [G, N, T] bool
    center_times: Tensor  # [G, N]
    center_valid: Tensor  # [G, N] bool
    activity: Tensor  # [G, N, S]
    context_activity: Tensor  # [G, N, S] bool: any non-center frame active
    source_valid: Tensor  # [G, S] bool
    signatures: Tensor  # [G, S, F]
    composition_ids: list[str]
    source_ids: list[list[str]]

    @property
    def num_groups(self) -> int:
        return int(self.features.shape[0])

    @property
    def windows_per_group(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_sources(self) -> int:
        return int(self.activity.shape[2])

    def flat(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Flatten groups/windows for the head: (features, times, valid, centers, center_valid)."""

        groups, windows = self.features.shape[:2]
        return (
            self.features.reshape(groups * windows, *self.features.shape[2:]),
            self.frame_times.reshape(groups * windows, self.frame_times.shape[2]),
            self.frame_valid.reshape(groups * windows, self.frame_valid.shape[2]),
            self.center_times.reshape(groups * windows),
            self.center_valid.reshape(groups * windows),
        )


def make_synthetic_batch(
    *,
    num_groups: int = 2,
    windows_per_group: int = 4,
    num_sources: int = 3,
    feature_dim: int = 8,
    frames_per_window: int = 5,
    hop_seconds: float = 0.7,
    noise: float = 0.03,
    seed: int = 0,
    silence: bool = False,
    boundary_fraction: float = 0.35,
) -> SyntheticFeatureBatch:
    """Build a deterministic fake batch.

    ``boundary_fraction`` controls how often a window shows "neighbors active,
    center silent" for a source; ``silence=True`` returns an all-silent batch
    with no identity information (only tiny noise in the features).
    """

    if frames_per_window < 3 or frames_per_window % 2 == 0:
        raise ValueError("frames_per_window must be odd and >= 3 so a unique center frame exists")
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=generator)

    center_index = frames_per_window // 2
    offsets = (torch.arange(frames_per_window) - center_index).to(torch.float32) * 0.04
    gaps = 0.5 + 1.5 * rand(num_groups, windows_per_group)
    center_times = torch.cumsum(gaps, dim=1) + 0.25 * rand(num_groups, 1)
    frame_times = center_times[:, :, None] + offsets[None, None, :]
    frame_valid = torch.ones(num_groups, windows_per_group, frames_per_window, dtype=torch.bool)
    center_valid = torch.ones(num_groups, windows_per_group, dtype=torch.bool)

    signatures = torch.stack(
        [make_signatures(num_sources, feature_dim, pitch_start=1 + group * num_sources) for group in range(num_groups)]
    )

    activity = torch.zeros(num_groups, windows_per_group, num_sources)
    local = torch.zeros(num_groups, windows_per_group, frames_per_window, num_sources)
    for group in range(num_groups):
        for window in range(windows_per_group):
            for source in range(num_sources):
                if silence:
                    continue
                roll = float(rand(()))
                if roll < boundary_fraction * 0.5:
                    # neighbors active, center silent: P must stay low.
                    local[group, window, :, source] = 1.0
                    local[group, window, center_index, source] = 0.0
                elif roll < boundary_fraction:
                    # center active, context silent: P must still fire.
                    local[group, window, center_index, source] = 1.0
                else:
                    value = 1.0 if float(rand(())) > 0.5 else 0.0
                    local[group, window, :, source] = value
            activity[group, window, :] = local[group, window, center_index, :]

    amplitudes = (0.6 + 0.4 * rand(num_groups, 1, 1, num_sources)).to(torch.float32)
    clean = torch.einsum("gnts,gsf->gntf", local * amplitudes, signatures)
    if silence:
        features = 0.01 * torch.randn(
            num_groups, windows_per_group, frames_per_window, feature_dim, generator=generator
        )
    else:
        features = clean + noise * torch.randn(
            num_groups, windows_per_group, frames_per_window, feature_dim, generator=generator
        )

    context = torch.zeros(num_groups, windows_per_group, num_sources, dtype=torch.bool)
    for group in range(num_groups):
        for window in range(windows_per_group):
            for source in range(num_sources):
                frames = [index for index in range(frames_per_window) if index != center_index]
                context[group, window, source] = bool(local[group, window, frames, source].any())

    source_valid = torch.ones(num_groups, num_sources, dtype=torch.bool)
    composition_ids = [f"comp-{group}" for group in range(num_groups)]
    source_ids = [[f"src-{source}" for source in range(num_sources)] for _ in range(num_groups)]
    return SyntheticFeatureBatch(
        features=features.to(torch.float32),
        frame_times=frame_times,
        frame_valid=frame_valid,
        center_times=center_times,
        center_valid=center_valid,
        activity=activity.to(torch.float32),
        context_activity=context,
        source_valid=source_valid,
        signatures=signatures,
        composition_ids=composition_ids,
        source_ids=source_ids,
    )


def make_silence_batch(**kwargs) -> SyntheticFeatureBatch:
    """All-silent fake batch: no identity may be forced from absence of evidence."""

    kwargs.pop("silence", None)
    return make_synthetic_batch(silence=True, **kwargs)
