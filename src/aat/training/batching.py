"""Turn real dataset blocks into the rectangular tensor batch the head needs.

One training group is one song block: several non-adjacent center windows with
**fixed source columns** (``sources.json`` order).  Groups are padded to the
maximum source count of the step, but the padded columns never enter the loss
(``source_valid`` is False and the targets are zero).  Padding windows inside
features are masked by ``frame_valid`` and never participate in attention or
the loss.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from aat.data import SourceWindowBlock

from .config import TrainingError, canonical_json
from .encoding import EncodedWindows


@dataclass(frozen=True)
class GroupSummary:
    """Per-song diagnostics recorded with each step."""

    sample_id: str
    split: str
    windows: int
    sources: int
    source_ids: tuple[str, ...]
    center_valid: int
    active_labels: int
    feature_dim: int
    token_count: int


@dataclass(frozen=True)
class TrainingBatch:
    """Rectangular group batch ready for ``SourceQueryHead`` and ``head_loss``."""

    features: Tensor  # [G, N, T, F]
    frame_times: Tensor  # [G, N, T]
    frame_valid: Tensor  # [G, N, T] bool
    center_times: Tensor  # [G, N]
    center_valid: Tensor  # [G, N] bool
    activity: Tensor  # [G, N, S] targets of real source columns
    source_valid: Tensor  # [G, S] bool
    composition_ids: tuple[str, ...]
    source_ids: tuple[tuple[str, ...], ...]
    sample_ids: tuple[str, ...]
    groups: tuple[GroupSummary, ...]
    windows: int
    source_columns: int
    content_sha256: str

    def to_device(self, device: torch.device | str) -> "TrainingBatch":
        target = torch.device(device)
        return TrainingBatch(
            features=self.features.to(target),
            frame_times=self.frame_times.to(target),
            frame_valid=self.frame_valid.to(target),
            center_times=self.center_times.to(target),
            center_valid=self.center_valid.to(target),
            activity=self.activity.to(target),
            source_valid=self.source_valid.to(target),
            composition_ids=self.composition_ids,
            source_ids=self.source_ids,
            sample_ids=self.sample_ids,
            groups=self.groups,
            windows=self.windows,
            source_columns=self.source_columns,
            content_sha256=self.content_sha256,
        )


def training_batch_from_blocks(
    blocks: Sequence[SourceWindowBlock],
    encoded: Sequence[EncodedWindows],
    *,
    dtype: torch.dtype = torch.float32,
) -> TrainingBatch:
    """Stack per-song blocks into one rectangular batch.

    All groups must have the same window count (``sample_batch`` guarantees it
    for ``on_unusable="error"``); source counts may differ and are padded.
    """

    if len(blocks) != len(encoded):
        raise TrainingError(
            f"blocks/encoded length mismatch: {len(blocks)} vs {len(encoded)}"
        )
    if not blocks:
        raise TrainingError("cannot build a training batch from zero groups")

    window_counts = {block.centers for block in blocks}
    if len(window_counts) != 1:
        raise TrainingError(
            f"groups must share one window count for the batched head; got {sorted(window_counts)}"
        )
    token_counts = {chunk.token_count for chunk in encoded}
    if len(token_counts) != 1:
        raise TrainingError(
            f"groups must share one token count (fixed windows); got {sorted(token_counts)}"
        )
    feature_dims = {chunk.feature_dim for chunk in encoded}
    if len(feature_dims) != 1:
        raise TrainingError(f"groups must share one feature dim; got {sorted(feature_dims)}")

    windows = blocks[0].centers
    feature_dim = encoded[0].feature_dim
    token_count = encoded[0].token_count
    source_columns = max(len(block.source_ids) for block in blocks)
    slots = blocks[0].slots
    if source_columns > slots:
        raise TrainingError(
            f"a group has {source_columns} sources but the dataset capacity is K={slots}"
        )

    groups = len(blocks)
    features = np.zeros((groups, windows, token_count, feature_dim), dtype=np.float32)
    frame_times = np.zeros((groups, windows, token_count), dtype=np.float64)
    frame_valid = np.zeros((groups, windows, token_count), dtype=bool)
    center_times = np.zeros((groups, windows), dtype=np.float64)
    center_valid = np.zeros((groups, windows), dtype=bool)
    activity = np.zeros((groups, windows, source_columns), dtype=np.float32)
    source_valid = np.zeros((groups, source_columns), dtype=bool)

    composition_ids: list[str] = []
    source_ids: list[tuple[str, ...]] = []
    sample_ids: list[str] = []
    summaries: list[GroupSummary] = []

    for index, (block, chunk) in enumerate(zip(blocks, encoded)):
        count = len(block.source_ids)
        if count > source_columns or block.centers != windows:
            raise TrainingError(f"group {index}: inconsistent block shape")
        features[index] = chunk.features
        frame_times[index] = chunk.frame_times
        frame_valid[index] = chunk.frame_valid
        center_times[index] = np.asarray(block.center_times, dtype=np.float64)
        center_valid[index] = np.asarray(block.center_valid, dtype=bool)
        # Real source columns only; padding columns stay zero/False.
        activity[index, :, :count] = block.activity[:, :count]
        source_valid[index, :count] = block.source_present[0, :count]

        composition_ids.append(block.sample_id)
        source_ids.append(tuple(block.source_ids))
        sample_ids.append(block.sample_id)
        summaries.append(
            GroupSummary(
                sample_id=block.sample_id,
                split=block.split,
                windows=windows,
                sources=count,
                source_ids=tuple(block.source_ids),
                center_valid=int(center_valid[index].sum()),
                active_labels=int(
                    (block.activity[:, :count] >= 0.5).sum()
                ),
                feature_dim=feature_dim,
                token_count=token_count,
            )
        )

    batch = TrainingBatch(
        features=torch.from_numpy(features).to(dtype=dtype),
        frame_times=torch.from_numpy(frame_times),
        frame_valid=torch.from_numpy(frame_valid),
        center_times=torch.from_numpy(center_times),
        center_valid=torch.from_numpy(center_valid),
        activity=torch.from_numpy(activity).to(dtype=dtype),
        source_valid=torch.from_numpy(source_valid),
        composition_ids=tuple(composition_ids),
        source_ids=tuple(source_ids),
        sample_ids=tuple(sample_ids),
        groups=tuple(summaries),
        windows=windows,
        source_columns=source_columns,
        content_sha256="",
    )
    return replace(batch, content_sha256=training_batch_sha256(batch))


def training_batch_sha256(batch: TrainingBatch) -> str:
    """Deterministic hash of the actual tensors and identity of a batch."""

    digest = hashlib.sha256()
    digest.update(
        canonical_json(
            {
                "sample_ids": list(batch.sample_ids),
                "source_ids": [list(ids) for ids in batch.source_ids],
                "composition_ids": list(batch.composition_ids),
                "windows": batch.windows,
                "source_columns": batch.source_columns,
            }
        ).encode("utf-8")
    )
    for name in (
        "features",
        "frame_times",
        "frame_valid",
        "center_times",
        "center_valid",
        "activity",
        "source_valid",
    ):
        tensor = getattr(batch, name)
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


__all__ = [
    "GroupSummary",
    "TrainingBatch",
    "training_batch_from_blocks",
    "training_batch_sha256",
]
