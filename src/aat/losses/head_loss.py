"""Permutation-invariant training loss for the K-query E/P head.

The loss treats one training group as "one composition, several center windows
with fixed source columns".  A single exact one-to-one matching is solved per
group from the whole-group activity sequence; then the loss exposes separate
terms:

* ``activity``      - matched center-activity BCE over valid centers;
* ``empty_slots``   - unmatched capacity pushed to activity 0 (not a silent
  source: a matched-but-silent source keeps its identity memory);
* ``positive``      - same identity (composition_id, source_id) cross-window
  prototype contrast, only on windows where that source is actually active;
* ``negative``      - hinge against other identities' prototypes.

Padding is excluded everywhere.  Ambiguous groups (multiple optimal
assignments) are reported and either symmetrically averaged (activity /
empty-slot terms) or masked out of identity supervision; truncated enumeration
skips identity and empty-slot terms so nothing depends on an arbitrary
assignment choice.  Every term keeps a zero-gradient path, so empty sources,
all-silent batches and all-invalid centers backpropagate finite zeros instead
of raising or fabricating success.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .matching import MatchingError, MatchingResult, match_sources

__all__ = [
    "HeadLoss",
    "HeadLossStats",
    "LossWeights",
    "head_loss",
]


@dataclass(frozen=True)
class LossWeights:
    """Default component weights; all four are exposed for tuning."""

    activity: float = 1.0
    empty_slots: float = 0.5
    positive: float = 1.0
    negative: float = 1.0


@dataclass(frozen=True)
class HeadLossStats:
    """Effective term counts and matching diagnostics for one batch."""

    groups: int
    matched_groups: int
    skipped_groups: int
    ambiguous_groups: int
    truncated_groups: int
    identity_masked_groups: int
    activity_terms: int
    empty_terms: int
    positive_terms: int
    negative_terms: int


@dataclass
class HeadLoss:
    """Weighted total plus every exposed component and matching result."""

    total: Tensor
    activity: Tensor
    empty_slots: Tensor
    positive: Tensor
    negative: Tensor
    weights: Mapping[str, float]
    stats: HeadLossStats
    matching: MatchingResult

    def components(self) -> dict[str, Tensor]:
        return {
            "activity": self.activity,
            "empty_slots": self.empty_slots,
            "positive": self.positive,
            "negative": self.negative,
        }

    def summary(self) -> dict[str, float]:
        values = {name: float(component.detach().cpu()) for name, component in self.components().items()}
        values["total"] = float(self.total.detach().cpu())
        return values


def _as_bool_tensor(value: Tensor | Any, shape: Sequence[int], *, name: str, device: torch.device) -> Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=torch.bool)
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name}: expected shape {list(shape)}, got {list(tensor.shape)}")
    return tensor


def head_loss(
    embeddings: Tensor,
    activity_logits: Tensor,
    *,
    activity_target: Tensor,
    source_valid: Tensor | None = None,
    composition_ids: Sequence[Hashable] | None = None,
    source_ids: Sequence[Sequence[Hashable]] | None = None,
    center_valid: Tensor | None = None,
    slot_valid: Tensor | None = None,
    weights: LossWeights | None = None,
    identity_activity_threshold: float = 0.5,
    negative_margin: float = 0.25,
    max_optimal_assignments: int = 64,
    match_cost_atol: float = 1e-9,
    match_cost_rtol: float = 1e-9,
    max_exact_slots: int = 16,
) -> HeadLoss:
    """Compute the arranged head loss.

    Shapes (``G`` groups, ``N`` center windows per group, ``K`` slots,
    ``S`` source columns, ``D`` = 128 embedding dimension):

    * ``embeddings``: ``[G, N, K, D]`` (unit rows expected; re-normalized here);
    * ``activity_logits``: ``[G, N, K]``;
    * ``activity_target``: ``[G, N, S]`` in ``[0, 1]``;
    * ``source_valid``: ``[G, S]`` bool (default all valid);
    * ``center_valid`` / ``slot_valid``: ``[G, N]`` / ``[G, N, K]`` bool
      (default all valid);
    * ``composition_ids``: length ``G``; ``source_ids[g]``: length ``S``.
      Identity positives only ever join equal ``(composition_id, source_id)``,
      so the same ``source_id`` string in two compositions is a negative, not
      a positive.  Omitting both disables identity terms.

    Raises ``ValueError`` when the head emits more valid sources than ``K``
    (capacity overflow) or ``K`` exceeds ``max_exact_slots``.
    """

    if not isinstance(embeddings, Tensor) or not isinstance(activity_logits, Tensor):
        raise TypeError("embeddings and activity_logits must be torch tensors")
    if embeddings.dim() != 4:
        raise ValueError(f"embeddings must be [G, N, K, D]; got {list(embeddings.shape)}")
    num_groups, num_windows, num_slots, embed_dim = embeddings.shape
    if activity_logits.shape != (num_groups, num_windows, num_slots):
        raise ValueError(
            f"activity_logits must match embeddings[..., K]; expected "
            f"{(num_groups, num_windows, num_slots)}, got {list(activity_logits.shape)}"
        )
    if num_slots > max_exact_slots:
        raise ValueError(
            f"K={num_slots} exceeds max_exact_slots={max_exact_slots}; exact group matching "
            "is intentionally limited to small K"
        )
    if embed_dim < 1:
        raise ValueError("embedding dimension must be >= 1")

    device = embeddings.device
    dtype = embeddings.dtype
    if activity_logits.dtype != dtype:
        activity_logits = activity_logits.to(dtype=dtype)
    if activity_logits.device != device:
        activity_logits = activity_logits.to(device=device)

    target = torch.as_tensor(activity_target, device=device, dtype=dtype)
    if target.dim() != 3:
        raise ValueError(f"activity_target must be [G, N, S]; got {list(target.shape)}")
    if target.shape[:2] != (num_groups, num_windows):
        raise ValueError("activity_target must share [G, N] with the head output")
    num_sources = int(target.shape[2])
    if not bool(torch.isfinite(target).all()) or bool(((target < 0) | (target > 1)).any()):
        raise ValueError("activity_target must be finite and inside [0, 1]")

    if source_valid is None:
        source_valid_t = torch.ones(num_groups, num_sources, dtype=torch.bool, device=device)
    else:
        source_valid_t = _as_bool_tensor(source_valid, (num_groups, num_sources), name="source_valid", device=device)
    if center_valid is None:
        center_valid_t = torch.ones(num_groups, num_windows, dtype=torch.bool, device=device)
    else:
        center_valid_t = _as_bool_tensor(center_valid, (num_groups, num_windows), name="center_valid", device=device)
    if slot_valid is None:
        slot_valid_t = torch.ones(num_groups, num_windows, num_slots, dtype=torch.bool, device=device)
    else:
        slot_valid_t = _as_bool_tensor(
            slot_valid, (num_groups, num_windows, num_slots), name="slot_valid", device=device
        )

    valid_sources_per_group = source_valid_t.sum(dim=1)
    overflow = valid_sources_per_group > num_slots
    if bool(overflow.any()):
        worst = int(overflow.nonzero(as_tuple=False)[0, 0])
        raise ValueError(
            f"group {worst} has {int(valid_sources_per_group[worst])} valid sources but K={num_slots}; "
            "source count must not exceed K"
        )

    if weights is None:
        weights = LossWeights()
    weight_map = {
        "activity": float(weights.activity),
        "empty_slots": float(weights.empty_slots),
        "positive": float(weights.positive),
        "negative": float(weights.negative),
    }

    use_identity = composition_ids is not None or source_ids is not None
    if use_identity:
        if composition_ids is None or source_ids is None:
            raise ValueError("composition_ids and source_ids must be provided together")
        if len(composition_ids) != num_groups:
            raise ValueError(f"composition_ids must have {num_groups} entries")
        if len(source_ids) != num_groups:
            raise ValueError(f"source_ids must have {num_groups} entries")
        for g, ids in enumerate(source_ids):
            if len(ids) < num_sources:
                raise ValueError(f"source_ids[{g}] must have at least {num_sources} entries")
    if not 0.0 <= identity_activity_threshold <= 1.0:
        raise ValueError("identity_activity_threshold must be inside [0, 1]")
    if not np.isfinite(negative_margin):
        raise ValueError("negative_margin must be finite")

    # ---- per-(group, window, slot, source) BCE and validity mask ----------
    logits_expanded = activity_logits.unsqueeze(-1).expand(
        num_groups, num_windows, num_slots, num_sources
    )
    target_expanded = target.unsqueeze(2).expand(
        num_groups, num_windows, num_slots, num_sources
    )
    bce_all = F.binary_cross_entropy_with_logits(
        logits_expanded, target_expanded, reduction="none"
    )
    bce_empty = F.binary_cross_entropy_with_logits(
        activity_logits, torch.zeros_like(activity_logits), reduction="none"
    )
    term_mask = (
        center_valid_t[:, :, None, None]
        & slot_valid_t[:, :, :, None]
        & source_valid_t[:, None, None, :]
    )

    # ---- group costs and exact matching -----------------------------------
    cost_num = (bce_all.detach() * term_mask).sum(dim=1)  # [G, K, S]
    cost_den = term_mask.sum(dim=1)  # [G, K, S]
    cost = (cost_num / cost_den.clamp_min(1)).transpose(1, 2)  # [G, S, K]
    cost = cost.masked_fill(cost_den.transpose(1, 2) == 0, float("inf"))
    slot_ok = ((~center_valid_t[:, :, None]) | slot_valid_t).all(dim=1)  # [G, K]
    cost = cost.masked_fill(~slot_ok[:, None, :], float("inf"))
    valid_center_per_group = center_valid_t.any(dim=1)
    # Groups with no valid center carry no evidence: mask their source rows out
    # of matching instead of failing on an all-invalid cost matrix.
    source_valid_for_matching = source_valid_t & valid_center_per_group[:, None]
    try:
        matching = match_sources(
            cost,
            source_valid_for_matching,
            slot_ok,
            max_optimal=max_optimal_assignments,
            atol=match_cost_atol,
            rtol=match_cost_rtol,
        )
    except MatchingError as error:  # pragma: no cover - surfaced with context
        raise ValueError(str(error)) from error

    # ---- components --------------------------------------------------------
    zero = embeddings.sum() * 0.0
    activity_numerator = zero
    empty_numerator = zero
    activity_denominator = 0.0
    empty_denominator = 0.0

    matched_groups = 0
    skipped_groups = 0
    ambiguous_groups = 0
    truncated_groups = 0
    identity_masked_groups = 0
    reliable_groups: list[int] = []

    for g in range(num_groups):
        group = matching.groups[g]
        source_count = len(group.source_indices)
        if not bool(valid_center_per_group[g]):
            if bool(source_valid_t[g].any()):
                skipped_groups += 1
            continue
        if source_count == 0:
            # All capacity is empty; still push those slots to P = 0.
            for slot in group.slot_indices:
                mask = center_valid_t[g] & slot_valid_t[g, :, slot]
                empty_numerator = empty_numerator + (bce_empty[g, :, slot] * mask).sum()
                empty_denominator += float(mask.sum())
            continue

        matched_groups += 1
        if group.optimal.truncated:
            truncated_groups += 1
            identity_masked_groups += 1
        elif group.optimal.num_optimal > 1:
            ambiguous_groups += 1
            identity_masked_groups += 1
        else:
            reliable_groups.append(g)

        if group.optimal.truncated:
            assignments = group.optimal.assignments[:1]
            assignment_weight = 1.0
        else:
            assignments = group.optimal.assignments
            assignment_weight = 1.0 / len(assignments)

        for assignment in assignments:
            matched_slots = [group.slot_indices[local] for local in assignment]
            for local_source, source in enumerate(group.source_indices):
                slot = matched_slots[local_source]
                mask = center_valid_t[g] & slot_valid_t[g, :, slot]
                activity_numerator = (
                    activity_numerator + assignment_weight * (bce_all[g, :, slot, source] * mask).sum()
                )
                activity_denominator += assignment_weight * float(mask.sum())
            if not group.optimal.truncated:
                for slot in group.slot_indices:
                    if slot in matched_slots:
                        continue
                    mask = center_valid_t[g] & slot_valid_t[g, :, slot]
                    empty_numerator = empty_numerator + assignment_weight * (bce_empty[g, :, slot] * mask).sum()
                    empty_denominator += assignment_weight * float(mask.sum())

    # ---- identity contrast (unique assignments only) -----------------------
    positive_numerator = zero
    negative_numerator = zero
    positive_denominator = 0
    negative_denominator = 0
    if use_identity and reliable_groups:
        anchors: dict[Hashable, list[tuple[int, int, int]]] = {}
        for g in reliable_groups:
            group = matching.groups[g]
            assignment = group.optimal.assignments[0]
            for local_source, source in enumerate(group.source_indices):
                slot = group.slot_indices[assignment[local_source]]
                key = (composition_ids[g], source_ids[g][source])
                active = (
                    center_valid_t[g]
                    & slot_valid_t[g, :, slot]
                    & (target[g, :, source] >= identity_activity_threshold)
                )
                for window in torch.nonzero(active, as_tuple=False).flatten().tolist():
                    anchors.setdefault(key, []).append((g, int(window), slot))

        normalized: dict[Hashable, Tensor] = {}
        prototypes: dict[Hashable, Tensor] = {}
        for key, entries in anchors.items():
            stacked = F.normalize(
                torch.stack([embeddings[g, window, slot] for g, window, slot in entries]), dim=-1, eps=1e-6
            )
            normalized[key] = stacked
            prototypes[key] = F.normalize(stacked.mean(dim=0), dim=-1, eps=1e-6).detach()

        for key in anchors:
            stacked = normalized[key]
            count = stacked.shape[0]
            if count >= 2:
                for index in range(count):
                    if count == 2:
                        others = stacked[1 - index].unsqueeze(0)
                    else:
                        others = torch.cat([stacked[:index], stacked[index + 1 :]], dim=0)
                    prototype = F.normalize(others.mean(dim=0), dim=-1, eps=1e-6).detach()
                    cosine = (stacked[index] * prototype).sum().clamp(min=-1.0, max=1.0)
                    positive_numerator = positive_numerator + (1.0 - cosine).clamp_min(0.0)
                    positive_denominator += 1
            for index in range(count):
                for other_key, other_prototype in prototypes.items():
                    if other_key == key:
                        continue
                    cosine = (stacked[index] * other_prototype).sum().clamp(min=-1.0, max=1.0)
                    negative_numerator = negative_numerator + F.relu(cosine - negative_margin)
                    negative_denominator += 1

    # ---- normalize and combine --------------------------------------------
    activity = activity_numerator / max(activity_denominator, 1.0)
    empty_slots = empty_numerator / max(empty_denominator, 1.0)
    positive = positive_numerator / max(positive_denominator, 1)
    negative = negative_numerator / max(negative_denominator, 1)

    total = (
        weight_map["activity"] * activity
        + weight_map["empty_slots"] * empty_slots
        + weight_map["positive"] * positive
        + weight_map["negative"] * negative
    )

    stats = HeadLossStats(
        groups=num_groups,
        matched_groups=matched_groups,
        skipped_groups=skipped_groups,
        ambiguous_groups=ambiguous_groups,
        truncated_groups=truncated_groups,
        identity_masked_groups=identity_masked_groups,
        activity_terms=int(round(activity_denominator)),
        empty_terms=int(round(empty_denominator)),
        positive_terms=int(positive_denominator),
        negative_terms=int(negative_denominator),
    )
    return HeadLoss(
        total=total,
        activity=activity,
        empty_slots=empty_slots,
        positive=positive,
        negative=negative,
        weights=weight_map,
        stats=stats,
        matching=matching,
    )
