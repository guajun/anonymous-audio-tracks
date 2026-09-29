"""Permutation-invariant training loss for the multi-source E/P head.

The loss operates on one training group: several center-time windows of the
same composition with S supervised sources.  Its pieces mirror the issue #7
checklist and are all exposed separately:

* matched center-activity BCE (balanced over positive/negative entries),
* same-source cross-window contrastive consistency,
* hinge negatives between different sources,
* empty-capacity suppression for unmatched slots.

Slot numbers are not identities, so the activity loss is computed after the
group-level one-to-one matching in :mod:`aat.losses.matching`; ambiguous
matches are averaged over the equivalence-class permutations instead of being
hard-picked at random.  Ground-truth source order therefore never changes the
total loss (see ``docs/MODEL_HEAD.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .matching import (
    MatchingResult,
    activity_pair_cost,
    descriptor_pair_cost,
    find_ambiguous_classes,
    iter_ambiguous_assignments,
    solve_matching,
)

__all__ = [
    "GroupTargets",
    "HeadLoss",
    "LossOutput",
    "PermutationInvariantHeadLoss",
]

_EPS = 1e-8


@dataclass
class GroupTargets:
    """Supervision for one same-composition training group.

    ``activity`` is ``[N, S]`` with values in ``[0, 1]`` (hard 0/1 or soft),
    ``window_valid`` is ``[N]`` (default all True) and optionally
    ``source_descriptors`` is ``[S, D]`` auxiliary identity information used by
    matching and identity terms.
    """

    activity: Tensor
    window_valid: Tensor | None = None
    source_descriptors: Tensor | None = None

    def __post_init__(self) -> None:
        activity = torch.as_tensor(self.activity, dtype=torch.float32)
        if activity.dim() != 2:
            raise ValueError(f"activity must be [N, S], got {tuple(activity.shape)}")
        if not torch.isfinite(activity).all():
            raise ValueError("activity contains NaN or Inf")
        if ((activity < 0.0) | (activity > 1.0)).any():
            raise ValueError("activity must stay inside [0, 1]")
        self.activity = activity

        windows, sources = activity.shape
        if self.window_valid is None:
            self.window_valid = torch.ones(windows, dtype=torch.bool)
        else:
            window_valid = torch.as_tensor(self.window_valid, dtype=torch.bool)
            if window_valid.shape != (windows,):
                raise ValueError(
                    f"window_valid must be [{windows}], got {tuple(window_valid.shape)}"
                )
            self.window_valid = window_valid

        if self.source_descriptors is not None:
            descriptors = torch.as_tensor(self.source_descriptors, dtype=torch.float32)
            if descriptors.dim() != 2 or descriptors.shape[0] != sources:
                raise ValueError(
                    f"source_descriptors must be [S, D] with S={sources}, "
                    f"got {tuple(descriptors.shape)}"
                )
            if not torch.isfinite(descriptors).all():
                raise ValueError("source_descriptors contains NaN or Inf")
            self.source_descriptors = descriptors

    @property
    def num_windows(self) -> int:
        return int(self.activity.shape[0])

    @property
    def num_sources(self) -> int:
        return int(self.activity.shape[1])


@dataclass
class LossOutput:
    """Loss terms plus the matching used to compute them."""

    total: Tensor
    activity: Tensor
    same_source: Tensor
    diff_source: Tensor
    empty_slots: Tensor
    matching: MatchingResult
    ambiguous_assignments: int
    ambiguity_truncated: bool

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "activity": float(self.activity.detach()),
            "same_source": float(self.same_source.detach()),
            "diff_source": float(self.diff_source.detach()),
            "empty_slots": float(self.empty_slots.detach()),
        }


class PermutationInvariantHeadLoss(nn.Module):
    """Group-level training loss; see ``docs/MODEL_HEAD.md`` for the formulas."""

    def __init__(
        self,
        *,
        activity_weight: float = 1.0,
        same_source_weight: float = 1.0,
        diff_source_weight: float = 1.0,
        empty_slot_weight: float = 1.0,
        descriptor_weight: float = 1.0,
        negative_margin: float = 0.25,
        ambiguity_atol: float = 1e-6,
        max_ambiguous_assignments: int = 64,
        max_match_permutations: int = 200_000,
    ) -> None:
        super().__init__()
        if negative_margin < 0.0:
            raise ValueError("negative_margin must be >= 0")
        if max_ambiguous_assignments < 1:
            raise ValueError("max_ambiguous_assignments must be >= 1")
        self.activity_weight = float(activity_weight)
        self.same_source_weight = float(same_source_weight)
        self.diff_source_weight = float(diff_source_weight)
        self.empty_slot_weight = float(empty_slot_weight)
        self.descriptor_weight = float(descriptor_weight)
        self.negative_margin = float(negative_margin)
        self.ambiguity_atol = float(ambiguity_atol)
        self.max_ambiguous_assignments = int(max_ambiguous_assignments)
        self.max_match_permutations = int(max_match_permutations)

    def forward(
        self,
        embeddings: Tensor,
        activity_logits: Tensor,
        targets: GroupTargets,
    ) -> LossOutput:
        if embeddings.dim() != 3:
            raise ValueError(f"embeddings must be [N, K, D], got {tuple(embeddings.shape)}")
        if activity_logits.dim() != 2:
            raise ValueError(
                f"activity_logits must be [N, K], got {tuple(activity_logits.shape)}"
            )
        windows, slots, _ = embeddings.shape
        if activity_logits.shape != (windows, slots):
            raise ValueError(
                f"activity_logits must be [{windows}, {slots}], "
                f"got {tuple(activity_logits.shape)}"
            )
        if targets.num_windows != windows:
            raise ValueError(
                f"group targets have {targets.num_windows} windows but predictions "
                f"have {windows}"
            )
        if targets.num_sources > slots:
            raise ValueError(
                f"capacity exceeded: {targets.num_sources} sources cannot be matched "
                f"to {slots} slots"
            )
        if not torch.isfinite(embeddings).all():
            raise ValueError("embeddings contain NaN or Inf")
        if not torch.isfinite(activity_logits).all():
            raise ValueError("activity_logits contain NaN or Inf")

        normalized = F.normalize(embeddings.to(torch.float32), dim=-1, eps=_EPS)
        logits = activity_logits.to(torch.float32)

        pair_cost = activity_pair_cost(logits, targets.activity, targets.window_valid)
        if targets.source_descriptors is not None:
            pair_cost = pair_cost + self.descriptor_weight * descriptor_pair_cost(
                normalized, targets.source_descriptors, targets.window_valid
            )

        matching = solve_matching(pair_cost, max_permutations=self.max_match_permutations)
        classes = find_ambiguous_classes(
            targets.activity,
            targets.window_valid,
            targets.source_descriptors,
            atol=self.ambiguity_atol,
        )
        assignments, total_assignments, truncated = iter_ambiguous_assignments(
            matching.slot_for_source,
            classes,
            max_assignments=self.max_ambiguous_assignments,
        )

        terms = [
            self._assignment_terms(normalized, logits, targets, assignment, pair_cost)
            for assignment in assignments
        ]
        activity = torch.stack([term[0] for term in terms]).mean()
        same_source = torch.stack([term[1] for term in terms]).mean()
        diff_source = torch.stack([term[2] for term in terms]).mean()
        empty_slots = torch.stack([term[3] for term in terms]).mean()

        total = (
            self.activity_weight * activity
            + self.same_source_weight * same_source
            + self.diff_source_weight * diff_source
            + self.empty_slot_weight * empty_slots
        )
        return LossOutput(
            total=total,
            activity=activity,
            same_source=same_source,
            diff_source=diff_source,
            empty_slots=empty_slots,
            matching=matching,
            ambiguous_assignments=total_assignments,
            ambiguity_truncated=truncated,
        )

    def _assignment_terms(
        self,
        embeddings: Tensor,
        logits: Tensor,
        targets: GroupTargets,
        assignment: Tensor,
        pair_cost: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Terms for one concrete source→slot assignment."""

        windows, slots, _ = embeddings.shape
        sources = targets.num_sources
        valid = targets.window_valid
        num_valid = int(valid.sum().item())
        zero = logits.sum() * 0.0

        if sources == 0:
            empty = self._empty_slots_term(logits, valid, list(range(slots)), num_valid)
            return zero, zero, zero, empty

        assignment = assignment.to(torch.long)
        activity = pair_cost[assignment, torch.arange(sources, device=pair_cost.device)].mean()

        matched = embeddings[:, assignment, :]  # [N, S, D]
        prototypes = self._prototypes(matched, valid)

        if num_valid == 0:
            same_source = zero
            diff_source = zero
        else:
            valid_matched = matched[valid]  # [Nv, S, D]
            similarities = (valid_matched * prototypes.unsqueeze(0)).sum(dim=-1)  # [Nv, S]
            same_source = (1.0 - similarities).mean()
            if sources >= 2:
                cross = torch.einsum("nid,jd->nij", valid_matched, prototypes)  # [Nv, S, S]
                off_diagonal = 1.0 - torch.eye(
                    sources, device=cross.device, dtype=cross.dtype
                )
                hinge = torch.relu(cross - self.negative_margin) * off_diagonal
                diff_source = hinge.sum() / (num_valid * sources * (sources - 1))
            else:
                diff_source = zero

        unmatched = [slot for slot in range(slots) if slot not in set(assignment.tolist())]
        empty = self._empty_slots_term(logits, valid, unmatched, num_valid)
        return activity, same_source, diff_source, empty

    @staticmethod
    def _prototypes(embeddings: Tensor, window_valid: Tensor) -> Tensor:
        weights = window_valid.to(torch.float32).unsqueeze(-1)  # [N, 1]
        means = (embeddings * weights.unsqueeze(-1)).sum(dim=0) / weights.sum().clamp_min(1.0)
        return F.normalize(means, dim=-1, eps=_EPS)

    @staticmethod
    def _empty_slots_term(
        logits: Tensor,
        valid: Tensor,
        unmatched: list[int],
        num_valid: int,
    ) -> Tensor:
        if not unmatched:
            return logits.sum() * 0.0
        selected = logits[:, unmatched]
        if num_valid == 0:
            return selected.sum() * 0.0
        bce = F.binary_cross_entropy_with_logits(
            selected, torch.zeros_like(selected), reduction="none"
        )
        return (bce * valid.to(bce.dtype).unsqueeze(-1)).sum() / (
            num_valid * len(unmatched)
        )


#: Short alias used in tests and future training code.
HeadLoss = PermutationInvariantHeadLoss
