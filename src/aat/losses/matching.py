"""Permutation matching between K candidate slots and S supervised sources.

Slot numbers are not identities, so every training group must first be assigned
one-to-one to its sources.  The matching cost is built from the group's activity
sequence (and optional auxiliary descriptors), never from the predicted
embeddings themselves: letting embeddings pick the matching would let the model
trivially align its own representation with itself.

K is small (protocol default 8), so the optimal injection is found by exhaustive
enumeration in deterministic lexicographic order.  When several sources are
indistinguishable on the whole group the assignment is ambiguous; the helpers
here expose those equivalence classes so the loss can average over them instead
of silently hard-picking one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations, product

import torch
from torch import Tensor
import torch.nn.functional as F

__all__ = [
    "MatchingResult",
    "activity_pair_cost",
    "balanced_bce",
    "descriptor_pair_cost",
    "find_ambiguous_classes",
    "iter_ambiguous_assignments",
    "slot_prototypes",
    "solve_matching",
]

_BCE_EPS = 1e-8
_DEFAULT_MAX_MATCH_PERMUTATIONS = 200_000


def balanced_bce(
    logits: Tensor,
    target: Tensor,
    *,
    valid: Tensor | None = None,
    dim: int = 0,
    keepdim: bool = False,
) -> Tensor:
    """BCE with positive and negative terms averaged separately.

    ``valid`` marks entries that participate.  A term whose side has no weight
    falls back to the other side; with no weighted entries at all the result is
    zero.  This keeps all-silent groups and all-zero predictions finite, and
    stops "predict everything silent" from scoring well when positives exist.
    """

    logits = torch.as_tensor(logits, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    if logits.shape != target.shape:
        raise ValueError(
            f"balanced_bce requires matching shapes, got {tuple(logits.shape)} "
            f"and {tuple(target.shape)}"
        )
    if valid is None:
        weight_mask = torch.ones_like(target)
    else:
        weight_mask = torch.as_tensor(valid, dtype=torch.float32)
        if weight_mask.shape != target.shape:
            raise ValueError(
                f"valid must have shape {tuple(target.shape)}, got {tuple(weight_mask.shape)}"
            )

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive_weight = target * weight_mask
    negative_weight = (1.0 - target) * weight_mask
    positive_mass = positive_weight.sum(dim=dim, keepdim=keepdim)
    negative_mass = negative_weight.sum(dim=dim, keepdim=keepdim)
    positive = (bce * positive_weight).sum(dim=dim, keepdim=keepdim) / positive_mass.clamp_min(
        _BCE_EPS
    )
    negative = (bce * negative_weight).sum(dim=dim, keepdim=keepdim) / negative_mass.clamp_min(
        _BCE_EPS
    )
    has_positive = positive_mass > 0
    has_negative = negative_mass > 0
    return torch.where(
        has_positive & has_negative,
        0.5 * (positive + negative),
        torch.where(has_positive, positive, negative),
    )


def activity_pair_cost(
    logits: Tensor,
    targets: Tensor,
    window_valid: Tensor | None = None,
) -> Tensor:
    """Balanced activity BCE for every ``(slot, source)`` pair.

    ``logits`` is ``[N, K]``, ``targets`` is ``[N, S]`` and the result is
    ``[K, S]``, reduced over the group's valid windows only.
    """

    if logits.dim() != 2:
        raise ValueError(f"logits must be [N, K], got {tuple(logits.shape)}")
    if targets.dim() != 2:
        raise ValueError(f"targets must be [N, S], got {tuple(targets.shape)}")
    windows, slots = logits.shape
    target_windows, sources = targets.shape
    if target_windows != windows:
        raise ValueError(
            f"logits has {windows} windows but targets has {target_windows}"
        )
    pair_logits = logits.unsqueeze(-1).expand(windows, slots, sources)
    pair_targets = targets.unsqueeze(1).expand(windows, slots, sources)
    if window_valid is None:
        valid = torch.ones(windows, dtype=torch.bool, device=logits.device)
    else:
        valid = torch.as_tensor(window_valid, dtype=torch.bool, device=logits.device)
        if valid.shape != (windows,):
            raise ValueError(f"window_valid must be [{windows}], got {tuple(valid.shape)}")
    valid3 = valid[:, None, None].expand(windows, slots, sources).to(torch.float32)
    return balanced_bce(pair_logits, pair_targets, valid=valid3, dim=0)


def slot_prototypes(embeddings: Tensor, window_valid: Tensor | None = None) -> Tensor:
    """Normalized mean identity per slot over the group's valid windows."""

    if embeddings.dim() != 3:
        raise ValueError(f"embeddings must be [N, K, D], got {tuple(embeddings.shape)}")
    windows = embeddings.shape[0]
    if window_valid is None:
        weights = torch.ones(windows, 1, dtype=torch.float32, device=embeddings.device)
    else:
        valid = torch.as_tensor(window_valid, dtype=torch.bool, device=embeddings.device)
        if valid.shape != (windows,):
            raise ValueError(f"window_valid must be [{windows}], got {tuple(valid.shape)}")
        weights = valid[:, None].to(torch.float32)
    mean = (embeddings.to(torch.float32) * weights.unsqueeze(-1)).sum(dim=0)
    mean = mean / weights.sum().clamp_min(1.0)
    return F.normalize(mean, dim=-1, eps=_BCE_EPS)


def descriptor_pair_cost(
    embeddings: Tensor,
    descriptors: Tensor,
    window_valid: Tensor | None = None,
) -> Tensor:
    """``1 - cosine`` between slot prototypes and normalized descriptors."""

    if descriptors.dim() != 2:
        raise ValueError(f"descriptors must be [S, D], got {tuple(descriptors.shape)}")
    prototypes = slot_prototypes(embeddings, window_valid)
    normalized = F.normalize(descriptors.to(torch.float32), dim=-1, eps=_BCE_EPS)
    return 1.0 - prototypes @ normalized.transpose(0, 1)


@dataclass(frozen=True)
class MatchingResult:
    """Deterministic one-to-one source→slot assignment for one training group."""

    slot_for_source: Tensor
    """``[S]`` long tensor: ``slot_for_source[s]`` is the slot matched to source s."""

    unmatched_slots: Tensor
    """``[U]`` long tensor with the remaining capacity, ascending."""

    cost: Tensor
    """``[K, S]`` pair cost used for the assignment."""

    best_cost: Tensor
    """Scalar cost of the chosen assignment (detached)."""

    num_sources: int
    num_slots: int


def solve_matching(
    cost: Tensor,
    *,
    max_permutations: int = _DEFAULT_MAX_MATCH_PERMUTATIONS,
) -> MatchingResult:
    """Find the minimum-cost injection ``source → slot`` by enumeration.

    Ties are broken by lexicographically smallest slot tuple, so the result is
    deterministic (no random hard matching).
    """

    if cost.dim() != 2:
        raise ValueError(f"cost must be [K, S], got {tuple(cost.shape)}")
    if not torch.isfinite(cost).all():
        raise ValueError("cost contains NaN or Inf")
    slots, sources = cost.shape
    if sources > slots:
        raise ValueError(
            f"capacity exceeded: {sources} sources cannot be matched to {slots} slots"
        )

    cost_numpy = cost.detach().to(torch.float64).cpu().numpy()
    if sources == 0:
        assignment = torch.empty(0, dtype=torch.long)
        unmatched = torch.arange(slots, dtype=torch.long)
        return MatchingResult(
            slot_for_source=assignment,
            unmatched_slots=unmatched,
            cost=cost,
            best_cost=cost.sum() * 0.0,
            num_sources=0,
            num_slots=slots,
        )

    total_permutations = math.perm(slots, sources)
    if total_permutations > max_permutations:
        raise ValueError(
            f"{slots} slots and {sources} sources need {total_permutations} candidate "
            f"assignments, above max_permutations={max_permutations}"
        )

    best_assignment: tuple[int, ...] | None = None
    best_value = math.inf
    for candidate in permutations(range(slots), sources):
        value = 0.0
        for source, slot in enumerate(candidate):
            value += float(cost_numpy[slot, source])
        if value < best_value:
            best_value = value
            best_assignment = candidate

    assert best_assignment is not None
    assignment_tensor = torch.tensor(best_assignment, dtype=torch.long)
    used = set(best_assignment)
    unmatched = torch.tensor(
        [slot for slot in range(slots) if slot not in used], dtype=torch.long
    )
    return MatchingResult(
        slot_for_source=assignment_tensor,
        unmatched_slots=unmatched,
        cost=cost,
        best_cost=torch.tensor(best_value, dtype=torch.float32),
        num_sources=sources,
        num_slots=slots,
    )


def _columns_equal(activity: Tensor, valid: Tensor, first: int, second: int, atol: float) -> bool:
    left = activity[valid, first]
    right = activity[valid, second]
    if left.numel() == 0:
        # No valid window carries any information: everything is ambiguous.
        return True
    return bool(torch.allclose(left, right, atol=atol, rtol=0.0))


def find_ambiguous_classes(
    activity: Tensor,
    window_valid: Tensor | None = None,
    descriptors: Tensor | None = None,
    *,
    atol: float = 1e-6,
) -> tuple[tuple[int, ...], ...]:
    """Group sources that are indistinguishable on this whole group.

    Two sources belong to the same class when their activity columns coincide
    on all valid windows and, when descriptors are given, their descriptors are
    also equal within ``atol``.  Only classes with more than one member are
    returned, sorted by their smallest source index.
    """

    if activity.dim() != 2:
        raise ValueError(f"activity must be [N, S], got {tuple(activity.shape)}")
    windows, sources = activity.shape
    activity = activity.to(torch.float32)
    if window_valid is None:
        valid = torch.ones(windows, dtype=torch.bool)
    else:
        valid = torch.as_tensor(window_valid, dtype=torch.bool)
        if valid.shape != (windows,):
            raise ValueError(f"window_valid must be [{windows}], got {tuple(valid.shape)}")
    if descriptors is not None:
        descriptors = descriptors.to(torch.float32)
        if descriptors.dim() != 2 or descriptors.shape[0] != sources:
            raise ValueError(
                f"descriptors must be [S, D] with S={sources}, "
                f"got {tuple(descriptors.shape)}"
            )

    parent = list(range(sources))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[max(root_first, root_second)] = min(root_first, root_second)

    for first in range(sources):
        for second in range(first + 1, sources):
            if not _columns_equal(activity, valid, first, second, atol):
                continue
            if descriptors is not None and not bool(
                torch.allclose(descriptors[first], descriptors[second], atol=atol, rtol=0.0)
            ):
                continue
            union(first, second)

    groups: dict[int, list[int]] = {}
    for source in range(sources):
        groups.setdefault(find(source), []).append(source)
    classes = [tuple(members) for members in groups.values() if len(members) > 1]
    classes.sort(key=lambda members: members[0])
    return tuple(classes)


def iter_ambiguous_assignments(
    slot_for_source: Tensor,
    classes: tuple[tuple[int, ...], ...],
    *,
    max_assignments: int = 64,
) -> tuple[list[Tensor], int, bool]:
    """Enumerate optimal assignments obtained by permuting slot images.

    Returns ``(assignments, total_count, truncated)``.  The first assignment is
    the base one; the order is deterministic.  When more than ``max_assignments``
    permutations exist the list is deterministically truncated and ``truncated``
    is ``True``.  All supported loss terms are symmetric on indistinguishable
    classes, so the truncation only affects adversarial weightings, not the
    documented objective (see ``docs/MODEL_HEAD.md`` §4).
    """

    slot_for_source = torch.as_tensor(slot_for_source, dtype=torch.long).reshape(-1)
    if not classes:
        return [slot_for_source], 1, False

    images = [[int(slot_for_source[source]) for source in members] for members in classes]
    per_class = [list(permutations(image)) for image in images]
    total = 1
    for candidates in per_class:
        total *= len(candidates)

    assignments: list[Tensor] = []
    truncated = total > max_assignments
    for combination in product(*per_class):
        assignment = slot_for_source.clone()
        for members, image in zip(classes, combination):
            assignment[list(members)] = torch.tensor(image, dtype=torch.long)
        assignments.append(assignment)
        if len(assignments) >= max_assignments:
            break
    return assignments, total, truncated
