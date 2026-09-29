"""Permutation-invariant matching and losses for the multi-source E/P head.

See ``docs/MODEL_HEAD.md`` for the matching cost, ambiguity handling and the
individual loss terms.  This package never loads audio, weights or datasets.
"""

from __future__ import annotations

from .head_loss import (
    GroupTargets,
    HeadLoss,
    LossOutput,
    PermutationInvariantHeadLoss,
)
from .matching import (
    MatchingResult,
    activity_pair_cost,
    balanced_bce,
    descriptor_pair_cost,
    find_ambiguous_classes,
    iter_ambiguous_assignments,
    slot_prototypes,
    solve_matching,
)

__all__ = [
    "GroupTargets",
    "HeadLoss",
    "LossOutput",
    "MatchingResult",
    "PermutationInvariantHeadLoss",
    "activity_pair_cost",
    "balanced_bce",
    "descriptor_pair_cost",
    "find_ambiguous_classes",
    "iter_ambiguous_assignments",
    "slot_prototypes",
    "solve_matching",
]
