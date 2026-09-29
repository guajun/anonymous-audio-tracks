"""Permutation-invariant matching and losses for the K-query E/P head (#7).

Public entry points:

* :func:`aat.losses.head_loss` - group-level arranged loss with separately
  exposed ``activity`` / ``empty_slots`` / ``positive`` / ``negative`` terms;
* :func:`aat.losses.match_sources` / :func:`enumerate_optimal_assignments` -
  exact one-to-one group matching with explicit ambiguity reporting.

See ``docs/MODEL_HEAD.md`` for every formula, mask and weight.
"""

from __future__ import annotations

from .head_loss import HeadLoss, HeadLossStats, LossWeights, head_loss
from .matching import (
    GroupMatching,
    MatchingError,
    MatchingResult,
    OptimalAssignments,
    enumerate_optimal_assignments,
    match_sources,
)

__all__ = [
    "GroupMatching",
    "HeadLoss",
    "HeadLossStats",
    "LossWeights",
    "MatchingError",
    "MatchingResult",
    "OptimalAssignments",
    "enumerate_optimal_assignments",
    "head_loss",
    "match_sources",
]
