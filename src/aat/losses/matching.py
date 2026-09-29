"""Exact group-level one-to-one source/slot matching.

A training group contains several center windows of one composition; source
columns are fixed across windows.  Matching builds one cost matrix from the
whole group activity sequence and solves the injection ``sources -> slots``
exactly (bit-mask DP).  There is deliberately no per-frame re-matching, no
greedy fallback and no online-tracker cardinality heuristic: those would hide
identity errors the loss is supposed to expose.

All optimal assignments are enumerated (up to a configurable cap).  Multiple
optima mean the group is ambiguous: the loss layer uses symmetric averaging for
assignment-symmetric components and masks identity supervision instead of
picking an arbitrary fixed column order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "MatchingError",
    "OptimalAssignments",
    "GroupMatching",
    "MatchingResult",
    "enumerate_optimal_assignments",
    "match_sources",
]

_INF = float("inf")


class MatchingError(ValueError):
    """Exact matching is impossible or unsupported for this group."""


@dataclass(frozen=True)
class OptimalAssignments:
    """All optimal assignments of one cost matrix (capped enumeration).

    ``assignments`` is a tuple of tuples: element ``i`` is the slot used by the
    ``i``-th row of the cost matrix passed to this function, i.e. a *local*
    index into ``[0, K)``.  :func:`match_sources` remaps these to the original
    batch slot indices when it builds :class:`GroupMatching`.
    ``num_optimal`` is the true count saturated at ``len(assignments) + 1`` when
    more than the cap exist, and ``truncated`` says whether enumeration stopped
    at the cap.
    """

    assignments: tuple[tuple[int, ...], ...]
    num_optimal: int
    truncated: bool
    optimal_cost: float

    @property
    def unique(self) -> bool:
        return self.num_optimal == 1 and not self.truncated


def enumerate_optimal_assignments(
    cost: Any,
    *,
    max_optimal: int = 64,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> OptimalAssignments:
    """Enumerate every optimal source->slot injection for a 2-D cost matrix.

    ``cost`` has shape ``[S, K]`` with ``S <= K``; ``+inf`` entries forbid an
    edge.  Costs are exact floating-point sums, and ``atol/rtol`` define how
    close an edge must be to the optimum to count as part of an optimal
    assignment.  The enumeration order is lexicographic by slot index, so
    repeated runs on the same matrix return the same first assignment.
    """

    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2-D [S, K]; got shape {list(cost.shape)}")
    if max_optimal < 1:
        raise ValueError("max_optimal must be >= 1")
    if np.isnan(cost).any() or np.isneginf(cost).any():
        raise ValueError("cost must not contain NaN or -inf")
    num_sources, num_slots = cost.shape
    if num_sources > num_slots:
        raise MatchingError(
            f"{num_sources} sources exceed {num_slots} available slots; "
            "refusing to truncate sources"
        )
    if num_sources == 0:
        return OptimalAssignments(assignments=((),), num_optimal=1, truncated=False, optimal_cost=0.0)

    full_mask = (1 << num_slots) - 1

    # dp[i, mask] = min cost assigning sources i..S-1 to the free slots in mask
    # (mask is a subset of all slots; the answer is dp[0, full_mask]).
    dp = np.full((num_sources + 1, 1 << num_slots), _INF, dtype=np.float64)
    dp[num_sources, :] = 0.0
    for i in range(num_sources - 1, -1, -1):
        for mask in range(1 << num_slots):
            best = _INF
            bits = mask
            while bits:
                bit = bits & -bits
                slot = bit.bit_length() - 1
                candidate = cost[i, slot] + dp[i + 1, mask ^ bit]
                if candidate < best:
                    best = candidate
                bits ^= bit
            dp[i, mask] = best

    if not np.isfinite(dp[0, full_mask]):
        raise MatchingError("no feasible source->slot assignment (all edges forbidden)")

    # Saturating count of optimal paths, then bounded DFS enumeration.
    cap = max_optimal + 1
    counts = np.zeros((num_sources + 1, 1 << num_slots), dtype=np.int64)
    counts[num_sources, :] = 1

    def edge_is_optimal(i: int, mask: int, slot: int) -> bool:
        bit = 1 << slot
        return bool(
            cost[i, slot] + dp[i + 1, mask ^ bit]
            <= dp[i, mask] + atol + rtol * abs(dp[i, mask])
        )

    for i in range(num_sources - 1, -1, -1):
        for mask in range(1 << num_slots):
            total = 0
            bits = mask
            while bits:
                bit = bits & -bits
                slot = bit.bit_length() - 1
                if edge_is_optimal(i, mask, slot):
                    total += int(counts[i + 1, mask ^ bit])
                    if total >= cap:
                        total = cap
                        break
                bits ^= bit
            counts[i, mask] = total

    found: list[tuple[int, ...]] = []
    accumulator: list[int] = []

    def visit(i: int, mask: int) -> None:
        if len(found) >= max_optimal:
            return
        if i == num_sources:
            found.append(tuple(accumulator))
            return
        bits = mask
        while bits:
            bit = bits & -bits
            slot = bit.bit_length() - 1
            bits ^= bit
            if not edge_is_optimal(i, mask, slot):
                continue
            accumulator.append(slot)
            visit(i + 1, mask ^ bit)
            accumulator.pop()
            if len(found) >= max_optimal:
                return

    visit(0, full_mask)
    num_optimal = int(counts[0, full_mask])
    return OptimalAssignments(
        assignments=tuple(found),
        num_optimal=num_optimal,
        truncated=num_optimal > len(found),
        optimal_cost=float(dp[0, full_mask]),
    )


@dataclass(frozen=True)
class GroupMatching:
    """Matching result of one training group, in original batch indices.

    ``source_indices[j]`` is the original source column matched to
    ``optimal.assignments[j]``, and every entry of every assignment tuple is an
    **original slot index**: :func:`match_sources` already remapped the solver's
    local indices.  Consumers must use ``assignment[j]`` directly and must not
    index ``slot_indices`` with it again.
    """

    source_indices: tuple[int, ...]
    slot_indices: tuple[int, ...]
    optimal: OptimalAssignments

    @property
    def ambiguous(self) -> bool:
        return self.optimal.num_optimal > 1 and not self.optimal.truncated

    @property
    def truncated(self) -> bool:
        return self.optimal.truncated

    @property
    def identity_masked(self) -> bool:
        """Whether identity supervision is masked for this group.

        Both complete-enumeration ambiguity and truncated enumeration leave the
        identity assignment unsupported, so the arranged loss masks identity
        terms in either case (see ``docs/MODEL_HEAD.md``).
        """

        return self.optimal.truncated or self.optimal.num_optimal > 1


@dataclass(frozen=True)
class MatchingResult:
    """Per-group matching results in batch order."""

    groups: tuple[GroupMatching, ...]


def _to_numpy(array: Any, *, dtype: Any = np.float64) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=dtype)


def match_sources(
    cost: Any,
    source_valid: Any | None = None,
    slot_valid: Any | None = None,
    *,
    max_optimal: int = 64,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> MatchingResult:
    """Match every group's valid sources to available slots exactly.

    ``cost`` is ``[G, S, K]`` (already group-averaged and detached).  Invalid
    sources / unavailable slots are excluded per group before solving, so
    padding cannot change the result.  Raises :class:`MatchingError` when a
    group has more valid sources than available slots.

    The returned :class:`GroupMatching` assignments are expressed in original
    batch slot indices; the solver's local indices never escape this function.
    """

    cost = _to_numpy(cost)
    if cost.ndim != 3:
        raise ValueError(f"cost must be 3-D [G, S, K]; got shape {list(cost.shape)}")
    num_groups, num_sources, num_slots = cost.shape

    if source_valid is None:
        source_valid = np.ones((num_groups, num_sources), dtype=bool)
    else:
        source_valid = _to_numpy(source_valid, dtype=bool)
        if source_valid.shape != (num_groups, num_sources):
            raise ValueError("source_valid must have shape [G, S]")
    if slot_valid is None:
        slot_valid = np.ones((num_groups, num_slots), dtype=bool)
    else:
        slot_valid = _to_numpy(slot_valid, dtype=bool)
        if slot_valid.shape != (num_groups, num_slots):
            raise ValueError("slot_valid must have shape [G, K]")

    groups: list[GroupMatching] = []
    for g in range(num_groups):
        sources = np.nonzero(source_valid[g])[0]
        candidate_slots = np.nonzero(slot_valid[g])[0]
        if sources.size == 0:
            # No sources: every requested slot is free capacity (all of them are
            # "empty slots" for the loss), nothing to assign.
            slots = candidate_slots
            submatrix = np.zeros((0, slots.size), dtype=np.float64)
        else:
            submatrix = cost[g][np.ix_(sources, candidate_slots)] if candidate_slots.size else np.zeros(
                (sources.size, 0)
            )
            usable = np.isfinite(submatrix).any(axis=0) if candidate_slots.size else np.zeros(0, dtype=bool)
            slots = candidate_slots[usable]
            submatrix = submatrix[:, usable]

        if sources.size > slots.size:
            raise MatchingError(
                f"group {g}: {sources.size} valid sources exceed {slots.size} available slots; "
                "source count must not exceed K"
            )
        optimal = enumerate_optimal_assignments(
            submatrix, max_optimal=max_optimal, atol=atol, rtol=rtol
        )
        mapped = tuple(
            tuple(int(slots[local]) for local in assignment) for assignment in optimal.assignments
        )
        groups.append(
            GroupMatching(
                source_indices=tuple(int(index) for index in sources),
                slot_indices=tuple(int(index) for index in slots),
                optimal=OptimalAssignments(
                    assignments=mapped,
                    num_optimal=optimal.num_optimal,
                    truncated=optimal.truncated,
                    optimal_cost=optimal.optimal_cost,
                ),
            )
        )
    return MatchingResult(groups=tuple(groups))
