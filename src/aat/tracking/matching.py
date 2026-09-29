"""Cosine gating and exact one-to-one assignment for cross-window association.

Rule (from issue #9): disable pairs outside the threshold first, then choose a
one-to-one assignment that maximises the number of feasible matches, and break
ties by the largest total cosine similarity.  K is small (default 8), so when
the slot count is at most ``max_exact_slots`` the problem is solved exactly by a
bitmask dynamic program; larger searches fall back to a deterministic greedy
assignment (documented limitation).

``-inf`` scores are allowed and mean "infeasible"; NaN and ``+inf`` are
rejected.  Two objectives are available:

``cardinality_then_score`` (default)
    Maximise the number of matched pairs, then the total score.  This is the
    online tracker rule.
``total_score``
    Maximise the total score, allowing pairs to stay unmatched; ties prefer
    more pairs.  Used by whole-song evaluation because it is equivalent to
    maximising micro-F1 under a fixed source/track mapping.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import TrackingError

#: Guard for the exact bitmask search: rows * 2**cols must stay below this.
MAX_EXACT_STATES = 2_000_000

#: Matching objectives accepted by :func:`maximum_assignment`.
OBJECTIVE_CARDINALITY_THEN_SCORE = "cardinality_then_score"
OBJECTIVE_TOTAL_SCORE = "total_score"
OBJECTIVES = (OBJECTIVE_CARDINALITY_THEN_SCORE, OBJECTIVE_TOTAL_SCORE)


def _as_2d(value: Any, path: str, *, allow_neg_inf: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise TrackingError(f"{path}: expected a 2-D array, got shape {tuple(array.shape)}")
    if np.any(np.isnan(array)) or np.any(np.isposinf(array)):
        raise TrackingError(f"{path}: contains NaN or +Inf")
    if not allow_neg_inf and np.any(np.isneginf(array)):
        raise TrackingError(f"{path}: contains -Inf")
    return array


def cosine_similarity(prototypes: Any, candidates: Any) -> np.ndarray:
    """Cosine similarity between ``(T, D)`` prototypes and ``(C, D)`` candidates.

    Returns a ``(T, C)`` float64 matrix clipped to ``[-1, 1]`` (floating-point
    error can otherwise produce 1 + 1e-16).
    """

    left = _as_2d(prototypes, "prototypes")
    right = _as_2d(candidates, "candidates")
    if left.shape[1] != right.shape[1]:
        raise TrackingError(
            "prototypes/candidates: embedding dimensions differ "
            f"({left.shape[1]} vs {right.shape[1]})"
        )
    if left.shape[0] == 0 or right.shape[0] == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)
    similarity = left @ right.T
    return np.clip(similarity, -1.0, 1.0)


def maximum_assignment(
    scores: Any,
    threshold: float,
    *,
    objective: str = OBJECTIVE_CARDINALITY_THEN_SCORE,
    max_exact_slots: int = 16,
    max_exact_rows: int = 256,
    max_exact_states: int = MAX_EXACT_STATES,
) -> list[tuple[int, int]]:
    """Gated one-to-one assignment under the requested objective.

    Parameters
    ----------
    scores:
        ``(rows, cols)`` matrix; a pair is feasible when ``score >= threshold``.
        Use ``-inf`` to mark a pair infeasible regardless of the threshold.
    threshold:
        Gate applied before assignment.  Pairs below it are disabled.
    objective:
        ``"cardinality_then_score"`` maximises (match count, total score);
        ``"total_score"`` maximises (total score, match count) and allows
        rows/columns to stay unmatched.
    max_exact_slots, max_exact_rows, max_exact_states:
        Bounds for the exact bitmask search.  Outside them the deterministic
        greedy fallback is used; the fallback is still one-to-one and gated but
        does not guarantee the optimum.

    Returns
    -------
    list[tuple[int, int]]
        Sorted ``(row, col)`` pairs; every row and column appears at most once.
    """

    score_matrix = _as_2d(scores, "scores", allow_neg_inf=True)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TrackingError(f"threshold: expected a number, got {type(threshold).__name__}")
    gate = float(threshold)
    if not math.isfinite(gate):
        raise TrackingError(f"threshold: must be finite, got {threshold!r}")
    if objective not in OBJECTIVES:
        raise TrackingError(
            f"objective: expected one of {list(OBJECTIVES)}, got {objective!r}"
        )
    if isinstance(max_exact_slots, bool) or not isinstance(max_exact_slots, int):
        raise TrackingError("max_exact_slots: expected an integer")
    if max_exact_slots < 0:
        raise TrackingError("max_exact_slots: must be >= 0")

    rows, cols = score_matrix.shape
    if rows == 0 or cols == 0:
        return []
    if _exact_possible(rows, cols, max_exact_slots, max_exact_rows, max_exact_states):
        return _exact_assignment(score_matrix, gate, objective)
    # Assignment is symmetric: a tall, narrow problem can still be solved
    # exactly with the bitmask over the small side.
    if _exact_possible(cols, rows, max_exact_slots, max_exact_rows, max_exact_states):
        transposed = _exact_assignment(
            np.ascontiguousarray(score_matrix.T), gate, objective
        )
        return sorted((col, row) for row, col in transposed)
    return _greedy_assignment(score_matrix, gate)


def _exact_possible(
    rows: int,
    cols: int,
    max_exact_slots: int,
    max_exact_rows: int,
    max_exact_states: int,
) -> bool:
    return (
        cols <= max_exact_slots
        and rows <= max_exact_rows
        and rows * (1 << cols) <= max_exact_states
    )


def _exact_assignment(
    scores: np.ndarray, threshold: float, objective: str
) -> list[tuple[int, int]]:
    rows, cols = scores.shape
    # node = (count, total_score, previous_mask, slot, created_row)
    initial = (0, 0.0, None, None, -1)
    best: dict[int, tuple] = {0: initial}
    # history[mask] = [(row_created, node), ...] so reconstruction never follows
    # a parent pointer that was later overwritten by a different DP path.
    history: dict[int, list[tuple[int, tuple]]] = {0: [(0, initial)]}

    for row in range(rows):
        updated = dict(best)
        row_scores = scores[row]
        for mask, node in best.items():
            count, total = node[0], node[1]
            for col in range(cols):
                if mask & (1 << col):
                    continue
                value = float(row_scores[col])
                if value < threshold:
                    continue
                new_mask = mask | (1 << col)
                candidate = (count + 1, total + value, mask, col, row)
                existing = updated.get(new_mask)
                if existing is None or _objective_key(
                    candidate, objective
                ) > _objective_key(existing, objective):
                    updated[new_mask] = candidate
        for mask, node in updated.items():
            if best.get(mask) is not node:
                history.setdefault(mask, []).append((row, node))
        best = updated

    final_mask: int | None = None
    final_node: tuple | None = None
    for mask, node in best.items():
        if final_node is None or _objective_key(node, objective) > _objective_key(
            final_node, objective
        ):
            final_mask, final_node = mask, node
    if final_node is None or final_node[0] == 0 or final_mask is None:
        return []

    pairs: list[tuple[int, int]] = []
    mask = final_mask
    row = rows - 1
    while row >= 0:
        entries = history.get(mask)
        if not entries:
            break
        node = None
        for entry_row, entry_node in reversed(entries):
            if entry_row <= row:
                node = entry_node
                break
        if node is None or node[4] < 0:
            break
        pairs.append((int(node[4]), int(node[3])))
        mask = int(node[2])
        row = int(node[4]) - 1
    pairs.sort()
    return pairs


def _objective_key(node: tuple, objective: str) -> tuple[float, int]:
    if objective == OBJECTIVE_TOTAL_SCORE:
        return (node[1], node[0])
    return (node[0], node[1])


def _greedy_assignment(scores: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    rows, cols = scores.shape
    feasible: list[tuple[float, int, int]] = []
    for row in range(rows):
        for col in range(cols):
            value = float(scores[row, col])
            if value >= threshold:
                feasible.append((-value, row, col))
    feasible.sort()
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, row, col in feasible:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        pairs.append((row, col))
    pairs.sort()
    return pairs


__all__ = [
    "MAX_EXACT_STATES",
    "OBJECTIVES",
    "OBJECTIVE_CARDINALITY_THEN_SCORE",
    "OBJECTIVE_TOTAL_SCORE",
    "cosine_similarity",
    "maximum_assignment",
]
