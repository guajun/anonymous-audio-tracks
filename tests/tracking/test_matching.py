"""Assignment and cosine tests: gating, exact cardinality and fallback."""

from __future__ import annotations

import numpy as np
import pytest

from aat.tracking import TrackingError, cosine_similarity, maximum_assignment


def test_pairs_below_threshold_are_disabled():
    scores = np.array([[0.2, 0.4], [0.1, 0.3]])
    assert maximum_assignment(scores, 0.5) == []


def test_exact_matching_prefers_cardinality_over_largest_cosine():
    # Greedy similarity would take (1, 0) = 0.99 and leave row 0 unmatched;
    # maximum cardinality matches both rows at total 1.955.
    scores = np.array(
        [
            [0.985, -np.inf],
            [0.99, 0.97],
        ]
    )
    assert maximum_assignment(scores, 0.7) == [(0, 0), (1, 1)]


def test_exact_matching_prefers_highest_total_cosine_at_equal_cardinality():
    # Both assignments match two rows; total cosine prefers the diagonal.
    scores = np.array(
        [
            [0.95, 1.0],
            [1.0, 0.95],
        ]
    )
    assert maximum_assignment(scores, 0.7) == [(0, 1), (1, 0)]


def test_ties_are_deterministic_and_keep_the_lower_slot():
    scores = np.array([[0.9, 0.9]])
    assert maximum_assignment(scores, 0.7) == [(0, 0)]


def test_greedy_fallback_is_still_gated_and_one_to_one():
    scores = np.array([[0.9, 0.1], [0.1, 0.8]])
    pairs = maximum_assignment(scores, 0.5, max_exact_slots=1)
    assert pairs == [(0, 0), (1, 1)]
    # Forced fallback can lose cardinality; that is the documented limitation.
    colliding = np.array([[0.9, np.nan]])
    with pytest.raises(TrackingError):
        maximum_assignment(colliding, 0.5)
    collide = np.array([[0.9, -np.inf], [0.85, -np.inf]])
    assert maximum_assignment(collide, 0.5, max_exact_slots=1) == [(0, 0)]


def test_wide_problem_is_solved_exactly_via_transpose():
    rng = np.random.default_rng(11)
    scores = rng.uniform(0.0, 1.0, size=(3, 20))
    threshold = 0.5
    best = [0, 0.0]

    def enumerate_assignments(row: int, used: set[int], total: float, count: int) -> None:
        if row == scores.shape[0]:
            if (count, total) > (best[0], best[1]):
                best[0], best[1] = count, total
            return
        enumerate_assignments(row + 1, used, total, count)
        for col in range(scores.shape[1]):
            if col in used or scores[row, col] < threshold:
                continue
            enumerate_assignments(
                row + 1, used | {col}, total + float(scores[row, col]), count + 1
            )

    enumerate_assignments(0, set(), 0.0, 0)
    pairs = maximum_assignment(scores, threshold, max_exact_slots=16)
    total = sum(float(scores[row, col]) for row, col in pairs)
    assert (len(pairs), total) == (best[0], best[1])


def test_empty_shapes_return_no_pairs():
    assert maximum_assignment(np.zeros((0, 3)), 0.5) == []
    assert maximum_assignment(np.zeros((2, 0)), 0.5) == []


def test_cosine_similarity_clips_and_checks_dimensions():
    left = np.array([[1.0, 0.0]])
    right = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    similarity = cosine_similarity(left, right)
    assert similarity.shape == (1, 3)
    np.testing.assert_allclose(similarity[0, :2], [1.0, 0.0])
    assert similarity[0, 2] <= 1.0
    with pytest.raises(TrackingError):
        cosine_similarity(np.zeros((1, 3)), np.zeros((1, 4)))


def test_exact_assignment_matches_brute_force_on_small_random_matrices():
    rng = np.random.default_rng(20260929)
    for _ in range(120):
        rows = int(rng.integers(1, 5))
        cols = int(rng.integers(1, 5))
        scores = rng.uniform(-0.2, 1.0, size=(rows, cols))
        threshold = float(rng.choice([0.3, 0.5, 0.7]))
        keys: list[tuple[int, float]] = []

        def enumerate_assignments(row: int, used: set[int], total: float, count: int) -> None:
            if row == rows:
                keys.append((count, total))
                return
            enumerate_assignments(row + 1, used, total, count)
            for col in range(cols):
                if col in used or scores[row, col] < threshold:
                    continue
                enumerate_assignments(
                    row + 1, used | {col}, total + float(scores[row, col]), count + 1
                )

        enumerate_assignments(0, set(), 0.0, 0)
        best_cardinality = max(keys, key=lambda item: (item[0], item[1]))
        best_total = max(keys, key=lambda item: (item[1], item[0]))

        pairs = maximum_assignment(scores, threshold)
        total = sum(float(scores[row, col]) for row, col in pairs)
        assert len(pairs) == best_cardinality[0]
        assert abs(total - best_cardinality[1]) <= 1e-9
        assert len({row for row, _ in pairs}) == len(pairs)
        assert len({col for _, col in pairs}) == len(pairs)

        total_pairs = maximum_assignment(scores, threshold, objective="total_score")
        total_score = sum(float(scores[row, col]) for row, col in total_pairs)
        assert abs(total_score - best_total[1]) <= 1e-9
        assert len(total_pairs) == best_total[0]
        assert len({row for row, _ in total_pairs}) == len(total_pairs)
        assert len({col for _, col in total_pairs}) == len(total_pairs)
        assert all(scores[row, col] >= threshold for row, col in pairs)


def test_total_score_objective_allows_unmatched_pairs():
    # The cardinality-first online rule takes both non-zero pairs (total 2);
    # whole-song evaluation must be able to keep only the 100-overlap pair.
    scores = np.array([[100.0, 1.0], [1.0, -np.inf]])
    assert maximum_assignment(scores, 0.5) == [(0, 1), (1, 0)]
    assert maximum_assignment(scores, 0.5, objective="total_score") == [(0, 0)]


def test_unknown_objective_is_rejected():
    with pytest.raises(TrackingError):
        maximum_assignment(np.zeros((1, 1)), 0.5, objective="bogus")
