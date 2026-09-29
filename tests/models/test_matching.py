"""Matching tests: exactness, ties, truncation, padding, rejection."""

from __future__ import annotations

from itertools import permutations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from aat.losses.matching import (
    MatchingError,
    enumerate_optimal_assignments,
    match_sources,
)


def _brute_force(cost: np.ndarray) -> tuple[float, int]:
    sources, slots = cost.shape
    best = float("inf")
    count = 0
    for assignment in permutations(range(slots), sources):
        total = float(sum(cost[i, assignment[i]] for i in range(sources)))
        if total < best - 1e-9:
            best = total
            count = 1
        elif abs(total - best) <= 1e-9:
            count += 1
    return best, count


def test_matches_brute_force_on_random_matrices():
    rng = np.random.default_rng(7)
    for _ in range(40):
        sources = int(rng.integers(1, 5))
        slots = int(rng.integers(sources, 7))
        cost = rng.normal(size=(sources, slots))
        result = enumerate_optimal_assignments(cost, max_optimal=10_000)
        best, count = _brute_force(cost)
        assert np.isclose(result.optimal_cost, best)
        assert result.num_optimal == count
        assert not result.truncated
        assert len(result.assignments) == count


def test_unique_optimum_is_reported_as_unique():
    cost = np.array([[0.0, 5.0, 5.0], [5.0, 0.0, 5.0]])
    result = enumerate_optimal_assignments(cost)
    assert result.unique
    assert result.assignments == ((0, 1),)
    assert result.num_optimal == 1


def test_identical_cost_rows_are_ambiguous_and_stable():
    cost = np.zeros((2, 3))
    result = enumerate_optimal_assignments(cost, max_optimal=100)
    assert result.num_optimal == 6  # P(3, 2)
    assert not result.truncated
    assert len(result.assignments) == 6
    assert result.assignments[0] == (0, 1)  # lexicographic, deterministic
    assert enumerate_optimal_assignments(cost, max_optimal=100).assignments == result.assignments


def test_truncated_enumeration_reports_cap():
    cost = np.zeros((2, 3))
    result = enumerate_optimal_assignments(cost, max_optimal=4)
    assert result.truncated
    assert len(result.assignments) == 4
    assert result.num_optimal == 5  # saturated at max_optimal + 1


def test_forbidden_edges_and_infeasible_matrix():
    cost = np.array([[1.0, np.inf, 2.0], [2.0, np.inf, 1.0]])
    result = enumerate_optimal_assignments(cost)
    assert result.unique
    assert result.assignments == ((0, 2),)
    assert np.isclose(result.optimal_cost, 2.0)

    with pytest.raises(MatchingError):
        enumerate_optimal_assignments(np.full((2, 2), np.inf))

    with pytest.raises(MatchingError):
        enumerate_optimal_assignments(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        enumerate_optimal_assignments(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        enumerate_optimal_assignments(np.array([[float("nan")]]))


def test_match_sources_excludes_padding_and_uses_original_indices():
    rng = np.random.default_rng(3)
    cost = rng.normal(size=(1, 3, 4))
    source_valid = np.array([[True, False, True]])
    slot_valid = np.array([[True, True, False, True]])
    result = match_sources(cost, source_valid, slot_valid)
    group = result.groups[0]
    assert group.source_indices == (0, 2)
    assert group.slot_indices == (0, 1, 3)
    sub = cost[0][np.ix_((0, 2), (0, 1, 3))]
    assert np.isclose(group.optimal.optimal_cost, _brute_force(sub)[0])
    # Assignments are expressed in original slot indices, not positions inside
    # ``slot_indices``.
    for assignment in group.optimal.assignments:
        assert all(entry in group.slot_indices for entry in assignment)
        assert len(set(assignment)) == len(assignment)
        assert all(
            np.isfinite(cost[0, source, slot])
            for source, slot in zip(group.source_indices, assignment)
        )

    # Invalid source rows may contain anything (NaN), they are never used.
    cost_with_nan = cost.copy()
    cost_with_nan[0, 1, :] = np.nan
    padded = match_sources(cost_with_nan, source_valid, slot_valid)
    assert np.isclose(padded.groups[0].optimal.optimal_cost, group.optimal.optimal_cost)


def test_match_sources_rejects_more_sources_than_slots():
    cost = np.zeros((1, 3, 2))
    with pytest.raises(MatchingError):
        match_sources(cost)

    cost = np.zeros((2, 3, 4))
    source_valid = np.ones((2, 3), dtype=bool)
    source_valid[1] = [True, True, True]
    slot_valid = np.ones((2, 4), dtype=bool)
    slot_valid[0] = [True, True, False, False]
    with pytest.raises(MatchingError):
        match_sources(cost, source_valid, slot_valid)


def test_identity_masked_property_covers_ambiguity_and_truncation():
    unique = match_sources(np.array([[[0.0, 5.0, 5.0], [5.0, 0.0, 5.0]]]))
    assert unique.groups[0].identity_masked is False

    ambiguous = match_sources(np.zeros((1, 2, 3)))
    assert ambiguous.groups[0].ambiguous is True
    assert ambiguous.groups[0].identity_masked is True

    truncated = match_sources(np.zeros((1, 2, 3)), max_optimal=2)
    assert truncated.groups[0].ambiguous is False
    assert truncated.groups[0].truncated is True
    assert truncated.groups[0].identity_masked is True


def test_optimal_cost_is_permutation_invariant():
    rng = np.random.default_rng(11)
    cost = rng.normal(size=(1, 3, 5))
    base = match_sources(cost)
    permutation = np.array([2, 0, 1])
    permuted = match_sources(cost[:, permutation, :])
    assert np.isclose(
        base.groups[0].optimal.optimal_cost, permuted.groups[0].optimal.optimal_cost
    )
    # With all-zero cost the optimal set is permutation-closed.
    tied = match_sources(np.zeros((1, 2, 3)))
    assert tied.groups[0].optimal.num_optimal == 6
