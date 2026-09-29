"""Matching, balanced BCE and ambiguity-handling tests."""

from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F

from aat.losses import (
    activity_pair_cost,
    balanced_bce,
    descriptor_pair_cost,
    find_ambiguous_classes,
    iter_ambiguous_assignments,
    slot_prototypes,
    solve_matching,
)

from . import fakes


def test_balanced_bce_averages_positive_and_negative_sides():
    logits = torch.tensor([0.0, 2.0, -1.0])
    targets = torch.tensor([1.0, 0.0, 0.0])
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    expected = 0.5 * (bce[0] + (bce[1] + bce[2]) / 2.0)
    assert balanced_bce(logits, targets, dim=0).item() == pytest.approx(
        expected.item(), abs=1e-6
    )


def test_balanced_bce_all_silent_falls_back_to_negative_side():
    logits = torch.tensor([0.0, 0.0])
    targets = torch.zeros(2)
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    assert balanced_bce(logits, targets, dim=0).item() == pytest.approx(
        expected.item(), abs=1e-6
    )


def test_balanced_bce_ignores_invalid_entries():
    logits = torch.tensor([0.0, 50.0])
    targets = torch.tensor([1.0, 0.0])
    valid = torch.tensor([True, False])
    expected = balanced_bce(logits[:1], targets[:1], dim=0)
    assert balanced_bce(logits, targets, valid=valid, dim=0).item() == pytest.approx(
        expected.item(), abs=1e-6
    )


def test_balanced_bce_zero_when_nothing_is_valid():
    logits = torch.zeros(2)
    targets = torch.zeros(2)
    valid = torch.zeros(2, dtype=torch.bool)
    value = balanced_bce(logits, targets, valid=valid, dim=0)
    assert value.item() == 0.0


def test_activity_pair_cost_shapes_and_better_fit_is_cheaper():
    logits = torch.tensor([[0.0, 2.0], [0.0, -2.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    cost = activity_pair_cost(logits, targets)
    assert cost.shape == (2, 2)
    # slot 1 follows source 0 much better than slot 0 does.
    assert cost[1, 0].item() < cost[0, 0].item()
    # An all-silent source column reduces to the plain negative-side BCE.
    expected_silent = F.binary_cross_entropy_with_logits(
        logits[:, 0], torch.zeros(2), reduction="mean"
    )
    assert cost[0, 1].item() == pytest.approx(expected_silent.item(), abs=1e-6)


def test_activity_pair_cost_validates_shapes():
    with pytest.raises(ValueError, match="targets"):
        activity_pair_cost(torch.zeros(2, 3), torch.zeros(2, 2, 1))


def test_solve_matching_finds_the_brute_force_optimum():
    generator = torch.Generator().manual_seed(5)
    cost = torch.rand(5, 3, generator=generator)
    result = solve_matching(cost)

    best = min(
        sum(float(cost[candidate[source], source]) for source in range(3))
        for candidate in itertools.permutations(range(5), 3)
    )
    assert result.best_cost.item() == pytest.approx(best, abs=1e-6)
    chosen = sum(
        float(cost[int(result.slot_for_source[source]), source]) for source in range(3)
    )
    assert chosen == pytest.approx(best, abs=1e-6)
    assert result.slot_for_source.shape == (3,)
    assert result.unmatched_slots.shape == (2,)
    assert set(result.slot_for_source.tolist()).isdisjoint(result.unmatched_slots.tolist())


def test_solve_matching_ties_use_lexicographically_first_assignment():
    cost = torch.ones(4, 3)
    result = solve_matching(cost)
    assert result.slot_for_source.tolist() == [0, 1, 2]
    assert result.unmatched_slots.tolist() == [3]


def test_solve_matching_rejects_capacity_overflow():
    with pytest.raises(ValueError, match="capacity exceeded"):
        solve_matching(torch.zeros(2, 3))


def test_solve_matching_rejects_non_finite_cost():
    with pytest.raises(ValueError, match="NaN or Inf"):
        solve_matching(torch.tensor([[0.0, float("nan")]]))


def test_solve_matching_rejects_excessive_enumeration():
    with pytest.raises(ValueError, match="max_permutations"):
        solve_matching(torch.zeros(8, 8), max_permutations=100)


def test_solve_matching_handles_zero_sources():
    cost = torch.rand(3, 0)
    result = solve_matching(cost)
    assert result.slot_for_source.tolist() == []
    assert result.unmatched_slots.tolist() == [0, 1, 2]
    assert result.best_cost.item() == 0.0


def test_find_ambiguous_classes_groups_identical_activity_columns():
    activity = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    valid = torch.ones(4, dtype=torch.bool)
    assert find_ambiguous_classes(activity, valid) == ((0, 1),)


def test_find_ambiguous_classes_ignores_invalid_windows():
    activity = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.7, 0.2]])
    valid = torch.tensor([True, True, False])
    assert find_ambiguous_classes(activity, valid) == ((0, 1),)

    all_invalid = torch.zeros(3, dtype=torch.bool)
    assert find_ambiguous_classes(activity, all_invalid) == ((0, 1),)


def test_descriptors_break_ambiguous_activity_classes():
    activity = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    descriptors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    valid = torch.ones(2, dtype=torch.bool)
    assert find_ambiguous_classes(activity, valid, descriptors) == ()


def test_iter_ambiguous_assignments_enumerates_class_permutations():
    base = torch.tensor([0, 1, 2])
    assignments, total, truncated = iter_ambiguous_assignments(base, ((0, 1),))
    assert total == 2
    assert truncated is False
    assert torch.equal(assignments[0], base)
    assert [assignment.tolist() for assignment in assignments] == [[0, 1, 2], [1, 0, 2]]


def test_iter_ambiguous_assignments_truncates_deterministically():
    base = torch.tensor([0, 1, 2])
    assignments, total, truncated = iter_ambiguous_assignments(
        base, ((0, 1, 2),), max_assignments=2
    )
    assert total == 6
    assert truncated is True
    assert len(assignments) == 2
    assert torch.equal(assignments[0], base)


def test_iter_ambiguous_assignments_without_classes_returns_base():
    base = torch.tensor([2, 0])
    assignments, total, truncated = iter_ambiguous_assignments(base, ())
    assert total == 1
    assert truncated is False
    assert torch.equal(assignments[0], base)


def test_matching_is_invariant_under_source_permutation():
    generator = torch.Generator().manual_seed(11)
    cost = torch.rand(4, 3, generator=generator)
    order = [2, 0, 1]
    first = solve_matching(cost)
    second = solve_matching(cost[:, order])
    assert second.best_cost.item() == pytest.approx(first.best_cost.item(), abs=1e-6)


def test_slot_prototypes_respect_valid_windows():
    embeddings = fakes.random_unit_embeddings(n_windows=3, slots=2, dim=8, seed=1)
    valid = torch.tensor([True, False, True])
    prototypes = slot_prototypes(embeddings, valid)
    expected = F.normalize(embeddings[[0, 2]].mean(dim=0), dim=-1)
    torch.testing.assert_close(prototypes, expected)


def test_descriptor_pair_cost_is_low_for_aligned_directions():
    embeddings = fakes.orthogonal_embeddings(n_windows=3, slots=2, seed=2)
    descriptors = embeddings[0].clone()
    cost = descriptor_pair_cost(embeddings, descriptors)
    assert cost.shape == (2, 2)
    assert cost[0, 0].item() == pytest.approx(0.0, abs=1e-5)
    assert cost[1, 1].item() == pytest.approx(0.0, abs=1e-5)
    assert cost[0, 1].item() > 0.5
