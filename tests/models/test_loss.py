"""Permutation-invariance, loss-term and numerical-safety tests."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F

from aat.losses import (
    GroupTargets,
    PermutationInvariantHeadLoss,
)

from . import fakes

LOG2 = math.log(2.0)


def _loss() -> PermutationInvariantHeadLoss:
    return PermutationInvariantHeadLoss()


def _aligned_logits(activity: torch.Tensor, magnitude: float = 6.0) -> torch.Tensor:
    """Slot k follows source k: positive where active, negative elsewhere."""

    return torch.where(
        activity > 0.5,
        torch.full_like(activity, magnitude),
        torch.full_like(activity, -magnitude),
    )


def _one_hot_basis(sources: int, dim: int) -> torch.Tensor:
    basis = torch.zeros(sources, dim)
    for index in range(sources):
        basis[index, index] = 1.0
    return basis


def test_gt_source_permutation_does_not_change_total_loss():
    torch.manual_seed(0)
    embeddings = fakes.random_unit_embeddings(n_windows=5, slots=4, dim=16, seed=1)
    logits = torch.randn(5, 4, generator=torch.Generator().manual_seed(2))
    activity = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    descriptors = torch.randn(3, 16, generator=torch.Generator().manual_seed(3))
    order = [2, 0, 1]

    loss = _loss()
    reference = loss(embeddings, logits, GroupTargets(activity))
    permuted = loss(
        embeddings, logits, GroupTargets(activity[:, order])
    )
    torch.testing.assert_close(reference.total, permuted.total, atol=1e-5, rtol=1e-5)
    for name in ("activity", "same_source", "diff_source", "empty_slots"):
        torch.testing.assert_close(
            getattr(reference, name), getattr(permuted, name), atol=1e-5, rtol=1e-5
        )

    with_descriptors = loss(
        embeddings, logits, GroupTargets(activity, source_descriptors=descriptors)
    )
    permuted_descriptors = loss(
        embeddings,
        logits,
        GroupTargets(activity[:, order], source_descriptors=descriptors[order]),
    )
    torch.testing.assert_close(
        with_descriptors.total, permuted_descriptors.total, atol=1e-5, rtol=1e-5
    )


def test_ambiguous_targets_are_explicitly_symmetrized():
    torch.manual_seed(1)
    embeddings = fakes.random_unit_embeddings(n_windows=4, slots=3, dim=16, seed=4)
    logits = torch.randn(4, 3, generator=torch.Generator().manual_seed(5))
    # Sources 0 and 1 have identical activity sequences on every window.
    activity = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    loss = _loss()
    reference = loss(embeddings, logits, GroupTargets(activity))
    assert reference.ambiguous_assignments == 2
    assert reference.ambiguity_truncated is False

    swapped = loss(embeddings, logits, GroupTargets(activity[:, [1, 0, 2]]))
    torch.testing.assert_close(reference.total, swapped.total, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        reference.same_source, swapped.same_source, atol=1e-5, rtol=1e-5
    )


def test_cross_window_identity_swap_increases_same_source_loss():
    dim = 16
    basis = _one_hot_basis(2, dim)
    embeddings = basis.unsqueeze(0).expand(3, -1, -1).clone()  # constant per slot
    activity = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    logits = _aligned_logits(activity)
    targets = GroupTargets(activity)

    loss = _loss()
    clean = loss(embeddings, logits, targets)
    assert clean.matching.slot_for_source.tolist() == [0, 1]
    assert clean.same_source.item() == pytest.approx(0.0, abs=1e-6)

    swapped = embeddings.clone()
    swapped[0, 0], swapped[0, 1] = embeddings[0, 1].clone(), embeddings[0, 0].clone()
    result = loss(swapped, logits, targets)

    assert result.matching.slot_for_source.tolist() == [0, 1]
    assert result.same_source.item() > clean.same_source.item() + 1e-3
    assert result.total.item() > clean.total.item()


def test_diff_source_penalizes_collapsed_identities():
    dim = 16
    activity = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    logits = _aligned_logits(activity)
    targets = GroupTargets(activity)
    loss = _loss()

    collapsed = torch.zeros(3, 2, dim)
    collapsed[:, :, 0] = 1.0
    collapsed_result = loss(collapsed, logits, targets)
    assert collapsed_result.diff_source.item() > 0.5

    basis = _one_hot_basis(2, dim)
    separated = basis.unsqueeze(0).expand(3, -1, -1).clone()
    separated_result = loss(separated, logits, targets)
    assert separated_result.diff_source.item() == pytest.approx(0.0, abs=1e-6)


def test_empty_slots_are_suppressed_and_full_capacity_has_no_empty_term():
    torch.manual_seed(2)
    embeddings = fakes.random_unit_embeddings(n_windows=3, slots=3, dim=16, seed=6)
    activity = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    logits = torch.zeros(3, 3)

    loss = _loss()
    result = loss(embeddings, logits, GroupTargets(activity))
    assert result.matching.unmatched_slots.shape == (1,)
    assert result.empty_slots.item() == pytest.approx(LOG2, abs=1e-5)

    full = loss(embeddings[:, :2], logits[:, :2], GroupTargets(activity))
    assert full.matching.unmatched_slots.numel() == 0
    assert full.empty_slots.item() == 0.0


def test_all_zero_prediction_is_not_counted_as_success():
    torch.manual_seed(3)
    embeddings = fakes.random_unit_embeddings(n_windows=3, slots=2, dim=16, seed=7)
    activity = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = GroupTargets(activity)
    loss = _loss()

    zero_logits = torch.full((3, 2), -20.0)
    zero_result = loss(embeddings, zero_logits, targets)
    assert zero_result.activity.item() > 1.0
    assert zero_result.total.item() > 0.0

    probe = zero_logits.clone().requires_grad_(True)
    probe_result = loss(embeddings, probe, targets)
    probe_result.total.backward()
    assert probe.grad is not None
    assert probe.grad.abs().sum().item() > 0.0

    perfect_logits = _aligned_logits(activity, magnitude=20.0)
    perfect_result = loss(embeddings, perfect_logits, targets)
    assert perfect_result.activity.item() < 0.01
    assert perfect_result.total.item() < zero_result.total.item()


def test_perfect_matched_predictions_are_near_zero():
    dim = 16
    basis = _one_hot_basis(2, dim)
    embeddings = basis.unsqueeze(0).expand(4, -1, -1).clone()
    activity = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    logits = _aligned_logits(activity, magnitude=20.0)

    result = _loss()(embeddings, logits, GroupTargets(activity))
    assert result.activity.item() < 1e-3
    assert result.same_source.item() < 1e-5
    assert result.diff_source.item() < 1e-6
    assert result.empty_slots.item() == 0.0
    assert result.total.item() < 1e-3


def test_all_invalid_windows_produce_finite_zero_terms_with_gradients():
    torch.manual_seed(4)
    embeddings = fakes.random_unit_embeddings(n_windows=3, slots=2, dim=16, seed=8)
    logits = torch.randn(3, 2, generator=torch.Generator().manual_seed(9)).requires_grad_(
        True
    )
    targets = GroupTargets(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        window_valid=torch.zeros(3, dtype=torch.bool),
    )

    result = _loss()(embeddings, logits, targets)
    for value in result.as_dict().values():
        assert math.isfinite(value)
    result.total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_empty_group_and_single_window_are_finite():
    torch.manual_seed(5)
    embeddings = fakes.random_unit_embeddings(n_windows=3, slots=2, dim=8, seed=10)
    logits = torch.zeros(3, 2)
    loss = _loss()

    empty = loss(embeddings, logits, GroupTargets(torch.zeros(3, 0)))
    assert empty.activity.item() == 0.0
    assert empty.same_source.item() == 0.0
    assert empty.diff_source.item() == 0.0
    assert empty.empty_slots.item() == pytest.approx(LOG2, abs=1e-5)

    single = loss(
        embeddings[:1],
        logits[:1],
        GroupTargets(torch.tensor([[1.0, 0.0]])),
    )
    assert single.same_source.item() == pytest.approx(0.0, abs=1e-6)


def test_all_silent_group_is_finite_and_prefers_zero_logits_over_positive():
    torch.manual_seed(6)
    embeddings = fakes.random_unit_embeddings(n_windows=3, slots=2, dim=8, seed=11)
    targets = GroupTargets(torch.zeros(3, 2))
    loss = _loss()

    silent = loss(embeddings, torch.zeros(3, 2), targets)
    loud = loss(embeddings, torch.full((3, 2), 5.0), targets)
    assert silent.activity.item() == pytest.approx(LOG2, abs=1e-5)
    assert silent.activity.item() < loud.activity.item()
    for value in silent.as_dict().values():
        assert math.isfinite(value)


def test_descriptors_drive_matching_when_activity_is_uninformative():
    dim = 16
    basis = _one_hot_basis(2, dim)
    embeddings = basis.unsqueeze(0).expand(3, -1, -1).clone()
    activity = torch.ones(3, 2)  # identical columns: activity cannot distinguish
    descriptors = basis.clone()
    targets = GroupTargets(activity, source_descriptors=descriptors)

    result = _loss()(embeddings, torch.zeros(3, 2), targets)
    assert result.matching.slot_for_source.tolist() == [0, 1]
    # Activity columns are identical but descriptors break the tie, so the
    # matching is no longer ambiguous.
    assert result.ambiguous_assignments == 1


def test_soft_activity_labels_are_supported():
    torch.manual_seed(7)
    embeddings = fakes.random_unit_embeddings(n_windows=2, slots=2, dim=8, seed=12)
    activity = torch.tensor([[0.5, 0.0], [0.0, 0.8]])
    result = _loss()(embeddings, torch.zeros(2, 2), GroupTargets(activity))
    for value in result.as_dict().values():
        assert math.isfinite(value)


def test_gradients_reach_model_parameters_and_inputs():
    group = fakes.synthetic_group(seed=3, n_windows=6, n_sources=3)
    head = fakes.make_head(feature_dim=8, slots=3, d_model=16, seed=4)
    features = group.features.clone().requires_grad_(True)

    output = head(features, group.feature_valid, group.frame_times)
    result = _loss()(output.embeddings, output.activity_logits, group.targets)
    result.total.backward()

    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum().item() > 0.0


def test_loss_rejects_capacity_overflow():
    embeddings = torch.zeros(2, 2, 8)
    activity = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="capacity exceeded"):
        _loss()(embeddings, torch.zeros(2, 2), GroupTargets(activity))


def test_group_targets_validation():
    with pytest.raises(ValueError, match="\\[N, S\\]"):
        GroupTargets(torch.zeros(3))
    with pytest.raises(ValueError, match="NaN or Inf"):
        GroupTargets(torch.tensor([[float("nan")]]))
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        GroupTargets(torch.tensor([[1.5]]))
    with pytest.raises(ValueError, match="window_valid"):
        GroupTargets(torch.zeros(3, 2), window_valid=torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="source_descriptors"):
        GroupTargets(torch.zeros(3, 2), source_descriptors=torch.zeros(3, 4))


def test_prediction_shapes_are_validated_against_targets():
    with pytest.raises(ValueError, match="embeddings"):
        _loss()(torch.zeros(2, 2), torch.zeros(2), GroupTargets(torch.zeros(2, 2)))
    with pytest.raises(ValueError, match="windows"):
        _loss()(torch.zeros(2, 2, 8), torch.zeros(2, 2), GroupTargets(torch.zeros(3, 2)))
