"""Arranged loss tests: matching, ambiguity, identity contrast, padding, silence."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch

from aat.losses import LossWeights, head_loss
from aat.models import SourceQueryHead

from .fakes import make_silence_batch, make_synthetic_batch

pytestmark = pytest.mark.ml


def _one_hot(index: int, dim: int = 128) -> torch.Tensor:
    vector = torch.zeros(dim)
    vector[index] = 1.0
    return vector


def _identity_ids(num_groups: int, num_sources: int, prefix: str = "c") -> tuple[list[str], list[list[str]]]:
    return (
        [f"{prefix}{group}" for group in range(num_groups)],
        [[f"src-{source}" for source in range(num_sources)] for _ in range(num_groups)],
    )


def test_gt_source_permutation_does_not_change_total_or_components():
    batch = make_synthetic_batch(num_groups=2, windows_per_group=3, num_sources=3, seed=5)
    torch.manual_seed(0)
    model = SourceQueryHead(batch.features.shape[-1], slots=4, d_model=32, num_heads=4)
    model.eval()
    features, times, valid, centers, center_valid = batch.flat()
    output = model(features, times, valid, centers, center_valid)
    groups, windows = batch.num_groups, batch.windows_per_group
    embeddings = output.embeddings.reshape(groups, windows, 4, 128)
    logits = output.activity_logits.reshape(groups, windows, 4)

    base = head_loss(
        embeddings,
        logits,
        activity_target=batch.activity,
        source_valid=batch.source_valid,
        composition_ids=batch.composition_ids,
        source_ids=batch.source_ids,
        center_valid=batch.center_valid,
    )

    permutation = [2, 0, 1]
    permuted = head_loss(
        embeddings,
        logits,
        activity_target=batch.activity[:, :, permutation],
        source_valid=batch.source_valid[:, permutation],
        composition_ids=batch.composition_ids,
        source_ids=[[ids[index] for index in permutation] for ids in batch.source_ids],
        center_valid=batch.center_valid,
    )
    for name in ("total", "activity", "empty_slots", "positive", "negative"):
        torch.testing.assert_close(getattr(base, name), getattr(permuted, name))
    assert base.stats == permuted.stats


def test_cross_window_identity_switch_increases_positive_and_total():
    embeddings = torch.zeros(1, 2, 2, 128)
    embeddings[0, 0, 0] = _one_hot(0)
    embeddings[0, 0, 1] = _one_hot(1)
    embeddings[0, 1, 0] = _one_hot(0)
    embeddings[0, 1, 1] = _one_hot(1)
    # Distinct activity sequences keep the matching unique (no ambiguity):
    # source a is active in both windows, source b only in window 0.
    target = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
    logits = torch.tensor([[[20.0, -20.0], [20.0, -20.0]]])

    baseline = head_loss(
        embeddings,
        logits,
        activity_target=target,
        composition_ids=["c0"],
        source_ids=[["a", "b"]],
    )
    assert baseline.stats.positive_terms == 2  # source a has two anchors
    assert float(baseline.positive) == 0.0

    switched = embeddings.clone()
    switched[0, 1, 0] = _one_hot(1)  # identity a appears as b's direction in window 1
    after = head_loss(
        switched,
        logits,
        activity_target=target,
        composition_ids=["c0"],
        source_ids=[["a", "b"]],
    )
    assert float(after.positive) > float(baseline.positive) + 0.5
    assert float(after.total) > float(baseline.total) + 0.1


def test_identical_activity_columns_report_ambiguity_and_mask_identity():
    target = torch.ones(1, 2, 2)
    logits = torch.zeros(1, 2, 2)
    embeddings = torch.zeros(1, 2, 2, 128)
    embeddings[..., 0] = 1.0

    result = head_loss(
        embeddings,
        logits,
        activity_target=target,
        composition_ids=["c0"],
        source_ids=[["a", "b"]],
    )
    assert result.matching.groups[0].optimal.num_optimal == 2
    assert result.stats.ambiguous_groups == 1
    assert result.stats.identity_masked_groups == 1
    assert result.stats.positive_terms == 0
    assert result.stats.negative_terms == 0

    permuted = head_loss(
        embeddings,
        logits,
        activity_target=target,
        composition_ids=["c0"],
        source_ids=[["b", "a"]],
    )
    torch.testing.assert_close(result.total, permuted.total)
    torch.testing.assert_close(result.activity, permuted.activity)
    torch.testing.assert_close(result.empty_slots, permuted.empty_slots)


def test_truncated_enumeration_skips_identity_and_empty_terms():
    target = torch.ones(1, 1, 4)
    logits = torch.zeros(1, 1, 5)
    embeddings = torch.randn(1, 1, 5, 128, requires_grad=True)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    embeddings.retain_grad()
    logits = torch.zeros(1, 1, 5)
    logits.requires_grad_(True)

    result = head_loss(
        embeddings,
        logits,
        activity_target=target,
        composition_ids=["c0"],
        source_ids=[["a", "b", "c", "d"]],
        max_optimal_assignments=5,
    )
    assert result.stats.truncated_groups == 1
    assert result.stats.identity_masked_groups == 1
    assert result.stats.empty_terms == 0
    assert result.stats.positive_terms == 0
    assert result.stats.negative_terms == 0
    result.total.backward()
    assert embeddings.grad is not None
    assert logits.grad is not None
    assert bool(torch.isfinite(embeddings.grad).all())
    assert bool(torch.isfinite(logits.grad).all())


def test_same_source_id_in_different_compositions_is_negative_not_positive():
    # One active window per identity so within-identity positives cannot fire;
    # only cross-group (composition_id, source_id) evidence is left.
    target = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
    logits = torch.full((2, 2, 1), 20.0)
    embeddings = torch.zeros(2, 2, 1, 128)
    embeddings[..., 0] = 1.0

    different = head_loss(
        embeddings,
        logits,
        activity_target=target,
        composition_ids=["comp-a", "comp-b"],
        source_ids=[["lead"], ["lead"]],
    )
    assert different.stats.positive_terms == 0
    assert different.stats.negative_terms == 2
    assert float(different.negative) > 0.0  # identical directions violate the hinge

    same_song = head_loss(
        embeddings,
        logits,
        activity_target=target,
        composition_ids=["comp-a", "comp-a"],
        source_ids=[["lead"], ["lead"]],
    )
    assert same_song.stats.positive_terms == 2  # one identity, two cross-group anchors
    assert same_song.stats.negative_terms == 0


def test_positive_pull_and_negative_push_geometry():
    target = torch.tensor([[[1.0], [1.0]]])
    logits = torch.full((1, 2, 1), 20.0)

    aligned = torch.zeros(1, 2, 1, 128)
    aligned[0, 0, 0] = _one_hot(0)
    aligned[0, 1, 0] = _one_hot(0)
    same = head_loss(
        aligned, logits, activity_target=target, composition_ids=["c"], source_ids=[["a"]]
    )
    assert float(same.positive) == 0.0

    flipped = aligned.clone()
    flipped[0, 1, 0] = _one_hot(1)
    apart = head_loss(
        flipped, logits, activity_target=target, composition_ids=["c"], source_ids=[["a"]]
    )
    assert float(apart.positive) > 0.9

    two_identities_target = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
    logits_two = torch.tensor([[[20.0, -20.0], [20.0, -20.0]]])
    embeddings = torch.zeros(1, 2, 2, 128)
    embeddings[..., 0] = 1.0  # both identities share a prototype
    negative = head_loss(
        embeddings,
        logits_two,
        activity_target=two_identities_target,
        composition_ids=["c"],
        source_ids=[["a", "b"]],
    )
    assert float(negative.negative) > 0.0

    embeddings[0, 0, 1, 0] = 0.0
    embeddings[0, 0, 1, 1] = 1.0
    separated = head_loss(
        embeddings,
        logits_two,
        activity_target=two_identities_target,
        composition_ids=["c"],
        source_ids=[["a", "b"]],
    )
    assert float(separated.negative) == 0.0


def test_empty_slot_penalty():
    target = torch.tensor([[[1.0], [0.0]]])
    embeddings = torch.zeros(1, 2, 2, 128)
    embeddings[..., 0] = 1.0

    high_empty_logits = torch.tensor([[[20.0, 20.0], [-20.0, 20.0]]])
    result = head_loss(
        embeddings,
        high_empty_logits,
        activity_target=target,
        composition_ids=["c"],
        source_ids=[["a"]],
    )
    assert result.stats.empty_terms == 2
    assert float(result.empty_slots.detach()) > 1.0

    quiet_empty_logits = torch.tensor([[[20.0, -20.0], [-20.0, -20.0]]])
    quiet = head_loss(
        embeddings,
        quiet_empty_logits,
        activity_target=target,
        composition_ids=["c"],
        source_ids=[["a"]],
    )
    assert float(quiet.empty_slots) < 0.01


def test_silence_batch_is_finite_and_does_not_force_identity():
    batch = make_silence_batch(num_groups=2, windows_per_group=3, num_sources=2, seed=3)
    torch.manual_seed(0)
    model = SourceQueryHead(batch.features.shape[-1], slots=4, d_model=32, num_heads=4)
    features, times, valid, centers, center_valid = batch.flat()
    output = model(features, times, valid, centers, center_valid)
    groups, windows = batch.num_groups, batch.windows_per_group
    embeddings = output.embeddings.reshape(groups, windows, 4, 128)
    logits = output.activity_logits.reshape(groups, windows, 4)

    result = head_loss(
        embeddings,
        logits,
        activity_target=batch.activity,
        source_valid=batch.source_valid,
        composition_ids=batch.composition_ids,
        source_ids=batch.source_ids,
        center_valid=batch.center_valid,
    )
    assert result.stats.positive_terms == 0
    assert result.stats.negative_terms == 0
    assert all(torch.isfinite(component).all() for component in result.components().values())
    result.total.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert bool(torch.isfinite(parameter.grad).all())


def test_no_valid_centers_skips_groups_without_failing_backward():
    embeddings = torch.randn(1, 2, 2, 128, requires_grad=True)
    logits = torch.randn(1, 2, 2)
    target = torch.ones(1, 2, 2)
    center_valid = torch.zeros(1, 2, dtype=torch.bool)

    result = head_loss(
        embeddings,
        logits,
        activity_target=target,
        composition_ids=["c"],
        source_ids=[["a", "b"]],
        center_valid=center_valid,
    )
    assert result.stats.activity_terms == 0
    assert result.stats.empty_terms == 0
    assert result.stats.skipped_groups == 1
    assert float(result.total.detach()) == 0.0
    result.total.backward()
    assert embeddings.grad is not None
    assert bool(torch.isfinite(embeddings.grad).all())


def test_empty_source_capacity_is_penalised_and_finite():
    embeddings = torch.randn(1, 2, 3, 128, requires_grad=True)
    logits = torch.full((1, 2, 3), 10.0)
    target = torch.zeros(1, 2, 0)

    result = head_loss(
        embeddings,
        logits,
        activity_target=target,
        composition_ids=["c"],
        source_ids=[[]],
    )
    assert result.stats.activity_terms == 0
    assert result.stats.empty_terms == 6
    assert float(result.empty_slots.detach()) > 1.0
    result.total.backward()
    assert bool(torch.isfinite(embeddings.grad).all())


def test_padding_windows_and_sources_do_not_change_loss():
    torch.manual_seed(4)
    embeddings = torch.nn.functional.normalize(torch.randn(1, 2, 3, 128), dim=-1)
    logits = torch.randn(1, 2, 3)
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    composition_ids = ["c"]
    source_ids = [["a", "b"]]

    base = head_loss(
        embeddings,
        logits,
        activity_target=target,
        composition_ids=composition_ids,
        source_ids=source_ids,
    )

    padded_embeddings = torch.cat([embeddings, torch.randn(1, 1, 3, 128)], dim=1)
    padded_logits = torch.cat([logits, torch.randn(1, 1, 3)], dim=1)
    padded_target = torch.cat([target, torch.zeros(1, 1, 2)], dim=1)
    padded_center_valid = torch.tensor([[True, True, False]])
    padded_sources_target = torch.cat([padded_target, torch.zeros(1, 3, 1)], dim=2)
    padded = head_loss(
        padded_embeddings,
        padded_logits,
        activity_target=padded_sources_target,
        source_valid=torch.tensor([[True, True, False]]),
        composition_ids=composition_ids,
        source_ids=[["a", "b", "unused"]],
        center_valid=padded_center_valid,
    )
    for name in ("total", "activity", "empty_slots", "positive", "negative"):
        torch.testing.assert_close(getattr(base, name), getattr(padded, name))


def test_capacity_overflow_and_exact_slot_limit_are_rejected():
    embeddings = torch.randn(1, 2, 2, 128)
    logits = torch.randn(1, 2, 2)
    target = torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match="must not exceed K"):
        head_loss(embeddings, logits, activity_target=target)

    small = torch.randn(1, 2, 3, 128)
    small_logits = torch.randn(1, 2, 3)
    with pytest.raises(ValueError, match="max_exact_slots"):
        head_loss(
            small,
            small_logits,
            activity_target=torch.ones(1, 2, 1),
            max_exact_slots=2,
        )


def test_identity_terms_are_optional_and_weight_override_is_exposed():
    batch = make_synthetic_batch(num_groups=1, windows_per_group=3, num_sources=2, seed=9)
    torch.manual_seed(0)
    model = SourceQueryHead(batch.features.shape[-1], slots=3, d_model=32, num_heads=4)
    features, times, valid, centers, center_valid = batch.flat()
    output = model(features, times, valid, centers, center_valid)
    embeddings = output.embeddings.reshape(1, 3, 3, 128)
    logits = output.activity_logits.reshape(1, 3, 3)

    without_identity = head_loss(embeddings, logits, activity_target=batch.activity)
    assert without_identity.stats.positive_terms == 0
    assert without_identity.stats.negative_terms == 0

    weighted = head_loss(
        embeddings,
        logits,
        activity_target=batch.activity,
        composition_ids=batch.composition_ids,
        source_ids=batch.source_ids,
        weights=LossWeights(activity=2.0, empty_slots=0.0, positive=3.0, negative=4.0),
    )
    assert weighted.weights == {"activity": 2.0, "empty_slots": 0.0, "positive": 3.0, "negative": 4.0}
    expected = (
        2.0 * weighted.activity
        + 3.0 * weighted.positive
        + 4.0 * weighted.negative
    )
    torch.testing.assert_close(weighted.total, expected)
