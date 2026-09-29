"""Real CPU overfit experiment on small distinguishable fake features.

This is the issue #7 end-to-end acceptance check: it actually optimizes the head
on a synthetic batch (not only forward mocks), then asserts that

* the loss really decreases and beats the zero-logits (P = 0.5) BCE baseline,
  while an all-inactive (P = 0) predictor gets F1 = 0 on positive-containing
  data - these are two different baselines and both are reported;
* center activity is learned as *center* activity (neighbors active + center
  silent must yield low ``P``; isolated center-active must yield high ``P``);
* same-identity embeddings are closer than different-identity embeddings;
* a sustained-silence batch stays finite through an optimization step.

``capsys`` output is part of the evidence reported in the PR.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import torch
import torch.nn.functional as F

from aat.losses import head_loss
from aat.models import SourceQueryHead

from .fakes import make_silence_batch, make_synthetic_batch

pytestmark = pytest.mark.ml

SLOTS = 4
STEPS = 250
LEARNING_RATE = 3e-3
D_MODEL = 64


def _train_head():
    batch = make_synthetic_batch(
        num_groups=3,
        windows_per_group=5,
        num_sources=3,
        feature_dim=8,
        frames_per_window=5,
        seed=11,
    )
    torch.manual_seed(0)
    model = SourceQueryHead(
        batch.features.shape[-1], slots=SLOTS, d_model=D_MODEL, num_heads=4
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    features, times, valid, centers, center_valid = batch.flat()
    groups, windows = batch.num_groups, batch.windows_per_group
    losses = []
    for _ in range(STEPS):
        output = model(features, times, valid, centers, center_valid)
        embeddings = output.embeddings.reshape(groups, windows, SLOTS, 128)
        logits = output.activity_logits.reshape(groups, windows, SLOTS)
        result = head_loss(
            embeddings,
            logits,
            activity_target=batch.activity,
            source_valid=batch.source_valid,
            composition_ids=batch.composition_ids,
            source_ids=batch.source_ids,
            center_valid=batch.center_valid,
        )
        optimizer.zero_grad()
        result.total.backward()
        optimizer.step()
        losses.append(result)
    return batch, model, losses


def _evaluate(batch, model):
    groups, windows = batch.num_groups, batch.windows_per_group
    features, times, valid, centers, center_valid = batch.flat()
    model.eval()
    with torch.no_grad():
        output = model(features, times, valid, centers, center_valid)
        embeddings = output.embeddings.reshape(groups, windows, SLOTS, 128)
        logits = output.activity_logits.reshape(groups, windows, SLOTS)
        result = head_loss(
            embeddings,
            logits,
            activity_target=batch.activity,
            source_valid=batch.source_valid,
            composition_ids=batch.composition_ids,
            source_ids=batch.source_ids,
            center_valid=batch.center_valid,
        )
        probabilities = torch.sigmoid(logits)

    matched = torch.zeros(groups, windows, batch.num_sources)
    usable = torch.zeros(groups, windows, batch.num_sources, dtype=torch.bool)
    for group in range(groups):
        matching = result.matching.groups[group]
        assert matching.optimal.num_optimal == 1 and not matching.optimal.truncated
        assignment = matching.optimal.assignments[0]
        for local, source in enumerate(matching.source_indices):
            slot = assignment[local]  # original slot index, already remapped
            matched[group, :, source] = probabilities[group, :, slot]
            usable[group, :, source] = batch.center_valid[group]
    return result, matched, usable, embeddings


def _pairwise_cosines(batch, embeddings, result):
    same: list[float] = []
    different: list[float] = []
    for group in range(batch.num_groups):
        matching = result.matching.groups[group]
        assignment = matching.optimal.assignments[0]
        slots = [assignment[local] for local in range(len(matching.source_indices))]
        unit = F.normalize(embeddings[group][:, slots, :], dim=-1)
        active = batch.activity[group] >= 0.5
        for left in range(active.shape[1]):
            for right in range(active.shape[1]):
                for window_left in range(active.shape[0]):
                    if not active[window_left, left]:
                        continue
                    for window_right in range(active.shape[0]):
                        if window_left == window_right or not active[window_right, right]:
                            continue
                        cosine = float((unit[window_left, left] * unit[window_right, right]).sum())
                        (same if left == right else different).append(cosine)
    return torch.tensor(same), torch.tensor(different)


@pytest.fixture(scope="module")
def trained_run():
    return _train_head()


def test_small_distinguishable_data_overfits_and_beats_zero_baseline(trained_run, capsys):
    batch, model, losses = trained_run
    first = losses[0].summary()
    last = losses[-1].summary()
    zero_logits_baseline = math.log(2.0)  # BCE of logits 0, i.e. P = 0.5 everywhere
    assert math.isfinite(last["total"])
    assert last["total"] < first["total"] * 0.5
    assert last["activity"] < 0.1
    assert last["activity"] < zero_logits_baseline - 0.4
    assert losses[-1].stats.positive_terms > 0

    result, matched, usable, _ = _evaluate(batch, model)
    targets = batch.activity[usable]
    probabilities = matched[usable]
    predicted = (probabilities >= 0.5).float()
    true_positive = float(((predicted == 1) & (targets == 1)).sum())
    false_positive = float(((predicted == 1) & (targets == 0)).sum())
    false_negative = float(((predicted == 0) & (targets == 1)).sum())
    f1 = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1.0)

    assert f1 > 0.9
    assert float(probabilities[targets >= 0.5].mean()) > 0.8
    assert float(probabilities[targets < 0.5].mean()) < 0.2

    # Explicit all-inactive (P = 0) baseline: with positives in the data it can
    # never be perfect, and its conventional F1 is 0.
    all_inactive = torch.zeros_like(targets)
    inactive_accuracy = float((all_inactive == targets).float().mean())
    inactive_f1 = 0.0
    assert inactive_accuracy < 1.0
    assert f1 > inactive_f1 + 0.9

    print(
        f"[overfit] loss {first['total']:.4f} -> {last['total']:.4f}; "
        f"activity {first['activity']:.4f} -> {last['activity']:.6f} "
        f"(zero-logits/P=0.5 BCE baseline {zero_logits_baseline:.4f}); "
        f"F1={f1:.3f} (all-inactive F1={inactive_f1:.3f} acc={inactive_accuracy:.3f}); "
        f"P(active)={float(probabilities[targets >= 0.5].mean()):.4f} "
        f"P(inactive)={float(probabilities[targets < 0.5].mean()):.4f}"
    )
    assert result.stats.negative_terms > 0


def test_center_activity_is_not_whole_window_activity(trained_run, capsys):
    batch, model, _ = trained_run
    _, matched, usable, _ = _evaluate(batch, model)
    targets = batch.activity[usable]
    probabilities = matched[usable]
    context = batch.context_activity[usable]

    boundary = (targets < 0.5) & context
    isolated = (targets >= 0.5) & ~context
    assert int(boundary.sum()) > 0 and int(isolated.sum()) > 0
    boundary_mean = float(probabilities[boundary].mean())
    isolated_mean = float(probabilities[isolated].mean())
    assert boundary_mean < 0.5  # neighbors active but the center is silent
    assert isolated_mean > 0.5  # center active while the context is silent
    print(
        f"[center] boundary P={boundary_mean:.6f} (n={int(boundary.sum())}); "
        f"isolated P={isolated_mean:.6f} (n={int(isolated.sum())})"
    )


def test_same_identity_embeddings_are_closer_than_different_identities(trained_run, capsys):
    batch, model, _ = trained_run
    result, _, _, embeddings = _evaluate(batch, model)
    same, different = _pairwise_cosines(batch, embeddings, result)
    assert same.numel() > 0 and different.numel() > 0
    same_mean = float(same.mean())
    different_mean = float(different.mean())
    assert same_mean > different_mean + 0.5
    print(
        f"[identity] mean cos same={same_mean:.4f} different={different_mean:.4f} "
        f"(pairs {same.numel()}/{different.numel()})"
    )


def test_sustained_silence_optimization_step_is_finite():
    batch = make_silence_batch(
        num_groups=2, windows_per_group=4, num_sources=2, feature_dim=8, seed=2
    )
    torch.manual_seed(1)
    model = SourceQueryHead(batch.features.shape[-1], slots=4, d_model=32, num_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features, times, valid, centers, center_valid = batch.flat()
    groups, windows = batch.num_groups, batch.windows_per_group
    output = model(features, times, valid, centers, center_valid)
    result = head_loss(
        output.embeddings.reshape(groups, windows, 4, 128),
        output.activity_logits.reshape(groups, windows, 4),
        activity_target=batch.activity,
        source_valid=batch.source_valid,
        composition_ids=batch.composition_ids,
        source_ids=batch.source_ids,
        center_valid=batch.center_valid,
    )
    assert result.stats.positive_terms == 0  # silence carries no identity evidence
    optimizer.zero_grad()
    result.total.backward()
    optimizer.step()
    for parameter in model.parameters():
        assert bool(torch.isfinite(parameter).all())
        assert parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
