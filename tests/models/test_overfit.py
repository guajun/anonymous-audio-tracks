"""Tiny synthetic overfit test: the loss has to be optimizable end to end."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F

from aat.losses import PermutationInvariantHeadLoss

from . import fakes


def test_tiny_synthetic_group_can_overfit():
    group = fakes.synthetic_group(
        seed=1, n_windows=8, n_sources=3, feature_dim=8, frames=2, noise=0.03
    )
    head = fakes.make_head(feature_dim=8, slots=3, d_model=32, seed=2)
    loss_fn = PermutationInvariantHeadLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)

    initial = None
    final = None
    for step in range(1000):
        optimizer.zero_grad()
        output = head(group.features, group.feature_valid, group.frame_times)
        result = loss_fn(output.embeddings, output.activity_logits, group.targets)
        result.total.backward()
        optimizer.step()
        if step == 0:
            initial = float(result.total.item())
        final = float(result.total.item())

    assert initial is not None and final is not None
    assert final < 0.5 * initial, f"loss did not drop enough: {initial} -> {final}"

    with torch.no_grad():
        output = head(group.features, group.feature_valid, group.frame_times)
        result = loss_fn(output.embeddings, output.activity_logits, group.targets)

    assignment = result.matching.slot_for_source
    predicted_activity = output.activity[:, assignment]
    target_activity = group.targets.activity
    accuracy = (((predicted_activity > 0.5) == (target_activity > 0.5)).float()).mean()
    assert accuracy.item() > 0.95

    matched = F.normalize(output.embeddings[:, assignment], dim=-1)
    prototype = F.normalize(matched.mean(dim=0), dim=-1)
    cosine = (matched * prototype.unsqueeze(0)).sum(dim=-1).mean()
    assert cosine.item() > 0.9
    assert result.activity.item() < 0.05
    assert result.same_source.item() < 0.05
