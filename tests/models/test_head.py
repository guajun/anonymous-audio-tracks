"""Head tests: shapes, masks, time sensitivity, slot validity, gradients."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np
import torch

from aat.models import HeadOutput, SourceQueryHead

pytestmark = pytest.mark.ml


def _inputs(batch: int = 5, frames: int = 5, feature_dim: int = 8, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(batch, frames, feature_dim, generator=generator)
    offsets = torch.linspace(-0.08, 0.08, frames)
    centers = 0.5 + torch.arange(batch, dtype=torch.float32) * 0.75
    frame_times = centers[:, None] + offsets[None, :]
    valid = torch.ones(batch, frames, dtype=torch.bool)
    return features, frame_times, valid, centers


def _model(slots: int = 8, feature_dim: int = 8) -> SourceQueryHead:
    torch.manual_seed(0)
    model = SourceQueryHead(feature_dim, slots=slots, d_model=32, num_heads=4)
    model.eval()
    return model


@pytest.mark.parametrize("slots", [1, 3, 8])
def test_shapes_units_and_probability_range(slots):
    model = _model(slots=slots)
    features, times, valid, centers = _inputs()
    out = model(features, times, valid, centers)

    assert isinstance(out, HeadOutput)
    assert out.embeddings.shape == (5, slots, 128)
    assert out.activity_logits.shape == (5, slots)
    assert out.slot_valid.shape == (5, slots)
    assert out.center_valid.shape == (5,)
    assert out.slots == slots and out.embedding_dim == 128
    assert bool(out.slot_valid.all())

    norms = out.embeddings[out.slot_valid].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    probabilities = out.activity_probabilities()
    assert probabilities.shape == (5, slots)
    assert bool((probabilities >= 0).all()) and bool((probabilities <= 1).all())


def test_fully_masked_and_center_invalid_slots_are_canonical_zeros():
    model = _model(slots=4)
    features, times, valid, centers = _inputs(batch=3)
    valid[1] = False
    out = model(features, times, valid, centers)
    assert not bool(out.slot_valid[1].any())
    assert float(out.embeddings[1].abs().max().detach()) == 0.0
    assert float(out.activity_probabilities()[1].abs().max().detach()) == 0.0
    assert bool(out.slot_valid[[0, 2]].all())

    center_flags = torch.tensor([True, False, True])
    out2 = model(features, times, valid, centers, center_flags)
    assert not bool(out2.slot_valid[1].any())
    assert float(out2.embeddings[1].abs().max().detach()) == 0.0
    assert float(out2.activity_probabilities()[1].abs().max().detach()) == 0.0


def test_zero_length_frames_and_empty_batch():
    model = _model(slots=3)
    empty_frames = torch.zeros(2, 0, 8)
    out = model(empty_frames, None, None, None)
    assert out.embeddings.shape == (2, 3, 128)
    assert not bool(out.slot_valid.any())
    assert float(out.embeddings.abs().max().detach()) == 0.0

    empty_batch = model(torch.zeros(0, 4, 8), None, None, None)
    assert empty_batch.embeddings.shape == (0, 3, 128)
    assert empty_batch.activity_logits.shape == (0, 3)


def test_padded_frames_do_not_affect_outputs():
    model = _model(slots=5)
    features, times, valid, centers = _inputs()
    valid[:, -1] = False
    baseline = model(features, times, valid, centers)
    altered = features.clone()
    altered[:, -1] = 123.0
    changed = model(altered, times, valid, centers)
    torch.testing.assert_close(baseline.embeddings, changed.embeddings)
    torch.testing.assert_close(baseline.activity_logits, changed.activity_logits)


def test_valid_mask_and_perturbed_valid_frames_change_outputs():
    model = _model(slots=4)
    features, times, valid, centers = _inputs()
    baseline = model(features, times, valid, centers)

    masked = valid.clone()
    masked[0, 0] = False
    after_mask = model(features, times, masked, centers)
    assert not torch.allclose(baseline.embeddings, after_mask.embeddings)

    perturbed = features.clone()
    perturbed[0, 0] += 10.0
    after_perturb = model(perturbed, times, valid, centers)
    assert not torch.allclose(baseline.embeddings, after_perturb.embeddings)


def test_time_positions_drive_center_logits_but_not_identity_vectors():
    model = _model(slots=4)
    features, times, valid, centers = _inputs()

    baseline = model(features, times, valid, centers)
    shifted_frames = times + 0.02  # center_times stay put: relative dt changes
    after = model(features, shifted_frames, valid, centers)
    assert not torch.allclose(baseline.activity_logits, after.activity_logits)
    # Identity attention is content-only by design: E must not drift with time.
    torch.testing.assert_close(baseline.embeddings, after.embeddings, rtol=1e-5, atol=1e-5)

    shifted_centers = centers + 0.02
    after_centers = model(features, times, valid, shifted_centers)
    assert not torch.allclose(baseline.activity_logits, after_centers.activity_logits)


def test_low_activity_still_has_unit_embeddings():
    model = _model(slots=3)
    with torch.no_grad():
        model.activity_head[-1].bias.fill_(-30.0)
    features, times, valid, centers = _inputs()
    out = model(features, times, valid, centers)
    assert float(out.activity_probabilities().max().detach()) < 1e-6
    norms = out.embeddings[out.slot_valid].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_all_masked_batch_backward_is_finite():
    model = _model(slots=4)
    features = torch.randn(2, 5, 8, requires_grad=True)
    valid = torch.zeros(2, 5, dtype=torch.bool)
    out = model(features, None, valid, None)
    assert not bool(out.slot_valid.any())
    loss = out.embeddings.sum() + out.activity_logits.sum()
    loss.backward()
    assert features.grad is not None
    assert bool(torch.isfinite(features.grad).all())
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert bool(torch.isfinite(parameter.grad).all())


def test_head_argument_validation():
    with pytest.raises(ValueError):
        SourceQueryHead(8, embedding_dim=64)
    with pytest.raises(ValueError):
        SourceQueryHead(0)
    with pytest.raises(ValueError):
        SourceQueryHead(8, slots=0)

    model = _model()
    features, times, valid, centers = _inputs()
    with pytest.raises(ValueError):
        model(features, times, valid.to(torch.float32), centers)  # wrong mask dtype/shape kept as bool cast
    with pytest.raises(ValueError):
        model(torch.randn(5, 5, 7), times, valid, centers)
    with pytest.raises(ValueError):
        model(torch.full((5, 5, 8), float("nan")), times, valid, centers)
    with pytest.raises(ValueError):
        model(features, times, valid[:, :4], centers)
    with pytest.raises(ValueError):
        model(features, times, valid, centers[:3])
