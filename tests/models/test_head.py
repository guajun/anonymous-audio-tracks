"""Head shape, mask, time-position and protocol-conversion tests.

Only synthetic features are used; no audio or backbone is loaded.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np
import torch.nn.functional as F

from aat.models import (
    HeadConfig,
    HeadOutput,
    MultiSourceHead,
    head_output_to_prediction_arrays,
)

from . import fakes


@pytest.mark.parametrize("slots", [1, 2, 8])
def test_head_output_shapes_ranges_and_unit_norms(slots):
    head = fakes.make_head(slots=slots, feature_dim=6)
    features = torch.randn(3, 4, 6)
    valid = torch.ones(3, 4, dtype=torch.bool)
    times = torch.arange(4, dtype=torch.float32).repeat(3, 1) * 0.08

    output = head(features, valid, times)

    assert output.embeddings.shape == (3, slots, 128)
    assert output.activity_logits.shape == (3, slots)
    assert output.activity.shape == (3, slots)
    assert output.slot_valid.shape == (3, slots)
    assert output.slot_valid.all()
    norms = output.embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert torch.isfinite(output.embeddings).all()
    assert torch.isfinite(output.activity_logits).all()
    assert bool((output.activity >= 0.0).all()) and bool((output.activity <= 1.0).all())


def test_head_requires_protocol_embedding_dim():
    with pytest.raises(ValueError, match="fixes embedding_dim at 128"):
        HeadConfig(feature_dim=4, embedding_dim=64)


@pytest.mark.parametrize("slots", [0, -1])
def test_head_rejects_non_positive_slots(slots):
    with pytest.raises(ValueError, match="slots must be >= 1"):
        HeadConfig(feature_dim=4, slots=slots)


def test_head_rejects_wrong_feature_dim():
    head = fakes.make_head(feature_dim=6)
    with pytest.raises(ValueError, match="does not match feature_dim"):
        head(torch.randn(2, 3, 5))


def test_head_rejects_wrong_valid_shape():
    head = fakes.make_head()
    with pytest.raises(ValueError, match="feature_valid"):
        head(torch.randn(2, 3, 8), torch.ones(2, 4, dtype=torch.bool))


def test_masked_frames_do_not_affect_output():
    head = fakes.make_head()
    features = torch.randn(1, 4, 8)
    valid = torch.tensor([[True, False, True, True]])
    times = torch.arange(4, dtype=torch.float32).unsqueeze(0)

    reference = head(features, valid, times)

    changed_features = features.clone()
    changed_features[0, 1] = 123.0
    changed_times = times.clone()
    changed_times[0, 1] = 999.0
    altered = head(changed_features, valid, changed_times)

    torch.testing.assert_close(reference.embeddings, altered.embeddings)
    torch.testing.assert_close(reference.activity_logits, altered.activity_logits)


def test_all_invalid_frames_still_produce_unit_finite_vectors():
    head = fakes.make_head()
    features = torch.randn(2, 3, 8)
    valid = torch.zeros(2, 3, dtype=torch.bool)

    output = head(features, valid)

    assert torch.isfinite(output.embeddings).all()
    assert torch.isfinite(output.activity_logits).all()
    norms = output.embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_time_positions_change_the_output():
    head = fakes.make_head()
    features = torch.randn(1, 3, 8)
    valid = torch.ones(1, 3, dtype=torch.bool)
    times = torch.tensor([[0.0, 0.08, 0.16]])

    first = head(features, valid, times)
    second = head(features, valid, times + 0.051)

    assert not torch.allclose(first.embeddings, second.embeddings)


def test_head_empty_batch_keeps_declared_shapes():
    head = fakes.make_head(slots=4)
    output = head(torch.zeros(0, 3, 8))
    assert output.embeddings.shape == (0, 4, 128)
    assert output.activity_logits.shape == (0, 4)
    assert output.slot_valid.shape == (0, 4)


def test_head_output_to_prediction_arrays_zeroes_invalid_slots():
    raw = torch.randn(2, 2, 128)
    embeddings = F.normalize(raw, dim=-1)
    logits = torch.tensor([[-3.0, 3.0], [-1.0, 2.0]])
    activity = torch.sigmoid(logits)
    output = HeadOutput(
        embeddings=embeddings,
        activity_logits=logits,
        activity=activity,
        slot_valid=torch.tensor([[True, False], [True, True]]),
    )

    arrays = head_output_to_prediction_arrays(
        output, np.array([0.0, 0.02]), np.ones(2, dtype=bool), hop_seconds=0.02
    )

    assert arrays["center_times"].dtype == np.float64
    assert arrays["embeddings"].dtype == np.float32
    assert arrays["activity"].dtype == np.float32
    np.testing.assert_array_equal(arrays["embeddings"][0, 1], np.zeros(128, dtype=np.float32))
    assert arrays["activity"][0, 1] == 0.0
    # Valid slots keep the unit identity vector.
    assert np.linalg.norm(arrays["embeddings"][0, 0]) == pytest.approx(1.0, abs=1e-5)


def test_head_output_to_prediction_arrays_validates_center_valid_length():
    head = fakes.make_head()
    output = head(torch.randn(2, 3, 8))
    with pytest.raises(ValueError, match="center_valid"):
        head_output_to_prediction_arrays(output, [0.0, 0.02], [True])


def test_sinusoidal_time_encoding_shape_and_finiteness():
    from aat.models import SinusoidalTimeEncoding

    encoding = SinusoidalTimeEncoding(16)
    times = torch.linspace(0.0, 10.0, 5)
    encoded = encoding(times)
    assert encoded.shape == (5, 16)
    assert torch.isfinite(encoded).all()
    assert bool((encoded.abs() <= 1.0 + 1e-6).all())


def test_head_config_is_required_to_match_feature_dim():
    config = HeadConfig(feature_dim=6)
    assert config.embedding_dim == 128
    head = MultiSourceHead(config)
    output = head(torch.randn(2, 3, 6))
    assert output.embeddings.shape == (2, config.slots, 128)
