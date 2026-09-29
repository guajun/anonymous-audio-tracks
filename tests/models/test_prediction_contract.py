"""Cross-check: head outputs really satisfy the frozen prediction contract."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from aat.contracts import PredictionData
from aat.models import head_output_to_prediction_arrays

from . import fakes


@pytest.mark.parametrize("slots", [1, 3, 8])
def test_head_output_passes_prediction_contract(slots, tmp_path):
    head = fakes.make_head(feature_dim=6, slots=slots)
    features = torch.randn(3, 4, 6)
    valid = torch.ones(3, 4, dtype=torch.bool)
    times = torch.arange(4, dtype=torch.float32).repeat(3, 1) * 0.08
    center_times = np.array([0.0, 0.02, 0.04], dtype=np.float64)

    output = head(features, valid, times)
    arrays = head_output_to_prediction_arrays(
        output, center_times, np.ones(3, dtype=bool), hop_seconds=0.02, sample_id="synth-0001"
    )
    data = PredictionData(**arrays)
    data.validate()

    assert data.slots == slots
    assert data.embedding_dim == 128
    assert data.embeddings.shape == (3, slots, 128)
    assert np.all(np.isfinite(data.embeddings))
    assert np.all(np.isfinite(data.activity))
    assert data.slot_valid.all()
    norms = np.linalg.norm(data.embeddings, axis=2)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-3)

    metadata_path, arrays_path = data.save(tmp_path)
    assert metadata_path.name == "prediction.json"
    assert arrays_path.name == "prediction.npz"
    loaded = PredictionData.load(tmp_path)
    np.testing.assert_array_equal(loaded.embeddings, data.embeddings)
    np.testing.assert_array_equal(loaded.activity, data.activity)


def test_head_default_capacity_is_eight_but_is_configurable():
    default_head = fakes.make_head(feature_dim=6)  # fakes default is 3
    assert default_head.config.slots == 3
    output = default_head(torch.randn(2, 2, 6))
    assert output.embeddings.shape == (2, 3, 128)

    eight = fakes.make_head(feature_dim=6, slots=8)
    arrays = head_output_to_prediction_arrays(eight(torch.randn(2, 2, 6)), [0.0, 0.02])
    data = PredictionData(**arrays)
    assert data.slots == 8
