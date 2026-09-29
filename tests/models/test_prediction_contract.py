"""Head output -> shared PredictionData contract mapping tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np
import torch

from aat.contracts import PredictionData, UNIT_NORM_TOLERANCE
from aat.models import SourceQueryHead, head_output_to_prediction_data

pytestmark = pytest.mark.ml


def _head_output():
    torch.manual_seed(0)
    model = SourceQueryHead(8, slots=8, d_model=32, num_heads=4)
    model.eval()
    features = torch.randn(3, 5, 8)
    offsets = torch.linspace(-0.08, 0.08, 5)
    center_times = torch.tensor([0.0, 0.5, 1.0])
    frame_times = center_times[:, None] + offsets[None, :]
    valid = torch.ones(3, 5, dtype=torch.bool)
    valid[1] = False  # fully padded window: invalid slots, zero E/P
    center_valid = torch.tensor([True, False, True])
    return model(features, frame_times, valid, center_times, center_valid), center_times


def test_head_output_maps_to_prediction_data_and_roundtrips(tmp_path):
    output, center_times = _head_output()
    data = head_output_to_prediction_data(
        output, center_times.numpy(), hop_seconds=0.02, sample_id="fake-features"
    )
    assert isinstance(data, PredictionData)
    assert data.slots == 8
    assert data.embedding_dim == 128
    assert not data.slot_valid[1].any()
    np.testing.assert_array_equal(data.embeddings[1], np.zeros((8, 128), dtype=np.float32))
    np.testing.assert_array_equal(data.activity[1], np.zeros(8, dtype=np.float32))

    valid_rows = data.slot_valid
    norms = np.linalg.norm(data.embeddings[valid_rows], axis=1)
    assert np.all(np.abs(norms - 1.0) <= UNIT_NORM_TOLERANCE)
    assert np.all((data.activity >= 0.0) & (data.activity <= 1.0))

    metadata_path, arrays_path = data.save(tmp_path)
    assert metadata_path.name == "prediction.json"
    loaded = PredictionData.load(tmp_path)
    np.testing.assert_allclose(loaded.embeddings, data.embeddings, atol=1e-6)
    np.testing.assert_allclose(loaded.activity, data.activity, atol=1e-6)
    np.testing.assert_array_equal(loaded.slot_valid, data.slot_valid)
    np.testing.assert_array_equal(loaded.center_valid, data.center_valid)


def test_prediction_mapping_rejects_wrong_center_times():
    output, _ = _head_output()
    with pytest.raises(ValueError):
        head_output_to_prediction_data(output, np.array([0.0, 0.5]))
