"""Prediction contract tests: E[N, K, 128], P[N, K] and slot semantics."""

from __future__ import annotations

import numpy as np
import pytest

from aat.contracts import (
    ContractError,
    PredictionData,
    RunProvenance,
    SchemaVersionError,
    load_json,
)

from . import fixtures


def test_prediction_default_shapes_roundtrip(tmp_path):
    data = PredictionData(**fixtures.prediction_arrays())
    assert data.slots == 8
    assert data.embedding_dim == 128
    assert data.embeddings.shape == (3, 8, 128)
    assert data.activity.shape == (3, 8)

    metadata_path, arrays_path = data.save(tmp_path)
    assert metadata_path.name == "prediction.json"
    assert arrays_path.name == "prediction.npz"

    loaded = PredictionData.load(tmp_path)
    assert loaded.slots == 8
    assert loaded.embedding_dim == 128
    np.testing.assert_array_equal(loaded.center_times, data.center_times)
    np.testing.assert_array_equal(loaded.embeddings, data.embeddings)
    np.testing.assert_array_equal(loaded.activity, data.activity)
    np.testing.assert_array_equal(loaded.slot_valid, data.slot_valid)
    np.testing.assert_array_equal(loaded.center_valid, data.center_valid)


def test_prediction_npz_is_a_plain_numpy_archive(tmp_path):
    PredictionData(**fixtures.prediction_arrays()).save(tmp_path)
    with np.load(tmp_path / "prediction.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "center_times",
            "embeddings",
            "activity",
            "slot_valid",
            "center_valid",
        }


def test_prediction_slots_and_embedding_dim_are_configurable(tmp_path):
    data = PredictionData(**fixtures.prediction_arrays(n_windows=4, slots=3, embedding_dim=16))
    assert data.slots == 3
    assert data.embedding_dim == 16
    assert data.embeddings.shape == (4, 3, 16)
    data.save(tmp_path)
    loaded = PredictionData.load(tmp_path)
    assert loaded.slots == 3
    assert loaded.embedding_dim == 16
    assert loaded.activity.shape == (4, 3)


def test_prediction_center_times_describe_window_centers(tmp_path):
    data = PredictionData(**fixtures.prediction_arrays())
    np.testing.assert_allclose(data.center_times, [0.0, 0.02, 0.04])
    assert data.center_valid.dtype == np.bool_
    assert data.center_valid[0] == np.bool_(False)
    data.save(tmp_path)
    metadata = load_json(tmp_path / "prediction.json")
    assert metadata["arrays"]["center_times"]["origin"] == "original_track_start"
    assert metadata["arrays"]["center_times"]["unit"] == "seconds"
    assert metadata["arrays"]["embeddings"]["shape"] == [3, 8, 128]
    assert metadata["arrays"]["activity"]["range"] == [0.0, 1.0]


def test_prediction_metadata_records_valid_slot_semantics(tmp_path):
    PredictionData(**fixtures.prediction_arrays()).save(tmp_path)
    metadata = load_json(tmp_path / "prediction.json")
    assert metadata["kind"] == "prediction"
    assert metadata["slots"] == 8
    assert metadata["embedding_dim"] == 128
    assert metadata["arrays"]["embeddings"]["dtype"] == "float32"
    assert metadata["arrays"]["embeddings"]["unit"] == "l2_normalized"
    assert metadata["arrays"]["slot_valid"]["true_means"].startswith("slot holds a usable")


def test_prediction_valid_slots_must_be_unit_norm():
    arrays = fixtures.prediction_arrays()
    arrays["embeddings"] = arrays["embeddings"].copy()
    arrays["embeddings"][0, 2] = arrays["embeddings"][0, 2] * 2.0
    with pytest.raises(ContractError, match="L2 norm"):
        PredictionData(**arrays)


def test_prediction_inactive_slots_must_be_zero_probability():
    arrays = fixtures.prediction_arrays()
    arrays["activity"] = arrays["activity"].copy()
    arrays["activity"][1, 0] = 0.5  # slot 0 of window 1 is marked invalid
    with pytest.raises(ContractError, match="inactive slots"):
        PredictionData(**arrays)


def test_prediction_inactive_slots_must_be_zero_vectors():
    arrays = fixtures.prediction_arrays()
    arrays["embeddings"] = arrays["embeddings"].copy()
    arrays["embeddings"][1, 0] = arrays["embeddings"][0, 1]
    with pytest.raises(ContractError, match="all-zero"):
        PredictionData(**arrays)


def test_prediction_valid_slot_may_be_silent_with_identity_memory():
    arrays = fixtures.prediction_arrays()
    arrays["activity"] = arrays["activity"].copy()
    arrays["activity"][0, 0] = 0.0  # valid candidate, currently silent
    data = PredictionData(**arrays)
    assert data.slot_valid[0, 0]
    assert data.activity[0, 0] == 0.0
    assert np.linalg.norm(data.embeddings[0, 0]) == pytest.approx(1.0, abs=1e-3)


def test_prediction_rejects_nan_and_infinite_values():
    arrays = fixtures.prediction_arrays()
    arrays["embeddings"] = arrays["embeddings"].copy()
    arrays["embeddings"][0, 0, 0] = np.nan
    with pytest.raises(ContractError, match="NaN"):
        PredictionData(**arrays)

    arrays = fixtures.prediction_arrays()
    arrays["activity"] = arrays["activity"].copy()
    arrays["activity"][0, 0] = np.inf
    with pytest.raises(ContractError, match="NaN"):
        PredictionData(**arrays)


def test_prediction_rejects_probability_out_of_range():
    arrays = fixtures.prediction_arrays()
    arrays["activity"] = arrays["activity"].copy()
    arrays["activity"][0, 0] = 1.2
    with pytest.raises(ContractError, match="\\[0, 1\\]"):
        PredictionData(**arrays)


def test_prediction_rejects_non_increasing_center_times():
    arrays = fixtures.prediction_arrays()
    arrays["center_times"] = np.array([0.0, 0.04, 0.02])
    with pytest.raises(ContractError, match="strictly increasing"):
        PredictionData(**arrays)


def test_prediction_rejects_shape_mismatch():
    arrays = fixtures.prediction_arrays()
    with pytest.raises(ContractError, match="expected shape"):
        PredictionData(**arrays, slots=4)

    arrays = fixtures.prediction_arrays()
    arrays["slot_valid"] = np.ones((3, 7), dtype=bool)
    with pytest.raises(ContractError, match="slot_valid"):
        PredictionData(**arrays)

    arrays = fixtures.prediction_arrays()
    arrays["center_valid"] = np.ones(2, dtype=bool)
    with pytest.raises(ContractError, match="center_valid"):
        PredictionData(**arrays)


def test_prediction_empty_window_axis_is_allowed():
    data = PredictionData(
        center_times=np.empty(0, dtype=np.float64),
        embeddings=np.zeros((0, 8, 128), dtype=np.float32),
        activity=np.zeros((0, 8), dtype=np.float32),
        slot_valid=np.zeros((0, 8), dtype=bool),
        center_valid=np.empty(0, dtype=bool),
    )
    assert data.embeddings.shape == (0, 8, 128)
    assert data.activity.shape == (0, 8)


def test_prediction_accepts_and_serialises_run_provenance(tmp_path):
    data = PredictionData(
        **fixtures.prediction_arrays(), provenance=fixtures.provenance_dict()
    )
    assert isinstance(data.provenance, RunProvenance)
    data.save(tmp_path)
    metadata = load_json(tmp_path / "prediction.json")
    assert metadata["provenance"]["run_id"] == "run-20260929-01"
    loaded = PredictionData.load(tmp_path)
    assert isinstance(loaded.provenance, RunProvenance)


def test_prediction_rejects_unsupported_schema_version(tmp_path):
    data = PredictionData(**fixtures.prediction_arrays())
    data.save(tmp_path)
    metadata = load_json(tmp_path / "prediction.json")
    metadata["schema_version"] = "999.0"
    from aat.contracts import dump_json

    dump_json(tmp_path / "prediction.json", metadata)
    with pytest.raises(SchemaVersionError):
        PredictionData.load(tmp_path)
