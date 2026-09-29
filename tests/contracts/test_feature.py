"""Feature contract tests (frame times + valid mask + feature matrix)."""

from __future__ import annotations

import numpy as np
import pytest

from aat.contracts import ContractError, FeatureData, dump_json, load_json

from . import fixtures


def test_feature_save_load_roundtrip(tmp_path):
    data = FeatureData(**fixtures.feature_arrays())
    metadata_path, arrays_path = data.save(tmp_path)
    assert metadata_path.name == "feature.json"
    assert arrays_path.name == "feature.npz"

    loaded = FeatureData.load(tmp_path)
    np.testing.assert_array_equal(loaded.frame_times, data.frame_times)
    np.testing.assert_array_equal(loaded.features, data.features)
    np.testing.assert_array_equal(loaded.valid, data.valid)
    assert loaded.feature_name == "synth-test-feature"
    assert loaded.feature_dim == 4
    assert loaded.frame_origin_seconds == 0.0
    assert loaded.hop_seconds == pytest.approx(0.04)


def test_feature_npz_is_a_plain_numpy_archive(tmp_path):
    FeatureData(**fixtures.feature_arrays()).save(tmp_path)
    with np.load(tmp_path / "feature.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"frame_times", "features", "valid"}


def test_feature_metadata_keeps_frame_grid_and_origin(tmp_path):
    FeatureData(**fixtures.feature_arrays()).save(tmp_path)
    metadata = load_json(tmp_path / "feature.json")
    assert metadata["kind"] == "feature"
    assert metadata["frame_origin_seconds"] == 0.0
    assert metadata["hop_seconds"] == pytest.approx(0.04)
    assert metadata["arrays"]["frame_times"]["origin"] == "original_track_start"
    assert metadata["arrays"]["frame_times"]["unit"] == "seconds"
    assert metadata["arrays"]["features"]["shape"] == [5, 4]
    assert metadata["arrays"]["valid"]["true_means"].startswith("frame was computed")


def test_feature_empty_frame_axis_is_allowed():
    data = FeatureData(
        frame_times=np.empty(0, dtype=np.float64),
        features=np.zeros((0, 4), dtype=np.float32),
        valid=np.empty(0, dtype=bool),
        feature_name="empty-feature",
        sample_rate=16000,
        frame_origin_seconds=0.0,
        hop_seconds=0.08,
        feature_dim=4,
    )
    assert data.features.shape == (0, 4)


def test_feature_rejects_nan_values():
    arrays = fixtures.feature_arrays()
    arrays["features"] = arrays["features"].copy()
    arrays["features"][2, 1] = np.nan
    with pytest.raises(ContractError, match="NaN"):
        FeatureData(**arrays)


def test_feature_rejects_non_increasing_frame_times():
    arrays = fixtures.feature_arrays()
    arrays["frame_times"] = np.array([0.0, 0.04, 0.04, 0.12, 0.16])
    with pytest.raises(ContractError, match="strictly increasing"):
        FeatureData(**arrays)


def test_feature_rejects_dimension_mismatch():
    arrays = fixtures.feature_arrays()
    arrays["feature_dim"] = 8
    with pytest.raises(ContractError, match="feature_dim"):
        FeatureData(**arrays)


def test_feature_rejects_row_length_mismatch():
    arrays = fixtures.feature_arrays()
    arrays["valid"] = np.array([True, True])
    with pytest.raises(ContractError, match="frame_times length"):
        FeatureData(**arrays)


def test_feature_requires_boolean_valid_mask():
    arrays = fixtures.feature_arrays()
    arrays["valid"] = np.ones(5, dtype=np.int8)
    with pytest.raises(ContractError, match="boolean"):
        FeatureData(**arrays)


def test_feature_requires_positive_hop_and_origin():
    arrays = fixtures.feature_arrays()
    arrays["hop_seconds"] = 0.0
    with pytest.raises(ContractError):
        FeatureData(**arrays)

    arrays = fixtures.feature_arrays()
    arrays["frame_origin_seconds"] = -0.1
    with pytest.raises(ContractError):
        FeatureData(**arrays)


def test_feature_feature_dimension_stays_free():
    # Only the prediction embedding dimension is fixed at 128; features may use
    # any extractor dimension declared by ``feature_dim``.
    arrays = fixtures.feature_arrays()
    arrays["features"] = np.arange(5 * 16, dtype=np.float32).reshape(5, 16)
    arrays["feature_dim"] = 16
    assert FeatureData(**arrays).feature_dim == 16


# --------------------------------------------------------------------------- #
# sidecar must match the protocol itself
# --------------------------------------------------------------------------- #


def _tamper(metadata_path, mutate):
    metadata = load_json(metadata_path)
    mutate(metadata)
    dump_json(metadata_path, metadata)


def test_feature_load_rejects_wrong_declared_unit(tmp_path):
    FeatureData(**fixtures.feature_arrays()).save(tmp_path)
    _tamper(tmp_path / "feature.json", lambda meta: meta["arrays"]["frame_times"].update(unit="milliseconds"))
    with pytest.raises(ContractError, match="unit"):
        FeatureData.load(tmp_path)


def test_feature_load_rejects_wrong_time_origin(tmp_path):
    FeatureData(**fixtures.feature_arrays()).save(tmp_path)
    _tamper(tmp_path / "feature.json", lambda meta: meta["arrays"]["frame_times"].update(origin="clip_start"))
    with pytest.raises(ContractError, match="origin"):
        FeatureData.load(tmp_path)


def test_feature_load_rejects_wrong_declared_dtype(tmp_path):
    FeatureData(**fixtures.feature_arrays()).save(tmp_path)
    _tamper(tmp_path / "feature.json", lambda meta: meta["arrays"]["features"].update(dtype="float64"))
    with pytest.raises(ContractError, match="dtype"):
        FeatureData.load(tmp_path)


def test_feature_load_requires_arrays_path(tmp_path):
    FeatureData(**fixtures.feature_arrays()).save(tmp_path)
    _tamper(tmp_path / "feature.json", lambda meta: meta.pop("arrays_path"))
    with pytest.raises(ContractError, match="arrays_path"):
        FeatureData.load(tmp_path)


# --------------------------------------------------------------------------- #
# mutable state must be revalidated on save
# --------------------------------------------------------------------------- #


def test_feature_save_revalidates_mutated_features(tmp_path):
    data = FeatureData(**fixtures.feature_arrays())
    data.features[0, 0] = np.nan
    with pytest.raises(ContractError, match="NaN"):
        data.save(tmp_path)


def test_feature_save_revalidates_mutated_frame_times(tmp_path):
    data = FeatureData(**fixtures.feature_arrays())
    data.frame_times[1] = data.frame_times[0]
    with pytest.raises(ContractError, match="strictly increasing"):
        data.save(tmp_path)


def test_feature_save_rejects_nan_preprocessing(tmp_path):
    data = FeatureData(**fixtures.feature_arrays())
    data.preprocessing = {"mean": float("nan")}
    with pytest.raises(ContractError, match="JSON-serialisable"):
        data.save(tmp_path)
