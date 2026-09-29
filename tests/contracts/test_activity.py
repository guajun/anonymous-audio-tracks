"""Activity label contract tests (center-time probabilities per raw source)."""

from __future__ import annotations

import numpy as np
import pytest

from aat.contracts import ActivityData, ContractError, dump_json, load_json

from . import fixtures


def test_activity_save_load_roundtrip(tmp_path):
    data = ActivityData(**fixtures.activity_arrays())
    metadata_path, arrays_path = data.save(tmp_path)
    assert metadata_path.name == "activity.json"
    assert arrays_path.name == "activity.npz"

    loaded = ActivityData.load(tmp_path)
    np.testing.assert_array_equal(loaded.center_times, data.center_times)
    np.testing.assert_array_equal(loaded.activity, data.activity)
    np.testing.assert_array_equal(loaded.valid, data.valid)
    assert loaded.source_ids == ("s01", "s02")
    assert loaded.sample_rate == 16000
    assert loaded.center_times.dtype == np.float64
    assert loaded.activity.dtype == np.float32
    assert loaded.valid.dtype == np.bool_


def test_activity_npz_is_a_plain_numpy_archive(tmp_path):
    ActivityData(**fixtures.activity_arrays()).save(tmp_path)
    with np.load(tmp_path / "activity.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"center_times", "activity", "valid"}


def test_activity_metadata_documents_units_and_shapes(tmp_path):
    data = ActivityData(**fixtures.activity_arrays())
    metadata_path, _ = data.save(tmp_path)
    metadata = load_json(metadata_path)
    assert metadata["schema_version"] == "0.1.0"
    assert metadata["kind"] == "activity"
    assert metadata["source_ids"] == ["s01", "s02"]
    assert metadata["arrays"]["center_times"]["unit"] == "seconds"
    assert metadata["arrays"]["center_times"]["origin"] == "original_track_start"
    assert metadata["arrays"]["activity"]["unit"] == "probability"
    assert metadata["arrays"]["activity"]["range"] == [0.0, 1.0]
    assert metadata["arrays"]["valid"]["dtype"] == "bool"
    assert metadata["arrays"]["activity"]["shape"] == [4, 2]


def test_activity_all_zero_rows_are_no_activity():
    arrays = fixtures.activity_arrays()
    arrays["activity"] = np.zeros((4, 2), dtype=np.float32)
    data = ActivityData(**arrays)
    assert not data.activity.any()


def test_activity_empty_time_axis_and_empty_sources_are_allowed():
    no_times = ActivityData(
        center_times=np.empty(0, dtype=np.float64),
        activity=np.zeros((0, 2), dtype=np.float32),
        valid=np.empty(0, dtype=bool),
        source_ids=("s01", "s02"),
        sample_rate=16000,
    )
    assert no_times.activity.shape == (0, 2)

    no_sources = ActivityData(
        center_times=np.array([0.0, 0.02], dtype=np.float64),
        activity=np.zeros((2, 0), dtype=np.float32),
        valid=np.array([True, True]),
        source_ids=(),
        sample_rate=16000,
    )
    assert no_sources.activity.shape == (2, 0)


def test_activity_rejects_nan_and_inf():
    arrays = fixtures.activity_arrays()
    arrays["activity"] = arrays["activity"].copy()
    arrays["activity"][0, 0] = np.nan
    with pytest.raises(ContractError, match="NaN"):
        ActivityData(**arrays)

    arrays = fixtures.activity_arrays()
    arrays["center_times"] = arrays["center_times"].copy()
    arrays["center_times"][1] = np.inf
    with pytest.raises(ContractError, match="NaN or infinite"):
        ActivityData(**arrays)


def test_activity_rejects_probability_out_of_range():
    arrays = fixtures.activity_arrays()
    arrays["activity"] = arrays["activity"].copy()
    arrays["activity"][1, 1] = 1.2
    with pytest.raises(ContractError, match="\\[0, 1\\]"):
        ActivityData(**arrays)


def test_activity_rejects_non_increasing_center_times():
    arrays = fixtures.activity_arrays()
    arrays["center_times"] = np.array([0.0, 0.04, 0.02, 0.06])
    with pytest.raises(ContractError, match="strictly increasing"):
        ActivityData(**arrays)

    arrays = fixtures.activity_arrays()
    arrays["center_times"] = np.array([0.0, 0.02, 0.02, 0.06])
    with pytest.raises(ContractError, match="strictly increasing"):
        ActivityData(**arrays)


def test_activity_rejects_inconsistent_lengths():
    arrays = fixtures.activity_arrays()
    arrays["activity"] = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ContractError, match="leading length"):
        ActivityData(**arrays)

    arrays = fixtures.activity_arrays()
    arrays["activity"] = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ContractError, match="source length"):
        ActivityData(**arrays)

    arrays = fixtures.activity_arrays()
    arrays["valid"] = np.array([True, True, True])
    with pytest.raises(ContractError, match="valid"):
        ActivityData(**arrays)


def test_activity_rejects_duplicate_source_ids():
    arrays = fixtures.activity_arrays()
    arrays["source_ids"] = ("s01", "s01")
    with pytest.raises(ContractError, match="duplicate"):
        ActivityData(**arrays)


def test_activity_requires_boolean_valid_mask():
    arrays = fixtures.activity_arrays()
    arrays["valid"] = np.array([0, 1, 1, 1])
    with pytest.raises(ContractError, match="boolean"):
        ActivityData(**arrays)


# --------------------------------------------------------------------------- #
# sidecar must match the protocol itself, not only the NPZ payload
# --------------------------------------------------------------------------- #


def _tamper(metadata_path, mutate):
    metadata = load_json(metadata_path)
    mutate(metadata)
    dump_json(metadata_path, metadata)


def test_activity_load_rejects_wrong_declared_unit(tmp_path):
    ActivityData(**fixtures.activity_arrays()).save(tmp_path)
    _tamper(tmp_path / "activity.json", lambda meta: meta["arrays"]["center_times"].update(unit="milliseconds"))
    with pytest.raises(ContractError, match="unit"):
        ActivityData.load(tmp_path)


def test_activity_load_rejects_wrong_time_origin(tmp_path):
    ActivityData(**fixtures.activity_arrays()).save(tmp_path)
    _tamper(tmp_path / "activity.json", lambda meta: meta["arrays"]["center_times"].update(origin="clip_start"))
    with pytest.raises(ContractError, match="origin"):
        ActivityData.load(tmp_path)


def test_activity_load_rejects_wrong_declared_dtype(tmp_path):
    ActivityData(**fixtures.activity_arrays()).save(tmp_path)
    _tamper(tmp_path / "activity.json", lambda meta: meta["arrays"]["activity"].update(dtype="float64"))
    with pytest.raises(ContractError, match="dtype"):
        ActivityData.load(tmp_path)


def test_activity_load_requires_arrays_path(tmp_path):
    ActivityData(**fixtures.activity_arrays()).save(tmp_path)
    _tamper(tmp_path / "activity.json", lambda meta: meta.pop("arrays_path"))
    with pytest.raises(ContractError, match="arrays_path"):
        ActivityData.load(tmp_path)


def test_activity_load_rejects_declared_shape_mismatch(tmp_path):
    ActivityData(**fixtures.activity_arrays()).save(tmp_path)
    _tamper(tmp_path / "activity.json", lambda meta: meta["arrays"]["activity"].update(shape=[99, 2]))
    with pytest.raises(ContractError, match="shape"):
        ActivityData.load(tmp_path)


def test_activity_load_rejects_unexpected_npz_keys(tmp_path):
    ActivityData(**fixtures.activity_arrays()).save(tmp_path)
    arrays_path = tmp_path / "activity.npz"
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["extra"] = np.zeros(1, dtype=np.float32)
    np.savez(arrays_path, **arrays)
    with pytest.raises(ContractError, match="keys mismatch"):
        ActivityData.load(tmp_path)


# --------------------------------------------------------------------------- #
# mutable state must be revalidated on save
# --------------------------------------------------------------------------- #


def test_activity_save_revalidates_mutated_probability(tmp_path):
    data = ActivityData(**fixtures.activity_arrays())
    data.activity[0, 0] = np.nan
    with pytest.raises(ContractError, match="NaN"):
        data.save(tmp_path)


def test_activity_save_revalidates_mutated_times(tmp_path):
    data = ActivityData(**fixtures.activity_arrays())
    data.center_times[1] = data.center_times[0]
    with pytest.raises(ContractError, match="strictly increasing"):
        data.save(tmp_path)


def test_activity_save_revalidates_mutated_source_ids(tmp_path):
    data = ActivityData(**fixtures.activity_arrays())
    data.source_ids = ("s01", "s01")
    with pytest.raises(ContractError, match="duplicate"):
        data.save(tmp_path)


def test_activity_save_rejects_nan_label_params(tmp_path):
    data = ActivityData(**fixtures.activity_arrays())
    data.label_params = {"threshold": float("nan")}
    with pytest.raises(ContractError, match="JSON-serialisable"):
        data.save(tmp_path)
