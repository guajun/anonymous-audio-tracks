"""Cross-document protocol tests: versions, kinds, defaults and file layout."""

from __future__ import annotations

import pytest

from aat.contracts import (
    ActivityData,
    ContractError,
    Controls,
    FeatureData,
    PredictionData,
    SampleManifest,
    SchemaVersionError,
    SourceRegistry,
    Trajectory,
    check_schema_header,
    check_schema_version,
)
from aat.contracts.version import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_HOP_SECONDS,
    DEFAULT_SLOTS,
    DEFAULT_WINDOW_SECONDS,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    UNIT_NORM_TOLERANCE,
)

from . import fixtures


def test_protocol_defaults_are_frozen():
    assert SCHEMA_VERSION == "0.1.0"
    assert SUPPORTED_SCHEMA_VERSIONS == ("0.1.0",)
    assert DEFAULT_SLOTS == 8
    assert DEFAULT_EMBEDDING_DIM == 128
    assert DEFAULT_WINDOW_SECONDS == 2.0
    assert DEFAULT_HOP_SECONDS == 0.02
    assert UNIT_NORM_TOLERANCE == 1e-3


@pytest.mark.parametrize(
    ("document", "kind"),
    [
        (fixtures.manifest_dict(), "sample_manifest"),
        (fixtures.sources_dict(), "sources"),
        (fixtures.controls_dict(), "controls"),
        (fixtures.trajectory_dict(), "trajectory"),
    ],
)
def test_json_documents_carry_version_and_kind(document, kind):
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["kind"] == kind
    check_schema_header(document, kind)


@pytest.mark.parametrize(
    ("data_factory", "loader"),
    [
        (fixtures.manifest_dict, SampleManifest.from_json_dict),
        (fixtures.sources_dict, SourceRegistry.from_json_dict),
        (fixtures.controls_dict, Controls.from_json_dict),
        (fixtures.trajectory_dict, Trajectory.from_json_dict),
    ],
)
def test_json_documents_reject_unknown_schema_version(data_factory, loader):
    data = data_factory()
    data["schema_version"] = "999.0"
    with pytest.raises(SchemaVersionError):
        loader(data)


def test_check_schema_version_rejects_unknown_value():
    with pytest.raises(SchemaVersionError):
        check_schema_version("2.0")


def test_wrong_kind_is_rejected():
    data = fixtures.manifest_dict()
    data["kind"] = "controls"
    with pytest.raises(ContractError, match="kind"):
        SampleManifest.from_json_dict(data)


def test_array_documents_use_documented_sidecar_names(tmp_path):
    ActivityData(**fixtures.activity_arrays()).save(tmp_path)
    FeatureData(**fixtures.feature_arrays()).save(tmp_path)
    PredictionData(**fixtures.prediction_arrays()).save(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "activity.json",
        "activity.npz",
        "feature.json",
        "feature.npz",
        "prediction.json",
        "prediction.npz",
    ]


def test_activity_columns_follow_sources_json_order():
    registry = SourceRegistry.from_json_dict(fixtures.sources_dict())
    activity = ActivityData(**fixtures.activity_arrays())
    assert activity.source_ids == registry.source_ids


def test_array_metadata_paths_are_relative_to_their_document():
    activity_metadata = ActivityData(**fixtures.activity_arrays()).to_metadata()
    feature_metadata = FeatureData(**fixtures.feature_arrays()).to_metadata()
    prediction_metadata = PredictionData(**fixtures.prediction_arrays()).to_metadata()
    assert activity_metadata["arrays_path"] == "activity.npz"
    assert feature_metadata["arrays_path"] == "feature.npz"
    assert prediction_metadata["arrays_path"] == "prediction.npz"
