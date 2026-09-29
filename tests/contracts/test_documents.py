"""JSON document contract tests: manifest, sources, controls, trajectory."""

from __future__ import annotations

import copy

import pytest

from aat.contracts import (
    ContractError,
    Controls,
    RunProvenance,
    SampleManifest,
    SchemaVersionError,
    SourceRegistry,
    Track,
    Trajectory,
)

from . import fixtures


def test_manifest_roundtrip_through_file(tmp_path):
    manifest = SampleManifest.from_json_dict(fixtures.manifest_dict())
    path = manifest.save(tmp_path / "manifest.json")
    assert path.name == "manifest.json"
    assert SampleManifest.load(path) == manifest


def test_manifest_requires_content_hash_for_every_artifact():
    data = fixtures.manifest_dict()
    del data["content_sha256"]["stems/s02.wav"]
    with pytest.raises(ContractError, match="content_sha256"):
        SampleManifest.from_json_dict(data)


def test_manifest_rejects_bad_digest():
    data = fixtures.manifest_dict()
    data["content_sha256"]["mix.wav"] = "not-a-digest"
    with pytest.raises(ContractError, match="sha256"):
        SampleManifest.from_json_dict(data)


def test_manifest_rejects_absolute_and_backslash_paths():
    data = fixtures.manifest_dict()
    data["mix_path"] = "C:/audio/mix.wav"
    with pytest.raises(ContractError, match="relative"):
        SampleManifest.from_json_dict(data)

    data = fixtures.manifest_dict()
    data["stem_paths"]["s01"] = "stems\\s01.wav"
    with pytest.raises(ContractError, match="separators"):
        SampleManifest.from_json_dict(data)


def test_manifest_rejects_bad_rate_duration_and_seed():
    data = fixtures.manifest_dict()
    data["sample_rate"] = 0
    with pytest.raises(ContractError):
        SampleManifest.from_json_dict(data)

    data = fixtures.manifest_dict()
    data["duration_seconds"] = 0.0
    with pytest.raises(ContractError):
        SampleManifest.from_json_dict(data)

    data = fixtures.manifest_dict()
    data["seed"] = -1
    with pytest.raises(ContractError):
        SampleManifest.from_json_dict(data)


def test_manifest_rejects_nan_duration():
    data = fixtures.manifest_dict()
    data["duration_seconds"] = float("nan")
    with pytest.raises(ContractError):
        SampleManifest.from_json_dict(data)


def test_manifest_requires_split_groups():
    data = fixtures.manifest_dict()
    del data["groups"]["preset"]
    with pytest.raises(ContractError, match="groups"):
        SampleManifest.from_json_dict(data)


def test_manifest_rejects_unsupported_schema_version():
    data = fixtures.manifest_dict()
    data["schema_version"] = "999.0"
    with pytest.raises(SchemaVersionError):
        SampleManifest.from_json_dict(data)


def test_sources_roundtrip_and_column_order(tmp_path):
    registry = SourceRegistry.from_json_dict(fixtures.sources_dict())
    assert registry.source_ids == ("s01", "s02")
    path = registry.save(tmp_path / "sources.json")
    assert SourceRegistry.load(path) == registry


def test_sources_rejects_duplicate_ids():
    data = fixtures.sources_dict()
    data["sources"][1]["source_id"] = "s01"
    with pytest.raises(ContractError, match="duplicate"):
        SourceRegistry.from_json_dict(data)


def test_sources_index_must_match_position():
    data = fixtures.sources_dict()
    data["sources"][1]["index"] = 3
    with pytest.raises(ContractError, match="position"):
        SourceRegistry.from_json_dict(data)


def test_sources_allows_empty_registry():
    data = {"schema_version": "0.1.0", "kind": "sources", "sources": []}
    assert SourceRegistry.from_json_dict(data).source_ids == ()


def test_controls_roundtrip_allows_simultaneous_events(tmp_path):
    controls = Controls.from_json_dict(fixtures.controls_dict())
    assert len(controls.events) == 4
    path = controls.save(tmp_path / "controls.json")
    assert Controls.load(path) == controls


def test_controls_rejects_decreasing_times():
    data = fixtures.controls_dict()
    data["events"][2]["time_seconds"] = 0.1  # after 0.25
    with pytest.raises(ContractError, match="non-decreasing"):
        Controls.from_json_dict(data)


def test_controls_rejects_negative_or_nan_time():
    data = fixtures.controls_dict()
    data["events"][0]["time_seconds"] = -0.1
    with pytest.raises(ContractError):
        Controls.from_json_dict(data)

    data = fixtures.controls_dict()
    data["events"][0]["time_seconds"] = float("nan")
    with pytest.raises(ContractError):
        Controls.from_json_dict(data)


def test_controls_validate_source_references():
    controls = Controls.from_json_dict(fixtures.controls_dict())
    controls.validate_source_references(("s01", "s02"))
    with pytest.raises(ContractError, match="s02"):
        controls.validate_source_references(("s01",))


def test_controls_empty_events_allowed():
    data = {"schema_version": "0.1.0", "kind": "controls", "events": []}
    assert Controls.from_json_dict(data).events == ()


def test_trajectory_roundtrip_through_file(tmp_path):
    trajectory = Trajectory.from_json_dict(fixtures.trajectory_dict())
    assert [track.track_id for track in trajectory.tracks] == ["trk-0001", "trk-0002"]
    path = trajectory.save(tmp_path / "trajectory.json")
    assert Trajectory.load(path) == trajectory


def test_trajectory_rejects_duplicate_track_ids():
    data = fixtures.trajectory_dict()
    data["tracks"][1]["track_id"] = "trk-0001"
    with pytest.raises(ContractError, match="duplicate"):
        Trajectory.from_json_dict(data)


def test_trajectory_rejects_zero_length_curve():
    data = fixtures.trajectory_dict()
    data["tracks"][0]["center_times"] = []
    data["tracks"][0]["activity"] = []
    with pytest.raises(ContractError, match="empty curve"):
        Trajectory.from_json_dict(data)


def test_trajectory_rejects_non_increasing_times():
    data = fixtures.trajectory_dict()
    data["tracks"][0]["center_times"] = [0.0, 0.04, 0.02]
    with pytest.raises(ContractError, match="strictly increasing"):
        Trajectory.from_json_dict(data)


def test_trajectory_rejects_invalid_probability():
    data = fixtures.trajectory_dict()
    data["tracks"][0]["activity"][1] = 1.5
    with pytest.raises(ContractError, match="must be <= 1"):
        Trajectory.from_json_dict(data)


def test_trajectory_rejects_inconsistent_lengths():
    data = fixtures.trajectory_dict()
    data["tracks"][0]["activity"] = [0.0, 0.8]
    with pytest.raises(ContractError, match="length"):
        Trajectory.from_json_dict(data)

    data = fixtures.trajectory_dict()
    data["tracks"][0]["confidence"] = [0.5, 0.9]
    with pytest.raises(ContractError, match="confidence"):
        Trajectory.from_json_dict(data)


def test_trajectory_requires_run_provenance():
    data = fixtures.trajectory_dict()
    del data["provenance"]["run_id"]
    with pytest.raises(ContractError, match="run_id"):
        Trajectory.from_json_dict(data)


def test_trajectory_rejects_bad_timestamp_and_commit():
    data = fixtures.trajectory_dict()
    data["provenance"]["created_at_utc"] = "not-a-timestamp"
    with pytest.raises(ContractError, match="ISO-8601"):
        Trajectory.from_json_dict(data)

    data = fixtures.trajectory_dict()
    data["provenance"]["git_commit"] = "NOTHEX"
    with pytest.raises(ContractError, match="git commit"):
        Trajectory.from_json_dict(data)


def test_trajectory_slot_indices_must_be_inside_capacity():
    data = fixtures.trajectory_dict()
    data["tracks"][0]["slot_indices"] = [1, 1, 9]
    with pytest.raises(ContractError, match="outside"):
        Trajectory.from_json_dict(data)


def test_trajectory_all_zero_activity_track_is_allowed():
    trajectory = Trajectory.from_json_dict(fixtures.trajectory_dict())
    silent = trajectory.tracks[1]
    assert all(value == 0.0 for value in silent.activity)


def test_trajectory_empty_tracks_are_no_tracks():
    data = fixtures.trajectory_dict()
    data["tracks"] = []
    trajectory = Trajectory.from_json_dict(data)
    assert trajectory.tracks == ()


def test_track_without_slot_indices_keeps_identity_separate_from_slots():
    track = Track(
        track_id="trk-x",
        center_times=(0.0, 0.02),
        activity=(0.0, 0.9),
    )
    assert track.slot_indices is None
    assert track.track_id != "0"


def test_run_provenance_requires_utc_timestamp():
    with pytest.raises(ContractError, match="UTC"):
        RunProvenance(run_id="run-1", created_at_utc="2026-09-29T12:00:00+02:00")
    with pytest.raises(ContractError, match="UTC"):
        RunProvenance(run_id="run-1", created_at_utc="2026-09-29T12:00:00")


def test_documents_mutate_their_own_copies_of_input_dicts():
    data = fixtures.manifest_dict()
    snapshot = copy.deepcopy(data)
    manifest = SampleManifest.from_json_dict(data)
    manifest.groups["composition"] = "mutated"
    assert data == snapshot
