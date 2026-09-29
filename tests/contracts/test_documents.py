"""JSON document contract tests: manifest, sources, controls, trajectory."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from aat.contracts import (
    DATA_KINDS,
    MANIFEST_STAGES,
    TRAJECTORY_TIME_TOLERANCE_SECONDS,
    AudioSpan,
    ContractError,
    Controls,
    RunProvenance,
    SampleManifest,
    SchemaVersionError,
    SourceRegistry,
    Track,
    Trajectory,
    check_schema_header,
)

from . import fixtures

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# manifest.json
# --------------------------------------------------------------------------- #


def test_manifest_rendered_roundtrip_through_file(tmp_path):
    manifest = SampleManifest.from_json_dict(fixtures.manifest_dict())
    assert manifest.stage == "rendered"
    assert manifest.activity_metadata_path is None
    assert manifest.activity_arrays_path is None
    path = manifest.save(tmp_path / "manifest.json")
    assert SampleManifest.load(path) == manifest


def test_manifest_labeled_roundtrip_through_file(tmp_path):
    manifest = SampleManifest.from_json_dict(fixtures.labeled_manifest_dict())
    assert manifest.stage == "labeled"
    assert manifest.activity_metadata_path == "activity.json"
    assert manifest.activity_arrays_path == "activity.npz"
    path = manifest.save(tmp_path / "manifest.json")
    assert SampleManifest.load(path) == manifest


def test_manifest_stage_enum_is_frozen():
    assert MANIFEST_STAGES == ("rendered", "labeled")


def test_manifest_rendered_rejects_label_paths():
    data = fixtures.manifest_dict()
    data["activity_metadata_path"] = "activity.json"
    with pytest.raises(ContractError, match="rendered"):
        SampleManifest.from_json_dict(data)


def test_manifest_labeled_requires_both_label_paths():
    data = fixtures.labeled_manifest_dict()
    del data["activity_arrays_path"]
    with pytest.raises(ContractError, match="labeled"):
        SampleManifest.from_json_dict(data)

    data = fixtures.labeled_manifest_dict()
    del data["activity_metadata_path"]
    with pytest.raises(ContractError, match="labeled"):
        SampleManifest.from_json_dict(data)


def test_manifest_rendered_hashes_cover_common_artifacts():
    data = fixtures.manifest_dict()
    del data["content_sha256"]["stems/s02.wav"]
    with pytest.raises(ContractError, match="content_sha256"):
        SampleManifest.from_json_dict(data)

    data = fixtures.manifest_dict()
    del data["content_sha256"]["controls.json"]
    with pytest.raises(ContractError, match="content_sha256"):
        SampleManifest.from_json_dict(data)


def test_manifest_labeled_requires_label_hashes():
    data = fixtures.labeled_manifest_dict()
    del data["content_sha256"]["activity.npz"]
    with pytest.raises(ContractError, match="content_sha256"):
        SampleManifest.from_json_dict(data)

    data = fixtures.labeled_manifest_dict()
    del data["content_sha256"]["activity.json"]
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


def test_manifest_composition_is_a_single_non_empty_id():
    data = fixtures.manifest_dict()
    data["groups"]["composition"] = ""
    with pytest.raises(ContractError, match="composition"):
        SampleManifest.from_json_dict(data)

    data = fixtures.manifest_dict()
    data["groups"]["composition"] = ["comp-a", "comp-b"]
    with pytest.raises(ContractError, match="composition"):
        SampleManifest.from_json_dict(data)


def test_manifest_asset_lists_are_required_and_validated():
    data = fixtures.manifest_dict()
    del data["groups"]["preset"]
    with pytest.raises(ContractError, match="groups"):
        SampleManifest.from_json_dict(data)

    data = fixtures.manifest_dict()
    data["groups"]["preset"] = "preset-a"
    with pytest.raises(ContractError, match="preset"):
        SampleManifest.from_json_dict(data)

    data = fixtures.manifest_dict()
    data["groups"]["preset"] = ["preset-a", ""]
    with pytest.raises(ContractError, match="preset"):
        SampleManifest.from_json_dict(data)

    data = fixtures.manifest_dict()
    data["groups"]["preset"] = ["preset-a", "preset-a"]
    with pytest.raises(ContractError, match="duplicate"):
        SampleManifest.from_json_dict(data)


def test_manifest_empty_asset_lists_mean_no_such_asset():
    data = fixtures.manifest_dict()
    data["groups"]["preset"] = []
    data["groups"]["sample_origin"] = []
    manifest = SampleManifest.from_json_dict(data)
    assert manifest.groups["preset"] == []
    assert manifest.groups["sample_origin"] == []


def test_manifest_groups_represent_shared_presets_and_origins():
    a = SampleManifest.from_json_dict(
        fixtures.manifest_for_assets("comp-a", ["p1", "p2"], ["o1"])
    )
    b = SampleManifest.from_json_dict(
        fixtures.manifest_for_assets("comp-b", ["p2", "p3"], ["o1"])
    )
    c = SampleManifest.from_json_dict(
        fixtures.manifest_for_assets("comp-c", ["p3", "p4"], ["o2"])
    )
    assert a.groups["composition"] != b.groups["composition"]
    assert set(a.groups["preset"]) & set(b.groups["preset"]) == {"p2"}
    assert set(b.groups["preset"]) & set(c.groups["preset"]) == {"p3"}
    assert set(a.groups["preset"]) & set(c.groups["preset"]) == set()
    assert set(a.groups["sample_origin"]) & set(c.groups["sample_origin"]) == set()


def test_manifest_rejects_unsupported_schema_version():
    data = fixtures.manifest_dict()
    data["schema_version"] = "999.0"
    with pytest.raises(SchemaVersionError):
        SampleManifest.from_json_dict(data)


def test_manifest_save_revalidates_after_mutation(tmp_path):
    manifest = SampleManifest.from_json_dict(fixtures.manifest_dict())
    manifest.groups["composition"] = ""
    with pytest.raises(ContractError):
        manifest.save(tmp_path / "manifest.json")

    manifest = SampleManifest.from_json_dict(fixtures.manifest_dict())
    manifest.content_sha256["mix.wav"] = "bad"
    with pytest.raises(ContractError):
        manifest.save(tmp_path / "manifest.json")


# --------------------------------------------------------------------------- #
# sources.json / controls.json
# --------------------------------------------------------------------------- #


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


def test_controls_save_rejects_nan_payload_after_mutation(tmp_path):
    controls = Controls.from_json_dict(fixtures.controls_dict())
    controls.events[0].data["velocity"] = float("nan")
    with pytest.raises(ContractError, match="JSON-serialisable"):
        controls.save(tmp_path / "controls.json")


# --------------------------------------------------------------------------- #
# trajectory.json
# --------------------------------------------------------------------------- #


def test_trajectory_roundtrip_through_file(tmp_path):
    trajectory = Trajectory.from_json_dict(fixtures.trajectory_dict())
    assert [track.track_id for track in trajectory.tracks] == ["trk-0001", "trk-0002"]
    assert trajectory.audio == AudioSpan(duration_seconds=4.0, track_start_seconds=0.0)
    path = trajectory.save(tmp_path / "trajectory.json")
    assert Trajectory.load(path) == trajectory


def test_trajectory_null_slot_index_marks_silence_memory():
    trajectory = Trajectory.from_json_dict(fixtures.trajectory_dict())
    assert trajectory.tracks[0].slot_indices == (1, None, 1)
    assert trajectory.tracks[1].slot_indices is None


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

    data = fixtures.trajectory_dict()
    del data["provenance"]["data_kind"]
    with pytest.raises(ContractError, match="data_kind"):
        Trajectory.from_json_dict(data)

    data = fixtures.trajectory_dict()
    data["provenance"]["data_kind"] = "guessed"
    with pytest.raises(ContractError, match="data_kind"):
        Trajectory.from_json_dict(data)


def test_provenance_data_kind_enum_is_frozen():
    assert DATA_KINDS == ("model", "annotation", "mock")
    for kind in DATA_KINDS:
        assert RunProvenance(run_id="run-1", data_kind=kind).data_kind == kind


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


def test_trajectory_rejects_points_outside_audio_span():
    data = fixtures.trajectory_dict()
    data["tracks"][0]["center_times"] = [0.0, 0.02, 4.5]  # end is 4.0
    with pytest.raises(ContractError, match="outside the audio span"):
        Trajectory.from_json_dict(data)

    data = fixtures.clip_trajectory_dict()
    data["tracks"][0]["center_times"] = [0.0, 0.24, 0.48]  # clip-relative times
    with pytest.raises(ContractError, match="outside the audio span"):
        Trajectory.from_json_dict(data)


def test_trajectory_accepts_span_tolerance_boundary():
    data = fixtures.trajectory_dict()
    tolerance = TRAJECTORY_TIME_TOLERANCE_SECONDS
    data["tracks"] = [
        {
            "track_id": "trk-boundary",
            "center_times": [0.0, 4.0 + tolerance / 2.0],
            "activity": [0.0, 0.9],
        }
    ]
    trajectory = Trajectory.from_json_dict(data)
    assert trajectory.tracks[0].center_times[1] > 4.0


def test_trajectory_empty_audio_must_not_have_tracks():
    data = fixtures.empty_trajectory_dict()
    trajectory = Trajectory.from_json_dict(data)
    assert trajectory.audio.duration_seconds == 0.0
    assert trajectory.tracks == ()

    data = fixtures.trajectory_dict()
    data["audio"]["duration_seconds"] = 0.0
    with pytest.raises(ContractError, match="empty audio"):
        Trajectory.from_json_dict(data)


def test_trajectory_clip_keeps_absolute_original_track_times():
    trajectory = Trajectory.from_json_dict(fixtures.clip_trajectory_dict())
    assert trajectory.audio.track_start_seconds == 12.0
    assert trajectory.tracks[0].center_times[0] == 12.0
    assert trajectory.tracks[0].slot_indices == (2, 2, None)


def test_trajectory_all_zero_activity_track_is_allowed():
    trajectory = Trajectory.from_json_dict(fixtures.trajectory_dict())
    silent = trajectory.tracks[1]
    assert all(value == 0.0 for value in silent.activity)


def test_trajectory_empty_tracks_are_no_tracks():
    data = fixtures.trajectory_dict()
    data["tracks"] = []
    trajectory = Trajectory.from_json_dict(data)
    assert trajectory.tracks == ()


def test_trajectory_save_revalidates_after_mutation(tmp_path):
    trajectory = Trajectory.from_json_dict(fixtures.trajectory_dict())
    trajectory.params["activity_threshold"] = float("nan")
    with pytest.raises(ContractError, match="JSON-serialisable"):
        trajectory.save(tmp_path / "trajectory.json")


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
        RunProvenance(
            run_id="run-1",
            data_kind="model",
            created_at_utc="2026-09-29T12:00:00+02:00",
        )
    with pytest.raises(ContractError, match="UTC"):
        RunProvenance(
            run_id="run-1",
            data_kind="model",
            created_at_utc="2026-09-29T12:00:00",
        )


# --------------------------------------------------------------------------- #
# documented examples
# --------------------------------------------------------------------------- #


def test_documented_json_examples_are_valid_pure_data():
    text = (REPO_ROOT / "docs" / "SCHEMAS.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(.*?)```", text, flags=re.DOTALL)
    assert blocks, "docs/SCHEMAS.md must contain JSON examples"
    kinds_seen: set[str] = set()
    for block in blocks:
        data = json.loads(block)
        kind = data.get("kind")
        kinds_seen.add(kind)
        if kind == "sample_manifest":
            SampleManifest.from_json_dict(data)
        elif kind == "sources":
            SourceRegistry.from_json_dict(data)
        elif kind == "controls":
            Controls.from_json_dict(data)
        elif kind == "trajectory":
            Trajectory.from_json_dict(data)
        elif kind in ("activity", "feature", "prediction"):
            check_schema_header(data, kind)
            assert data["arrays_path"], kind
            assert data["arrays"], kind
        else:
            pytest.fail(f"documented example has unknown kind: {kind!r}")
    assert "sample_manifest" in kinds_seen
