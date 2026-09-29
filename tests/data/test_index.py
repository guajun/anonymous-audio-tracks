"""Index building, validation, digests and load-time change detection."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from aat.contracts import ActivityData
from aat.data import (
    DatasetError,
    DatasetIndex,
    build_dataset_index,
    discover_samples,
    sha256_file,
    verify_dataset,
)

from . import fixtures

RATE = fixtures.RATE


def _sample(data_root: Path, relative: str, sample_id: str, composition: str, **kwargs):
    stems = {"s01": fixtures.tone_bursts(4.0, RATE, [(0.0, 4.0)])}
    return fixtures.make_sample(
        data_root,
        relative,
        sample_id=sample_id,
        composition=composition,
        stems=stems,
        **kwargs,
    )


def _build(data_root: Path, **kwargs) -> DatasetIndex:
    options = {
        "seed": 11,
        "ratios": [0.8, 0.1, 0.1],
        "slots": 4,
        "window_seconds": 2.0,
        "index_dir": data_root.parent / "index",
    }
    options.update(kwargs)
    return build_dataset_index(data_root, **options)


def test_build_records_digests_seed_and_split(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _sample(data, "a/s1", "s1", "comp-a", presets=["preset-a"])
    _sample(data, "b/s2", "s2", "comp-b", presets=["preset-a"])
    _sample(data, "c/s3", "s3", "comp-c", sample_origins=["origin-x"])

    index = _build(data)
    assert index.plan["seed"] == 11
    assert index.plan["slots"] == 4
    assert index.plan["window_seconds"] == 2.0
    assert index.summary["sample_count"] == 3
    assert index.summary["leak_free"] is True
    assert index.summary["cross_split_assets"] == {
        "composition": 0,
        "preset": 0,
        "sample_origin": 0,
    }

    entries = {entry.sample_id: entry for entry in index.samples}
    assert entries["s1"].path == "a/s1"
    assert entries["s1"].content_sha256["mix.wav"] == sha256_file(data / "a/s1" / "mix.wav")
    assert entries["s1"].record_sha256() == entries["s1"].sample_sha256
    assert entries["s1"].labels["center_count"] == 201
    assert entries["s1"].labels["valid_count"] == 101
    assert entries["s1"].usable is True
    assert entries["s1"].groups["composition"] == "comp-a"
    assert entries["s1"].groups["preset"] == ("preset-a",)
    assert entries["s1"].groups["sample_origin"] == ()

    # A and B share a preset, so the whole component is in one split.
    assert entries["s1"].split == entries["s2"].split
    assert {entry.split for entry in index.samples} <= {"train", "val", "test"}

    path = tmp_path / "index" / "index.json"
    index.save(path)
    loaded = DatasetIndex.load(path)
    assert loaded.to_json_dict() == index.to_json_dict()
    assert loaded.resolve_data_root(index_path=path) == data.resolve()
    assert verify_dataset(loaded, data).ok


def test_rebuild_with_same_seed_is_byte_identical(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for number in range(3):
        _sample(data, f"s{number:02d}", f"sample-{number}", f"comp-{number}")
    out_dir = tmp_path / "index"
    first = _build(data, index_dir=out_dir, seed=5)
    first.save(out_dir / "first.json")
    second = _build(data, index_dir=out_dir, seed=5)
    second.save(out_dir / "second.json")
    assert (out_dir / "first.json").read_bytes() == (out_dir / "second.json").read_bytes()


def test_different_seed_changes_split_but_stays_leak_free(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for number in range(9):
        _sample(data, f"s{number:02d}", f"sample-{number}", f"comp-{number}")
    ratios = [1, 1, 1]
    first = _build(data, seed=1, ratios=ratios)
    second = _build(data, seed=2, ratios=ratios)
    first_map = {entry.sample_id: entry.split for entry in first.samples}
    second_map = {entry.sample_id: entry.split for entry in second.samples}
    assert first_map != second_map
    for index in (first, second):
        assert index.summary["cross_split_assets"] == {
            "composition": 0,
            "preset": 0,
            "sample_origin": 0,
        }


def test_no_cross_split_crossover_for_all_group_keys(tmp_path: Path) -> None:
    data = tmp_path / "data"
    # composition shared by a1/a2; preset shared by b1/b2; origin shared by c1/c2.
    _sample(data, "a1", "a1", "comp-shared", presets=["p1"])
    _sample(data, "a2", "a2", "comp-shared", presets=["p2"])
    _sample(data, "b1", "b1", "comp-b1", presets=["p-shared"])
    _sample(data, "b2", "b2", "comp-b2", presets=["p-shared"])
    _sample(data, "c1", "c1", "comp-c1", sample_origins=["o-shared"])
    _sample(data, "c2", "c2", "comp-c2", sample_origins=["o-shared"])
    _sample(data, "d", "d", "comp-d")
    index = _build(data, ratios=[1, 1, 1])
    assert index.summary["cross_split_assets"] == {
        "composition": 0,
        "preset": 0,
        "sample_origin": 0,
    }
    by_id = {entry.sample_id: entry.split for entry in index.samples}
    assert by_id["a1"] == by_id["a2"]
    assert by_id["b1"] == by_id["b2"]
    assert by_id["c1"] == by_id["c2"]
    for split in ("train", "val", "test"):
        assert index.summary["split_counts"][split] >= 0


def test_transitive_shared_assets_stay_in_one_split(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _sample(data, "a", "a", "comp-a", presets=["p"])
    _sample(data, "b", "b", "comp-b", presets=["p"], sample_origins=["q"])
    _sample(data, "c", "c", "comp-c", sample_origins=["q"])
    index = _build(data, ratios=[1, 1, 1])
    splits = {entry.split for entry in index.samples}
    assert len(splits) == 1
    assert index.summary["component_count"] == 1
    assert index.summary["largest_component_size"] == 3
    assert index.summary["cross_split_assets"] == {
        "composition": 0,
        "preset": 0,
        "sample_origin": 0,
    }


def test_duplicate_sample_id_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _sample(data, "a/s1", "duplicate", "comp-a")
    _sample(data, "b/s1", "duplicate", "comp-b")
    with pytest.raises(DatasetError, match="duplicate sample_id"):
        _build(data)


def test_rendered_stage_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _sample(data, "a/s1", "s1", "comp-a", labeled=False)
    with pytest.raises(DatasetError, match="stage is 'rendered'"):
        _build(data)


def test_missing_label_arrays_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    sample = _sample(data, "a/s1", "s1", "comp-a")
    (sample / "activity.npz").unlink()
    with pytest.raises(DatasetError, match="file not found"):
        _build(data)


def test_corrupted_stem_digest_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    sample = _sample(data, "a/s1", "s1", "comp-a")
    with (sample / "stems" / "s01.wav").open("ab") as handle:
        handle.write(b"\x00\x00")
    with pytest.raises(DatasetError, match="content_sha256"):
        _build(data)


def test_unknown_schema_version_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    sample = _sample(data, "a/s1", "s1", "comp-a")
    fixtures.rewrite_json(
        sample / "manifest.json", lambda payload: payload.__setitem__("schema_version", "9.9.9")
    )
    with pytest.raises(DatasetError, match="unsupported version"):
        _build(data)


def test_unknown_label_config_version_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    sample = _sample(data, "a/s1", "s1", "comp-a")

    def mutate(payload):
        payload["label_params"]["version"] = "activity-label-v9"

    fixtures.rewrite_json(sample / "activity.json", mutate)
    fixtures.rehash_manifest(sample, "activity.json")
    with pytest.raises(DatasetError, match="unsupported activity label config"):
        _build(data)


def test_missing_label_params_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    sample = _sample(data, "a/s1", "s1", "comp-a")
    fixtures.rewrite_json(sample / "activity.json", lambda payload: payload.pop("label_params"))
    fixtures.rehash_manifest(sample, "activity.json")
    with pytest.raises(DatasetError, match="no label_params snapshot"):
        _build(data)


def test_inconsistent_source_columns_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    stems = {
        "s01": fixtures.tone_bursts(4.0, RATE, [(0.0, 4.0)]),
        "s02": fixtures.tone_bursts(4.0, RATE, [(1.0, 2.0)]),
    }
    sample = fixtures.make_sample(
        data, "a/s1", sample_id="s1", composition="comp-a", stems=stems
    )

    def mutate(payload):
        payload["source_ids"] = list(reversed(payload["source_ids"]))

    fixtures.rewrite_json(sample / "activity.json", mutate)
    fixtures.rehash_manifest(sample, "activity.json")
    with pytest.raises(DatasetError, match="do not match the sources.json order"):
        _build(data)


def test_capacity_rejects_silent_sources_too(tmp_path: Path) -> None:
    data = tmp_path / "data"
    stems = {
        "s01": fixtures.tone_bursts(4.0, RATE, [(0.0, 4.0)]),
        "s02": np.zeros(round(4.0 * RATE), dtype=np.float64),
        "s03": fixtures.tone_bursts(4.0, RATE, [(2.0, 3.0)]),
    }
    fixtures.make_sample(
        data, "a/s1", sample_id="s1", composition="comp-a", stems=stems
    )
    with pytest.raises(DatasetError, match="exceed the slots capacity K=2"):
        _build(data, slots=2)


def test_audio_duration_mismatch_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    sample = _sample(data, "a/s1", "s1", "comp-a")
    fixtures.rewrite_json(
        sample / "manifest.json",
        lambda payload: payload.__setitem__("duration_seconds", 4.5),
    )
    with pytest.raises(DatasetError, match="mix duration"):
        _build(data)


def test_center_time_out_of_range_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    sample = _sample(data, "a/s1", "s1", "comp-a")
    activity = ActivityData.load(sample)
    times = np.append(activity.center_times, activity.center_times[-1] + 0.02)
    values = np.vstack([activity.activity, activity.activity[-1:]])
    valid = np.append(activity.valid, False)
    ActivityData(
        center_times=times,
        activity=values,
        valid=valid,
        source_ids=activity.source_ids,
        sample_rate=activity.sample_rate,
        hop_seconds=activity.hop_seconds,
        sample_id=activity.sample_id,
        label_params=activity.label_params,
    ).save(sample)
    fixtures.rehash_manifest(sample, "activity.json", "activity.npz")
    with pytest.raises(DatasetError, match="lie outside the rendered span"):
        _build(data)


def test_labeled_center_window_must_match_plan(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _sample(data, "a/s1", "s1", "comp-a")
    with pytest.raises(DatasetError, match="center_window_seconds"):
        _build(data, window_seconds=1.0)


def test_empty_data_root_is_supported(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    index = _build(data)
    assert index.summary["sample_count"] == 0
    assert index.summary["split_counts"] == {"train": 0, "val": 0, "test": 0}
    assert index.summary["empty_splits"] == ["train", "val", "test"]
    assert any("no labeled samples" in warning for warning in index.summary["warnings"])
    path = tmp_path / "index" / "index.json"
    index.save(path)
    loaded = DatasetIndex.load(path)
    assert loaded.samples == ()


def test_data_root_itself_must_not_be_a_sample(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _sample(tmp_path, "data", "s1", "comp-a")
    with pytest.raises(DatasetError, match="itself a sample directory"):
        discover_samples(data)


def test_single_component_reports_real_ratios_and_empty_splits(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for number in range(4):
        _sample(
            data,
            f"s{number}",
            f"sample-{number}",
            f"comp-{number}",
            presets=["shared-preset"],
        )
    index = _build(data, ratios=[0.8, 0.1, 0.1])
    assert index.summary["component_count"] == 1
    assert index.summary["largest_component_size"] == 4
    assert index.summary["empty_splits"] == ["val", "test"]
    assert index.summary["split_counts"] == {"train": 4, "val": 0, "test": 0}
    assert index.summary["split_ratios_actual"] == {"train": 1.0, "val": 0.0, "test": 0.0}
    assert any("cannot fit any split target" in warning for warning in index.summary["warnings"])
    assert any("cannot be met" in warning for warning in index.summary["warnings"])


@pytest.mark.parametrize("seed", [-1, True, 1.5, "7"])
def test_invalid_seed_rejected(tmp_path: Path, seed) -> None:
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(DatasetError, match="seed"):
        _build(data, seed=seed)


@pytest.mark.parametrize("slots", [0, -1, True, 2.5])
def test_invalid_slots_rejected(tmp_path: Path, slots) -> None:
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(DatasetError, match="slots"):
        _build(data, slots=slots)


@pytest.mark.parametrize("ratios", [[0.5, -0.5, 1.0], [0, 0, 0], [1, 2]])
def test_invalid_ratios_rejected(tmp_path: Path, ratios) -> None:
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(DatasetError, match="ratios"):
        _build(data, ratios=ratios)


def test_verify_detects_changed_inputs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    sample = _sample(data, "a/s1", "s1", "comp-a")
    index = _build(data)
    path = tmp_path / "index" / "index.json"
    index.save(path)
    assert verify_dataset(index, data).ok

    with (sample / "mix.wav").open("ab") as handle:
        handle.write(b"\x00")
    verification = verify_dataset(index, data)
    assert not verification.ok
    assert any("content_sha256['mix.wav']" in error for error in verification.errors)
    with pytest.raises(DatasetError, match="verification failed"):
        DatasetIndex.load(path)


def test_load_detects_tampered_index_record(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _sample(data, "a/s1", "s1", "comp-a")
    index = _build(data)
    path = tmp_path / "index" / "index.json"
    index.save(path)

    def mutate(payload):
        payload["samples"][0]["split"] = "val"

    fixtures.rewrite_json(path, mutate)
    with pytest.raises(DatasetError, match="sample_sha256 mismatch"):
        DatasetIndex.load(path, verify_files=False)


def test_load_without_portable_root_requires_data_root(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _sample(data, "a/s1", "s1", "comp-a")
    index = build_dataset_index(
        data, seed=1, ratios=[1, 0, 0], slots=4, window_seconds=2.0, index_dir=None
    )
    path = tmp_path / "index" / "index.json"
    index.save(path)
    assert index.layout["data_root"] is None
    with pytest.raises(DatasetError, match="pass data_root explicitly"):
        DatasetIndex.load(path, verify_files=True)
    loaded = DatasetIndex.load(path, data_root=data, verify_files=True)
    assert loaded.summary["sample_count"] == 1


def test_index_and_data_root_can_move_together(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _sample(data, "a/s1", "s1", "comp-a")
    index_dir = tmp_path / "index"
    index = _build(data, index_dir=index_dir)
    path = index_dir / "index.json"
    index.save(path)
    assert index.layout["data_root"] == "../data"

    moved = tmp_path / "moved"
    shutil.copytree(data, moved / "data")
    shutil.copytree(index_dir, moved / "index")
    loaded = DatasetIndex.load(moved / "index" / "index.json")
    assert loaded.resolve_data_root(index_path=moved / "index" / "index.json") == (
        moved / "data"
    ).resolve()


def test_discover_samples_is_sorted(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _sample(data, "z/one", "one", "comp-z")
    _sample(data, "a/two", "two", "comp-a")
    _sample(data, "m/three", "three", "comp-m")
    assert discover_samples(data) == ("a/two", "m/three", "z/one")
