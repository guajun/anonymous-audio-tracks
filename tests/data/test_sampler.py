"""Multi-center batch sampling: shapes, masks, pairs, capacity and determinism."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from aat.contracts import ActivityData
from aat.data import DatasetError, build_dataset_index, sample_batch
from aat.labels import read_wav
from aat.windowing import center_times, extract_windows_at_times, window_sample_count

from . import fixtures

RATE = fixtures.RATE


def _build(data_root, **options):
    settings = {
        "seed": 1,
        "ratios": [1, 0, 0],
        "slots": 8,
        "window_seconds": 2.0,
        "index_dir": data_root.parent / "index",
    }
    settings.update(options)
    return build_dataset_index(data_root, **settings)


def _full_source(rate: int = RATE, duration: float = 4.0) -> np.ndarray:
    return fixtures.tone_bursts(duration, rate, [(0.0, duration)])


def _mono(path) -> np.ndarray:
    wav = read_wav(path)
    return wav.samples.mean(axis=1) if wav.channels > 1 else wav.samples[:, 0]


@pytest.mark.parametrize("rate", [RATE, 44100])
def test_batch_shapes_masks_and_traceability(tmp_path, rate: int) -> None:
    data = tmp_path / "data"
    duration = 4.0
    stems = {
        "s01": _full_source(rate, duration),
        "s02": fixtures.tone_bursts(duration, rate, [(1.0, 2.0)]),
    }
    fixtures.make_sample(
        data,
        "song-a",
        sample_id="song-a",
        composition="comp-a",
        presets=["preset-a"],
        stems=stems,
        sample_rate=rate,
        duration_seconds=duration,
    )
    index = _build(data, slots=4)
    batch = sample_batch(
        index, data, seed=3, items=1, centers_per_item=3, min_center_gap=5
    )
    assert len(batch) == 1
    assert batch.skipped == ()
    assert batch.sample_ids == ("song-a",)

    block = batch.blocks[0]
    assert block.source_ids == ("s01", "s02")
    assert block.slot_ids == ("s01", "s02", None, None)
    assert block.slots == 4
    assert block.sample_rate == rate
    assert block.window_seconds == 2.0
    assert block.window_samples == window_sample_count(2.0, rate)
    assert block.hop_seconds == 0.02
    assert block.center_times.shape == (3,)
    assert block.center_valid.all()
    assert block.center_indices.shape == (3,)

    assert block.activity.shape == (3, 4)
    assert block.activity.dtype == np.float32
    assert block.audio.shape == (3, block.window_samples)
    assert block.audio.dtype == np.float32
    assert block.audio_valid.shape == block.audio.shape
    assert block.audio_valid.dtype == np.bool_
    assert block.audio_valid.all()

    assert np.array_equal(block.source_present[:, :2], np.ones((3, 2), dtype=bool))
    assert not block.source_present[:, 2:].any()
    assert np.all(block.activity[:, 2:] == 0.0)
    assert np.all(np.diff(block.center_indices) >= 5)

    activity = ActivityData.load(data / "song-a")
    assert np.array_equal(block.activity[:, :2], activity.activity[block.center_indices])
    assert np.allclose(block.center_times, activity.center_times[block.center_indices])
    assert np.array_equal(block.center_valid, activity.valid[block.center_indices])
    assert block.activity_sha256 == fixtures.sha256_file(data / "song-a" / "activity.npz")
    assert block.mix_sha256 == fixtures.sha256_file(data / "song-a" / "mix.wav")

    windows, valid = extract_windows_at_times(
        _mono(data / "song-a" / "mix.wav"),
        block.center_times,
        rate,
        2.0,
        origin_seconds=0.0,
    )
    assert np.array_equal(block.audio, windows.astype(np.float32))
    assert np.array_equal(block.audio_valid, valid)


def test_silent_source_is_not_a_padding_slot(tmp_path) -> None:
    data = tmp_path / "data"
    duration = 4.0
    stems = {
        "s01": _full_source(),
        "s02": np.zeros(round(duration * RATE), dtype=np.float64),
    }
    fixtures.make_sample(
        data, "song", sample_id="song", composition="comp", stems=stems
    )
    index = _build(data, slots=3)
    batch = sample_batch(index, data, seed=2, items=1, centers_per_item=4)
    block = batch.blocks[0]

    # The silent source exists in every row but has zero activity.
    assert block.source_present[:, 1].all()
    assert np.all(block.activity[:, 1] == 0.0)
    # The padding slot does not exist and is also zero.
    assert not block.source_present[:, 2].any()
    assert np.all(block.activity[:, 2] == 0.0)
    entry = index.samples[0]
    assert entry.groups["composition"] == "comp"
    assert entry.controls["sources"]["s02"]["note_on_count"] == 0


def test_invalid_centers_are_masked_not_supervised(tmp_path) -> None:
    data = tmp_path / "data"
    duration = 4.0
    manual = fixtures.manual_activity(
        sample_id="song",
        source_ids=["s01"],
        duration_seconds=duration,
        activity=np.ones((len(center_times(duration, 0.02)), 1), dtype=np.float32),
    )
    fixtures.make_sample(
        data,
        "song",
        sample_id="song",
        composition="comp",
        stems={"s01": _full_source()},
        activity=manual,
    )
    index = _build(data, slots=2)
    batch = sample_batch(
        index,
        data,
        seed=3,
        items=1,
        centers_per_item=2,
        min_center_gap=200,
        valid_only=False,
    )
    block = batch.blocks[0]
    assert block.center_indices.tolist() == [0, 200]
    assert not block.center_valid.any()
    assert not block.audio_valid.all()
    # Labels exist for the real source rows but the masks announce that
    # supervision is invalid; the padding slot is still zero.
    assert block.activity[:, 0].min() == 1.0
    assert np.all(block.activity[:, 1] == 0.0)
    assert block.activity.shape == (2, 2)


def test_center_valid_does_not_imply_full_audio_window(tmp_path) -> None:
    data = tmp_path / "data"
    duration = 4.0
    manual = fixtures.manual_activity(
        sample_id="song",
        source_ids=["s01"],
        duration_seconds=duration,
        activity=np.ones((len(center_times(duration, 0.02)), 1), dtype=np.float32),
    )
    fixtures.make_sample(
        data,
        "song",
        sample_id="song",
        composition="comp",
        stems={"s01": _full_source()},
        activity=manual,
    )
    index = _build(data, slots=2)
    batch = sample_batch(
        index,
        data,
        seed=1,
        items=1,
        centers_per_item=2,
        min_center_gap=100,
        window_seconds=4.0,
    )
    block = batch.blocks[0]
    assert block.window_seconds == 4.0
    assert block.center_valid.all()
    assert not block.audio_valid.all()
    assert any(not row.all() for row in block.audio_valid)


@pytest.mark.parametrize("rate", [RATE, 44100])
def test_nonzero_track_start_uses_absolute_times(tmp_path, rate: int) -> None:
    data = tmp_path / "data"
    duration = 3.0
    track_start = 12.34
    stems = {"s01": fixtures.tone_bursts(duration, rate, [(0.5, 2.5)])}
    fixtures.make_sample(
        data,
        "song",
        sample_id="song",
        composition="comp",
        stems=stems,
        sample_rate=rate,
        duration_seconds=duration,
        track_start_seconds=track_start,
    )
    index = _build(data, slots=2)
    batch = sample_batch(index, data, seed=4, items=1, centers_per_item=2, min_center_gap=10)
    block = batch.blocks[0]
    assert block.track_start_seconds == pytest.approx(track_start)
    assert block.center_times[0] >= track_start
    assert block.window_samples == window_sample_count(2.0, rate)
    assert block.audio_valid.all()

    windows, valid = extract_windows_at_times(
        _mono(data / "song" / "mix.wav"),
        block.center_times,
        rate,
        2.0,
        origin_seconds=track_start,
    )
    assert np.array_equal(block.audio, windows.astype(np.float32))
    assert np.array_equal(block.audio_valid, valid)


def test_same_source_non_adjacent_pairs(tmp_path) -> None:
    data = tmp_path / "data"
    duration = 4.0
    rows = len(center_times(duration, 0.02))
    activity = np.zeros((rows, 2), dtype=np.float32)
    activity[:, 0] = 1.0
    manual = fixtures.manual_activity(
        sample_id="song",
        source_ids=["s01", "s02"],
        duration_seconds=duration,
        activity=activity,
    )
    fixtures.make_sample(
        data,
        "song",
        sample_id="song",
        composition="comp",
        stems={"s01": _full_source(), "s02": np.zeros(round(duration * RATE))},
        activity=manual,
    )
    index = _build(data, slots=3)
    batch = sample_batch(
        index, data, seed=5, items=1, centers_per_item=4, min_center_gap=3
    )
    block = batch.blocks[0]
    assert np.all(np.diff(block.center_indices) >= 3)
    assert block.same_source_pairs
    for slot, left, right in block.same_source_pairs:
        assert slot == 0
        assert block.activity[left, slot] >= 0.5
        assert block.activity[right, slot] >= 0.5
        assert block.center_indices[right] - block.center_indices[left] >= 3
    assert {slot for slot, _, _ in block.same_source_pairs} == {0}


def test_pitch_evidence_never_rewrites_labels(tmp_path) -> None:
    data = tmp_path / "data"
    events = [
        {
            "time_seconds": 0.0,
            "source_id": "s01",
            "event_type": "note_on",
            "data": {"note": 60, "velocity": 100},
        },
        {
            "time_seconds": 1.0,
            "source_id": "s01",
            "event_type": "note_on",
            "data": {"note": 67, "velocity": 100},
        },
    ]
    sample = fixtures.make_sample(
        data,
        "song",
        sample_id="song",
        composition="comp",
        stems={"s01": _full_source()},
        controls_events=events,
    )
    index = _build(data, slots=2)
    entry = index.samples[0]
    assert entry.controls["note_min"] == 60
    assert entry.controls["note_max"] == 67
    assert entry.controls["sources"]["s01"]["note_min"] == 60
    assert entry.controls["sources"]["s01"]["note_max"] == 67

    batch = sample_batch(index, data, seed=1, items=1, centers_per_item=2)
    assert batch.blocks[0].source_note_ranges == ((60, 67),)

    before = np.load(sample / "activity.npz")["activity"]
    fixtures.rewrite_json(
        sample / "controls.json",
        lambda payload: [
            event["data"].__setitem__("note", 40 + position)
            for position, event in enumerate(payload["events"])
        ],
    )
    fixtures.rehash_manifest(sample, "controls.json")
    rebuilt = _build(data, slots=2)
    rebuilt_entry = rebuilt.samples[0]
    assert (rebuilt_entry.controls["note_min"], rebuilt_entry.controls["note_max"]) == (40, 41)
    after = np.load(sample / "activity.npz")["activity"]
    assert np.array_equal(before, after)


def test_batch_is_seeded_and_deterministic(tmp_path) -> None:
    data = tmp_path / "data"
    for number in range(4):
        fixtures.make_sample(
            data,
            f"s{number}",
            sample_id=f"sample-{number}",
            composition=f"comp-{number}",
            stems={"s01": _full_source()},
        )
    index = _build(data)
    first = sample_batch(index, data, seed=7, items=2, centers_per_item=2, min_center_gap=1)
    second = sample_batch(index, data, seed=7, items=2, centers_per_item=2, min_center_gap=1)
    assert first.sample_ids == second.sample_ids
    for left, right in zip(first.blocks, second.blocks):
        assert np.array_equal(left.center_times, right.center_times)
        assert np.array_equal(left.audio, right.audio)
        assert np.array_equal(left.activity, right.activity)
        assert np.array_equal(left.audio_valid, right.audio_valid)
    choices = {
        sample_batch(index, data, seed=seed, items=2, centers_per_item=2).sample_ids
        for seed in range(6)
    }
    assert len(choices) > 1


def test_blocks_never_mix_two_songs(tmp_path) -> None:
    data = tmp_path / "data"
    for number in range(4):
        fixtures.make_sample(
            data,
            f"s{number}",
            sample_id=f"sample-{number}",
            composition=f"comp-{number}",
            stems={"s01": _full_source()},
        )
    index = _build(data, slots=3)
    batch = sample_batch(index, data, seed=9, items=4, centers_per_item=2, min_center_gap=5)
    assert len(batch) == 4
    assert len(set(batch.sample_ids)) == 4
    for block in batch.blocks:
        activity = ActivityData.load(data / block.path)
        assert np.array_equal(
            block.activity[:, :1], activity.activity[block.center_indices]
        )
        windows, valid = extract_windows_at_times(
            _mono(data / block.path / "mix.wav"),
            block.center_times,
            block.sample_rate,
            block.window_seconds,
            origin_seconds=block.track_start_seconds,
        )
        assert np.array_equal(block.audio, windows.astype(np.float32))
        assert np.array_equal(block.audio_valid, valid)


def test_multichannel_mix_is_averaged(tmp_path) -> None:
    data = tmp_path / "data"
    duration = 4.0
    sample = fixtures.make_sample(
        data,
        "song",
        sample_id="song",
        composition="comp",
        stems={"s01": _full_source()},
    )
    wav = read_wav(sample / "mix.wav")
    stereo = np.column_stack([wav.samples[:, 0], 0.5 * wav.samples[:, 0]])
    fixtures.write_pcm16(sample / "mix.wav", stereo, RATE)
    fixtures.rehash_manifest(sample, "mix.wav")

    index = _build(data, slots=2)
    assert index.samples[0].audio["channels"] == 2
    batch = sample_batch(index, data, seed=1, items=1, centers_per_item=2)
    block = batch.blocks[0]
    mixed = read_wav(sample / "mix.wav")
    assert mixed.channels == 2
    windows, valid = extract_windows_at_times(
        mixed.samples.mean(axis=1),
        block.center_times,
        RATE,
        2.0,
        origin_seconds=0.0,
    )
    assert np.array_equal(block.audio, windows.astype(np.float32))
    assert np.array_equal(block.audio_valid, valid)


def test_no_valid_centers_is_explicit(tmp_path) -> None:
    data = tmp_path / "data"
    duration = 4.0
    rows = len(center_times(duration, 0.02))
    manual = fixtures.manual_activity(
        sample_id="song",
        source_ids=["s01"],
        duration_seconds=duration,
        activity=np.zeros((rows, 1), dtype=np.float32),
        valid=np.zeros(rows, dtype=bool),
    )
    fixtures.make_sample(
        data,
        "song",
        sample_id="song",
        composition="comp",
        stems={"s01": _full_source()},
        activity=manual,
    )
    index = _build(data, slots=2)
    entry = index.samples[0]
    assert entry.usable is False
    assert any("no valid center windows" in warning for warning in entry.warnings)
    assert any("all center activity labels are 0" in warning for warning in entry.warnings)
    with pytest.raises(DatasetError, match="no valid center windows"):
        sample_batch(index, data, seed=1, items=1)
    batch = sample_batch(index, data, seed=1, items=1, on_unusable="skip")
    assert batch.blocks == ()
    assert batch.skipped
    assert "no valid center windows" in batch.skipped[0][1]
    assert batch.requested_items == 1
    assert len(batch) == 0


def test_infeasible_min_gap_is_explicit(tmp_path) -> None:
    data = tmp_path / "data"
    duration = 4.0
    rows = len(center_times(duration, 0.02))
    valid = np.zeros(rows, dtype=bool)
    valid[[0, 100, 200]] = True
    manual = fixtures.manual_activity(
        sample_id="song",
        source_ids=["s01"],
        duration_seconds=duration,
        activity=np.ones((rows, 1), dtype=np.float32),
        valid=valid,
    )
    fixtures.make_sample(
        data,
        "song",
        sample_id="song",
        composition="comp",
        stems={"s01": _full_source()},
        activity=manual,
    )
    index = _build(data, slots=2)
    with pytest.raises(DatasetError, match="cannot select"):
        sample_batch(index, data, seed=1, items=1, centers_per_item=3, min_center_gap=150)
    batch = sample_batch(
        index, data, seed=1, items=1, centers_per_item=3, min_center_gap=150, on_unusable="skip"
    )
    assert batch.blocks == ()
    assert batch.skipped


def test_capacity_override_rejects_before_sampling(tmp_path) -> None:
    data = tmp_path / "data"
    stems = {
        "s01": _full_source(),
        "s02": fixtures.tone_bursts(4.0, RATE, [(1.0, 2.0)]),
        "s03": fixtures.tone_bursts(4.0, RATE, [(2.0, 3.0)]),
    }
    fixtures.make_sample(
        data, "song", sample_id="song", composition="comp", stems=stems
    )
    index = _build(data, slots=3)
    with pytest.raises(DatasetError, match="exceed the slots capacity K=2"):
        sample_batch(index, data, seed=1, items=1, slots=2)


def test_empty_split_returns_empty_batch(tmp_path) -> None:
    data = tmp_path / "data"
    fixtures.make_sample(
        data, "song", sample_id="song", composition="comp", stems={"s01": _full_source()}
    )
    index = _build(data, ratios=[1, 0, 0])
    batch = sample_batch(index, data, seed=1, split="val")
    assert len(batch) == 0
    assert batch.requested_items == 0
    with pytest.raises(DatasetError, match="only 0 sample"):
        sample_batch(index, data, seed=1, split="val", items=1)


def test_consumed_mix_digest_is_verified_by_default(tmp_path) -> None:
    data = tmp_path / "data"
    duration = 4.0
    sample = fixtures.make_sample(
        data,
        "song",
        sample_id="song",
        composition="comp",
        stems={"s01": np.zeros(round(duration * RATE), dtype=np.float64)},
    )
    index = _build(data, slots=2)
    recorded = index.samples[0].content_sha256["mix.wav"]
    assert recorded == fixtures.sha256_file(sample / "mix.wav")

    # Same frame count, same manifest, different bytes: load/verify happened
    # before this rewrite, so only the per-batch check can catch it.
    fixtures.write_pcm16(
        sample / "mix.wav",
        np.full((round(duration * RATE), 1), 0.2),
        RATE,
    )
    assert fixtures.sha256_file(sample / "mix.wav") != recorded

    with pytest.raises(DatasetError, match="mix 'mix.wav' sha256 mismatch"):
        sample_batch(index, data, seed=1, items=1, centers_per_item=2)

    batch = sample_batch(
        index, data, seed=1, items=1, centers_per_item=2, verify_digests=False
    )
    block = batch.blocks[0]
    # The block reports the index digest, not the rewritten bytes: opt-out is
    # explicit unverified consumption, never a silent re-baseline.
    assert block.mix_sha256 == recorded
    assert block.mix_sha256 != fixtures.sha256_file(sample / "mix.wav")
    assert float(np.abs(block.audio).max()) > 0.15


def test_consumed_activity_arrays_digest_is_verified(tmp_path) -> None:
    data = tmp_path / "data"
    sample = fixtures.make_sample(
        data,
        "song",
        sample_id="song",
        composition="comp",
        stems={"s01": _full_source()},
    )
    index = _build(data, slots=2)

    with np.load(sample / "activity.npz", allow_pickle=False) as archive:
        center_times = archive["center_times"]
        valid = archive["valid"]
        mutated = np.where(archive["activity"] > 0.5, 0.25, 0.75).astype(np.float32)
    np.savez(
        sample / "activity.npz",
        center_times=center_times,
        activity=mutated,
        valid=valid,
    )
    with pytest.raises(DatasetError, match="activity arrays 'activity.npz' sha256 mismatch"):
        sample_batch(index, data, seed=1, items=1, centers_per_item=2)
    # The opt-out consumes the changed labels explicitly (documented as unverified);
    # the block digest stays the index value and is not silently re-baselined.
    batch = sample_batch(
        index, data, seed=1, items=1, centers_per_item=2, verify_digests=False
    )
    block = batch.blocks[0]
    assert np.array_equal(block.activity[:, :1], mutated[block.center_indices])
    assert block.activity_sha256 != fixtures.sha256_file(sample / "activity.npz")


def test_consumed_activity_metadata_digest_is_verified(tmp_path) -> None:
    data = tmp_path / "data"
    sample = fixtures.make_sample(
        data,
        "song",
        sample_id="song",
        composition="comp",
        stems={"s01": _full_source()},
    )
    index = _build(data, slots=2)
    fixtures.rewrite_json(
        sample / "activity.json",
        lambda payload: payload["label_params"]["config"].__setitem__("hop_seconds", 0.021),
    )
    with pytest.raises(
        DatasetError, match="activity metadata 'activity.json' sha256 mismatch"
    ):
        sample_batch(index, data, seed=1, items=1, centers_per_item=2)


def test_sampling_from_another_root_is_digest_checked(tmp_path) -> None:
    root_a = tmp_path / "a"
    fixtures.make_sample(
        root_a,
        "song",
        sample_id="song",
        composition="comp",
        stems={"s01": _full_source()},
    )
    index = _build(root_a, slots=2)
    root_b = tmp_path / "b"
    shutil.copytree(root_a / "song", root_b / "song")
    fixtures.write_pcm16(
        root_b / "song" / "mix.wav",
        np.full((round(4.0 * RATE), 1), 0.2),
        RATE,
    )

    # Same relative path, same shapes, different bytes under another root.
    with pytest.raises(DatasetError, match="mix 'mix.wav' sha256 mismatch"):
        sample_batch(index, root_b, seed=1, items=1, centers_per_item=2)
    batch = sample_batch(index, root_a, seed=1, items=1, centers_per_item=2)
    assert len(batch) == 1
    assert batch.blocks[0].mix_sha256 == fixtures.sha256_file(root_a / "song" / "mix.wav")


def test_include_invalid_allows_boundary_only_sample(tmp_path) -> None:
    data = tmp_path / "data"
    fixtures.make_sample(
        data,
        "short",
        sample_id="short",
        composition="comp",
        stems={"s01": fixtures.tone_bursts(0.5, RATE, [(0.0, 0.5)])},
        duration_seconds=0.5,
    )
    index = _build(data, slots=2)
    entry = index.samples[0]
    assert entry.labels["center_count"] == 26
    assert entry.labels["valid_count"] == 0
    assert entry.usable is False
    assert any("no valid center windows" in warning for warning in entry.warnings)

    # Default valid_only=True still rejects (or records with on_unusable=skip).
    with pytest.raises(DatasetError, match="no valid center windows"):
        sample_batch(index, data, seed=1, items=1, centers_per_item=2)
    skipped = sample_batch(index, data, seed=1, items=1, on_unusable="skip")
    assert skipped.blocks == ()
    assert "no valid center windows" in skipped.skipped[0][1]

    batch = sample_batch(
        index, data, seed=1, items=1, centers_per_item=2, valid_only=False
    )
    assert len(batch) == 1
    block = batch.blocks[0]
    assert block.center_indices.size == 2
    assert not block.center_valid.any()
    assert not block.audio_valid.all()
    assert block.same_source_pairs == ()
    assert block.source_present[:, :1].all()
    assert np.all(block.activity[:, 1] == 0.0)
    assert block.activity.shape == (2, 2)


def test_zero_center_sample_is_unusable_even_with_include_invalid(tmp_path) -> None:
    data = tmp_path / "data"
    manual = fixtures.manual_activity(
        sample_id="song",
        source_ids=["s01"],
        duration_seconds=0.0,
    )
    assert manual.center_times.size == 0
    fixtures.make_sample(
        data,
        "song",
        sample_id="song",
        composition="comp",
        stems={"s01": _full_source()},
        activity=manual,
    )
    index = _build(data, slots=2)
    entry = index.samples[0]
    assert entry.labels["center_count"] == 0
    assert entry.usable is False
    for valid_only in (True, False):
        with pytest.raises(DatasetError, match="no center times"):
            sample_batch(index, data, seed=1, items=1, valid_only=valid_only)
        skipped = sample_batch(
            index, data, seed=1, items=1, valid_only=valid_only, on_unusable="skip"
        )
        assert skipped.blocks == ()
        assert "no center times" in skipped.skipped[0][1]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"items": 5}, "only 4 sample"),
        ({"items": 0}, "items: must be >= 1"),
        ({"centers_per_item": 0}, "centers_per_item: must be >= 1"),
        ({"seed": -1}, "seed: must be >= 0"),
        ({"seed": True}, "seed: expected an integer"),
        ({"split": "eval"}, "split: expected one of"),
        ({"on_unusable": "ignore"}, "on_unusable"),
        ({"activity_threshold": 1.5}, "activity_threshold"),
        ({"window_seconds": 0.0}, "window_seconds"),
        ({"slots": 0}, "slots: must be >= 1"),
        ({"valid_only": "yes"}, "valid_only"),
        ({"verify_digests": "yes"}, "verify_digests"),
    ],
)
def test_parameter_validation(tmp_path, kwargs, match: str) -> None:
    data = tmp_path / "data"
    for number in range(4):
        fixtures.make_sample(
            data,
            f"s{number}",
            sample_id=f"sample-{number}",
            composition=f"comp-{number}",
            stems={"s01": _full_source()},
        )
    index = _build(data)
    with pytest.raises(DatasetError, match=match):
        sample_batch(index, data, **{"seed": 1, "items": 1, **kwargs})
