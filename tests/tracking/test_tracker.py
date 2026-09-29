"""Tracker state-machine tests: identity over slots, silence, birth and expiry."""

from __future__ import annotations

import numpy as np
import pytest

from aat.contracts import AudioSpan, Trajectory
from aat.evaluation import evaluate_trajectory
from aat.tracking import PredictionSequence, TrackingConfig, TrackingError, associate_sequence

from . import helpers


def _run(windows, times, *, config=None, span_duration=10.0, center_valid=None):
    sequence = helpers.sequence(times, windows, center_valid=center_valid)
    return associate_sequence(
        sequence,
        AudioSpan(duration_seconds=span_duration, track_start_seconds=0.0),
        helpers.provenance(),
        config=config,
    )


def test_slot_permutation_keeps_identity():
    e0, e1 = helpers.basis(0), helpers.basis(1)
    times = np.arange(4) * 0.02
    windows = [
        helpers.window(2, {0: (e0, 0.9), 1: (e1, 0.9)}),
        helpers.window(2, {0: (e1, 0.9), 1: (e0, 0.9)}),
        helpers.window(2, {0: (e0, 0.9), 1: (e1, 0.9)}),
        helpers.window(2, {0: (e1, 0.9), 1: (e0, 0.9)}),
    ]

    trajectory = _run(windows, times)

    assert [track.track_id for track in trajectory.tracks] == ["trk-0001", "trk-0002"]
    first, second = trajectory.tracks
    assert first.slot_indices == (0, 1, 0, 1)
    assert second.slot_indices == (1, 0, 1, 0)
    np.testing.assert_allclose(first.activity, (0.9, 0.9, 0.9, 0.9), atol=1e-6)
    np.testing.assert_allclose(second.activity, (0.9, 0.9, 0.9, 0.9), atol=1e-6)


def test_similar_embeddings_stay_separate_identities():
    e0 = helpers.basis(0)
    e1 = helpers.blend(0, 1, 0.95)
    times = np.arange(3) * 0.02
    windows = [
        helpers.window(2, {0: (e0, 0.9), 1: (e1, 0.9)}),
        helpers.window(2, {0: (e1, 0.9), 1: (e0, 0.9)}),
        helpers.window(2, {0: (e1, 0.9), 1: (e0, 0.9)}),
    ]

    trajectory = _run(windows, times)

    assert len(trajectory.tracks) == 2
    first, second = trajectory.tracks
    assert first.slot_indices == (0, 1, 1)
    assert second.slot_indices == (1, 0, 0)


def test_low_probability_candidate_is_matched_but_does_not_drag_prototype():
    e0 = helpers.basis(0)
    e1 = helpers.blend(0, 1, 0.8)
    times = np.arange(3) * 0.02
    windows = [
        helpers.window(2, {0: (e0, 0.9)}),
        helpers.window(2, {0: (e1, 0.1)}),
        helpers.window(2, {0: (e1, 0.9), 1: (e0, 0.9)}),
    ]
    # alpha=0 would replace the prototype after any active update, so staying on
    # slot 1 in window 2 proves the low-P candidate did not move it.
    config = TrackingConfig(prototype_alpha=0.0, match_threshold=0.7)

    trajectory = _run(windows, times, config=config)

    first = trajectory.tracks[0]
    assert first.slot_indices == (0, 0, 1)
    np.testing.assert_allclose(first.activity, (0.9, 0.1, 0.9), atol=1e-6)
    assert trajectory.tracks[1].slot_indices == (0,)
    assert first.confidence is not None
    assert first.confidence[1] == pytest.approx(0.8, abs=1e-6)


def test_active_candidate_updates_prototype_when_alpha_is_zero():
    e0 = helpers.basis(0)
    e1 = helpers.blend(0, 1, 0.8)
    times = np.arange(3) * 0.02
    windows = [
        helpers.window(2, {0: (e0, 0.9)}),
        helpers.window(2, {0: (e1, 0.9)}),
        helpers.window(2, {0: (e1, 0.9), 1: (e0, 0.9)}),
    ]
    config = TrackingConfig(prototype_alpha=0.0, match_threshold=0.7)

    trajectory = _run(windows, times, config=config)

    first = trajectory.tracks[0]
    assert first.slot_indices == (0, 0, 0)
    assert trajectory.tracks[1].slot_indices == (1,)


def test_candidate_below_gate_starts_new_track_and_old_one_keeps_memory():
    e0, e1 = helpers.basis(0), helpers.basis(1)
    times = np.arange(2) * 0.02
    windows = [
        helpers.window(2, {0: (e0, 0.9)}),
        helpers.window(2, {0: (e1, 0.9)}),
    ]

    trajectory = _run(windows, times)

    assert len(trajectory.tracks) == 2
    first, second = trajectory.tracks
    assert first.slot_indices == (0, None)
    np.testing.assert_allclose(first.activity, (0.9, 0.0), atol=1e-6)
    assert second.slot_indices == (0,)


def test_reappearance_within_retention_reconnects_same_track():
    e0 = helpers.basis(0)
    times = np.arange(6) * 0.02
    windows = [
        helpers.window(1, {0: (e0, 0.9)}),
        helpers.window(1, {0: (e0, 0.9)}),
        helpers.window(1, {}),
        helpers.window(1, {}),
        helpers.window(1, {0: (e0, 0.9)}),
        helpers.window(1, {0: (e0, 0.9)}),
    ]
    config = TrackingConfig(retention_seconds=0.1)

    trajectory = _run(windows, times, config=config)

    assert len(trajectory.tracks) == 1
    track = trajectory.tracks[0]
    assert track.slot_indices == (0, 0, None, None, 0, 0)
    np.testing.assert_allclose(
        track.activity, (0.9, 0.9, 0.0, 0.0, 0.9, 0.9), atol=1e-6
    )


def test_track_expires_after_retention_and_cumulative_ids_exceed_k():
    e0 = helpers.basis(0)
    times = np.arange(10) * 0.02
    windows = [
        helpers.window(1, {0: (e0, 0.9)}),
        helpers.window(1, {0: (e0, 0.9)}),
        helpers.window(1, {}),
        helpers.window(1, {}),
        helpers.window(1, {}),
        helpers.window(1, {}),
        helpers.window(1, {}),
        helpers.window(1, {}),
        helpers.window(1, {0: (e0, 0.9)}),
        helpers.window(1, {0: (e0, 0.9)}),
    ]
    config = TrackingConfig(retention_seconds=0.05)

    trajectory = _run(windows, times, config=config)

    assert trajectory.slots == 1
    assert len(trajectory.tracks) == 2
    first, second = trajectory.tracks
    assert first.track_id == "trk-0001"
    assert second.track_id == "trk-0002"
    assert first.center_times == (0.0, 0.02, 0.04, 0.06)
    np.testing.assert_allclose(first.activity, (0.9, 0.9, 0.0, 0.0), atol=1e-6)
    assert second.center_times == (0.16, 0.18)


def test_random_slot_permutations_follow_embeddings():
    e0, e1 = helpers.basis(0), helpers.basis(1)
    rng = np.random.default_rng(20260929)
    window_count = 8
    times = np.arange(window_count) * 0.02
    windows = []
    first_activity: list[float] = []
    second_activity: list[float] = []
    for index in range(window_count):
        first_is_active = index % 2 == 0
        slot_for_first = 0 if rng.random() < 0.5 else 1
        slot_for_second = 1 - slot_for_first
        first_probability = 0.9 if first_is_active else 0.0
        second_probability = 0.0 if first_is_active else 0.9
        windows.append(
            helpers.window(
                2,
                {
                    slot_for_first: (e0, first_probability),
                    slot_for_second: (e1, second_probability),
                },
            )
        )
        first_activity.append(first_probability)
        second_activity.append(second_probability)

    trajectory = _run(windows, times)

    assert len(trajectory.tracks) == 2
    first, second = trajectory.tracks
    # The second source is only born where it first becomes active (window 1).
    np.testing.assert_allclose(first.activity, first_activity, atol=1e-6)
    np.testing.assert_allclose(second.activity, second_activity[1:], atol=1e-6)

    reference = helpers.activity(times, [first_activity, second_activity], ["s1", "s2"])
    result = evaluate_trajectory(reference, trajectory)
    assert result.f1 == pytest.approx(1.0)
    assert result.id_switch_count == 0


def test_birth_threshold_is_configurable_below_activity_threshold():
    e0 = helpers.basis(0)
    times = np.arange(2) * 0.02
    windows = [
        helpers.window(1, {0: (e0, 0.3)}),
        helpers.window(1, {0: (e0, 0.3)}),
    ]

    default_trajectory = _run(windows, times)
    assert default_trajectory.tracks == ()

    config = TrackingConfig(activity_threshold=0.5, birth_threshold=0.2)
    configured_trajectory = _run(windows, times, config=config)
    assert len(configured_trajectory.tracks) == 1
    np.testing.assert_allclose(
        configured_trajectory.tracks[0].activity, (0.3, 0.3), atol=1e-6
    )


def test_padded_windows_are_skipped():
    e0 = helpers.basis(0)
    times = np.arange(3) * 0.02
    windows = [
        helpers.window(1, {0: (e0, 0.9)}),
        helpers.window(1, {0: (e0, 0.9)}),
        helpers.window(1, {0: (e0, 0.9)}),
    ]

    trajectory = _run(windows, times, center_valid=[False, True, True])

    assert len(trajectory.tracks) == 1
    assert trajectory.tracks[0].center_times == (0.02, 0.04)


def test_empty_sequence_produces_empty_trajectory(tmp_path):
    sequence = PredictionSequence(
        center_times=np.zeros(0, dtype=np.float64),
        embeddings=np.zeros((0, 2, helpers.EMBEDDING_DIM), dtype=np.float32),
        activity=np.zeros((0, 2), dtype=np.float32),
        slot_valid=np.zeros((0, 2), dtype=bool),
        center_valid=np.zeros(0, dtype=bool),
    )

    trajectory = associate_sequence(
        sequence,
        AudioSpan(duration_seconds=1.0, track_start_seconds=0.0),
        helpers.provenance(),
    )

    assert trajectory.tracks == ()
    path = trajectory.save(tmp_path / "trajectory.json")
    loaded = Trajectory.load(path)
    assert loaded.tracks == ()
    assert loaded.slots == 2


def test_all_silent_sequence_produces_no_tracks():
    e0, e1 = helpers.basis(0), helpers.basis(1)
    times = np.arange(3) * 0.02
    windows = [
        helpers.window(2, {0: (e0, 0.0), 1: (e1, 0.0)}),
        helpers.window(2, {0: (e0, 0.0), 1: (e1, 0.0)}),
        helpers.window(2, {0: (e0, 0.0), 1: (e1, 0.0)}),
    ]

    trajectory = _run(windows, times)

    assert trajectory.tracks == ()


@pytest.mark.parametrize("slots", [1, 3, 5])
def test_slot_count_is_configurable(slots):
    e0 = helpers.basis(0)
    times = np.arange(2) * 0.02
    windows = [
        helpers.window(slots, {slots - 1: (e0, 0.9)}),
        helpers.window(slots, {0: (e0, 0.9)}),
    ]

    trajectory = _run(windows, times)

    assert trajectory.slots == slots
    assert len(trajectory.tracks) == 1
    assert trajectory.tracks[0].slot_indices == (slots - 1, 0)


def test_trajectory_roundtrips_through_protocol(tmp_path):
    e0, e1 = helpers.basis(0), helpers.basis(1)
    times = np.arange(4) * 0.02
    windows = [
        helpers.window(2, {0: (e0, 0.9), 1: (e1, 0.9)}),
        helpers.window(2, {0: (e0, 0.9), 1: (e1, 0.9)}),
        helpers.window(2, {0: (e0, 0.0), 1: (e1, 0.9)}),
        helpers.window(2, {0: (e0, 0.9), 1: (e1, 0.9)}),
    ]
    config = TrackingConfig(
        activity_threshold=0.4,
        match_threshold=0.6,
        retention_seconds=3.0,
        prototype_alpha=0.5,
        birth_threshold=0.3,
    )

    trajectory = _run(windows, times, config=config)
    path = trajectory.save(tmp_path / "trajectory.json")
    loaded = Trajectory.load(path)

    assert loaded.to_json_dict() == trajectory.to_json_dict()
    assert loaded.params["match_threshold"] == pytest.approx(0.6)
    assert loaded.params["birth_threshold"] == pytest.approx(0.3)
    # A valid candidate with P=0 is identity memory: a real slot, zero activity.
    assert loaded.tracks[0].slot_indices[2] == 0
    assert loaded.tracks[0].activity[2] == pytest.approx(0.0)


def test_centers_outside_audio_span_are_rejected():
    e0 = helpers.basis(0)
    times = np.arange(3) * 0.02
    windows = [
        helpers.window(1, {0: (e0, 0.9)}),
        helpers.window(1, {0: (e0, 0.9)}),
        helpers.window(1, {0: (e0, 0.9)}),
    ]

    with pytest.raises(TrackingError):
        _run(windows, times, span_duration=0.01)


def test_sequence_validation_rejects_non_unit_valid_candidate():
    with pytest.raises(TrackingError):
        PredictionSequence(
            center_times=np.array([0.0]),
            embeddings=np.zeros((1, 1, helpers.EMBEDDING_DIM), dtype=np.float32),
            activity=np.zeros((1, 1), dtype=np.float32),
            slot_valid=np.ones((1, 1), dtype=bool),
        )
