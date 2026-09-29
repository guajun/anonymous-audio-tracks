"""Whole-song evaluation tests: mapping, P/R/F1, ID switch, counts, boundaries."""

from __future__ import annotations

import json

import numpy as np
import pytest

from aat.evaluation import EvaluationConfig, evaluate_trajectory

from . import helpers


def test_perfect_tracking_scores_one_and_no_switches():
    times = np.arange(8) * 0.02
    first = [0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0]
    second = [0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.9, 0.9]
    reference = helpers.activity(times, [first, second], ["s1", "s2"])
    prediction = helpers.trajectory(
        [
            {"track_id": "trk-0001", "times": times, "activity": first},
            {"track_id": "trk-0002", "times": times, "activity": second},
        ]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.true_positives == 8
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.id_switch_count == 0
    assert result.reference_source_count == 2
    assert result.predicted_track_count == 2
    assert result.source_count_error == 0
    assert result.boundary_segments_compared == 4
    assert result.boundary_segments_missed == 0
    assert result.boundary_onset_mae == pytest.approx(0.0)
    assert result.boundary_offset_mae == pytest.approx(0.0)
    assert result.mapping == {"s1": "trk-0001", "s2": "trk-0002"}
    # Every reappearance is reported through the same track id: no switch.
    assert result.per_source["s1"].id_switch_count == 0
    assert result.per_source["s2"].id_switch_count == 0
    json.dumps(result.to_dict(), allow_nan=False)


def test_intentional_identity_swap_is_detected():
    times = np.arange(8) * 0.02
    first = [0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0]
    second = [0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.9, 0.9]
    # track A takes s1's first run and s2's second run; track B the reverse.
    track_a = [0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9]
    track_b = [0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0]
    reference = helpers.activity(times, [first, second], ["s1", "s2"])
    prediction = helpers.trajectory(
        [
            {"track_id": "trk-0001", "times": times, "activity": track_a},
            {"track_id": "trk-0002", "times": times, "activity": track_b},
        ]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.id_switch_count == 2
    assert result.per_source["s1"].id_switch_count == 1
    assert result.per_source["s2"].id_switch_count == 1
    assert result.f1 == pytest.approx(0.5)
    assert result.true_positives == 4
    assert result.false_positives == 4
    assert result.false_negatives == 4
    assert result.boundary_segments_compared == 2
    assert result.boundary_segments_missed == 2
    assert len(result.mapping) == 2
    assert set(result.mapping.values()) == {"trk-0001", "trk-0002"}


def test_source_count_error_is_signed_and_absolute():
    times = np.arange(4) * 0.02
    active = [0.9, 0.9, 0.9, 0.9]
    reference = helpers.activity(times, [active, active], ["s1", "s2"])

    over = helpers.trajectory(
        [
            {"track_id": "trk-0001", "times": times, "activity": active},
            {"track_id": "trk-0002", "times": times, "activity": active},
            {"track_id": "trk-0003", "times": times, "activity": active},
        ]
    )
    over_result = evaluate_trajectory(reference, over)
    assert over_result.predicted_track_count == 3
    assert over_result.source_count_error == 1
    assert over_result.source_count_abs_error == 1
    assert over_result.unmapped_track_ids == ("trk-0003",)

    under = helpers.trajectory(
        [{"track_id": "trk-0001", "times": times, "activity": active}]
    )
    under_result = evaluate_trajectory(reference, under)
    assert under_result.predicted_track_count == 1
    assert under_result.source_count_error == -1
    assert under_result.source_count_abs_error == 1
    assert under_result.false_negatives == 4


def test_boundary_error_reports_onset_and_offset_offsets():
    times = np.arange(5) * 0.02
    reference = helpers.activity(times, [[0.9, 0.9, 0.9, 0.0, 0.0]], ["s1"])
    prediction = helpers.trajectory(
        [
            {
                "track_id": "trk-0001",
                "times": times,
                "activity": [0.0, 0.9, 0.9, 0.9, 0.0],
            }
        ]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.boundary_segments_compared == 1
    assert result.boundary_segments_missed == 0
    assert result.boundary_onset_mae == pytest.approx(0.02)
    assert result.boundary_offset_mae == pytest.approx(0.02)
    assert result.true_positives == 2
    assert result.false_positives == 1
    assert result.false_negatives == 1


def test_empty_and_all_silent_reference_do_not_fabricate_metrics():
    times = np.arange(3) * 0.02
    silent = helpers.activity(times, [[0.0, 0.0, 0.0]], ["s1"])
    empty_prediction = helpers.trajectory([])

    result = evaluate_trajectory(silent, empty_prediction)

    assert result.precision is None
    assert result.recall is None
    assert result.f1 is None
    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.id_switch_count == 0
    assert result.reference_source_count == 0
    assert result.predicted_track_count == 0
    assert result.source_count_error == 0
    assert result.boundary_onset_mae is None
    assert result.boundary_offset_mae is None
    assert result.boundary_segments_compared == 0
    assert result.boundary_segments_missed == 0
    json.dumps(result.to_dict(), allow_nan=False)


def test_single_point_boundary_is_measured_not_fabricated():
    times = np.array([0.0])
    reference = helpers.activity(times, [[0.9]], ["s1"])
    prediction = helpers.trajectory(
        [{"track_id": "trk-0001", "times": times, "activity": [0.9]}]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.f1 == pytest.approx(1.0)
    assert result.boundary_segments_compared == 1
    assert result.boundary_onset_mae == pytest.approx(0.0)
    assert result.boundary_offset_mae == pytest.approx(0.0)


def test_low_probability_noise_counts_as_false_positive_at_threshold():
    times = np.arange(4) * 0.02
    reference = helpers.activity(times, [[0.9, 0.9, 0.0, 0.0]], ["s1"])
    prediction = helpers.trajectory(
        [
            {
                "track_id": "trk-0001",
                "times": times,
                "activity": [0.9, 0.9, 0.6, 0.0],
            }
        ]
    )

    at_half = evaluate_trajectory(reference, prediction)
    assert at_half.false_positives == 1
    assert at_half.precision == pytest.approx(2 / 3)

    above_noise = evaluate_trajectory(
        reference, prediction, EvaluationConfig(activity_threshold=0.7)
    )
    assert above_noise.false_positives == 0
    assert above_noise.precision == pytest.approx(1.0)


def test_predicted_only_frames_are_false_positives_without_mapping():
    reference = helpers.activity([0.0, 0.02], [[0.9, 0.9]], ["s1"])
    prediction = helpers.trajectory(
        [{"track_id": "trk-0001", "times": [1.0], "activity": [0.9]}],
        duration_seconds=2.0,
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.mapping == {}
    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 2
    assert result.precision == pytest.approx(0.0)
    assert result.recall == pytest.approx(0.0)
    assert result.f1 == pytest.approx(0.0)
    assert result.predicted_track_count == 1


def test_time_tolerance_aligns_almost_equal_grids():
    reference = helpers.activity([0.0, 0.02, 0.04], [[0.9, 0.9, 0.0]], ["s1"])
    prediction = helpers.trajectory(
        [
            {
                "track_id": "trk-0001",
                "times": [5e-7, 0.02 + 5e-7],
                "activity": [0.9, 0.9],
            }
        ]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.f1 == pytest.approx(1.0)
    assert result.true_positives == 2
    assert result.false_negatives == 0


def test_f1_is_zero_when_positives_exist_but_nothing_is_predicted():
    times = np.arange(4) * 0.02
    reference = helpers.activity(times, [[1.0, 1.0, 1.0, 1.0]], ["s1"])
    prediction = helpers.trajectory([])

    result = evaluate_trajectory(reference, prediction)

    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.false_negatives == 4
    assert result.precision is None
    assert result.recall == pytest.approx(0.0)
    assert result.f1 == pytest.approx(0.0)
    assert result.boundary_segments_compared == 0
    assert result.boundary_segments_missed == 1
    assert result.boundary_onset_mae is None
    assert result.boundary_offset_mae is None


def test_f1_is_zero_when_there_are_only_false_positives():
    times = np.arange(4) * 0.02
    reference = helpers.activity(times, [[0.0, 0.0, 0.0, 0.0]], ["s1"])
    prediction = helpers.trajectory(
        [{"track_id": "trk-0001", "times": times, "activity": [1.0, 1.0, 1.0, 1.0]}]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.true_positives == 0
    assert result.false_positives == 4
    assert result.false_negatives == 0
    assert result.precision == pytest.approx(0.0)
    assert result.recall is None
    assert result.f1 == pytest.approx(0.0)


def test_invalid_reference_frames_are_masked_and_not_predicted_only():
    times = np.array([0.0, 0.02, 0.04, 0.06])
    valid = [False, True, True, False]
    reference = helpers.activity(times, [[1.0, 1.0, 1.0, 1.0]], ["s1"], valid=valid)
    prediction = helpers.trajectory(
        [{"track_id": "trk-0001", "times": times, "activity": [1.0, 1.0, 1.0, 1.0]}]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.f1 == pytest.approx(1.0)
    # The only run touches invalid frames on both sides, so its edges are not
    # acoustic boundaries and stay undefined instead of being scored.
    assert result.boundary_segments_compared == 1
    assert result.boundary_segments_missed == 0
    assert result.boundary_onset_mae is None
    assert result.boundary_offset_mae is None


def test_predicted_points_on_invalid_frames_stay_ignored_but_real_predicted_only_frames_count():
    times = np.array([0.0, 0.02, 0.04, 0.06])
    valid = [False, True, True, False]
    reference = helpers.activity(times, [[1.0, 1.0, 1.0, 1.0]], ["s1"], valid=valid)
    prediction = helpers.trajectory(
        [
            {
                "track_id": "trk-0001",
                "times": [0.0, 0.02, 0.04, 0.06, 0.08],
                "activity": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        ]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.true_positives == 2
    assert result.false_positives == 1  # only the genuine predicted-only 0.08 point
    assert result.false_negatives == 0


def test_id_switch_within_continuous_activity_is_counted():
    times = np.arange(4) * 0.02
    reference = helpers.activity(times, [[1.0, 1.0, 1.0, 1.0]], ["s1"])
    prediction = helpers.trajectory(
        [
            {"track_id": "a", "times": times, "activity": [1.0, 1.0, 0.0, 0.0]},
            {"track_id": "b", "times": times, "activity": [0.0, 0.0, 1.0, 1.0]},
        ]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.id_switch_count == 1
    assert result.per_source["s1"].id_switch_count == 1
    assert result.ambiguous_owner_frames == 0


def test_ambiguous_owner_frames_are_reported_not_resolved():
    times = np.arange(4) * 0.02
    reference = helpers.activity(times, [[1.0, 1.0, 1.0, 1.0]], ["s1"])
    prediction = helpers.trajectory(
        [
            {"track_id": "a", "times": times, "activity": [1.0, 1.0, 1.0, 1.0]},
            {"track_id": "b", "times": times, "activity": [1.0, 1.0, 1.0, 1.0]},
        ]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.id_switch_count == 0
    assert result.ambiguous_owner_frames == 4
    assert result.per_source["s1"].ambiguous_owner_frames == 4


def test_id_switch_memory_survives_missing_and_ambiguous_frames():
    times = np.arange(5) * 0.02
    reference = helpers.activity(times, [[1.0, 1.0, 1.0, 1.0, 1.0]], ["s1"])
    # frame 0 owner a; frame 1 no track; frame 2 ambiguous (a and b);
    # frames 3-4 owner a again: no identity change.
    prediction = helpers.trajectory(
        [
            {"track_id": "a", "times": times, "activity": [1.0, 0.0, 1.0, 1.0, 1.0]},
            {"track_id": "b", "times": times, "activity": [0.0, 0.0, 1.0, 0.0, 0.0]},
        ]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.id_switch_count == 0
    assert result.ambiguous_owner_frames == 1


def test_owner_change_after_gap_is_a_switch():
    times = np.arange(5) * 0.02
    reference = helpers.activity(times, [[1.0, 1.0, 1.0, 1.0, 1.0]], ["s1"])
    prediction = helpers.trajectory(
        [
            {"track_id": "a", "times": times, "activity": [1.0, 0.0, 0.0, 0.0, 0.0]},
            {"track_id": "b", "times": times, "activity": [0.0, 0.0, 0.0, 0.0, 1.0]},
        ]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.id_switch_count == 1


def test_overlapping_sources_do_not_fabricate_owner_switches():
    times = np.arange(3) * 0.02
    reference = helpers.activity(
        times, [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], ["s1", "s2"]
    )
    prediction = helpers.trajectory(
        [
            {"track_id": "a", "times": times, "activity": [1.0, 0.0, 1.0]},
            {"track_id": "b", "times": times, "activity": [0.0, 1.0, 0.0]},
        ]
    )

    result = evaluate_trajectory(reference, prediction)

    # The middle frame has two active GT sources, so b cannot be attributed to
    # either of them by activity alone: it is a missed detection for s1 and a
    # correct detection for s2, not two identity switches.
    assert result.mapping == {"s1": "a", "s2": "b"}
    assert result.id_switch_count == 0
    assert result.per_source["s1"].id_switch_count == 0
    assert result.per_source["s2"].id_switch_count == 0
    assert result.ambiguous_owner_frames == 2


def test_same_owner_before_and_after_gt_overlap_is_not_a_switch():
    times = np.arange(5) * 0.02
    reference = helpers.activity(
        times,
        [[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0]],
        ["s1", "s2"],
    )
    prediction = helpers.trajectory(
        [
            {"track_id": "a", "times": times, "activity": [1.0, 1.0, 0.0, 1.0, 1.0]},
            {"track_id": "b", "times": times, "activity": [0.0, 0.0, 1.0, 0.0, 0.0]},
        ]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.id_switch_count == 0
    assert result.ambiguous_owner_frames == 2  # frame 2 affects both sources


def test_switch_after_ambiguity_is_counted_once():
    times = np.arange(5) * 0.02
    reference = helpers.activity(
        times,
        [[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0]],
        ["s1", "s2"],
    )
    prediction = helpers.trajectory(
        [
            {"track_id": "a", "times": times, "activity": [1.0, 1.0, 1.0, 0.0, 0.0]},
            {"track_id": "b", "times": times, "activity": [0.0, 0.0, 0.0, 1.0, 1.0]},
        ]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.id_switch_count == 1
    assert result.ambiguous_owner_frames == 2


def test_zero_threshold_does_not_fabricate_predictions_on_invalid_frames():
    times = np.arange(3) * 0.02
    reference = helpers.activity(
        times, [[1.0, 1.0, 1.0]], ["s1"], valid=[False, False, False]
    )
    prediction = helpers.trajectory(
        [{"track_id": "a", "times": times, "activity": [1.0, 1.0, 1.0]}]
    )

    result = evaluate_trajectory(
        reference, prediction, EvaluationConfig(activity_threshold=0.0)
    )

    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.reference_source_count == 0
    assert result.predicted_track_count == 0
    assert result.precision is None
    assert result.recall is None
    assert result.f1 is None
    assert result.mapping == {}


def test_zero_threshold_keeps_valid_observed_activity_active():
    times = np.arange(3) * 0.02
    reference = helpers.activity(
        times, [[1.0, 1.0, 1.0]], ["s1"], valid=[False, True, True]
    )
    prediction = helpers.trajectory(
        [{"track_id": "a", "times": times, "activity": [1.0, 1.0, 1.0]}]
    )

    result = evaluate_trajectory(
        reference, prediction, EvaluationConfig(activity_threshold=0.0)
    )

    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.f1 == pytest.approx(1.0)


def test_zero_threshold_does_not_activate_frames_without_prediction_points():
    times = np.arange(3) * 0.02
    reference = helpers.activity(times, [[1.0, 1.0, 1.0]], ["s1"])
    prediction = helpers.trajectory(
        [{"track_id": "a", "times": [0.02], "activity": [1.0]}]
    )

    result = evaluate_trajectory(
        reference, prediction, EvaluationConfig(activity_threshold=0.0)
    )

    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 2


def test_zero_threshold_keeps_observed_zero_activity_active():
    times = np.arange(2) * 0.02
    reference = helpers.activity(times, [[1.0, 1.0]], ["s1"])
    prediction = helpers.trajectory(
        [{"track_id": "a", "times": times, "activity": [0.0, 0.0]}]
    )

    result = evaluate_trajectory(
        reference, prediction, EvaluationConfig(activity_threshold=0.0)
    )

    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_global_mapping_maximises_total_overlap_not_pair_count():
    times = np.arange(102) * 0.02
    reference = helpers.activity(
        times, [[1.0] * 101 + [0.0], [0.0] * 101 + [1.0]], ["s0", "s1"]
    )
    prediction = helpers.trajectory(
        [
            {"track_id": "a", "times": times, "activity": [1.0] * 100 + [0.0, 1.0]},
            {"track_id": "b", "times": times, "activity": [0.0] * 100 + [1.0, 0.0]},
        ],
        duration_seconds=3.0,
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.mapping == {"s0": "a"}
    assert result.true_positives == 100
    assert result.false_positives == 2
    assert result.false_negatives == 2
    assert result.precision == pytest.approx(100 / 102)
    assert result.recall == pytest.approx(100 / 102)
    assert result.f1 == pytest.approx(200 / 204)


def test_evaluation_is_invariant_to_source_and_track_order():
    times = np.arange(102) * 0.02
    reference = helpers.activity(
        times, [[1.0] * 101 + [0.0], [0.0] * 101 + [1.0]], ["s0", "s1"]
    )
    prediction = helpers.trajectory(
        [
            {"track_id": "a", "times": times, "activity": [1.0] * 100 + [0.0, 1.0]},
            {"track_id": "b", "times": times, "activity": [0.0] * 100 + [1.0, 0.0]},
        ],
        duration_seconds=3.0,
    )
    permuted_reference = helpers.activity(
        times, [[0.0] * 101 + [1.0], [1.0] * 101 + [0.0]], ["s1", "s0"]
    )
    permuted_prediction = helpers.trajectory(
        [
            {"track_id": "b", "times": times, "activity": [0.0] * 100 + [1.0, 0.0]},
            {"track_id": "a", "times": times, "activity": [1.0] * 100 + [0.0, 1.0]},
        ],
        duration_seconds=3.0,
    )

    baseline = evaluate_trajectory(reference, prediction)
    permuted = evaluate_trajectory(permuted_reference, permuted_prediction)

    assert permuted.mapping == baseline.mapping
    for name in ("true_positives", "false_positives", "false_negatives"):
        assert getattr(permuted, name) == getattr(baseline, name)
    assert permuted.precision == pytest.approx(baseline.precision)
    assert permuted.recall == pytest.approx(baseline.recall)
    assert permuted.f1 == pytest.approx(baseline.f1)


def test_invalid_gaps_split_boundary_runs_and_masked_points_are_ignored():
    times = np.arange(5) * 0.02
    valid = [True, True, False, True, True]
    reference = helpers.activity(times, [[1.0, 1.0, 1.0, 1.0, 1.0]], ["s1"], valid=valid)
    prediction = helpers.trajectory(
        [{"track_id": "trk-0001", "times": times, "activity": [1.0, 1.0, 1.0, 1.0, 1.0]}]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.true_positives == 4
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.f1 == pytest.approx(1.0)
    assert result.boundary_segments_compared == 2
    assert result.boundary_segments_missed == 0
    assert result.boundary_onset_mae == pytest.approx(0.0)
    assert result.boundary_offset_mae == pytest.approx(0.0)


def test_boundary_missed_counts_fully_missed_sources():
    times = np.arange(4) * 0.02
    reference = helpers.activity(
        times, [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], ["s1", "s2"]
    )
    prediction = helpers.trajectory(
        [{"track_id": "trk-0001", "times": times, "activity": [1.0, 1.0, 1.0, 1.0]}]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.reference_source_count == 2
    assert result.boundary_segments_compared == 1
    assert result.boundary_segments_missed == 1


def test_all_invalid_reference_frames_yield_no_metrics():
    times = np.arange(3) * 0.02
    reference = helpers.activity(
        times, [[1.0, 1.0, 1.0]], ["s1"], valid=[False, False, False]
    )
    prediction = helpers.trajectory(
        [{"track_id": "trk-0001", "times": times, "activity": [1.0, 1.0, 1.0]}]
    )

    result = evaluate_trajectory(reference, prediction)

    assert result.reference_source_count == 0
    assert result.predicted_track_count == 0
    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision is None
    assert result.recall is None
    assert result.f1 is None
    assert result.boundary_segments_compared == 0
    assert result.boundary_segments_missed == 0
    assert result.boundary_onset_mae is None
    assert result.boundary_offset_mae is None
