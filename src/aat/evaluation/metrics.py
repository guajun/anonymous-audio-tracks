"""Whole-song trajectory evaluation against center-activity labels.

The activity metrics use one whole-song one-to-one mapping between reference
sources and predicted tracks.  Reference labels are never re-permuted per frame
to hide identity changes; a swapped-identity output therefore loses activity
F1 instead of being remapped into a perfect score.

Reported quantities

* activity true/false positives and negatives, precision, recall, F1 (micro and
  per reference source);
* ID switches, tracked frame by frame along each source's active span: a frame
  with a single active track has a discernible owner, frames with several
  active tracks are counted as ambiguous (never resolved arbitrarily) and
  preserve the previous owner across them and across silence;
* source-count error between active reference sources and active predicted
  tracks;
* activity-boundary error for every reference source, including fully missed
  ones; run edges that touch an explicitly invalid reference frame are not
  acoustic boundaries and are not scored.

Alignment

Predictions are first aligned to the *full* reference grid (valid and invalid
frames), then the ``valid`` mask decides what is evaluated: a predicted point
on an explicitly invalid frame is ignored, while a point that falls on no
reference grid time is a genuine predicted-only frame (false positive when
active).  Invalid frames split reference runs.

Undefined metrics are ``None`` rather than a fabricated 0: F1 is ``None`` only
when ``2*TP + FP + FN`` is zero (no positives and no predictions), while
precision/recall are independently ``None`` when their own denominator is zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from aat.contracts.arrays import ActivityData
from aat.contracts.documents import Trajectory
from aat.tracking.matching import OBJECTIVE_TOTAL_SCORE, maximum_assignment

from .errors import EvaluationError


def _number(value: Any, path: str, *, minimum: float | None, maximum: float | None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{path}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationError(f"{path}: must be finite, got {value!r}")
    if minimum is not None and number < minimum:
        raise EvaluationError(f"{path}: must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise EvaluationError(f"{path}: must be <= {maximum}, got {number}")
    return number


@dataclass(frozen=True)
class EvaluationConfig:
    """Binarisation threshold and time-grid alignment tolerance."""

    activity_threshold: float = 0.5
    time_tolerance_seconds: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "activity_threshold",
            _number(
                self.activity_threshold,
                "config.activity_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "time_tolerance_seconds",
            _number(
                self.time_tolerance_seconds,
                "config.time_tolerance_seconds",
                minimum=0.0,
                maximum=None,
            ),
        )
        if self.time_tolerance_seconds <= 0.0:
            raise EvaluationError("config.time_tolerance_seconds: must be > 0")


@dataclass(frozen=True)
class SourceEvaluation:
    """Per-reference-source activity metrics and identity switches."""

    source_id: str
    track_id: str | None
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    id_switch_count: int
    ambiguous_owner_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "track_id": self.track_id,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "id_switch_count": self.id_switch_count,
            "ambiguous_owner_frames": self.ambiguous_owner_frames,
        }


@dataclass(frozen=True)
class TrajectoryEvaluation:
    """Whole-song evaluation result; ``None`` means "undefined, not zero"."""

    activity_threshold: float
    time_tolerance_seconds: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    id_switch_count: int
    ambiguous_owner_frames: int
    reference_source_count: int
    predicted_track_count: int
    source_count_error: int
    source_count_abs_error: int
    boundary_onset_mae: float | None
    boundary_offset_mae: float | None
    boundary_segments_compared: int
    boundary_segments_missed: int
    mapping: Mapping[str, str]
    unmapped_track_ids: tuple[str, ...]
    per_source: Mapping[str, SourceEvaluation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "activity_threshold": self.activity_threshold,
            "time_tolerance_seconds": self.time_tolerance_seconds,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "id_switch_count": self.id_switch_count,
            "ambiguous_owner_frames": self.ambiguous_owner_frames,
            "reference_source_count": self.reference_source_count,
            "predicted_track_count": self.predicted_track_count,
            "source_count_error": self.source_count_error,
            "source_count_abs_error": self.source_count_abs_error,
            "boundary_onset_mae": self.boundary_onset_mae,
            "boundary_offset_mae": self.boundary_offset_mae,
            "boundary_segments_compared": self.boundary_segments_compared,
            "boundary_segments_missed": self.boundary_segments_missed,
            "mapping": dict(self.mapping),
            "unmapped_track_ids": list(self.unmapped_track_ids),
            "per_source": {
                source_id: evaluation.to_dict()
                for source_id, evaluation in self.per_source.items()
            },
        }


def evaluate_trajectory(
    reference: ActivityData,
    prediction: Trajectory,
    config: EvaluationConfig | None = None,
) -> TrajectoryEvaluation:
    """Evaluate a whole-song ``trajectory`` against ``activity`` labels."""

    if not isinstance(reference, ActivityData):
        raise EvaluationError(
            f"reference: expected ActivityData, got {type(reference).__name__}"
        )
    if not isinstance(prediction, Trajectory):
        raise EvaluationError(
            f"prediction: expected Trajectory, got {type(prediction).__name__}"
        )
    config = EvaluationConfig() if config is None else config
    if not isinstance(config, EvaluationConfig):
        raise EvaluationError(
            f"config: expected EvaluationConfig, got {type(config).__name__}"
        )

    source_ids = tuple(reference.source_ids)
    threshold = config.activity_threshold
    tolerance = config.time_tolerance_seconds

    reference_times = np.asarray(reference.center_times, dtype=np.float64)
    reference_valid = np.asarray(reference.valid, dtype=bool)
    reference_activity = np.asarray(reference.activity, dtype=np.float64)
    reference_active = reference_activity >= threshold
    # Labels on invalid frames are masked out everywhere: activity metrics,
    # runs, ID switches and boundaries.
    reference_truth = reference_active & reference_valid[:, None]

    tracks = tuple(prediction.tracks)
    aligned, predicted_only_active, masked_points = _align_tracks(
        tracks, reference_times, reference_valid, tolerance, threshold
    )
    predicted_active = aligned >= threshold

    mapping_pairs = _global_mapping(reference_truth, predicted_active)
    mapping = {source_ids[source]: tracks[track].track_id for source, track in mapping_pairs}
    source_to_track = dict(mapping_pairs)
    mapped_tracks = {track for _, track in mapping_pairs}

    switches, ambiguous = _id_switch_counts(reference_truth, predicted_active)

    boundary_start_errors: list[float] = []
    boundary_end_errors: list[float] = []
    boundary_segments_compared = 0
    boundary_segments_missed = 0

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    per_source: dict[str, SourceEvaluation] = {}
    for source_index, source_id in enumerate(source_ids):
        ground_truth = reference_truth[:, source_index]
        track_index = source_to_track.get(source_index)
        if track_index is None:
            true_positives = 0
            false_positives = 0
            false_negatives = int(np.count_nonzero(ground_truth))
        else:
            predicted = predicted_active[track_index]
            true_positives = int(np.count_nonzero(ground_truth & predicted))
            false_positives = int(np.count_nonzero(~ground_truth & predicted)) + int(
                predicted_only_active[track_index]
            )
            false_negatives = int(np.count_nonzero(ground_truth & ~predicted))
        precision, recall, f1 = _precision_recall_f1(
            true_positives, false_positives, false_negatives
        )
        per_source[source_id] = SourceEvaluation(
            source_id=source_id,
            track_id=tracks[track_index].track_id if track_index is not None else None,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1=f1,
            id_switch_count=int(switches[source_index]),
            ambiguous_owner_frames=int(ambiguous[source_index]),
        )
        micro_tp += true_positives
        micro_fp += false_positives
        micro_fn += false_negatives

    for track_index in range(len(tracks)):
        if track_index in mapped_tracks:
            continue
        micro_fp += int(np.count_nonzero(predicted_active[track_index])) + int(
            predicted_only_active[track_index]
        )

    # Boundaries cover every reference source, including fully missed ones; the
    # fixed mapping is never re-optimised here.
    for source_index, _source_id in enumerate(source_ids):
        ground_truth_segments = _segments(reference_truth[:, source_index], reference_times)
        track_index = source_to_track.get(source_index)
        if track_index is None:
            boundary_segments_missed += len(ground_truth_segments)
            continue
        predicted_segments = _track_segments(
            tracks[track_index], threshold, masked_points[track_index]
        )
        for start, end, start_index, end_index in ground_truth_segments:
            best_overlap = 0.0
            best_segment: tuple[float, float] | None = None
            for candidate_start, candidate_end, _, _ in predicted_segments:
                overlap = _segment_overlap(
                    start, end, candidate_start, candidate_end, tolerance
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_segment = (candidate_start, candidate_end)
            if best_segment is None:
                boundary_segments_missed += 1
                continue
            boundary_segments_compared += 1
            if _edge_is_real(
                start_index, -1, reference_valid, reference_active[:, source_index]
            ):
                boundary_start_errors.append(abs(best_segment[0] - start))
            if _edge_is_real(
                end_index, +1, reference_valid, reference_active[:, source_index]
            ):
                boundary_end_errors.append(abs(best_segment[1] - end))

    micro_precision, micro_recall, micro_f1 = _precision_recall_f1(
        micro_tp, micro_fp, micro_fn
    )
    reference_source_count = int(np.count_nonzero(reference_truth.sum(axis=0)))
    predicted_track_count = int(
        np.count_nonzero(predicted_active.sum(axis=1) + predicted_only_active)
    )
    unmapped_track_ids = tuple(
        track.track_id for index, track in enumerate(tracks) if index not in mapped_tracks
    )

    return TrajectoryEvaluation(
        activity_threshold=threshold,
        time_tolerance_seconds=tolerance,
        true_positives=micro_tp,
        false_positives=micro_fp,
        false_negatives=micro_fn,
        precision=micro_precision,
        recall=micro_recall,
        f1=micro_f1,
        id_switch_count=int(switches.sum()),
        ambiguous_owner_frames=int(ambiguous.sum()),
        reference_source_count=reference_source_count,
        predicted_track_count=predicted_track_count,
        source_count_error=predicted_track_count - reference_source_count,
        source_count_abs_error=abs(predicted_track_count - reference_source_count),
        boundary_onset_mae=float(np.mean(boundary_start_errors)) if boundary_start_errors else None,
        boundary_offset_mae=float(np.mean(boundary_end_errors)) if boundary_end_errors else None,
        boundary_segments_compared=boundary_segments_compared,
        boundary_segments_missed=boundary_segments_missed,
        mapping=mapping,
        unmapped_track_ids=unmapped_track_ids,
        per_source=per_source,
    )


def _precision_recall_f1(
    true_positives: int, false_positives: int, false_negatives: int
) -> tuple[float | None, float | None, float | None]:
    """Independent undefined-ness for precision/recall, F1 from its own denominator.

    ``F1 = 2TP / (2TP + FP + FN)`` is defined whenever that denominator is
    non-zero, so "positives but no predictions" and "predictions but no
    positives" both give F1 = 0 instead of ``None``.
    """

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else None
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else None
    )
    denominator = 2 * true_positives + false_positives + false_negatives
    f1 = (2.0 * true_positives / denominator) if denominator > 0 else None
    return precision, recall, f1


def _align_tracks(
    tracks: tuple[Any, ...],
    reference_times: np.ndarray,
    reference_valid: np.ndarray,
    tolerance: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Align every track to the full reference grid, then apply the valid mask.

    Returns ``(aligned, predicted_only_active, masked_points)`` where
    ``aligned`` is ``(tracks, frames)`` activity on the full grid (invalid
    columns stay zero), ``predicted_only_active`` counts active points with no
    reference frame within tolerance, and ``masked_points[track]`` marks points
    that landed on an explicitly invalid frame (ignored, not false positives).
    """

    aligned = np.zeros((len(tracks), reference_times.shape[0]), dtype=np.float64)
    predicted_only_active = np.zeros(len(tracks), dtype=np.int64)
    masked_points: list[np.ndarray] = []
    for index, track in enumerate(tracks):
        track_times = np.asarray(track.center_times, dtype=np.float64)
        track_activity = np.asarray(track.activity, dtype=np.float64)
        mapped, unmatched = _map_point_times(track_times, reference_times, tolerance)
        masked = np.zeros(track_times.shape[0], dtype=bool)
        for point in range(track_times.shape[0]):
            frame = mapped[point]
            if frame < 0:
                continue
            if not reference_valid[frame]:
                masked[point] = True
                continue
            if track_activity[point] > aligned[index, frame]:
                aligned[index, frame] = track_activity[point]
        if np.any(unmatched):
            predicted_only_active[index] = int(
                np.count_nonzero(track_activity[unmatched] >= threshold)
            )
        masked_points.append(masked)
    return aligned, predicted_only_active, masked_points


def _map_point_times(
    times: np.ndarray, reference_times: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    count = times.shape[0]
    mapped = np.full(count, -1, dtype=np.int64)
    unmatched = np.ones(count, dtype=bool)
    if count == 0 or reference_times.shape[0] == 0:
        return mapped, unmatched
    positions = np.searchsorted(reference_times, times)
    for point in range(count):
        value = float(times[point])
        best = -1
        best_delta = math.inf
        left = int(positions[point]) - 1
        right = int(positions[point])
        if left >= 0:
            delta = abs(value - float(reference_times[left]))
            if delta <= tolerance:
                best, best_delta = left, delta
        if right < reference_times.shape[0]:
            delta = abs(value - float(reference_times[right]))
            if delta <= tolerance and delta < best_delta:
                best, best_delta = right, delta
        if best >= 0:
            mapped[point] = best
            unmatched[point] = False
    return mapped, unmatched


def _global_mapping(
    reference_truth: np.ndarray, predicted_active: np.ndarray
) -> list[tuple[int, int]]:
    """One whole-song source-to-track mapping maximising total overlap (TP).

    Feasible pairs need at least one frame where both are active.  Unlike the
    online tracker, the evaluation objective is total overlap rather than pair
    count: an unmatched source/track is allowed, because maximising total TP is
    exactly maximising micro-F1 for fixed activity totals.
    """

    source_count = reference_truth.shape[1]
    track_count = predicted_active.shape[0]
    if source_count == 0 or track_count == 0:
        return []
    scores = reference_truth.astype(np.int64).T @ predicted_active.astype(np.int64).T
    feasible = scores.astype(np.float64)
    feasible[scores == 0] = -np.inf
    return maximum_assignment(feasible, 0.0, objective=OBJECTIVE_TOTAL_SCORE)


def _id_switch_counts(
    reference_truth: np.ndarray, predicted_active: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Frame-wise owner changes per source with explicit ambiguity counting.

    A frame with exactly one active track gives a discernible owner; several
    active tracks are ambiguous and never resolved arbitrarily.  The last
    discernible owner is kept across silence, masked frames and ambiguous
    frames, so a reappearance on a different track is counted as a switch.
    """

    frame_count, source_count = reference_truth.shape
    switches = np.zeros(source_count, dtype=np.int64)
    ambiguous = np.zeros(source_count, dtype=np.int64)
    for source in range(source_count):
        last_owner: int | None = None
        for frame in range(frame_count):
            if not reference_truth[frame, source]:
                continue
            active_tracks = np.flatnonzero(predicted_active[:, frame])
            if active_tracks.size == 0:
                continue
            if active_tracks.size > 1:
                ambiguous[source] += 1
                continue
            owner = int(active_tracks[0])
            if last_owner is not None and owner != last_owner:
                switches[source] += 1
            last_owner = owner
    return switches, ambiguous


def _segments(
    active: np.ndarray, times: np.ndarray
) -> list[tuple[float, float, int, int]]:
    """Contiguous runs as ``(start_time, end_time, start_index, end_index)``.

    Invalid reference frames are ``False`` in ``active``, so they split runs.
    """

    segments: list[tuple[float, float, int, int]] = []
    index = 0
    frame_count = active.shape[0]
    while index < frame_count:
        if not active[index]:
            index += 1
            continue
        start = index
        while index < frame_count and active[index]:
            index += 1
        end = index - 1
        segments.append((float(times[start]), float(times[end]), start, end))
    return segments


def _track_segments(
    track: Any, threshold: float, masked_points: np.ndarray
) -> list[tuple[float, float, int, int]]:
    times = np.asarray(track.center_times, dtype=np.float64)
    activity = np.asarray(track.activity, dtype=np.float64)
    active = (activity >= threshold) & ~masked_points
    return _segments(active, times)


def _edge_is_real(
    index: int,
    direction: int,
    reference_valid: np.ndarray,
    reference_truth: np.ndarray,
) -> bool:
    """Whether a run edge is an actual acoustic boundary.

    An edge next to an explicitly invalid frame is unknown, not a boundary, and
    is skipped; the outer edges of the analyzed grid count as boundaries.
    """

    neighbour = index + direction
    if neighbour < 0 or neighbour >= reference_truth.shape[0]:
        return True
    return bool(reference_valid[neighbour]) and not bool(reference_truth[neighbour])


def _segment_overlap(
    ground_truth_start: float,
    ground_truth_end: float,
    predicted_start: float,
    predicted_end: float,
    tolerance: float,
) -> float:
    ground_truth_point = ground_truth_end <= ground_truth_start
    predicted_point = predicted_end <= predicted_start
    if ground_truth_point and predicted_point:
        return 1.0 if abs(ground_truth_start - predicted_start) <= tolerance else 0.0
    if ground_truth_point:
        if predicted_start - tolerance <= ground_truth_start <= predicted_end + tolerance:
            return 1.0
        return 0.0
    if predicted_point:
        if ground_truth_start - tolerance <= predicted_start <= ground_truth_end + tolerance:
            return 1.0
        return 0.0
    return max(0.0, min(ground_truth_end, predicted_end) - max(ground_truth_start, predicted_start))


__all__ = [
    "EvaluationConfig",
    "SourceEvaluation",
    "TrajectoryEvaluation",
    "evaluate_trajectory",
]
