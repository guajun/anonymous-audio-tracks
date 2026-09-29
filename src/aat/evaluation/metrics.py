"""Whole-song trajectory evaluation against center-activity labels.

The activity metrics use one whole-song one-to-one mapping between reference
sources and predicted tracks.  Reference labels are never re-permuted per frame
to hide identity changes; a swapped-identity output therefore loses activity
F1 instead of being remapped into a perfect score.

Reported quantities

* activity true/false positives and negatives, precision, recall, F1 (micro and
  per reference source);
* ID switches: for each reference source, its active frames are split into
  contiguous runs; each run is attributed to the predicted track with the most
  active frames (ties by summed predicted activity, then track order).  A switch
  is counted when consecutive attributed runs change track, which is exactly the
  "source reappeared and did not reconnect to its original track" failure;
* source-count error between active reference sources and active predicted
  tracks;
* activity-boundary error: for each reference source's active run, the mapped
  track's overlapping predicted run gives onset/offset absolute errors.

Undefined metrics are ``None`` rather than a fabricated 0: an empty song, an
all-silent reference or a run with no overlapping prediction doesn't have a
precision or a boundary error to report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from aat.contracts.arrays import ActivityData
from aat.contracts.documents import Trajectory
from aat.tracking.matching import maximum_assignment

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

    reference_valid = np.asarray(reference.valid, dtype=bool)
    valid_indices = np.flatnonzero(reference_valid)
    reference_times = np.asarray(reference.center_times, dtype=np.float64)[valid_indices]
    reference_activity = np.asarray(reference.activity, dtype=np.float64)[valid_indices]
    reference_active = reference_activity >= threshold

    tracks = tuple(prediction.tracks)
    aligned, predicted_only_active = _align_tracks(
        tracks, reference_times, tolerance, threshold
    )
    predicted_active = aligned >= threshold

    mapping_pairs = _global_mapping(reference_active, predicted_active)
    mapping = {source_ids[source]: tracks[track].track_id for source, track in mapping_pairs}
    source_to_track = dict(mapping_pairs)
    mapped_tracks = {track for _, track in mapping_pairs}

    switches = _id_switch_counts(reference_active, aligned, predicted_active)

    boundary_start_errors: list[float] = []
    boundary_end_errors: list[float] = []
    boundary_segments_missed = 0

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    per_source: dict[str, SourceEvaluation] = {}
    for source_index, source_id in enumerate(source_ids):
        ground_truth = reference_active[:, source_index]
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

    # Boundaries are measured on the mapped pair only, never by remapping GT.
    for source_index, track_index in mapping_pairs:
        ground_truth_segments = _segments(reference_active[:, source_index], reference_times)
        predicted_segments = _track_segments(tracks[track_index], threshold)
        for start, end in ground_truth_segments:
            best_overlap = 0.0
            best_segment: tuple[float, float] | None = None
            for candidate_start, candidate_end in predicted_segments:
                overlap = _segment_overlap(
                    start, end, candidate_start, candidate_end, tolerance
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_segment = (candidate_start, candidate_end)
            if best_segment is None:
                boundary_segments_missed += 1
                continue
            boundary_start_errors.append(abs(best_segment[0] - start))
            boundary_end_errors.append(abs(best_segment[1] - end))

    micro_precision, micro_recall, micro_f1 = _precision_recall_f1(
        micro_tp, micro_fp, micro_fn
    )
    reference_source_count = int(np.count_nonzero(reference_active.sum(axis=0)))
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
        reference_source_count=reference_source_count,
        predicted_track_count=predicted_track_count,
        source_count_error=predicted_track_count - reference_source_count,
        source_count_abs_error=abs(predicted_track_count - reference_source_count),
        boundary_onset_mae=float(np.mean(boundary_start_errors)) if boundary_start_errors else None,
        boundary_offset_mae=float(np.mean(boundary_end_errors)) if boundary_end_errors else None,
        boundary_segments_compared=len(boundary_start_errors),
        boundary_segments_missed=boundary_segments_missed,
        mapping=mapping,
        unmapped_track_ids=unmapped_track_ids,
        per_source=per_source,
    )


def _precision_recall_f1(
    true_positives: int, false_positives: int, false_negatives: int
) -> tuple[float | None, float | None, float | None]:
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
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _align_tracks(
    tracks: tuple[Any, ...],
    reference_times: np.ndarray,
    tolerance: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project track curves onto the valid reference grid.

    Returns the aligned activity matrix ``(tracks, reference_frames)`` plus the
    number of active predicted points that had no reference frame within
    tolerance (they still count as false positives).
    """

    aligned = np.zeros((len(tracks), reference_times.shape[0]), dtype=np.float64)
    predicted_only_active = np.zeros(len(tracks), dtype=np.int64)
    for index, track in enumerate(tracks):
        track_times = np.asarray(track.center_times, dtype=np.float64)
        track_activity = np.asarray(track.activity, dtype=np.float64)
        mapped, unmatched = _map_point_times(track_times, reference_times, tolerance)
        for point in range(track_times.shape[0]):
            frame = mapped[point]
            if frame >= 0 and track_activity[point] > aligned[index, frame]:
                aligned[index, frame] = track_activity[point]
        if np.any(unmatched):
            predicted_only_active[index] = int(
                np.count_nonzero(track_activity[unmatched] >= threshold)
            )
    return aligned, predicted_only_active


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
    reference_active: np.ndarray, predicted_active: np.ndarray
) -> list[tuple[int, int]]:
    """One whole-song source-to-track mapping.

    Feasible pairs need at least one frame where both are active; the mapping
    maximises the number of overlapping pairs and then the total overlap.
    """

    source_count = reference_active.shape[1]
    track_count = predicted_active.shape[0]
    if source_count == 0 or track_count == 0:
        return []
    scores = reference_active.astype(np.int64).T @ predicted_active.astype(np.int64).T
    feasible = scores.astype(np.float64)
    feasible[scores == 0] = -np.inf
    return maximum_assignment(feasible, 0.0)


def _id_switch_counts(
    reference_active: np.ndarray, aligned: np.ndarray, predicted_active: np.ndarray
) -> np.ndarray:
    source_count = reference_active.shape[1]
    switches = np.zeros(source_count, dtype=np.int64)
    for source in range(source_count):
        column = reference_active[:, source]
        last_owner: int | None = None
        index = 0
        while index < column.shape[0]:
            if not column[index]:
                index += 1
                continue
            start = index
            while index < column.shape[0] and column[index]:
                index += 1
            owner = _covering_track(slice(start, index), aligned, predicted_active)
            if owner is None:
                continue
            if last_owner is not None and owner != last_owner:
                switches[source] += 1
            last_owner = owner
    return switches


def _covering_track(
    run: slice, aligned: np.ndarray, predicted_active: np.ndarray
) -> int | None:
    best_track: int | None = None
    best_key: tuple[int, float, int] | None = None
    for track in range(aligned.shape[0]):
        active_count = int(np.count_nonzero(predicted_active[track, run]))
        if active_count == 0:
            continue
        key = (active_count, float(np.sum(aligned[track, run])), -track)
        if best_key is None or key > best_key:
            best_key = key
            best_track = track
    return best_track


def _segments(active: np.ndarray, times: np.ndarray) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    index = 0
    frame_count = active.shape[0]
    while index < frame_count:
        if not active[index]:
            index += 1
            continue
        start = index
        while index < frame_count and active[index]:
            index += 1
        segments.append((float(times[start]), float(times[index - 1])))
    return segments


def _track_segments(track: Any, threshold: float) -> list[tuple[float, float]]:
    times = np.asarray(track.center_times, dtype=np.float64)
    activity = np.asarray(track.activity, dtype=np.float64)
    return _segments(activity >= threshold, times)


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
