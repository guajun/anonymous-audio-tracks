"""Gated cross-window association: E/P sequence in, ``trajectory.json`` out.

Per window:

1. expire tracks whose last gated match is older than ``retention_seconds``;
2. gate candidate pairs by cosine similarity (``match_threshold``) and solve a
   one-to-one assignment that maximises (match count, total cosine);
3. record matched tracks, update prototypes only from candidates with
   ``P >= activity_threshold``, and add zero-activity memory points for alive
   unmatched tracks;
4. start a new track for every unmatched candidate with
   ``P >= birth_threshold``.

Slot indices are never identities: matching uses only embeddings, and a track
may be carried by different slots in different windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aat.contracts.arrays import PredictionData
from aat.contracts.documents import AudioSpan, RunProvenance, Track, Trajectory

from .config import TrackingConfig
from .errors import TrackingError
from .matching import cosine_similarity, maximum_assignment
from .sequence import PredictionSequence

#: Tolerance when checking centers against the recorded audio span (matches the
#: protocol tolerance used by :class:`aat.contracts.documents.Trajectory`).
_AUDIO_SPAN_TOLERANCE_SECONDS = 1e-6
#: Safety margin for the "gap is still within retention" comparison.
_RETENTION_EPSILON_SECONDS = 1e-9


@dataclass
class _TrackState:
    track_id: str
    prototype: np.ndarray
    last_match_time: float
    points: list[tuple[float, float, float, int | None]] = field(default_factory=list)


def as_prediction_sequence(value: PredictionSequence | PredictionData) -> PredictionSequence:
    """Coerce a contract ``PredictionData`` document or accept a sequence as-is."""

    if isinstance(value, PredictionSequence):
        return value
    if isinstance(value, PredictionData):
        return PredictionSequence.from_prediction(value)
    raise TrackingError(
        "sequence: expected a PredictionSequence or PredictionData, got "
        f"{type(value).__name__}"
    )


def associate_sequence(
    sequence: PredictionSequence | PredictionData,
    audio: AudioSpan,
    provenance: RunProvenance,
    *,
    config: TrackingConfig | None = None,
    sample_id: str | None = None,
) -> Trajectory:
    """Associate a whole prediction sequence into stable source trajectories.

    ``audio`` is the original-track span of the analyzed audio; every center
    time must lie inside it and the trajectory records the span unchanged so a
    viewer can reject misaligned clips.  ``provenance`` is copied into the
    output; ``mock``/``model``/``annotation`` stay distinguishable.
    """

    sequence = as_prediction_sequence(sequence)
    if not isinstance(audio, AudioSpan):
        raise TrackingError(f"audio: expected an AudioSpan, got {type(audio).__name__}")
    if not isinstance(provenance, RunProvenance):
        raise TrackingError(
            f"provenance: expected a RunProvenance, got {type(provenance).__name__}"
        )
    config = TrackingConfig() if config is None else config
    if not isinstance(config, TrackingConfig):
        raise TrackingError(f"config: expected a TrackingConfig, got {type(config).__name__}")

    times = sequence.center_times
    if times.size:
        start = audio.track_start_seconds
        end = audio.end_seconds
        if times[0] < start - _AUDIO_SPAN_TOLERANCE_SECONDS or (
            times[-1] > end + _AUDIO_SPAN_TOLERANCE_SECONDS
        ):
            raise TrackingError(
                f"center_times: [{times[0]!r}, {times[-1]!r}] lies outside the audio span "
                f"[{start!r}, {end!r}]"
            )

    activity_threshold = config.activity_threshold
    birth_threshold = config.effective_birth_threshold
    retention = config.retention_seconds
    alpha = config.prototype_alpha

    alive: list[_TrackState] = []
    finished: list[_TrackState] = []
    next_track_number = 1

    for window in range(sequence.n_windows):
        time_seconds = float(times[window])

        # 1. Terminate tracks whose silence gap exceeded the retention time.
        still_alive: list[_TrackState] = []
        for track in alive:
            if time_seconds - track.last_match_time > retention + _RETENTION_EPSILON_SECONDS:
                finished.append(track)
            else:
                still_alive.append(track)
        alive = still_alive

        # Padded first/last windows carry no trustworthy evidence: no match, no
        # birth and no memory points.
        if not bool(sequence.center_valid[window]):
            continue

        valid_slots = np.flatnonzero(sequence.slot_valid[window])
        if valid_slots.size == 0:
            for track in alive:
                track.points.append((time_seconds, 0.0, 0.0, None))
            continue

        candidates = sequence.embeddings[window, valid_slots]
        candidate_activity = sequence.activity[window, valid_slots]

        if alive:
            prototypes = np.stack([track.prototype for track in alive])
            similarities = cosine_similarity(prototypes, candidates)
            assignment = maximum_assignment(
                similarities,
                config.match_threshold,
                max_exact_slots=config.max_exact_slots,
            )
        else:
            similarities = np.zeros((0, valid_slots.size), dtype=np.float64)
            assignment = []

        matched_track_to_candidate: dict[int, int] = {}
        matched_candidate_to_track: dict[int, int] = {}
        for track_index, candidate_index in assignment:
            matched_track_to_candidate[track_index] = candidate_index
            matched_candidate_to_track[candidate_index] = track_index

        for track_index, track in enumerate(alive):
            if track_index not in matched_track_to_candidate:
                track.points.append((time_seconds, 0.0, 0.0, None))
                continue
            candidate_index = matched_track_to_candidate[track_index]
            slot = int(valid_slots[candidate_index])
            probability = float(candidate_activity[candidate_index])
            confidence = float(np.clip(similarities[track_index, candidate_index], 0.0, 1.0))
            track.points.append((time_seconds, probability, confidence, slot))
            track.last_match_time = time_seconds
            # Identity memory: low-probability candidates are recorded but never
            # move the prototype.
            if probability >= activity_threshold:
                _update_prototype(track, candidates[candidate_index], alpha)

        # Gated candidates without a one-to-one match may start new identities.
        for candidate_index in range(valid_slots.size):
            if candidate_index in matched_candidate_to_track:
                continue
            probability = float(candidate_activity[candidate_index])
            if probability < birth_threshold:
                continue
            slot = int(valid_slots[candidate_index])
            track = _TrackState(
                track_id=f"trk-{next_track_number:04d}",
                prototype=candidates[candidate_index].astype(np.float64).copy(),
                last_match_time=time_seconds,
            )
            track.points.append((time_seconds, probability, 1.0, slot))
            alive.append(track)
            next_track_number += 1

    states = finished + alive
    states.sort(key=_track_order)
    tracks = tuple(_to_track(state) for state in states)
    return Trajectory(
        audio=audio,
        provenance=provenance,
        tracks=tracks,
        sample_id=sample_id,
        slots=sequence.slots,
        params=config.as_params(),
    )


def track_prediction(
    prediction: PredictionData,
    audio: AudioSpan,
    provenance: RunProvenance,
    *,
    config: TrackingConfig | None = None,
    sample_id: str | None = None,
) -> Trajectory:
    """Convenience wrapper around :func:`associate_sequence`."""

    return associate_sequence(
        prediction, audio, provenance, config=config, sample_id=sample_id
    )


def _update_prototype(track: _TrackState, candidate: np.ndarray, alpha: float) -> None:
    if alpha <= 0.0:
        updated = candidate.astype(np.float64).copy()
    else:
        updated = alpha * track.prototype + (1.0 - alpha) * candidate.astype(np.float64)
    norm = float(np.linalg.norm(updated))
    if norm <= 0.0:
        return
    track.prototype = updated / norm


def _track_order(state: _TrackState) -> int:
    try:
        return int(state.track_id.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 0


def _to_track(state: _TrackState) -> Track:
    center_times = tuple(point[0] for point in state.points)
    activity = tuple(point[1] for point in state.points)
    confidence = tuple(point[2] for point in state.points)
    slot_indices = tuple(point[3] for point in state.points)
    return Track(
        track_id=state.track_id,
        center_times=center_times,
        activity=activity,
        confidence=confidence,
        slot_indices=slot_indices,
    )


__all__ = ["as_prediction_sequence", "associate_sequence", "track_prediction"]
