"""Score expansion: patterns and controls on the original-track time axis.

The configuration stores compact, repeating patterns (step grid + MIDI notes).
This module expands them into explicit absolute-time events exactly once, so the
rendered graph, ``controls.json`` and the render report all derive from the same
deterministic score.  All times are absolute seconds on the original track:
``score_time = track_start_seconds + pattern-local time``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from aat.contracts.documents import ControlEvent, Controls

from .config import RenderConfig


@dataclass(frozen=True)
class ScoreEvent:
    """One MIDI note with absolute start/duration in seconds."""

    source_id: str
    source_index: int
    note: int
    velocity: int
    start_seconds: float
    duration_seconds: float
    repetition: int
    step: int


def _iter_notes(config: RenderConfig, source):
    pattern = source.pattern
    loop_seconds = (
        pattern.loop_steps * pattern.step_seconds if pattern.loop_steps is not None else None
    )
    repetitions = (
        max(1, int(math.ceil((config.duration_seconds - 1e-9) / loop_seconds)))
        if loop_seconds is not None
        else 1
    )
    offset = config.track_start_seconds
    for repetition in range(repetitions):
        base = repetition * (loop_seconds or 0.0)
        for note in pattern.notes:
            local_start = base + note.step * pattern.step_seconds
            if local_start >= config.duration_seconds:
                continue
            start = offset + local_start
            duration = note.length_steps * pattern.step_seconds
            yield ScoreEvent(
                source_id=source.source_id,
                source_index=source.index,
                note=note.note,
                velocity=note.velocity,
                start_seconds=start,
                duration_seconds=duration,
                repetition=repetition,
                step=note.step,
            )


def expand_score(config: RenderConfig) -> tuple[ScoreEvent, ...]:
    """Expand every source pattern into an ordered tuple of score events."""

    events: list[ScoreEvent] = []
    for source in config.sources:
        events.extend(_iter_notes(config, source))
    events.sort(
        key=lambda event: (
            event.start_seconds,
            event.source_index,
            event.step,
            event.note,
        )
    )
    return tuple(events)


def source_events(config: RenderConfig, source_id: str) -> tuple[ScoreEvent, ...]:
    """Score events for one source, preserving the global ordering."""

    return tuple(event for event in expand_score(config) if event.source_id == source_id)


def build_controls(config: RenderConfig, score: tuple[ScoreEvent, ...]) -> Controls:
    """Build the protocol ``controls.json`` document from the score.

    Each note yields a ``note_on`` at its start and a ``note_off`` at its end;
    times are non-decreasing, which matches the protocol requirement (several
    events may share one timestamp).
    """

    events: list[tuple[float, int, ControlEvent]] = []
    for position, event in enumerate(score):
        note_on = ControlEvent(
            time_seconds=event.start_seconds,
            source_id=event.source_id,
            event_type="note_on",
            data={
                "note": event.note,
                "velocity": event.velocity,
                "duration_seconds": event.duration_seconds,
                "step": event.step,
                "repetition": event.repetition,
            },
        )
        note_off = ControlEvent(
            time_seconds=event.start_seconds + event.duration_seconds,
            source_id=event.source_id,
            event_type="note_off",
            data={"note": event.note},
        )
        events.append((event.start_seconds, position, note_on))
        events.append((event.start_seconds + event.duration_seconds, position, note_off))
    events.sort(key=lambda item: (item[0], item[2].source_id, item[2].event_type, item[1]))
    return Controls(events=tuple(item[2] for item in events), sample_id=config.sample_id)


__all__ = ["ScoreEvent", "build_controls", "expand_score", "source_events"]
