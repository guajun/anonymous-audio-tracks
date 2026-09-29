"""Hysteresis state machine, release tail and segmentation.

The state machine is a Schmitt trigger over the dB envelope: it turns on when
the level reaches ``on_dbfs`` and only turns off when the level falls below
``off_dbfs``.  A configurable release hold keeps a falling edge active for a
short tail (``release_hold_seconds``), which lets a note tail stay active after
note-off without turning the whole model window into an OR.

The code never reads MIDI/control events; note-on/note-off only exist in
``controls.json`` as reference data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import LabelError


def hysteresis_states(level_db: Any, on_dbfs: float, off_dbfs: float) -> np.ndarray:
    """Boolean per-sample state of a two-threshold Schmitt trigger."""

    levels = np.asarray(level_db, dtype=np.float64)
    if levels.ndim != 1:
        raise LabelError("level_db: expected a 1-D array")
    if off_dbfs > on_dbfs:
        raise LabelError(
            f"thresholds: off_dbfs ({off_dbfs}) must be <= on_dbfs ({on_dbfs})"
        )
    if levels.size == 0:
        return np.zeros(0, dtype=bool)
    index = np.arange(levels.size, dtype=np.int64)
    last_on = np.maximum.accumulate(np.where(levels >= on_dbfs, index, -1))
    last_off = np.maximum.accumulate(np.where(levels < off_dbfs, index, -1))
    return last_on > last_off


def release_extension(
    states: Any, hold_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extend falling edges by ``hold_samples`` samples.

    Returns ``(released, active)`` where ``released`` marks frames that are
    only active because of the tail and ``active`` is ``states | released``.
    """

    mask = np.asarray(states, dtype=bool)
    if mask.ndim != 1:
        raise LabelError("states: expected a 1-D array")
    if isinstance(hold_samples, bool) or not isinstance(hold_samples, (int, np.integer)):
        raise LabelError(f"hold_samples: expected an integer, got {hold_samples!r}")
    hold = int(hold_samples)
    if hold < 0:
        raise LabelError(f"hold_samples: must be >= 0, got {hold}")
    if mask.size == 0 or hold == 0:
        return np.zeros(mask.shape, dtype=bool), mask.copy()
    index = np.arange(mask.size, dtype=np.int64)
    falling = np.zeros(mask.size, dtype=bool)
    falling[1:] = mask[:-1] & ~mask[1:]
    last_falling = np.maximum.accumulate(np.where(falling, index, -1))
    within = (last_falling >= 0) & (index - last_falling < hold)
    released = within & ~mask
    return released, mask | released


@dataclass(frozen=True)
class EnvelopeLabels:
    """Per-sample labels from one envelope."""

    state: np.ndarray
    released: np.ndarray
    active: np.ndarray


def label_envelope(
    level_db: Any,
    on_dbfs: float,
    off_dbfs: float,
    release_hold_samples: int,
) -> EnvelopeLabels:
    """Apply hysteresis plus release and return all per-sample masks."""

    states = hysteresis_states(level_db, on_dbfs, off_dbfs)
    released, active = release_extension(states, release_hold_samples)
    return EnvelopeLabels(state=states, released=released, active=active)


def active_segments(active: Any) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` sample runs where ``active`` is true."""

    mask = np.asarray(active, dtype=bool)
    if mask.ndim != 1:
        raise LabelError("active: expected a 1-D array")
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(edges[position]), int(edges[position + 1]))
        for position in range(0, edges.size, 2)
    ]
