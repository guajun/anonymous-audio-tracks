"""Configurable thresholds for cross-window association.

All association thresholds live here so that no magic number is buried in the
tracker.  ``birth_threshold`` defaults to ``activity_threshold``; resolving the
default is explicit via :attr:`TrackingConfig.effective_birth_threshold`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import TrackingError


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackingError(f"{path}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise TrackingError(f"{path}: must be finite, got {value!r}")
    if minimum is not None:
        too_small = number <= minimum if strict_minimum else number < minimum
        if too_small:
            comparator = ">" if strict_minimum else ">="
            raise TrackingError(f"{path}: must be {comparator} {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise TrackingError(f"{path}: must be <= {maximum}, got {number}")
    return number


@dataclass(frozen=True)
class TrackingConfig:
    """Thresholds and rules used by :func:`aat.tracking.associate_sequence`.

    Parameters
    ----------
    activity_threshold:
        ``P >= activity_threshold`` marks a matched candidate as *active*.  Only
        active candidates update the track prototype or give birth to a track.
        Candidates below it can still be matched and recorded (identity memory)
        but never move the prototype.
    match_threshold:
        Minimum cosine similarity between a track prototype and a candidate
        embedding for the pair to be considered by the one-to-one assignment.
        Pairs below the gate are disabled before assignment.
    retention_seconds:
        How long a track survives without any gated match.  While it survives,
        every window adds a zero-activity memory point; a source that reappears
        within the retention window is reconnected to the same track.
    prototype_alpha:
        Exponential moving-average weight of the previous prototype when an
        active candidate updates a track (``0`` replaces, close to ``1`` barely
        moves).  Updated prototypes are re-normalised to unit L2 norm.
    birth_threshold:
        Minimum ``P`` for an unmatched candidate to start a new track.  ``None``
        means the same value as ``activity_threshold``.
    max_exact_slots:
        Slot counts up to this value use the exact bitmask assignment in
        :func:`aat.tracking.matching.maximum_assignment`; larger searches fall
        back to a deterministic greedy assignment.
    """

    activity_threshold: float = 0.5
    match_threshold: float = 0.7
    retention_seconds: float = 1.0
    prototype_alpha: float = 0.9
    birth_threshold: float | None = None
    max_exact_slots: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "activity_threshold",
            _number(self.activity_threshold, "config.activity_threshold", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "match_threshold",
            _number(self.match_threshold, "config.match_threshold", minimum=-1.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "retention_seconds",
            _number(self.retention_seconds, "config.retention_seconds", minimum=0.0),
        )
        object.__setattr__(
            self,
            "prototype_alpha",
            _number(
                self.prototype_alpha,
                "config.prototype_alpha",
                minimum=0.0,
                maximum=1.0,
                strict_minimum=False,
            ),
        )
        if self.prototype_alpha >= 1.0:
            raise TrackingError("config.prototype_alpha: must be < 1")
        if self.birth_threshold is not None:
            object.__setattr__(
                self,
                "birth_threshold",
                _number(self.birth_threshold, "config.birth_threshold", minimum=0.0, maximum=1.0),
            )
        if isinstance(self.max_exact_slots, bool) or not isinstance(self.max_exact_slots, int):
            raise TrackingError(
                f"config.max_exact_slots: expected an integer, got "
                f"{type(self.max_exact_slots).__name__}"
            )
        if not 1 <= self.max_exact_slots <= 20:
            raise TrackingError(
                f"config.max_exact_slots: must be in [1, 20], got {self.max_exact_slots}"
            )

    @property
    def effective_birth_threshold(self) -> float:
        """Birth threshold with ``birth_threshold or activity_threshold`` resolved."""

        if self.birth_threshold is None:
            return self.activity_threshold
        return self.birth_threshold

    def as_params(self) -> dict[str, Any]:
        """JSON-serialisable snapshot recorded in ``trajectory.params``."""

        return {
            "activity_threshold": self.activity_threshold,
            "match_threshold": self.match_threshold,
            "retention_seconds": self.retention_seconds,
            "prototype_alpha": self.prototype_alpha,
            "birth_threshold": self.effective_birth_threshold,
            "max_exact_slots": self.max_exact_slots,
            "matching": "gated_max_cardinality_then_cosine",
        }


__all__ = ["TrackingConfig"]
