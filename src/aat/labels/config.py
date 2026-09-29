"""Explicit, serialisable configuration for center-time activity labels.

Every knob that can change a label is a field of :class:`LabelConfig`: the short
energy window and hop, the absolute/relative thresholds, the noise-floor
estimator, hysteresis, the release/tail policy, smoothing and the probability
mapping.  The labeler writes ``config.to_dict()`` plus ``config.fingerprint()``
into ``ActivityData.label_params`` so a run can be rebuilt and compared.

The defaults are M1 analysis starting points, not calibrated thresholds and not
a claim about subjective audibility.  See ``docs/ACTIVITY_LABELS.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

from .errors import LabelConfigError

#: Version of the labeling algorithm and default parameter set.  Bump for any
#: change that can alter labels even when the config dictionary is untouched.
LABELER_VERSION = "activity-labeler-0.1.0"

#: ``soft``: probability ramps between the off/on thresholds and is gated by the
#: hysteresis state.  ``binary``: probability equals the hysteresis state (0/1).
PROBABILITY_MODES = ("soft", "binary")


def _require_number(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabelConfigError(f"{name}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise LabelConfigError(f"{name}: must be finite, got {value!r}")
    if minimum is not None:
        if strict_minimum and number <= minimum:
            raise LabelConfigError(f"{name}: must be > {minimum}, got {number}")
        if not strict_minimum and number < minimum:
            raise LabelConfigError(f"{name}: must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise LabelConfigError(f"{name}: must be <= {maximum}, got {number}")
    return number


@dataclass(frozen=True)
class LabelConfig:
    """All parameters that define one deterministic labeling pass."""

    #: Algorithm/default-set version recorded next to the parameters.
    labeler_version: str = LABELER_VERSION
    #: Center grid step used when the caller does not supply center times.
    hop_seconds: float = 0.02
    #: Model window (``W``) used for the ``valid`` mask, not for the energy.
    model_window_seconds: float = 2.0
    #: Short-time energy window; must stay much shorter than the model window.
    frame_seconds: float = 0.02
    #: Step of the energy envelope before interpolation to center times.
    frame_hop_seconds: float = 0.01
    #: Reference amplitude for dBFS conversion (1.0 = full scale).
    full_scale: float = 1.0
    #: Numeric lower clamp for the dB envelope.
    floor_db: float = -120.0
    #: Percentile (0-100) of the frame dB values used as the noise floor.
    noise_floor_percentile: float = 10.0
    #: Explicit noise floor in dBFS; ``None`` uses the percentile estimate.
    noise_floor_db: float | None = None
    #: Absolute activation threshold in dBFS.
    abs_on_db: float = -45.0
    #: Required level above the noise floor (dB) to activate.
    rel_on_db: float = 6.0
    #: Hysteresis: deactivation threshold is ``on - hysteresis_db``.
    hysteresis_db: float = 3.0
    #: Release/tail: keep the state active for up to this long below ``off``.
    release_seconds: float = 0.05
    #: Symmetric moving-average smoothing of the dB envelope (0 disables).
    smoothing_seconds: float = 0.0
    #: ``soft`` (ramped probability) or ``binary`` (state only).
    probability_mode: str = "soft"
    #: Peak-to-noise-floor margin below which a non-silent stem is flagged
    #: low SNR / ambiguous (diagnostic only; never gates the labels).
    min_snr_db: float = 10.0
    #: Probability at or above which a center counts as active in the summary.
    summary_active_probability: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.labeler_version, str) or not self.labeler_version:
            raise LabelConfigError("labeler_version: expected a non-empty string")
        _require_number("hop_seconds", self.hop_seconds, minimum=0.0, strict_minimum=True)
        _require_number(
            "model_window_seconds",
            self.model_window_seconds,
            minimum=0.0,
            strict_minimum=True,
        )
        frame_seconds = _require_number(
            "frame_seconds", self.frame_seconds, minimum=0.0, strict_minimum=True
        )
        frame_hop = _require_number(
            "frame_hop_seconds", self.frame_hop_seconds, minimum=0.0, strict_minimum=True
        )
        if frame_hop > frame_seconds:
            raise LabelConfigError(
                "frame_hop_seconds: must be <= frame_seconds so the energy envelope "
                "has no gaps"
            )
        _require_number("full_scale", self.full_scale, minimum=0.0, strict_minimum=True)
        floor_db = _require_number("floor_db", self.floor_db)
        abs_on_db = _require_number("abs_on_db", self.abs_on_db, maximum=0.0)
        if abs_on_db <= floor_db:
            raise LabelConfigError(
                f"abs_on_db: must be above floor_db ({floor_db}), got {abs_on_db}"
            )
        _require_number(
            "noise_floor_percentile", self.noise_floor_percentile, minimum=0.0, maximum=100.0
        )
        if self.noise_floor_db is not None:
            noise_floor_db = _require_number("noise_floor_db", self.noise_floor_db)
            if noise_floor_db <= floor_db:
                raise LabelConfigError(
                    f"noise_floor_db: must be above floor_db ({floor_db}), got {noise_floor_db}"
                )
        _require_number("rel_on_db", self.rel_on_db, minimum=0.0)
        _require_number("hysteresis_db", self.hysteresis_db, minimum=0.0)
        _require_number("release_seconds", self.release_seconds, minimum=0.0)
        _require_number("smoothing_seconds", self.smoothing_seconds, minimum=0.0)
        if self.probability_mode not in PROBABILITY_MODES:
            raise LabelConfigError(
                f"probability_mode: expected one of {list(PROBABILITY_MODES)}, "
                f"got {self.probability_mode!r}"
            )
        _require_number("min_snr_db", self.min_snr_db, minimum=0.0)
        _require_number(
            "summary_active_probability",
            self.summary_active_probability,
            minimum=0.0,
            maximum=1.0,
        )

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-compatible parameter dictionary (stable field order)."""

        return {name: getattr(self, name) for name in self.field_names()}

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        allow_partial: bool = False,
    ) -> "LabelConfig":
        """Rebuild a config from :meth:`to_dict` output (or a partial override).

        Unknown keys are rejected instead of silently ignored so a typo in a
        calibration file cannot masquerade as a reproducible run.
        """

        if not isinstance(data, Mapping):
            raise LabelConfigError(f"config: expected a mapping, got {type(data).__name__}")
        known = cls.field_names()
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise LabelConfigError(f"config: unknown field(s): {', '.join(unknown)}")
        if not allow_partial:
            missing = sorted(set(known) - set(data))
            if missing:
                raise LabelConfigError(f"config: missing field(s): {', '.join(missing)}")
        base = cls() if allow_partial else None
        values: dict[str, Any] = {}
        for name in known:
            if name in data:
                values[name] = data[name]
            elif base is not None:
                values[name] = getattr(base, name)
        return cls(**values)

    def fingerprint(self) -> str:
        """SHA-256 over the canonical JSON form of :meth:`to_dict`."""

        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["LABELER_VERSION", "PROBABILITY_MODES", "LabelConfig"]
