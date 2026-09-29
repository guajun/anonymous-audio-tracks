"""Labeling configuration: windows, thresholds, hysteresis and tail policy.

Two different windows are deliberately kept apart:

``center_window_seconds``
    The model center window.  It is used only for the protocol ``valid`` mask
    ("the center window lies fully inside the rendered audio") and must not be
    confused with the local energy window.
``energy_window_seconds``
    The short local acoustic window used to compute the RMS envelope that all
    thresholds operate on.  It is much shorter than the model window.

The frozen config is serialized verbatim into ``activity.json`` under
``label_params`` together with a canonical SHA-256, so a label file can be
checked and the config rebuilt from its own metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from .errors import LabelError

#: Identifier of the threshold/labeling semantics; changing behavior requires a
#: new version string so old label files never silently change meaning.
CONFIG_VERSION = "activity-label-v1"


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabelError(f"{path}: expected a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise LabelError(f"{path}: must be finite, got {value!r}")
    return result


def _positive(value: Any, path: str) -> float:
    result = _number(value, path)
    if result <= 0.0:
        raise LabelError(f"{path}: must be > 0, got {result}")
    return result


def _non_negative(value: Any, path: str) -> float:
    result = _number(value, path)
    if result < 0.0:
        raise LabelError(f"{path}: must be >= 0, got {result}")
    return result


def _fraction(value: Any, path: str) -> float:
    result = _number(value, path)
    if not 0.0 <= result <= 1.0:
        raise LabelError(f"{path}: must be in [0, 1], got {result}")
    return result


@dataclass(frozen=True)
class LabelConfig:
    """Deterministic settings for center-time activity labels.

    The effective "on" threshold is
    ``max(absolute_threshold_dbfs, min(noise_floor_dbfs + noise_floor_margin_db,
    peak_dbfs - peak_relative_threshold_db))``: the absolute threshold keeps
    very quiet material quiet, the noise-floor gate keeps the label above the
    measured floor, and the relative-to-peak gate prevents an adaptive floor
    from swallowing a continuous signal.  The state machine then switches off
    only below ``on_dbfs - hysteresis_db`` and holds a short release tail.
    """

    # Output grid (also fixes the protocol ``hop_seconds`` metadata).
    hop_seconds: float = 0.02
    # Model center window, protocol ``valid`` only.
    center_window_seconds: float = 2.0
    # Local acoustic energy window, thresholding only.
    energy_window_seconds: float = 0.05
    # Thresholds.
    absolute_threshold_dbfs: float = -60.0
    peak_relative_threshold_db: float = 35.0
    noise_floor_percentile: float = 10.0
    noise_floor_margin_db: float = 6.0
    # Hysteresis and tail release.
    hysteresis_db: float = 3.0
    release_hold_seconds: float = 0.15
    # Readable summary flags.
    low_snr_threshold_db: float = 12.0
    ambiguity_margin_db: float = 3.0
    ambiguity_fraction_threshold: float = 0.2

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _positive(self.hop_seconds, "hop_seconds")
        _positive(self.center_window_seconds, "center_window_seconds")
        _positive(self.energy_window_seconds, "energy_window_seconds")
        _number(self.absolute_threshold_dbfs, "absolute_threshold_dbfs")
        _positive(self.peak_relative_threshold_db, "peak_relative_threshold_db")
        percentile = _number(self.noise_floor_percentile, "noise_floor_percentile")
        if not 0.0 <= percentile <= 100.0:
            raise LabelError(
                f"noise_floor_percentile: must be in [0, 100], got {percentile}"
            )
        _non_negative(self.noise_floor_margin_db, "noise_floor_margin_db")
        _non_negative(self.hysteresis_db, "hysteresis_db")
        _non_negative(self.release_hold_seconds, "release_hold_seconds")
        _non_negative(self.low_snr_threshold_db, "low_snr_threshold_db")
        _non_negative(self.ambiguity_margin_db, "ambiguity_margin_db")
        _fraction(self.ambiguity_fraction_threshold, "ambiguity_fraction_threshold")

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.field_names()}

    def canonical_payload(self) -> dict[str, Any]:
        return {"version": CONFIG_VERSION, **self.to_dict()}

    def config_sha256(self) -> str:
        canonical = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_label_params(self) -> dict[str, Any]:
        """Snapshot written into ``activity.json`` ``label_params``."""

        return {
            "version": CONFIG_VERSION,
            "sha256": self.config_sha256(),
            "config": self.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LabelConfig:
        """Rebuild a config from a mapping; unknown keys are rejected."""

        if not isinstance(data, Mapping):
            raise LabelError("label config: expected a mapping")
        known = set(cls.field_names())
        unknown = sorted(set(data) - known)
        if unknown:
            raise LabelError("label config: unknown key(s): " + ", ".join(unknown))
        return cls(**{name: data[name] for name in data})

    @classmethod
    def from_label_params(cls, label_params: Mapping[str, Any]) -> LabelConfig:
        """Rebuild and verify the config snapshot stored in ``label_params``."""

        if not isinstance(label_params, Mapping):
            raise LabelError("label_params: expected a mapping")
        version = label_params.get("version")
        if version != CONFIG_VERSION:
            raise LabelError(
                f"label_params.version: expected {CONFIG_VERSION!r}, got {version!r}"
            )
        raw = label_params.get("config")
        if not isinstance(raw, Mapping):
            raise LabelError("label_params.config: expected a mapping")
        config = cls.from_dict(raw)
        declared = label_params.get("sha256")
        actual = config.config_sha256()
        if declared is not None and declared != actual:
            raise LabelError(
                f"label_params.sha256: declared {declared!r} does not match "
                f"config {actual!r}"
            )
        return config
