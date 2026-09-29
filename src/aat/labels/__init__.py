"""Center-time activity labeling for rendered stems (protocol v0.1.0)."""

from __future__ import annotations

from .activity import (
    EnvelopeLabels,
    active_segments,
    hysteresis_states,
    label_envelope,
    release_extension,
)
from .config import CONFIG_VERSION, LabelConfig
from .energy import (
    SILENCE_DBFS,
    LevelThresholds,
    compute_thresholds,
    envelope_db,
    estimate_noise_floor_db,
    mean_square_envelope,
    rms_envelope,
)
from .errors import LabelError
from .pipeline import LabelResult, SourceSummary, label_stems
from .wav import WavFile, read_wav

__all__ = [
    "CONFIG_VERSION",
    "SILENCE_DBFS",
    "EnvelopeLabels",
    "LabelConfig",
    "LabelError",
    "LabelResult",
    "LevelThresholds",
    "SourceSummary",
    "WavFile",
    "active_segments",
    "compute_thresholds",
    "envelope_db",
    "estimate_noise_floor_db",
    "hysteresis_states",
    "label_envelope",
    "label_stems",
    "mean_square_envelope",
    "read_wav",
    "rms_envelope",
]
