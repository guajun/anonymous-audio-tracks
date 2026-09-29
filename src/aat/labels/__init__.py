"""Rendered-stem center-time activity labels (issue #4).

The package turns protocol stem audio into validated ``ActivityData`` documents.
It never uses MIDI/parameter control events as truth and never tracks identity
or re-detects repeated onsets.
"""

from __future__ import annotations

from .activity import (
    StemActivation,
    hysteresis_state,
    label_stem,
    probability_curve,
    resolve_thresholds,
)
from .config import LABELER_VERSION, PROBABILITY_MODES, LabelConfig
from .energy import EnergyEnvelope, compute_energy_envelope
from .errors import LabelAudioError, LabelConfigError, LabelError, WavError
from .pipeline import LabelResult, build_summary, label_stems
from .wav import WavFormat, decode_wav_bytes, read_wav

__all__ = [
    "LABELER_VERSION",
    "PROBABILITY_MODES",
    "EnergyEnvelope",
    "LabelAudioError",
    "LabelConfig",
    "LabelConfigError",
    "LabelError",
    "LabelResult",
    "StemActivation",
    "WavError",
    "WavFormat",
    "build_summary",
    "compute_energy_envelope",
    "decode_wav_bytes",
    "hysteresis_state",
    "label_stem",
    "label_stems",
    "probability_curve",
    "read_wav",
    "resolve_thresholds",
]
