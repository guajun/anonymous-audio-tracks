"""Errors raised by the activity labeler.

These are plain ``ValueError`` subclasses: a label run consumes renderer output
and never raises protocol errors from :mod:`aat.contracts`.  Protocol violations
of the produced ``ActivityData`` are still raised by the contract layer itself.
"""

from __future__ import annotations


class LabelError(ValueError):
    """Base class for activity-labeling input/configuration errors."""


class LabelConfigError(LabelError):
    """Invalid or unknown activity label configuration."""


class LabelAudioError(LabelError):
    """Unusable stem audio (empty, wrong shape, rate mismatch, ...)."""


class WavError(LabelAudioError):
    """Malformed or unsupported WAV payload."""


__all__ = ["LabelAudioError", "LabelConfigError", "LabelError", "WavError"]
