"""Errors raised by the activity-labeling package."""

from __future__ import annotations


class LabelError(ValueError):
    """An activity-labeling input, configuration or WAV payload is invalid."""
