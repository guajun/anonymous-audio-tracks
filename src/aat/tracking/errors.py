"""Exceptions for cross-window source tracking.

Tracking errors describe invalid configuration or invalid prediction input.
They are distinct from :class:`aat.contracts.errors.ContractError`, which is
raised when a produced document violates the shared protocol.
"""

from __future__ import annotations


class TrackingError(ValueError):
    """Invalid tracking configuration, prediction sequence or callback output."""
