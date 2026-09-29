"""Exception types raised by the shared contracts and windowing helpers."""

from __future__ import annotations


class ContractError(ValueError):
    """A document or array payload violates the shared protocol."""


class SchemaVersionError(ContractError):
    """A document declares an unsupported ``schema_version``."""


class WindowError(ValueError):
    """Invalid argument or state for center-time window extraction."""


class WindowRangeError(WindowError):
    """A window center lies outside the original-track audio bounds."""
