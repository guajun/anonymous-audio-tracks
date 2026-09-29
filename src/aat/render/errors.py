"""Error types for the deterministic renderer (issue #3).

All failures raise a subclass of :class:`RenderError` so callers can present a
single actionable message instead of a traceback.  The classes deliberately
mirror the ``aat.contracts`` error split: configuration, environment and
validation failures are distinct.
"""

from __future__ import annotations


class RenderError(Exception):
    """Base class for every renderer failure."""


class RenderConfigError(RenderError):
    """The render configuration is missing, malformed or out of range."""


class RenderDependencyError(RenderError):
    """The optional render extra (DawDreamer) is not installed."""


class RenderValidationError(RenderError):
    """Rendered audio violated a documented invariant (empty/NaN/clipping/tail)."""


class StemSumError(RenderValidationError):
    """Exported stems do not sum to the exported mix within the LSB tolerance."""


class SurgeProbeError(RenderError):
    """The Surge XT probe could not produce a trustworthy result."""


__all__ = [
    "RenderConfigError",
    "RenderDependencyError",
    "RenderError",
    "RenderValidationError",
    "StemSumError",
    "SurgeProbeError",
]
