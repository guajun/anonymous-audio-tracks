"""Anonymous Audio Tracks.

Shared, versioned data contracts (``aat.contracts``) and center-time windowing
(``aat.windowing``) for the data, model-head, trajectory and viewer issues.

Time convention used everywhere: seconds on the original-track axis, where
``t = 0`` is the start of the original composition.  See ``docs/SCHEMAS.md``.
"""

from __future__ import annotations

from .contracts.version import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION", "__version__"]

__version__ = "0.1.0"
