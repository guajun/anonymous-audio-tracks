"""Renderer version constants.

``RENDER_VERSION`` tracks the renderer implementation, while
``REPORT_SCHEMA_VERSION`` tracks the private ``render_report.json`` /
``surge_probe.json`` shapes (the shared data protocol keeps its own
``aat.contracts.version.SCHEMA_VERSION``).
"""

from __future__ import annotations

RENDER_VERSION = "0.1.0"
REPORT_SCHEMA_VERSION = "0.1.0"

#: Version string recorded in ``manifest.versions["renderer"]`` per engine.
RENDERER_ENGINE = "DawDreamer"

__all__ = ["RENDERER_ENGINE", "RENDER_VERSION", "REPORT_SCHEMA_VERSION"]
