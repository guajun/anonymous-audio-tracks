"""Protocol version, document kinds and shared defaults.

Bumping ``SCHEMA_VERSION`` is a deliberate, documented contract change: every
JSON/NPZ reader rejects versions it does not know.
"""

from __future__ import annotations

SCHEMA_VERSION = "0.1.0"
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION,)

KIND_SAMPLE_MANIFEST = "sample_manifest"
KIND_SOURCES = "sources"
KIND_CONTROLS = "controls"
KIND_ACTIVITY = "activity"
KIND_FEATURE = "feature"
KIND_PREDICTION = "prediction"
KIND_TRAJECTORY = "trajectory"

KINDS = frozenset(
    {
        KIND_SAMPLE_MANIFEST,
        KIND_SOURCES,
        KIND_CONTROLS,
        KIND_ACTIVITY,
        KIND_FEATURE,
        KIND_PREDICTION,
        KIND_TRAJECTORY,
    }
)

#: Default number of per-window candidate slots.  K stays configurable per run.
DEFAULT_SLOTS = 8
#: Embedding dimension fixed by protocol 0.1.0.  Only K is configurable.
DEFAULT_EMBEDDING_DIM = 128
#: Absolute tolerance for "valid candidate embedding is a unit vector".
UNIT_NORM_TOLERANCE = 1e-3
#: Nominal window/hop prototype from ``configs/experiment.toml`` (not final).
DEFAULT_WINDOW_SECONDS = 2.0
DEFAULT_HOP_SECONDS = 0.02
