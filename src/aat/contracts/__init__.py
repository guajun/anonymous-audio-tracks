"""Versioned shared contracts for samples, sources, controls, activity, feature,
prediction and trajectory documents.

See ``docs/SCHEMAS.md`` for the complete field-by-field specification.  The
Python objects here are thin, strict wrappers over the JSON/NPZ payloads; they
never load audio, model weights or run outputs.
"""

from __future__ import annotations

from .arrays import (
    ACTIVITY_ARRAYS_FILENAME,
    ACTIVITY_METADATA_FILENAME,
    FEATURE_ARRAYS_FILENAME,
    FEATURE_METADATA_FILENAME,
    PREDICTION_ARRAYS_FILENAME,
    PREDICTION_METADATA_FILENAME,
    ActivityData,
    FeatureData,
    PredictionData,
)
from .checks import check_schema_header, check_schema_version
from .documents import (
    ControlEvent,
    Controls,
    RunProvenance,
    SampleManifest,
    SourceEntry,
    SourceRegistry,
    Track,
    Trajectory,
)
from .errors import ContractError, SchemaVersionError
from .jsonio import dump_json, load_json
from .version import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_HOP_SECONDS,
    DEFAULT_SLOTS,
    DEFAULT_WINDOW_SECONDS,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    UNIT_NORM_TOLERANCE,
)

__all__ = [
    "ACTIVITY_ARRAYS_FILENAME",
    "ACTIVITY_METADATA_FILENAME",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_HOP_SECONDS",
    "DEFAULT_SLOTS",
    "DEFAULT_WINDOW_SECONDS",
    "FEATURE_ARRAYS_FILENAME",
    "FEATURE_METADATA_FILENAME",
    "PREDICTION_ARRAYS_FILENAME",
    "PREDICTION_METADATA_FILENAME",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "UNIT_NORM_TOLERANCE",
    "ActivityData",
    "ContractError",
    "ControlEvent",
    "Controls",
    "FeatureData",
    "PredictionData",
    "RunProvenance",
    "SampleManifest",
    "SchemaVersionError",
    "SourceEntry",
    "SourceRegistry",
    "Track",
    "Trajectory",
    "check_schema_header",
    "check_schema_version",
    "dump_json",
    "load_json",
]
