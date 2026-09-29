"""NPZ-backed array protocols: activity labels, features and predictions.

Each binary document is a JSON sidecar (metadata) plus an ``.npz`` payload with
exactly the documented keys.  The sidecar is authoritative for units, shapes and
provenance; the ``.npz`` payload is a plain ``numpy.savez`` archive with
``allow_pickle=False`` so that non-Python readers (including future browser
readers) can decode it without executing Python.

Time convention: ``center_times`` / ``frame_times`` are absolute seconds on the
original-track axis (``t = 0`` is the original track start).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .checks import (
    as_1d,
    as_2d,
    as_3d,
    as_bool_array,
    check_finite,
    check_non_negative_array,
    check_schema_header,
    check_schema_version,
    check_strictly_increasing_array,
    check_unit_interval,
    require_int,
    require_keys,
    require_mapping,
    require_number,
    require_relative_posix_path,
    require_sequence,
    require_str,
    require_unique,
)
from .documents import RunProvenance
from .errors import ContractError
from .jsonio import dump_json, load_json
from .version import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_SLOTS,
    KIND_ACTIVITY,
    KIND_FEATURE,
    KIND_PREDICTION,
    SCHEMA_VERSION,
    UNIT_NORM_TOLERANCE,
)

ACTIVITY_METADATA_FILENAME = "activity.json"
ACTIVITY_ARRAYS_FILENAME = "activity.npz"
ACTIVITY_ARRAY_KEYS = ("center_times", "activity", "valid")
ACTIVITY_UNITS = {
    "center_times": "seconds",
    "activity": "probability",
    "valid": "bool",
}

FEATURE_METADATA_FILENAME = "feature.json"
FEATURE_ARRAYS_FILENAME = "feature.npz"
FEATURE_ARRAY_KEYS = ("frame_times", "features", "valid")
FEATURE_UNITS = {
    "frame_times": "seconds",
    "features": "feature",
    "valid": "bool",
}

PREDICTION_METADATA_FILENAME = "prediction.json"
PREDICTION_ARRAYS_FILENAME = "prediction.npz"
PREDICTION_ARRAY_KEYS = ("center_times", "embeddings", "activity", "slot_valid", "center_valid")
PREDICTION_UNITS = {
    "center_times": "seconds",
    "embeddings": "l2_normalized",
    "activity": "probability",
    "slot_valid": "bool",
    "center_valid": "bool",
}


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


def _load_npz(path: Path, expected_keys: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        missing = sorted(set(expected_keys) - keys)
        unexpected = sorted(keys - set(expected_keys))
        if missing or unexpected:
            raise ContractError(
                f"{path}: NPZ keys mismatch "
                f"(missing={missing}, unexpected={unexpected}); expected {list(expected_keys)}"
            )
        return {key: archive[key] for key in expected_keys}


def _resolve_arrays_path(directory: Path, metadata: Mapping[str, Any], default: str, path: str) -> Path:
    raw = metadata.get("arrays_path", default)
    relative = require_relative_posix_path(raw, f"{path}.arrays_path")
    base = directory.resolve()
    candidate = (directory / relative).resolve()
    if not candidate.is_relative_to(base):
        raise ContractError(f"{path}.arrays_path: must resolve inside the document directory")
    return candidate


def _check_array_specs(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    units: Mapping[str, str],
    path: str,
) -> None:
    specs = require_mapping(metadata.get("arrays"), f"{path}.arrays")
    require_keys(specs, units.keys(), f"{path}.arrays")
    for key in units:
        spec = require_mapping(specs[key], f"{path}.arrays.{key}")
        require_keys(spec, ("dtype", "shape", "unit"), f"{path}.arrays.{key}")
        if np.dtype(spec["dtype"]) != arrays[key].dtype:
            raise ContractError(
                f"{path}.arrays.{key}.dtype: declared {spec['dtype']!r}, "
                f"payload has {arrays[key].dtype.name!r}"
            )
        if list(spec["shape"]) != list(arrays[key].shape):
            raise ContractError(
                f"{path}.arrays.{key}.shape: declared {list(spec['shape'])}, "
                f"payload has {list(arrays[key].shape)}"
            )
        require_str(spec["unit"], f"{path}.arrays.{key}.unit")


def _array_spec(dtype: Any, shape: Sequence[int], unit: str, **extra: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "dtype": np.dtype(dtype).name,
        "shape": [int(size) for size in shape],
        "unit": unit,
    }
    spec.update(extra)
    return spec


def _save_document(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    directory: str | Path,
    metadata_filename: str,
    arrays_filename: str,
) -> tuple[Path, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    arrays_path = target / arrays_filename
    _save_npz(arrays_path, arrays)
    metadata_path = dump_json(target / metadata_filename, metadata)
    return metadata_path, arrays_path


@dataclass
class ActivityData:
    """``activity.json`` + ``activity.npz`` (``kind = activity``).

    ``activity[t, s]`` is the center-time activity probability of raw source
    ``source_ids[s]`` at ``center_times[t]``.  ``valid[t]`` is ``False`` when the
    label window needed zero padding, i.e. the row should be masked by consumers.
    No activity is represented by zeros, never by NaN or by dropping rows.
    """

    center_times: Any
    activity: Any
    valid: Any
    source_ids: Sequence[str]
    sample_rate: int
    hop_seconds: float | None = None
    sample_id: str | None = None
    label_params: Mapping[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.center_times = as_1d(self.center_times, np.float64, "activity.center_times")
        self.activity = as_2d(self.activity, np.float32, "activity.activity")
        self.valid = as_bool_array(self.valid, 1, "activity.valid")
        source_ids = tuple(self.source_ids)
        for position, value in enumerate(source_ids):
            require_str(value, f"activity.source_ids[{position}]")
        require_unique(source_ids, "activity.source_ids")
        self.source_ids = source_ids
        self.validate()

    def validate(self) -> None:
        check_schema_version(self.schema_version)
        require_int(self.sample_rate, "activity.sample_rate", minimum=1)
        if self.hop_seconds is not None:
            require_number(
                self.hop_seconds, "activity.hop_seconds", minimum=0.0, strict_minimum=True
            )
        if self.sample_id is not None:
            require_str(self.sample_id, "activity.sample_id")
        if self.label_params is not None:
            require_mapping(self.label_params, "activity.label_params")

        rows = self.center_times.shape[0]
        if self.activity.shape[0] != rows:
            raise ContractError(
                f"activity.activity: leading length {self.activity.shape[0]} must match "
                f"center_times length {rows}"
            )
        if self.activity.shape[1] != len(self.source_ids):
            raise ContractError(
                f"activity.activity: source length {self.activity.shape[1]} must match "
                f"source_ids length {len(self.source_ids)}"
            )
        if self.valid.shape[0] != rows:
            raise ContractError(
                f"activity.valid: length {self.valid.shape[0]} must match "
                f"center_times length {rows}"
            )
        check_finite(self.center_times, "activity.center_times")
        check_non_negative_array(self.center_times, "activity.center_times")
        check_strictly_increasing_array(self.center_times, "activity.center_times")
        check_finite(self.activity, "activity.activity")
        check_unit_interval(self.activity, "activity.activity")

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "center_times": self.center_times,
            "activity": self.activity,
            "valid": self.valid,
        }

    def to_metadata(self, arrays_filename: str = ACTIVITY_ARRAYS_FILENAME) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": KIND_ACTIVITY,
            "sample_rate": self.sample_rate,
            "source_ids": list(self.source_ids),
            "arrays_path": arrays_filename,
            "arrays": {
                "center_times": _array_spec(
                    np.float64,
                    self.center_times.shape,
                    "seconds",
                    origin="original_track_start",
                ),
                "activity": _array_spec(
                    np.float32,
                    self.activity.shape,
                    "probability",
                    range=[0.0, 1.0],
                ),
                "valid": _array_spec(
                    np.bool_,
                    self.valid.shape,
                    "bool",
                    true_means="center window lies fully inside the rendered audio",
                ),
            },
        }
        for name in ("hop_seconds", "sample_id", "label_params"):
            value = getattr(self, name)
            if value is not None:
                metadata[name] = value
        return metadata

    def save(
        self,
        directory: str | Path,
        *,
        metadata_filename: str = ACTIVITY_METADATA_FILENAME,
        arrays_filename: str = ACTIVITY_ARRAYS_FILENAME,
    ) -> tuple[Path, Path]:
        return _save_document(
            self.to_metadata(arrays_filename),
            self.to_npz_dict(),
            directory,
            metadata_filename,
            arrays_filename,
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        metadata_filename: str = ACTIVITY_METADATA_FILENAME,
    ) -> ActivityData:
        target = Path(directory)
        metadata = load_json(target / metadata_filename)
        check_schema_header(metadata, KIND_ACTIVITY)
        require_keys(metadata, ("sample_rate", "source_ids"), "activity")
        arrays = _load_npz(
            _resolve_arrays_path(target, metadata, ACTIVITY_ARRAYS_FILENAME, "activity"),
            ACTIVITY_ARRAY_KEYS,
        )
        _check_array_specs(metadata, arrays, ACTIVITY_UNITS, "activity")
        source_ids = require_sequence(metadata["source_ids"], "activity.source_ids")
        return cls(
            center_times=arrays["center_times"],
            activity=arrays["activity"],
            valid=arrays["valid"],
            source_ids=source_ids,
            sample_rate=metadata["sample_rate"],
            hop_seconds=metadata.get("hop_seconds"),
            sample_id=metadata.get("sample_id"),
            label_params=metadata.get("label_params"),
            schema_version=metadata.get("schema_version", SCHEMA_VERSION),
        )


@dataclass
class FeatureData:
    """``feature.json`` + ``feature.npz`` (``kind = feature``).

    Per-frame audio features consumed by the output head.  ``features[t, d]`` is
    frame ``t`` of one extractor; ``features`` never carries source semantics.
    ``frame_times`` is authoritative and absolute on the original-track axis;
    ``frame_origin_seconds``/``hop_seconds`` are the nominal grid metadata.
    ``valid[t]`` is ``False`` when the frame was computed from zero-padded audio.
    """

    frame_times: Any
    features: Any
    valid: Any
    feature_name: str
    sample_rate: int
    frame_origin_seconds: float
    hop_seconds: float
    feature_dim: int | None = None
    backend: str | None = None
    preprocessing: Mapping[str, Any] | None = None
    sample_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.frame_times = as_1d(self.frame_times, np.float64, "feature.frame_times")
        self.features = as_2d(self.features, np.float32, "feature.features")
        self.valid = as_bool_array(self.valid, 1, "feature.valid")
        if self.feature_dim is None:
            self.feature_dim = int(self.features.shape[1])
        self.validate()

    def validate(self) -> None:
        check_schema_version(self.schema_version)
        require_str(self.feature_name, "feature.feature_name")
        require_int(self.sample_rate, "feature.sample_rate", minimum=1)
        require_number(self.frame_origin_seconds, "feature.frame_origin_seconds", minimum=0.0)
        require_number(
            self.hop_seconds, "feature.hop_seconds", minimum=0.0, strict_minimum=True
        )
        require_int(self.feature_dim, "feature.feature_dim", minimum=1)
        if self.backend is not None:
            require_str(self.backend, "feature.backend")
        if self.preprocessing is not None:
            require_mapping(self.preprocessing, "feature.preprocessing")
        if self.sample_id is not None:
            require_str(self.sample_id, "feature.sample_id")

        frames = self.frame_times.shape[0]
        if self.features.shape[0] != frames:
            raise ContractError(
                f"feature.features: leading length {self.features.shape[0]} must match "
                f"frame_times length {frames}"
            )
        if self.features.shape[1] != self.feature_dim:
            raise ContractError(
                f"feature.features: dimension {self.features.shape[1]} must match "
                f"feature_dim {self.feature_dim}"
            )
        if self.valid.shape[0] != frames:
            raise ContractError(
                f"feature.valid: length {self.valid.shape[0]} must match "
                f"frame_times length {frames}"
            )
        check_finite(self.frame_times, "feature.frame_times")
        check_non_negative_array(self.frame_times, "feature.frame_times")
        check_strictly_increasing_array(self.frame_times, "feature.frame_times")
        check_finite(self.features, "feature.features")

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "frame_times": self.frame_times,
            "features": self.features,
            "valid": self.valid,
        }

    def to_metadata(self, arrays_filename: str = FEATURE_ARRAYS_FILENAME) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": KIND_FEATURE,
            "feature_name": self.feature_name,
            "feature_dim": self.feature_dim,
            "sample_rate": self.sample_rate,
            "frame_origin_seconds": self.frame_origin_seconds,
            "hop_seconds": self.hop_seconds,
            "arrays_path": arrays_filename,
            "arrays": {
                "frame_times": _array_spec(
                    np.float64,
                    self.frame_times.shape,
                    "seconds",
                    origin="original_track_start",
                ),
                "features": _array_spec(
                    np.float32,
                    self.features.shape,
                    "feature",
                    axes=["frame", "feature_dim"],
                ),
                "valid": _array_spec(
                    np.bool_,
                    self.valid.shape,
                    "bool",
                    true_means="frame was computed from real audio, not zero padding",
                ),
            },
        }
        for name in ("backend", "preprocessing", "sample_id"):
            value = getattr(self, name)
            if value is not None:
                metadata[name] = value
        return metadata

    def save(
        self,
        directory: str | Path,
        *,
        metadata_filename: str = FEATURE_METADATA_FILENAME,
        arrays_filename: str = FEATURE_ARRAYS_FILENAME,
    ) -> tuple[Path, Path]:
        return _save_document(
            self.to_metadata(arrays_filename),
            self.to_npz_dict(),
            directory,
            metadata_filename,
            arrays_filename,
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        metadata_filename: str = FEATURE_METADATA_FILENAME,
    ) -> FeatureData:
        target = Path(directory)
        metadata = load_json(target / metadata_filename)
        check_schema_header(metadata, KIND_FEATURE)
        require_keys(
            metadata,
            (
                "feature_name",
                "feature_dim",
                "sample_rate",
                "frame_origin_seconds",
                "hop_seconds",
            ),
            "feature",
        )
        arrays = _load_npz(
            _resolve_arrays_path(target, metadata, FEATURE_ARRAYS_FILENAME, "feature"),
            FEATURE_ARRAY_KEYS,
        )
        _check_array_specs(metadata, arrays, FEATURE_UNITS, "feature")
        return cls(
            frame_times=arrays["frame_times"],
            features=arrays["features"],
            valid=arrays["valid"],
            feature_name=metadata["feature_name"],
            feature_dim=metadata["feature_dim"],
            sample_rate=metadata["sample_rate"],
            frame_origin_seconds=metadata["frame_origin_seconds"],
            hop_seconds=metadata["hop_seconds"],
            backend=metadata.get("backend"),
            preprocessing=metadata.get("preprocessing"),
            sample_id=metadata.get("sample_id"),
            schema_version=metadata.get("schema_version", SCHEMA_VERSION),
        )


@dataclass
class PredictionData:
    """``prediction.json`` + ``prediction.npz`` (``kind = prediction``).

    Core shapes: ``embeddings`` is ``E[N, K, embedding_dim]`` and ``activity`` is
    ``P[N, K]``.  ``P[n, k]`` is the activity probability at the *center* of
    window ``n`` (``center_times[n]``), not "active anywhere in the window".

    ``slot_valid[n, k]`` marks a usable candidate.  Active candidates hold a
    finite unit-norm embedding even when ``P`` is 0 (identity memory over short
    silence).  Inactive slots are canonical zeros with ``P = 0`` and carry no
    identity; a slot number is never an identity.
    """

    center_times: Any
    embeddings: Any
    activity: Any
    slot_valid: Any
    center_valid: Any
    slots: int | None = None
    embedding_dim: int | None = None
    hop_seconds: float | None = None
    sample_id: str | None = None
    provenance: RunProvenance | Mapping[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.center_times = as_1d(self.center_times, np.float64, "prediction.center_times")
        self.embeddings = as_3d(self.embeddings, np.float32, "prediction.embeddings")
        self.activity = as_2d(self.activity, np.float32, "prediction.activity")
        self.slot_valid = as_bool_array(self.slot_valid, 2, "prediction.slot_valid")
        self.center_valid = as_bool_array(self.center_valid, 1, "prediction.center_valid")
        if self.slots is None:
            self.slots = int(self.embeddings.shape[1])
        if self.embedding_dim is None:
            self.embedding_dim = int(self.embeddings.shape[2])
        if isinstance(self.provenance, Mapping):
            self.provenance = RunProvenance.from_json_dict(self.provenance)
        self.validate()

    def validate(self) -> None:
        check_schema_version(self.schema_version)
        require_int(self.slots, "prediction.slots", minimum=1)
        require_int(self.embedding_dim, "prediction.embedding_dim", minimum=1)
        if self.hop_seconds is not None:
            require_number(
                self.hop_seconds, "prediction.hop_seconds", minimum=0.0, strict_minimum=True
            )
        if self.sample_id is not None:
            require_str(self.sample_id, "prediction.sample_id")
        if self.provenance is not None:
            if not isinstance(self.provenance, RunProvenance):
                raise ContractError("prediction.provenance: expected a provenance object")
            self.provenance.validate()

        windows = self.center_times.shape[0]
        expected_embeddings = (windows, self.slots, self.embedding_dim)
        if self.embeddings.shape != expected_embeddings:
            raise ContractError(
                f"prediction.embeddings: expected shape {list(expected_embeddings)} "
                f"(N, slots, embedding_dim), got {list(self.embeddings.shape)}"
            )
        if self.activity.shape != (windows, self.slots):
            raise ContractError(
                f"prediction.activity: expected shape [{windows}, {self.slots}], "
                f"got {list(self.activity.shape)}"
            )
        if self.slot_valid.shape != (windows, self.slots):
            raise ContractError(
                f"prediction.slot_valid: expected shape [{windows}, {self.slots}], "
                f"got {list(self.slot_valid.shape)}"
            )
        if self.center_valid.shape != (windows,):
            raise ContractError(
                f"prediction.center_valid: expected shape [{windows}], "
                f"got {list(self.center_valid.shape)}"
            )
        check_finite(self.center_times, "prediction.center_times")
        check_non_negative_array(self.center_times, "prediction.center_times")
        check_strictly_increasing_array(self.center_times, "prediction.center_times")
        check_finite(self.embeddings, "prediction.embeddings")
        check_finite(self.activity, "prediction.activity")
        check_unit_interval(self.activity, "prediction.activity")

        inactive = ~self.slot_valid
        if np.any(self.activity[inactive] != 0.0):
            raise ContractError(
                "prediction.activity: inactive slots must have activity probability 0"
            )
        if np.any(self.embeddings[inactive] != 0.0):
            raise ContractError(
                "prediction.embeddings: inactive slots must be canonical all-zero vectors"
            )

        if np.any(self.slot_valid):
            norms = np.linalg.norm(self.embeddings[self.slot_valid], axis=1)
            deviation = np.abs(norms - 1.0)
            if np.any(deviation > UNIT_NORM_TOLERANCE):
                worst = int(np.argmax(deviation))
                rows, cols = np.nonzero(self.slot_valid)
                raise ContractError(
                    "prediction.embeddings: active slot embedding at window "
                    f"{int(rows[worst])}, slot {int(cols[worst])} has L2 norm "
                    f"{float(norms[worst]):.6g}; expected 1 ± {UNIT_NORM_TOLERANCE}"
                )

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "center_times": self.center_times,
            "embeddings": self.embeddings,
            "activity": self.activity,
            "slot_valid": self.slot_valid,
            "center_valid": self.center_valid,
        }

    def to_metadata(self, arrays_filename: str = PREDICTION_ARRAYS_FILENAME) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": KIND_PREDICTION,
            "slots": self.slots,
            "embedding_dim": self.embedding_dim,
            "arrays_path": arrays_filename,
            "arrays": {
                "center_times": _array_spec(
                    np.float64,
                    self.center_times.shape,
                    "seconds",
                    origin="original_track_start",
                ),
                "embeddings": _array_spec(
                    np.float32,
                    self.embeddings.shape,
                    "l2_normalized",
                    axes=["window", "slot", "embedding_dim"],
                    valid_slot_norm="unit L2; inactive slots are all zero",
                ),
                "activity": _array_spec(
                    np.float32,
                    self.activity.shape,
                    "probability",
                    range=[0.0, 1.0],
                ),
                "slot_valid": _array_spec(
                    np.bool_,
                    self.slot_valid.shape,
                    "bool",
                    true_means="slot holds a usable candidate (identity may still be silent)",
                ),
                "center_valid": _array_spec(
                    np.bool_,
                    self.center_valid.shape,
                    "bool",
                    true_means="center window lies fully inside the audio",
                ),
            },
        }
        for name in ("hop_seconds", "sample_id"):
            value = getattr(self, name)
            if value is not None:
                metadata[name] = value
        if isinstance(self.provenance, RunProvenance):
            metadata["provenance"] = self.provenance.to_json_dict()
        return metadata

    def save(
        self,
        directory: str | Path,
        *,
        metadata_filename: str = PREDICTION_METADATA_FILENAME,
        arrays_filename: str = PREDICTION_ARRAYS_FILENAME,
    ) -> tuple[Path, Path]:
        return _save_document(
            self.to_metadata(arrays_filename),
            self.to_npz_dict(),
            directory,
            metadata_filename,
            arrays_filename,
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        metadata_filename: str = PREDICTION_METADATA_FILENAME,
    ) -> PredictionData:
        target = Path(directory)
        metadata = load_json(target / metadata_filename)
        check_schema_header(metadata, KIND_PREDICTION)
        require_keys(metadata, ("slots", "embedding_dim"), "prediction")
        arrays = _load_npz(
            _resolve_arrays_path(target, metadata, PREDICTION_ARRAYS_FILENAME, "prediction"),
            PREDICTION_ARRAY_KEYS,
        )
        _check_array_specs(metadata, arrays, PREDICTION_UNITS, "prediction")
        return cls(
            center_times=arrays["center_times"],
            embeddings=arrays["embeddings"],
            activity=arrays["activity"],
            slot_valid=arrays["slot_valid"],
            center_valid=arrays["center_valid"],
            slots=metadata["slots"],
            embedding_dim=metadata["embedding_dim"],
            hop_seconds=metadata.get("hop_seconds"),
            sample_id=metadata.get("sample_id"),
            provenance=metadata.get("provenance"),
            schema_version=metadata.get("schema_version", SCHEMA_VERSION),
        )


__all__ = [
    "ACTIVITY_ARRAYS_FILENAME",
    "ACTIVITY_METADATA_FILENAME",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_SLOTS",
    "FEATURE_ARRAYS_FILENAME",
    "FEATURE_METADATA_FILENAME",
    "PREDICTION_ARRAYS_FILENAME",
    "PREDICTION_METADATA_FILENAME",
    "ActivityData",
    "FeatureData",
    "PredictionData",
]
