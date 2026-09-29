"""Reusable validation primitives for the aat protocol documents.

All helpers raise :class:`~aat.contracts.errors.ContractError` with the JSON-ish
location of the offending value so that validation failures are actionable in
tests and in later dataset/renderer issues.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .errors import ContractError, SchemaVersionError
from .version import SUPPORTED_SCHEMA_VERSIONS

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


def check_schema_version(value: Any, path: str = "document.schema_version") -> None:
    """Reject any ``schema_version`` this package does not know."""

    if value not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(SUPPORTED_SCHEMA_VERSIONS)
        raise SchemaVersionError(
            f"{path}: unsupported version {value!r}; supported: {supported}"
        )


def check_schema_header(data: Any, kind: str) -> Mapping[str, Any]:
    """Validate the shared ``schema_version``/``kind`` header of a document."""

    mapping = require_mapping(data, "document")
    check_schema_version(mapping.get("schema_version"))
    document_kind = mapping.get("kind")
    if document_kind != kind:
        raise ContractError(
            f"document.kind: expected {kind!r}, got {document_kind!r}"
        )
    return mapping


def require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path}: expected an object, got {type(value).__name__}")
    return value


def require_sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError(f"{path}: expected an array, got {type(value).__name__}")
    return value


def require_keys(data: Mapping[str, Any], keys: Iterable[str], path: str) -> None:
    missing = sorted(key for key in keys if key not in data)
    if missing:
        raise ContractError(f"{path}: missing required key(s): {', '.join(missing)}")


def require_str(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{path}: expected a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise ContractError(f"{path}: must be a non-empty string")
    return value


def require_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{path}: expected an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ContractError(f"{path}: must be >= {minimum}, got {value}")
    return value


def require_number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
    strict_maximum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{path}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{path}: must be finite, got {value!r}")
    if minimum is not None:
        too_small = number <= minimum if strict_minimum else number < minimum
        if too_small:
            comparator = ">" if strict_minimum else ">="
            raise ContractError(f"{path}: must be {comparator} {minimum}, got {number}")
    if maximum is not None:
        too_large = number >= maximum if strict_maximum else number > maximum
        if too_large:
            comparator = "<" if strict_maximum else "<="
            raise ContractError(f"{path}: must be {comparator} {maximum}, got {number}")
    return number


def require_probability(value: Any, path: str) -> float:
    return require_number(value, path, minimum=0.0, maximum=1.0)


def require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{path}: expected a boolean, got {type(value).__name__}")
    return value


def require_relative_posix_path(value: Any, path: str) -> str:
    """Accept only non-empty relative paths with ``/`` separators."""

    text = require_str(value, path)
    if "\\" in text:
        raise ContractError(f"{path}: must use '/' separators, got {text!r}")
    if text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        raise ContractError(f"{path}: must be a relative path, got {text!r}")
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ContractError(
            f"{path}: must not contain empty, '.' or '..' path components, got {text!r}"
        )
    return text


def require_sha256_hex(value: Any, path: str) -> str:
    text = require_str(value, path)
    if not _SHA256_RE.match(text):
        raise ContractError(f"{path}: expected a lowercase 64-character sha256 hex digest")
    return text


def require_git_commit(value: Any, path: str) -> str:
    text = require_str(value, path)
    if not _HEX_COMMIT_RE.match(text):
        raise ContractError(f"{path}: expected a 7-40 character lowercase hex git commit")
    return text


def require_unique(values: Iterable[Any], path: str) -> None:
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        rendered = ", ".join(repr(item) for item in sorted(duplicates, key=repr))
        raise ContractError(f"{path}: duplicate entries: {rendered}")


def check_strictly_increasing(values: Sequence[float], path: str) -> None:
    for index in range(1, len(values)):
        if not values[index] > values[index - 1]:
            raise ContractError(
                f"{path}: values must be strictly increasing; "
                f"index {index} is {values[index]!r} after {values[index - 1]!r}"
            )


def check_non_decreasing(values: Sequence[float], path: str) -> None:
    for index in range(1, len(values)):
        if values[index] < values[index - 1]:
            raise ContractError(
                f"{path}: values must be non-decreasing; "
                f"index {index} is {values[index]!r} after {values[index - 1]!r}"
            )


def as_1d(value: Any, dtype: Any, path: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ContractError(f"{path}: expected a 1-D array, got shape {tuple(array.shape)}")
    return np.ascontiguousarray(array)


def as_2d(value: Any, dtype: Any, path: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ContractError(f"{path}: expected a 2-D array, got shape {tuple(array.shape)}")
    return np.ascontiguousarray(array)


def as_3d(value: Any, dtype: Any, path: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 3:
        raise ContractError(f"{path}: expected a 3-D array, got shape {tuple(array.shape)}")
    return np.ascontiguousarray(array)


def as_bool_array(value: Any, ndim: int, path: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise ContractError(f"{path}: expected a boolean array, got dtype {array.dtype}")
    if array.ndim != ndim:
        raise ContractError(f"{path}: expected a {ndim}-D array, got shape {tuple(array.shape)}")
    return np.ascontiguousarray(array)


def check_finite(array: np.ndarray, path: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{path}: contains NaN or infinite values")


def check_unit_interval(array: np.ndarray, path: str) -> None:
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ContractError(f"{path}: probabilities must lie in [0, 1]")


def check_strictly_increasing_array(array: np.ndarray, path: str) -> None:
    if array.size > 1 and not np.all(np.diff(array) > 0):
        raise ContractError(f"{path}: values must be strictly increasing")


def check_non_negative_array(array: np.ndarray, path: str) -> None:
    if np.any(array < 0.0):
        raise ContractError(f"{path}: values must be >= 0")
