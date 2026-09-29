"""Small strict JSON helpers used by every contract document.

Writers refuse NaN/Infinity anywhere in the payload (``allow_nan=False``);
readers reject the non-standard ``NaN``/``Infinity`` literals and any float that
overflows to infinity, no matter how deeply it is nested.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .errors import ContractError


def dumps_json(data: Any) -> str:
    """Serialise ``data`` to a stable JSON string.

    Raises :class:`ContractError` for payloads JSON cannot represent safely,
    including NaN/Infinity at any nesting depth.
    """

    try:
        return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False)
    except ValueError as exc:
        raise ContractError(f"document: not JSON-serialisable: {exc}") from exc


def dump_json(path: str | Path, data: Any) -> Path:
    """Write ``data`` as UTF-8 JSON with a trailing newline and stable layout."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps_json(data) + "\n", encoding="utf-8", newline="\n")
    return target


def _reject_constant(name: str) -> Any:
    raise ContractError(f"document: non-finite JSON constant {name!r} is not allowed")


def _reject_non_finite(value: Any, path: str = "document") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{position}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{path}: non-finite float is not allowed")


def load_json(path: str | Path) -> Any:
    """Load a JSON document, rejecting non-finite numbers at any nesting depth."""

    source = Path(path)
    try:
        data = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ContractError(f"{source}: invalid JSON: {exc}") from exc
    _reject_non_finite(data)
    return data
