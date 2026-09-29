"""Small strict JSON helpers used by every contract document."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ContractError


def dump_json(path: str | Path, data: Any) -> Path:
    """Write ``data`` as UTF-8 JSON with a trailing newline and stable layout."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)
    target.write_text(text + "\n", encoding="utf-8", newline="\n")
    return target


def load_json(path: str | Path) -> Any:
    """Load a JSON document from ``path``."""

    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - message clarity only
        raise ContractError(f"{source}: invalid JSON: {exc}") from exc
