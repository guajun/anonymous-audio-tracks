"""严格 JSON 载入器。

Python 标准库 ``json`` 默认接受 ``NaN``/``Infinity``/``-Infinity`` 字面量、
静默接受重复键（last-wins）、把 ``1e999`` 解析成 ``inf``。本模块对这三类问题
零容忍，全部拒绝并返回带 JSON Pointer 的错误（重复键只报告键名，见下）。

策略（冻结）：

- ``NaN`` / ``Infinity`` / ``-Infinity`` 字面量 -> ``E_PARSE``（拒绝）
- 语法错误 / 非 UTF-8 / BOM -> ``E_PARSE``
- 对象重复键 -> ``E_DUPLICATE_KEY``（拒绝，不 last-wins；**只报告键名**，
  因为 ``object_pairs_hook`` 在自底向上构建对象时拿不到完整路径——这是已知限制）
- ``1e999`` 等语法合法但解析为 inf/nan 的数字 -> ``E_NONFINITE``（带精确 Pointer）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import Issue, pointer_of


class _ConstantError(ValueError):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class _DuplicateKeyError(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _reject_constant(name: str) -> Any:
    raise _ConstantError(name)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateKeyError(key)
        seen[key] = value
    return seen


def _walk_nonfinite(node: Any, parts: list[Any], out: list[Issue]) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        if node != node or node in (float("inf"), float("-inf")):
            out.append(
                Issue(
                    "parse",
                    "E_NONFINITE",
                    pointer_of(parts),
                    f"数字必须是有限数，得到 {node!r}（拒绝 NaN/Inf）",
                )
            )
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _walk_nonfinite(value, parts + [key], out)
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            _walk_nonfinite(value, parts + [index], out)


def load_text(text: str, *, source: str = "<text>") -> tuple[Any | None, list[Issue]]:
    """把 JSON 文本解析成 Python 对象；失败时对象为 ``None`` 并给出错误列表。"""
    issues: list[Issue] = []
    try:
        data = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except _ConstantError as exc:
        return None, [
            Issue(
                "parse",
                "E_PARSE",
                "",
                f"拒绝非标准 JSON 常量 {exc.name!r}（NaN/Infinity/-Infinity 不是合法 JSON）",
            )
        ]
    except _DuplicateKeyError as exc:
        return None, [
            Issue(
                "parse",
                "E_DUPLICATE_KEY",
                "",
                f"对象存在重复键 {exc.key!r}（拒绝 last-wins；重复键无法给出完整路径）",
            )
        ]
    except json.JSONDecodeError as exc:
        return None, [
            Issue(
                "parse",
                "E_PARSE",
                "",
                f"JSON 语法错误（{source} 第 {exc.lineno} 行第 {exc.colno} 列）: {exc.msg}",
            )
        ]
    except ValueError as exc:  # 其他解析期 ValueError（如 BOM）
        return None, [Issue("parse", "E_PARSE", "", f"JSON 解析失败: {exc}")]

    _walk_nonfinite(data, [], issues)
    if issues:
        return None, issues
    return data, []


def load_path(path: str | Path) -> tuple[Any | None, list[Issue]]:
    """读取文件（严格 UTF-8，拒绝 BOM/非 UTF-8）并解析。"""
    file_path = Path(path)
    try:
        raw = file_path.read_bytes()
    except OSError as exc:
        return None, [Issue("parse", "E_PARSE", "", f"无法读取文件 {file_path}: {exc}")]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, [
            Issue("parse", "E_PARSE", "", f"文件必须是 UTF-8 文本（{file_path}）: {exc}")
        ]
    return load_text(text, source=str(file_path))
