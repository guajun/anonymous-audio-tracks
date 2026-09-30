"""校验流水线（冻结接口）。

顺序：严格解析（parse） → 版本匹配（version） → 结构（structure） → 语义（semantic）。

- 版本不匹配时只返回一个 ``E_VERSION`` 错误并停止（未来版本兼容）。
- 解析失败时不跑后续层。
- 结构与语义都会执行并合并报错，逐条带 JSON Pointer。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import Issue, Report
from .loader import depth_issues, load_path, load_text, numeric_issues
from .schema_check import jsonschema_available, validate_with_jsonschema
from .semantic import validate_semantic
from .structure import validate_structure

#: 冻结协议版本
SCHEMA_VERSION = "agentic-audio-tracks/v1"

#: 冻结目录（agentic/schema）
BASE_DIR = Path(__file__).resolve().parent.parent
#: 结构 schema（Draft 2020-12 便携子集，可在浏览器使用）
SCHEMA_PATH = BASE_DIR / "schema" / "agentic-audio-tracks-v1.schema.json"
#: 语义规则表（机器可读）
SEMANTIC_RULES_PATH = BASE_DIR / "schema" / "semantic_rules.json"

_ENGINES = ("auto", "jsonschema", "stdlib")


class DependencyError(RuntimeError):
    """请求了不可用的校验引擎。"""


def load_schema(schema_path: Path | None = None) -> dict[str, Any]:
    path = schema_path or SCHEMA_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_engine(engine: str) -> str:
    if engine not in _ENGINES:
        raise DependencyError(f"unknown engine {engine!r}; expected one of {_ENGINES}")
    if engine == "auto":
        return "jsonschema" if jsonschema_available() else "stdlib"
    if engine == "jsonschema" and not jsonschema_available():
        raise DependencyError(
            "jsonschema 引擎不可用：请先安装 pin 的依赖 "
            "（uv run --no-project --with jsonschema==4.26.0 ... 或 uv pip install -r agentic/schema/requirements.txt）"
        )
    return engine


def _structure_issues(data: Any, schema: dict[str, Any], engine: str) -> list[Issue]:
    if engine == "jsonschema":
        return validate_with_jsonschema(data, schema)
    return validate_structure(data, schema)


def validate_document(
    data: Any,
    *,
    engine: str = "auto",
    schema_path: Path | None = None,
    source: str = "<document>",
) -> Report:
    """校验已解析的文档（跳过 JSON 语法解析，但仍执行深度/数值政策）。"""
    resolved = resolve_engine(engine)
    issues: list[Issue] = []

    # parse 层政策（与 validate_text/validate_file 一致）：超深/超范围数字直接受控拒绝
    issues.extend(depth_issues(data))
    if not issues:
        issues.extend(numeric_issues(data))
    if issues:
        return Report(source=source, ok=False, engine=resolved, issues=tuple(issues))

    version = data.get("schema_version") if isinstance(data, dict) else None
    if version != SCHEMA_VERSION:
        issues.append(
            Issue(
                "version",
                "E_VERSION",
                "/schema_version",
                f"schema_version 必须精确等于 {SCHEMA_VERSION!r}，得到 {version!r}（不匹配时只报本错误）",
            )
        )
        return Report(source=source, ok=False, engine=resolved, issues=tuple(issues))

    schema = load_schema(schema_path)
    try:
        issues.extend(_structure_issues(data, schema, resolved))
        issues.extend(validate_semantic(data))
    except RecursionError:
        return Report(
            source=source,
            ok=False,
            engine=resolved,
            issues=(Issue("parse", "E_PARSE", "", "文档嵌套过深（递归上限），拒绝"),),
        )
    issues.sort(key=lambda issue: (issue.pointer, issue.code, issue.message))
    return Report(source=source, ok=not issues, engine=resolved, issues=tuple(issues))


def validate_text(
    text: str,
    *,
    engine: str = "auto",
    schema_path: Path | None = None,
    source: str = "<text>",
) -> Report:
    """校验 JSON 文本（含严格 parse 层）。"""
    resolved = resolve_engine(engine)
    data, issues = load_text(text, source=source)
    if issues:
        return Report(source=source, ok=False, engine=resolved, issues=tuple(issues))
    return validate_document(data, engine=resolved, schema_path=schema_path, source=source)


def validate_file(
    path: str | Path,
    *,
    engine: str = "auto",
    schema_path: Path | None = None,
) -> Report:
    """校验 JSON 文件（路径任意，与 cwd 无关）。"""
    file_path = Path(path)
    resolved = resolve_engine(engine)
    data, issues = load_path(file_path)
    if issues:
        return Report(source=str(file_path), ok=False, engine=resolved, issues=tuple(issues))
    return validate_document(data, engine=resolved, schema_path=schema_path, source=str(file_path))
