"""结构校验：官方 ``jsonschema`` 引擎（可选依赖，requirements.txt 有 pin）。

与 structure.py 的零依赖实现互为交叉验证。错误码取 schema 关键字名，
错误位置取 JSON Pointer，便于和零依赖实现对齐比较。
"""

from __future__ import annotations

from typing import Any

from .errors import Issue, pointer_of


def jsonschema_available() -> bool:
    try:
        import jsonschema  # noqa: F401
    except Exception:
        return False
    return True


def jsonschema_version() -> str | None:
    try:
        from importlib import metadata

        return metadata.version("jsonschema")
    except Exception:
        return None


def validate_with_jsonschema(data: Any, schema: dict[str, Any]) -> list[Issue]:
    """用 ``jsonschema.Draft202012Validator`` 检查结构。"""
    import jsonschema

    validator = jsonschema.Draft202012Validator(schema)
    issues = [
        Issue(
            "structure",
            str(error.validator),
            pointer_of(list(error.absolute_path)),
            f"{error.message}（jsonschema {jsonschema_version() or '?'}）",
        )
        for error in validator.iter_errors(data)
    ]
    return sorted(issues, key=lambda issue: (issue.pointer, issue.code, issue.message))
