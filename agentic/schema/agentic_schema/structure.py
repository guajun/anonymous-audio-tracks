"""结构校验：Draft 2020-12 便携子集的零依赖参考实现。

只支持本协议 schema 用到的关键字：
``type`` ``const`` ``enum`` ``properties`` ``required`` ``additionalProperties(false)``
``items`` ``$defs`` ``$ref``（仅内部引用）``pattern`` ``minimum`` ``exclusiveMinimum``
``maximum`` ``minLength`` ``maxLength`` ``minItems``。

用途：

1. 在没有 ``jsonschema`` 依赖的 checkout 里也能做结构校验（错误同样带 JSON Pointer）；
2. 与 ``jsonschema``（schema_check.py）互为交叉验证（tests/test_engines_agree.py）。

布尔值不算数字（``true`` 不是 ``1``）；整数接受 ``2.0`` 这类整值浮点（与 jsonschema 一致）。
"""

from __future__ import annotations

import re
from typing import Any

from .errors import Issue, pointer_of

_SUPPORTED_KEYWORDS = {
    "$schema",
    "$id",
    "$defs",
    "$ref",
    "title",
    "description",
    "type",
    "const",
    "enum",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "pattern",
    "minimum",
    "exclusiveMinimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and value.is_integer()


def _type_ok(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "number":
        return _is_number(value)
    if name == "integer":
        return _is_integer(value)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    raise ValueError(f"unsupported type name: {name!r}")


def _json_equal(a: Any, b: Any) -> bool:
    if _is_number(a) and _is_number(b):
        return a == b
    return type(a) is type(b) and a == b


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/$defs/"):
        raise ValueError(f"only internal #/$defs/ refs are supported, got {ref!r}")
    name = ref.split("/")[-1]
    return root["$defs"][name]


def _validate(instance: Any, schema: dict[str, Any], root: dict[str, Any], parts: list[Any], out: list[Issue]) -> None:
    unknown = set(schema) - _SUPPORTED_KEYWORDS
    if unknown:
        raise ValueError(f"unsupported schema keywords: {sorted(unknown)}")

    if "$ref" in schema:
        _validate(instance, _resolve_ref(schema["$ref"], root), root, parts, out)
        return

    pointer = pointer_of(parts)

    if "const" in schema and not _json_equal(instance, schema["const"]):
        out.append(Issue("structure", "const", pointer, f"必须等于 {schema['const']!r}，得到 {instance!r}"))
    if "enum" in schema and not any(_json_equal(instance, choice) for choice in schema["enum"]):
        out.append(Issue("structure", "enum", pointer, f"必须是 {schema['enum']!r} 之一，得到 {instance!r}"))

    type_spec = schema.get("type")
    if type_spec is not None:
        names = type_spec if isinstance(type_spec, list) else [type_spec]
        if not any(_type_ok(instance, name) for name in names):
            out.append(
                Issue("structure", "type", pointer, f"类型应为 {' 或 '.join(names)}，得到 {type(instance).__name__} {instance!r}")
            )
            return  # 类型不对就不再往下查

    if isinstance(instance, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in instance]
        if missing:
            out.append(Issue("structure", "required", pointer, f"缺少必填字段 {missing!r}"))
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                _validate(value, properties[key], root, parts + [key], out)
        if schema.get("additionalProperties") is False:
            extras = [key for key in instance if key not in properties]
            if extras:
                out.append(
                    Issue(
                        "structure",
                        "additionalProperties",
                        pointer,
                        f"不允许出现未定义字段 {extras!r}（未知字段策略=拒绝；加字段=新版本）",
                    )
                )
        return

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            out.append(Issue("structure", "minItems", pointer, f"至少需要 {schema['minItems']} 项，得到 {len(instance)}"))
        if "items" in schema:
            for index, value in enumerate(instance):
                _validate(value, schema["items"], root, parts + [index], out)
        return

    if isinstance(instance, str):
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            out.append(Issue("structure", "pattern", pointer, f"不匹配模式 {schema['pattern']!r}，得到 {instance!r}"))
        if "minLength" in schema and len(instance) < schema["minLength"]:
            out.append(Issue("structure", "minLength", pointer, f"长度至少 {schema['minLength']}，得到 {len(instance)}"))
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            out.append(Issue("structure", "maxLength", pointer, f"长度最多 {schema['maxLength']}，得到 {len(instance)}"))
        return

    if _is_number(instance):
        if "minimum" in schema and instance < schema["minimum"]:
            out.append(Issue("structure", "minimum", pointer, f"必须 >= {schema['minimum']}，得到 {instance!r}"))
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            out.append(Issue("structure", "exclusiveMinimum", pointer, f"必须 > {schema['exclusiveMinimum']}，得到 {instance!r}"))
        if "maximum" in schema and instance > schema["maximum"]:
            out.append(Issue("structure", "maximum", pointer, f"必须 <= {schema['maximum']}，得到 {instance!r}"))


def validate_structure(data: Any, schema: dict[str, Any]) -> list[Issue]:
    """用零依赖子集校验器检查结构；错误按 (pointer, code) 稳定排序。"""
    out: list[Issue] = []
    _validate(data, schema, schema, [], out)
    return sorted(out, key=lambda issue: (issue.pointer, issue.code, issue.message))
