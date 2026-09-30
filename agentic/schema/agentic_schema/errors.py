"""错误/报告数据结构。所有错误都带 JSON Pointer，便于定位到具体字段。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 合法层级：parse=严格 JSON 解析，version=版本匹配，structure=Draft 2020-12 结构，semantic=跨字段语义
LAYERS = ("parse", "version", "structure", "semantic")


def escape_segment(segment: str) -> str:
    """JSON Pointer 转义：``~`` -> ``~0``，``/`` -> ``~1``（RFC 6901）。"""
    return str(segment).replace("~", "~0").replace("/", "~1")


def pointer_of(parts: list[Any]) -> str:
    """把路径片段列表拼成 JSON Pointer；空列表为根（``""``）。"""
    if not parts:
        return ""
    return "".join("/" + escape_segment(p) for p in parts)


@dataclass(frozen=True)
class Issue:
    """一条校验错误。

    - ``layer``: ``parse`` / ``version`` / ``structure`` / ``semantic``
    - ``code``: 语义层为 ``E_*`` 稳定错误码；结构层为 schema 关键字名（如 ``type``/``required``）；
      解析层为 ``E_PARSE`` / ``E_DUPLICATE_KEY`` / ``E_NONFINITE``
    - ``pointer``: 出错位置的 JSON Pointer（根为 ``""``）
    - ``message``: 人类可读说明（中文）
    """

    layer: str
    code: str
    pointer: str
    message: str

    def __post_init__(self) -> None:
        if self.layer not in LAYERS:
            raise ValueError(f"unknown layer: {self.layer!r}")

    def as_dict(self) -> dict[str, str]:
        return {
            "layer": self.layer,
            "code": self.code,
            "pointer": self.pointer,
            "message": self.message,
        }

    def format(self) -> str:
        where = self.pointer if self.pointer else "<root>"
        return f"{where} [{self.layer}/{self.code}] {self.message}"


@dataclass(frozen=True)
class Report:
    """一次校验的结果。"""

    source: str
    ok: bool
    engine: str
    issues: tuple[Issue, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ok": self.ok,
            "engine": self.engine,
            "schema_version": "agentic-audio-tracks/v1",
            "issues": [i.as_dict() for i in self.issues],
        }
