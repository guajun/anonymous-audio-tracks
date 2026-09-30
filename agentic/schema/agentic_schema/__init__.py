"""agentic-audio-tracks/v1 校验器（issue #32 冻结实现）。

公开 API（#33/#34 可直接依赖）::

    from agentic_schema import (
        SCHEMA_VERSION,      # "agentic-audio-tracks/v1"
        SCHEMA_PATH,         # 结构 schema 文件（Draft 2020-12，可在浏览器使用）
        SEMANTIC_RULES_PATH, # 语义规则表（机器可读）
        Issue, Report,
        validate_file, validate_text, validate_document,
    )

CLI（冻结命令，任意 cwd 可运行，路径相对脚本自身解析）::

    python agentic/schema/validate.py <file.json> [...] [--json] [--engine auto|jsonschema|stdlib]

退出码：0=全部合法，1=存在不合法文档，2=用法/IO/依赖错误。
"""

from __future__ import annotations

from pathlib import Path

from .errors import Issue, Report
from .pipeline import (
    SCHEMA_PATH,
    SCHEMA_VERSION,
    SEMANTIC_RULES_PATH,
    validate_document,
    validate_file,
    validate_text,
)

__all__ = [
    "Issue",
    "Report",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "SEMANTIC_RULES_PATH",
    "validate_document",
    "validate_file",
    "validate_text",
]

#: 包目录（agentic/schema/agentic_schema）
PACKAGE_DIR = Path(__file__).resolve().parent
#: 冻结接口目录（agentic/schema）
BASE_DIR = PACKAGE_DIR.parent
