#!/usr/bin/env python
"""agentic-audio-tracks/v1 校验器 CLI（issue #32 冻结命令）。

用法（任意目录可运行；schema 路径相对本脚本解析，与 cwd 无关）::

    python agentic/schema/validate.py <file.json> [more.json ...]
    python agentic/schema/validate.py --json <file.json>
    python agentic/schema/validate.py --engine stdlib <file.json>

退出码：0=全部合法；1=存在不合法文档；2=用法/IO/依赖错误。

错误输出每行一条：``<file>: <JSON Pointer> [<layer>/<code>] <message>``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 跨平台确定性输出：错误说明含中文，固定 UTF-8（Windows 默认 locale 会乱码）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from agentic_schema.pipeline import (  # noqa: E402
    SCHEMA_PATH,
    SCHEMA_VERSION,
    DependencyError,
    validate_file,
)
from agentic_schema.schema_check import jsonschema_version  # noqa: E402

USAGE_EXIT = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate.py",
        description=f"校验 {SCHEMA_VERSION} 文档（结构 + 跨字段语义，错误带 JSON Pointer）",
    )
    parser.add_argument("files", nargs="+", help="待校验的 JSON 文件路径")
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出机器可读 JSON（每文件一个对象）",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "jsonschema", "stdlib"),
        default="auto",
        help="结构校验引擎：auto=有 jsonschema 就用之，否则零依赖 stdlib（默认 auto）",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help=f"覆盖结构 schema 路径（默认 {SCHEMA_PATH.name}，相对本脚本）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    schema_path = Path(args.schema) if args.schema else None

    reports = []
    usage_error = False
    for file_name in args.files:
        path = Path(file_name)
        if not path.is_file():
            print(f"error: 文件不存在或不是普通文件: {file_name}", file=sys.stderr)
            usage_error = True
            continue
        try:
            report = validate_file(path, engine=args.engine, schema_path=schema_path)
        except DependencyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return USAGE_EXIT
        except OSError as exc:
            print(f"error: 无法读取 {file_name}: {exc}", file=sys.stderr)
            usage_error = True
            continue
        reports.append(report)

    if args.json:
        for report in reports:
            payload = report.as_dict()
            payload["engine_detail"] = jsonschema_version() if report.engine == "jsonschema" else "stdlib"
            print(json.dumps(payload, ensure_ascii=False))
    else:
        for report in reports:
            status = "OK" if report.ok else "FAIL"
            print(f"{status} {report.source} (engine={report.engine})")
            for issue in report.issues:
                print(f"  {issue.format()}")

    if usage_error:
        return USAGE_EXIT
    if any(not report.ok for report in reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
