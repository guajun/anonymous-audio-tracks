#!/usr/bin/env python3
"""Redact and extract evidence from real-run outputs (issue #30).

Takes raw artifacts (pi --mode json event stream, session JSONL, stderr) and
prints a sanitized excerpt suitable for `reports/`. Guarantees:

  * base64-looking payloads (>= 80 chars) -> ``<BASE64 redacted: N chars>``
  * API-key-looking tokens (sk-..., AIza..., Bearer ...) -> ``<REDACTED-TOKEN>``
  * home directory / repo root -> ``<HOME>`` / ``<REPO>``

Usage:
    python probe/extract_evidence.py tmp/bridge-events.jsonl
    python probe/extract_evidence.py tmp/bridge-events.jsonl --field text
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

BASE64_RE = re.compile(r"[A-Za-z0-9+/=]{80,}")
TOKEN_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._~+/=-]{8,})")


def redact(text: str) -> str:
    text = TOKEN_RE.sub("<REDACTED-TOKEN>", text)
    text = BASE64_RE.sub(lambda m: f"<BASE64 redacted: {len(m.group(0))} chars>", text)
    text = "".join(ch if (ch.isprintable() or ch in "\n\t") else "·" for ch in text)
    home = str(Path.home()).replace("\\", "/")
    text = text.replace(home, "<HOME>")
    text = text.replace(home.replace("/", "\\"), "<HOME>")
    cwd = str(Path(__file__).resolve().parents[2]).replace("\\", "/")
    text = text.replace(cwd, "<REPO>")
    text = text.replace(cwd.replace("/", "\\"), "<REPO>")
    return text


def iter_text_records(path: Path):
    """Yield (kind, text) from pi JSON event streams, session JSONL, or plain text.

    Walks nested records (message_end.message.content[].text,
    message_update.assistantMessageEvent.delta, tool_execution_end.result, ...)
    and yields every human-relevant string with a path-ish label.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    def walk(node, label):
        if isinstance(node, dict):
            role = node.get("role") or node.get("type")
            here = f"{label}/{role}" if isinstance(role, str) and label else (label or str(role))
            for key, value in node.items():
                if isinstance(value, str):
                    if key in ("text", "delta", "errorMessage", "command", "content") and value.strip():
                        yield f"{here}.{key}", value
                elif isinstance(value, (dict, list)):
                    yield from walk(value, f"{here}.{key}")
        elif isinstance(node, list):
            for item in node:
                yield from walk(item, label)

    jsonl_hits = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        jsonl_hits += 1
        yield from walk(record, "")

    if jsonl_hits == 0:
        yield "raw", raw


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="pi event stream / session JSONL / text file")
    parser.add_argument("--field", help="only show records whose key contains this substring")
    parser.add_argument("--max-chars", type=int, default=600, help="truncate each record")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"input not found: {path}", file=sys.stderr)
        return 2

    for key, text in iter_text_records(path):
        if args.field and args.field not in key:
            continue
        snippet = text if len(text) <= args.max_chars else text[: args.max_chars] + " …[truncated]"
        print(f"[{key}] {redact(snippet)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
