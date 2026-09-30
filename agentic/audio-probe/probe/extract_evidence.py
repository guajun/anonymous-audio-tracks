#!/usr/bin/env python3
"""Redact and extract evidence from real-run outputs (issue #30).

Takes raw artifacts (pi --mode json event stream, session JSONL, stderr) and
prints a sanitized excerpt suitable for `reports/`. Guarantees:

  * API-key-looking tokens (sk-..., AIza..., Bearer ...) -> ``<REDACTED-TOKEN>``
  * base64-looking payloads (>= 80 chars)               -> ``<BASE64 redacted: N chars>``
  * home directory / repo root (both separator flavors and JSON-escaped
    doubling) -> ``<HOME>`` / ``<REPO>``
  * any other absolute path (drive-letter or UNC, including JSON-escaped
    ``F:\\...`` forms) -> ``<PATH>``

Order matters (review item 4): the FULL text is redacted first and only then
truncated to ``--max-chars``, so a long secret that crosses the truncation
boundary cannot leak a prefix. Every printed excerpt is verified to contain no
unsanitized absolute path (exit code 6 otherwise).

Usage:
    python probe/extract_evidence.py tmp/bridge-events.jsonl
    python probe/extract_evidence.py tmp/bridge-events.jsonl --field text --max-chars 600
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE64_RE = re.compile(r"[A-Za-z0-9+/=]{80,}")
TOKEN_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._~+/=-]{8,})")
# Absolute paths: drive-letter form (not preceded by alphanumerics, so URL
# schemes like https:// do not match) and UNC form. Both cover JSON-escaped
# backslashes because we match on the raw characters of the input string.
DRIVE_PATH_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]:[\\/][^\s\"'*<>|]*)")
UNC_PATH_RE = re.compile(r"\\\\[A-Za-z0-9._$-]+(?:[\\/][^\s\"'*<>|]*)?")

EXIT_OK = 0
EXIT_NOT_FOUND = 2
EXIT_UNSANITIZED = 6


def repo_root() -> Path:
    # probe/extract_evidence.py -> probe -> audio-probe -> agentic -> repo root
    return Path(__file__).resolve().parents[3]


def _root_variants(root: str) -> set[str]:
    """Separator flavors plus JSON-escaped doubling of the root string."""
    forward = root.replace("\\", "/")
    backslash = root.replace("/", "\\")
    return {
        forward,
        backslash,
        forward.replace("\\", "\\\\"),
        backslash.replace("\\", "\\\\"),
    }


def redact(text: str) -> str:
    text = TOKEN_RE.sub("<REDACTED-TOKEN>", text)
    text = BASE64_RE.sub(lambda m: f"<BASE64 redacted: {len(m.group(0))} chars>", text)
    for root, label in ((str(Path.home()), "<HOME>"), (str(repo_root()), "<REPO>")):
        for variant in _root_variants(root):
            text = text.replace(variant, label)
    text = DRIVE_PATH_RE.sub("<PATH>", text)
    text = UNC_PATH_RE.sub("<PATH>", text)
    text = "".join(ch if (ch.isprintable() or ch in "\n\t") else "·" for ch in text)
    return text


def find_unsanitized(text: str) -> list[str]:
    """Absolute paths that survived sanitization (should never happen)."""
    hits = DRIVE_PATH_RE.findall(text) + UNC_PATH_RE.findall(text)
    for root in (str(Path.home()), str(repo_root())):
        for variant in _root_variants(root):
            if variant and variant in text:
                hits.append(variant)
    return hits


def format_record(text: str, max_chars: int) -> str:
    """Redact the FULL text first, then truncate (never the other way round)."""
    safe = redact(text)
    if max_chars > 0 and len(safe) > max_chars:
        safe = safe[:max_chars] + " …[truncated]"
    return safe


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
    parser.add_argument("--max-chars", type=int, default=600, help="truncate each record AFTER redaction")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"input not found: {path}", file=sys.stderr)
        return EXIT_NOT_FOUND

    for key, text in iter_text_records(path):
        if args.field and args.field not in key:
            continue
        snippet = format_record(text, args.max_chars)
        residue = find_unsanitized(snippet)
        if residue:
            print(f"refusing to print unsanitized excerpt: {residue[:2]}", file=sys.stderr)
            return EXIT_UNSANITIZED
        print(f"[{key}] {snippet}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
