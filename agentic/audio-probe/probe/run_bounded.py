#!/usr/bin/env python3
"""run_bounded.py — bounded subprocess runner for the real probe scripts.

Review item 30/1: real runs must have an explicit timeout, must preserve their
artifacts, and must FAIL (nonzero) when the run times out, the command fails,
or Pi reports a provider error/aborted final message even though it exited 0.

Usage:
    python probe/run_bounded.py --timeout SECS --stdout F --stderr F [--events F] -- <cmd> ...

Exit codes (stable):
    0  success (command ok, no provider error/abort found in events)
    2  usage error
    3  command failed (nonzero exit)
    4  timeout (command killed after --timeout seconds)
    5  provider error / aborted run detected in the event stream
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_COMMAND_FAILED = 3
EXIT_TIMEOUT = 4
EXIT_PROVIDER_ERROR = 5

BAD_STOP_REASONS = {"error", "aborted"}


def find_provider_failures(events_path: Path) -> list[str]:
    """Return descriptions of provider error / aborted assistant messages."""
    failures: list[str] = []
    try:
        lines = events_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [f"events file unreadable: {exc.__class__.__name__}"]

    def walk(node) -> list[str]:
        found: list[str] = []
        if isinstance(node, dict):
            if node.get("type") == "error":
                found.append(f"error event: {str(node.get('message') or node.get('error') or '')[:120]}")
            message = node.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                if message.get("stopReason") in BAD_STOP_REASONS:
                    found.append(f"assistant stopReason={message.get('stopReason')}")
                if message.get("errorMessage"):
                    found.append(f"assistant errorMessage: {str(message['errorMessage'])[:120]}")
            for value in node.values():
                if isinstance(value, (dict, list)):
                    found.extend(walk(value))
        elif isinstance(node, list):
            for item in node:
                found.extend(walk(item))
        return found

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        failures.extend(walk(record))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, required=True, help="hard timeout in seconds")
    parser.add_argument("--stdout", required=True, help="file to capture stdout")
    parser.add_argument("--stderr", required=True, help="file to capture stderr")
    parser.add_argument("--events", help="event JSONL to scan for provider errors after the run")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="command after --")
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("run_bounded: no command given", file=sys.stderr)
        return EXIT_USAGE

    stdout_path = Path(args.stdout)
    stderr_path = Path(args.stderr)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"run_bounded: timeout={args.timeout}s cmd={' '.join(cmd[:4])}…")
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        try:
            proc = subprocess.run(cmd, stdout=out, stderr=err, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            # Artifacts already written by the child stay on disk.
            print(f"run_bounded: TIMEOUT after {args.timeout}s (killed)", file=sys.stderr)
            return EXIT_TIMEOUT
        except OSError as exc:
            print(f"run_bounded: cannot start command ({exc.__class__.__name__})", file=sys.stderr)
            return EXIT_COMMAND_FAILED

    if proc.returncode != 0:
        print(f"run_bounded: command failed rc={proc.returncode}", file=sys.stderr)
        return EXIT_COMMAND_FAILED

    if args.events:
        failures = find_provider_failures(Path(args.events))
        if failures:
            for failure in failures:
                print(f"run_bounded: provider failure detected: {failure}", file=sys.stderr)
            return EXIT_PROVIDER_ERROR

    print("run_bounded: OK")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
