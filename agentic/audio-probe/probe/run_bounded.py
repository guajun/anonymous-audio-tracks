#!/usr/bin/env python3
"""run_bounded.py — bounded subprocess runner for the real probe scripts.

Review item 30/1 (round 2, blocker 1): real runs must have an explicit timeout,
must preserve their artifacts, and must FAIL (nonzero) when the run times out,
the command fails, or Pi reports a provider error/aborted/incomplete final
message even though it exited 0.

Launching (round 2, blocker 1): a command headed by `pi` is resolved through
`probe/pi_launcher.py` to a NATIVE argv (Node + CLI JS entry from install
metadata) — Python's Windows command lookup cannot run the `pi` shell shim, and
no shell string is used anywhere, so argv boundaries (spaces in paths/prompts)
are preserved and a timeout kill terminates the actual Pi process.

Usage:
    python probe/run_bounded.py --timeout SECS --stdout F --stderr F [--events F] -- <cmd> ...

Exit codes (stable):
    0  success (command ok, event stream ends with a successful terminal
       assistant message when --events is given)
    2  usage error (including non-finite or nonpositive --timeout)
    3  command failed (nonzero exit) or command could not be resolved/started
    4  timeout (command killed after --timeout seconds)
    5  provider error / aborted run detected in the event stream
    6  empty / garbled / incomplete event stream (no successful terminal
       assistant message) — never counts as a successful model completion
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from pi_launcher import LauncherError, resolve_command

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_COMMAND_FAILED = 3
EXIT_TIMEOUT = 4
EXIT_PROVIDER_ERROR = 5
EXIT_INCOMPLETE_EVENTS = 6

BAD_STOP_REASONS = {"error", "aborted"}
TERMINAL_SUCCESS_STOP_REASON = "stop"


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


def has_terminal_success(events_path: Path) -> bool:
    """True only when the event stream contains assistant messages AND the last
    one is a successful terminal message (stopReason == "stop"). Empty, garbled
    or tool-use-only (incomplete) streams return False."""
    last_assistant: dict | None = None
    try:
        lines = events_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False

    def walk(node) -> None:
        nonlocal last_assistant
        if isinstance(node, dict):
            message = node.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                last_assistant = message
            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        walk(record)

    return bool(last_assistant) and last_assistant.get("stopReason") == TERMINAL_SUCCESS_STOP_REASON


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

    timeout = args.timeout
    if not isinstance(timeout, float) or not math.isfinite(timeout) or timeout <= 0:
        print("run_bounded: --timeout must be a finite positive number", file=sys.stderr)
        return EXIT_USAGE

    try:
        cmd = resolve_command(cmd)
    except LauncherError as exc:
        print(f"run_bounded: cannot start command ({exc})", file=sys.stderr)
        return EXIT_COMMAND_FAILED

    stdout_path = Path(args.stdout)
    stderr_path = Path(args.stderr)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"run_bounded: timeout={timeout}s cmd={' '.join(cmd[:4])}…")
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        try:
            proc = subprocess.run(cmd, stdout=out, stderr=err, timeout=timeout)
        except subprocess.TimeoutExpired:
            # Artifacts already written by the child stay on disk. The launched
            # command is the native Pi process itself (no wrapper shell), so
            # the kill reaches Pi directly.
            print(f"run_bounded: TIMEOUT after {timeout}s (killed)", file=sys.stderr)
            return EXIT_TIMEOUT
        except OSError as exc:
            print(f"run_bounded: cannot start command ({exc.__class__.__name__})", file=sys.stderr)
            return EXIT_COMMAND_FAILED

    if proc.returncode != 0:
        print(f"run_bounded: command failed rc={proc.returncode}", file=sys.stderr)
        return EXIT_COMMAND_FAILED

    if args.events:
        events_path = Path(args.events)
        failures = find_provider_failures(events_path)
        if failures:
            for failure in failures:
                print(f"run_bounded: provider failure detected: {failure}", file=sys.stderr)
            return EXIT_PROVIDER_ERROR
        if not has_terminal_success(events_path):
            print(
                "run_bounded: event stream empty/garbled/incomplete: "
                "no successful terminal assistant message",
                file=sys.stderr,
            )
            return EXIT_INCOMPLETE_EVENTS

    print("run_bounded: OK")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
