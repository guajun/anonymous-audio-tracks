#!/usr/bin/env python3
"""pi_launcher.py — resolve the installed Pi CLI to a NATIVE process image.

Why (issue #30 review round 2, blocker 1): on Windows the `pi` command is a
POSIX shell shim (`~/.pi/agent/bin/pi`) plus `pi.cmd`/`pi.ps1`. Python's
`subprocess` cannot execute the shim directly (FileNotFoundError), and running
it through a shell would put a wrapper shell between the runner and Pi so a
timeout kill would not reach the real Node process.

This module mirrors Pi's own `pi-launcher.js` install-metadata resolution and
returns an argv array headed by a native executable (Node + CLI JS entry), so:

  * argv boundaries are preserved (paths/prompts with spaces are one argv),
  * no shell string is involved anywhere,
  * a timeout kill terminates the actual Pi process (the direct child).

Resolution order for a command headed by `pi`:
  1. `PI_BIN` env (explicit override; word-split — used by offline fake-pi
     regression tests),
  2. Node (`PI_NODE` env or `node` on PATH) + the CLI JS entry from install
     metadata:
       a. `PI_INSTALL_DIR` (explicit package dir),
       b. `PI_PACKAGE_DIR`/node_modules/@earendil-works/pi-coding-agent,
       c. `PI_CODING_AGENT_DIR` (default ~/.pi/agent)/install/current-version
          -> releases/<version>/node_modules/@earendil-works/pi-coding-agent
          (same validation as pi-launcher.js: version token regex),
       d. newest release directory scan,
     reading `package.json` `bin.pi` (e.g. dist/bundle/cli.js),
  3. a native `pi.exe` (Windows) or `pi` binary (POSIX) on PATH.

Commands not headed by `pi` are returned unchanged.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

PACKAGE_NAME = "@earendil-works/pi-coding-agent"
COMMAND_NAME = "pi"
VERSION_RE = re.compile(r"^[0-9A-Za-z._+-]+$")
PI_HEADS = {"pi", "pi.exe", "pi.cmd", "pi.ps1"}


class LauncherError(RuntimeError):
    pass


def find_node() -> str:
    node = os.environ.get("PI_NODE", "").strip() or shutil.which("node")
    if not node:
        raise LauncherError("node executable not found (install Node >= 22.19 or set PI_NODE)")
    return node


def _cli_from_package(package_dir: Path) -> Path:
    """Return the CLI JS entry declared by the package's `bin.pi`, validated
    exactly like pi-launcher.js (must stay inside the package dir)."""
    package_json = package_dir / "package.json"
    try:
        meta = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LauncherError(f"cannot read package metadata ({exc.__class__.__name__})")
    bin_field = meta.get("bin")
    rel = bin_field if isinstance(bin_field, str) else (bin_field or {}).get(COMMAND_NAME)
    if not rel or not isinstance(rel, str):
        raise LauncherError(f"{PACKAGE_NAME} does not declare a {COMMAND_NAME} bin")
    cli_path = (package_dir / rel).resolve()
    try:
        cli_path.relative_to(package_dir.resolve())
    except ValueError:
        raise LauncherError("managed Pi executable escapes the package directory")
    if not cli_path.is_file():
        raise LauncherError("managed Pi executable is missing")
    return cli_path


def _agent_dir() -> Path:
    override = os.environ.get("PI_CODING_AGENT_DIR", "").strip()
    return Path(override) if override else Path.home() / ".pi" / "agent"


def _candidate_package_dirs() -> list[Path]:
    candidates: list[Path] = []

    explicit = os.environ.get("PI_INSTALL_DIR", "").strip()
    if explicit:
        candidates.append(Path(explicit))
    pkg_dir = os.environ.get("PI_PACKAGE_DIR", "").strip()
    if pkg_dir:
        candidates.append(Path(pkg_dir) / "node_modules" / PACKAGE_NAME)

    agent_dir = _agent_dir()
    install_root = agent_dir / "install"

    # pi-launcher.js reads install/current-version first.
    current_file = install_root / "current-version"
    try:
        version = current_file.read_text(encoding="utf-8").strip()
    except OSError:
        version = ""
    if version and version not in (".", "..") and VERSION_RE.match(version):
        candidates.append(install_root / "releases" / version / "node_modules" / PACKAGE_NAME)

    # Fallback: newest release directory.
    releases = install_root / "releases"
    if releases.is_dir():
        versions = [d.name for d in releases.iterdir() if d.is_dir() and VERSION_RE.match(d.name)]
        versions.sort(key=lambda v: [int(x) if x.isdigit() else x for x in re.split(r"[._+-]", v)])
        for version in reversed(versions):
            candidates.append(releases / version / "node_modules" / PACKAGE_NAME)
    return candidates


def resolve_command(cmd: list[str]) -> list[str]:
    """Return the argv to execute natively. `cmd` is already split (never a
    shell string)."""
    if not cmd:
        raise LauncherError("empty command")
    head = Path(cmd[0]).name.lower()
    if head not in PI_HEADS:
        return list(cmd)

    override = os.environ.get("PI_BIN", "").strip()
    if override:
        return shlex.split(override) + list(cmd[1:])

    node = find_node()
    for package_dir in _candidate_package_dirs():
        try:
            return [node, str(_cli_from_package(package_dir)), *cmd[1:]]
        except LauncherError:
            continue

    exe = shutil.which(COMMAND_NAME)
    if exe and (os.name != "nt" or exe.lower().endswith(".exe")):
        return [exe, *cmd[1:]]
    raise LauncherError(
        "cannot resolve the installed pi CLI (checked install metadata and PATH; "
        "set PI_INSTALL_DIR to the @earendil-works/pi-coding-agent directory)"
    )


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--print":
        try:
            resolved = resolve_command(sys.argv[2:])
        except LauncherError as exc:
            print(f"pi_launcher: {exc}", file=sys.stderr)
            return 3
        print(json.dumps(resolved, ensure_ascii=False))
        return 0
    print("usage: python probe/pi_launcher.py --print <cmd> [args...]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
