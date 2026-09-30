"""Portable Pi launcher regressions (issue #30 review round 2, blocker 1).

`probe/pi_launcher.py` must resolve the installed Pi CLI to a NATIVE argv
(Node + CLI JS entry from install metadata) so that:

  * Python's Windows command lookup is never asked to run the `pi` shell shim,
  * argv boundaries (spaces / unicode in paths and prompts) survive exactly,
  * a timeout kill terminates the actual Pi process (the direct child).

Covered offline with fake install metadata + a stub CLI (no API, no network),
plus one ACTUAL `pi --version` smoke through the real runner (zero API calls).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]  # agentic/audio-probe
NODE = shutil.which("node")

sys.path.insert(0, str(BASE / "probe"))

pytestmark = pytest.mark.skipif(NODE is None, reason="node not available")

STUB_ECHO_ARGV = 'console.log(JSON.stringify(process.argv.slice(2)));'
STUB_SLEEP = "setTimeout(() => {}, 60000);"


def make_fake_install(root: Path, cli_js: str, version: str = "9.9.9") -> Path:
    pkg = root / "install" / "releases" / version / "node_modules" / "@earendil-works" / "pi-coding-agent"
    (pkg / "dist" / "bundle").mkdir(parents=True)
    (pkg / "package.json").write_text(json.dumps({"bin": {"pi": "dist/bundle/cli.js"}}), encoding="utf-8")
    (pkg / "dist" / "bundle" / "cli.js").write_text(cli_js, encoding="utf-8")
    (root / "install").mkdir(parents=True, exist_ok=True)
    (root / "install" / "current-version").write_text(version, encoding="utf-8")
    return pkg


def run_bounded(cmd, tmp_path: Path, env_extra, timeout: str = "10"):
    env = os.environ.copy()
    env.pop("PI_BIN", None)  # never let the fake-pi override leak into these tests
    for key in ("PI_INSTALL_DIR", "PI_PACKAGE_DIR"):
        env.pop(key, None)
    # Restrict PATH so no stray `pi` shim on the host can satisfy the fallback.
    env["PATH"] = os.path.dirname(NODE)
    env.update(env_extra)
    out, err = tmp_path / "o.txt", tmp_path / "e.txt"
    proc = subprocess.run(
        [sys.executable, "probe/run_bounded.py", "--timeout", timeout,
         "--stdout", str(out), "--stderr", str(err), "--", *cmd],
        cwd=BASE,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    return proc, out, err


class TestPortableLauncher:
    def test_argv_boundaries_preserved(self, tmp_path: Path) -> None:
        pkg = make_fake_install(tmp_path / "agent", STUB_ECHO_ARGV)
        args = ["arg with spaces", "C:\\path with space\\x.wav", "中文提示词 with spaces", "--flag=ok"]
        proc, out, _ = run_bounded(["pi", *args], tmp_path, {"PI_CODING_AGENT_DIR": str(tmp_path / "agent")})
        assert proc.returncode == 0, proc.stderr
        echoed = json.loads(out.read_text(encoding="utf-8"))
        assert echoed == args, "argv must survive exactly (no shell re-splitting)"
        # resolution used Node + the declared bin entry, not a shell shim
        assert str(pkg / "dist" / "bundle" / "cli.js").replace("\\", "/") in proc.stdout.replace("\\", "/")

    def test_install_dir_override(self, tmp_path: Path) -> None:
        pkg = make_fake_install(tmp_path / "agent", STUB_ECHO_ARGV)
        proc, out, _ = run_bounded(
            ["pi", "--version"],
            tmp_path,
            {"PI_INSTALL_DIR": str(pkg)},
        )
        assert proc.returncode == 0, proc.stderr
        assert json.loads(out.read_text(encoding="utf-8")) == ["--version"]

    def test_timeout_kills_the_launched_process(self, tmp_path: Path) -> None:
        """The killed child is Node running the CLI entry itself — no wrapper."""
        make_fake_install(tmp_path / "agent", STUB_SLEEP)
        proc, _, err = run_bounded(
            ["pi", "--anything"],
            tmp_path,
            {"PI_CODING_AGENT_DIR": str(tmp_path / "agent")},
            timeout="1",
        )
        assert proc.returncode == 4, proc.stdout + proc.stderr
        assert "TIMEOUT" in proc.stderr
        assert err.exists(), "child artifacts preserved on timeout"

    def test_malicious_current_version_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "agent"
        make_fake_install(root, STUB_ECHO_ARGV, version="9.9.9")
        (root / "install" / "current-version").write_text("..\\evil", encoding="utf-8")
        # empty releases so the newest-release fallback finds nothing either
        for child in (root / "install" / "releases").iterdir():
            shutil.rmtree(child)
        proc, _, _ = run_bounded(["pi", "--version"], tmp_path, {"PI_CODING_AGENT_DIR": str(root)})
        assert proc.returncode == 3, proc.stdout + proc.stderr
        assert "cannot" in proc.stderr.lower() or "cannot" in proc.stdout.lower()

    def test_non_pi_commands_pass_through(self, tmp_path: Path) -> None:
        import pi_launcher

        cmd = [sys.executable, "-c", "print('x')"]
        assert pi_launcher.resolve_command(cmd) == cmd


class TestActualPiSmoke:
    """No-network smoke: run the ACTUAL installed pi --version through the
    fixed runner (zero API calls). Skips only when no install is resolvable."""

    def test_actual_pi_version(self, tmp_path: Path) -> None:
        proc, out, _ = run_bounded(["pi", "--version"], tmp_path, {})
        if proc.returncode == 3 and "cannot resolve" in proc.stderr:
            pytest.skip("pi install metadata not found on this host")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        version = out.read_text(encoding="utf-8").strip()
        assert re.search(r"\d+\.\d+\.\d+", version), f"unexpected version output: {version!r}"
