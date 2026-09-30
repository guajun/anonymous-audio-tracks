"""Offline regression tests for the REAL probe runners (review item 1).

Drives `probe/run_real_native.sh` / `probe/run_real_bridge.sh` against
`tests/fake_pi.py` in four modes (ok / fail / hang / error_event). NO API calls
are made; the real-model evidence in reports/ is not regenerated here.

Covered semantics: explicit timeout (exit 4), command-failure propagation with
preserved artifacts (exit 3), provider-error detection even when the command
exits 0 (exit 5), and success (exit 0).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]  # agentic/audio-probe
FAKE = BASE / "tests" / "fake_pi.py"
BASH = shutil.which("bash")

sys.path.insert(0, str(BASE / "probe"))

pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def run_script(script: str, tmp_path: Path, mode: str, timeout: str = "3"):
    out_dir = tmp_path / "out"
    env = os.environ.copy()
    env.update(
        {
            "PI_BIN": f"{sys.executable} {FAKE}",
            "AUDIO_PROBE_TMP": str(out_dir),
            "AUDIO_PROBE_TIMEOUT": timeout,
            "FAKE_PI_MODE": mode,
        }
    )
    proc = subprocess.run(
        [BASH, f"probe/{script}"],
        cwd=BASE,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    return proc, out_dir


class TestRealRunScriptsWithFakePi:
    def test_fake_success(self, tmp_path: Path) -> None:
        proc, out_dir = run_script("run_real_native.sh", tmp_path, "ok")
        assert proc.returncode == 0, proc.stderr
        events = (out_dir / "native-events.jsonl").read_text(encoding="utf-8")
        assert "fake final answer" in events

    def test_fake_failure_propagates_and_keeps_artifacts(self, tmp_path: Path) -> None:
        proc, out_dir = run_script("run_real_native.sh", tmp_path, "fail")
        assert proc.returncode == 3, proc.stdout + proc.stderr
        assert (out_dir / "native-stderr.txt").exists(), "artifacts must be preserved on failure"
        assert "simulated command failure" in (out_dir / "native-stderr.txt").read_text(encoding="utf-8")

    def test_fake_hang_is_killed_by_timeout(self, tmp_path: Path) -> None:
        proc, out_dir = run_script("run_real_native.sh", tmp_path, "hang", timeout="1")
        assert proc.returncode == 4, proc.stdout + proc.stderr

    def test_provider_error_detected_despite_exit_zero(self, tmp_path: Path) -> None:
        proc, out_dir = run_script("run_real_native.sh", tmp_path, "error_event")
        assert proc.returncode == 5, proc.stdout + proc.stderr
        assert (out_dir / "native-events.jsonl").exists(), "failure artifacts preserved"

    def test_bridge_script_success(self, tmp_path: Path) -> None:
        proc, out_dir = run_script("run_real_bridge.sh", tmp_path, "ok")
        assert proc.returncode == 0, proc.stderr
        assert (out_dir / "bridge-events.jsonl").exists()


class TestProviderFailureScan:
    def test_scan_detects_stop_reason_and_error_message(self, tmp_path: Path) -> None:
        import run_bounded

        cases = {
            "error": {"role": "assistant", "stopReason": "error", "content": []},
            "aborted": {"role": "assistant", "stopReason": "aborted", "content": []},
            "message": {"role": "assistant", "stopReason": "stop", "errorMessage": "boom", "content": []},
        }
        for name, message in cases.items():
            events = tmp_path / f"{name}.jsonl"
            events.write_text(json.dumps({"type": "message_end", "message": message}), encoding="utf-8")
            assert run_bounded.find_provider_failures(events), f"{name} must be flagged"

    def test_scan_accepts_clean_stream(self, tmp_path: Path) -> None:
        import run_bounded

        events = tmp_path / "clean.jsonl"
        events.write_text(
            json.dumps(
                {
                    "type": "message_end",
                    "message": {"role": "assistant", "stopReason": "stop", "content": [{"type": "text", "text": "ok"}]},
                }
            ),
            encoding="utf-8",
        )
        assert run_bounded.find_provider_failures(events) == []

    def test_run_bounded_usage_exit_code(self) -> None:
        import run_bounded

        assert run_bounded.EXIT_COMMAND_FAILED == 3
        assert run_bounded.EXIT_TIMEOUT == 4
        assert run_bounded.EXIT_PROVIDER_ERROR == 5
