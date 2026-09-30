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

    @pytest.mark.parametrize("mode", ["empty", "garbled", "incomplete"])
    def test_event_stream_never_counts_as_success(self, tmp_path: Path, mode: str) -> None:
        """Empty/garbled/incomplete streams must fail (exit 6), preserving
        artifacts, instead of passing as a successful model completion."""
        proc, out_dir = run_script("run_real_native.sh", tmp_path, mode)
        assert proc.returncode == 6, proc.stdout + proc.stderr
        assert (out_dir / "native-events.jsonl").exists(), "failed artifacts preserved"


class TestTimeoutValidation:
    @pytest.mark.parametrize("bad", ["0", "-1", "nan", "inf", "-inf"])
    def test_nonpositive_or_nonfinite_timeout_rejected(self, tmp_path: Path, bad: str) -> None:
        env = os.environ.copy()
        proc = subprocess.run(
            [sys.executable, "probe/run_bounded.py", "--timeout", bad,
             "--stdout", str(tmp_path / "o"), "--stderr", str(tmp_path / "e"),
             "--", sys.executable, "-c", "print('should not run')"],
            cwd=BASE,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert not (tmp_path / "o").exists() or (tmp_path / "o").stat().st_size == 0

    def test_missing_timeout_rejected(self, tmp_path: Path) -> None:
        proc = subprocess.run(
            [sys.executable, "probe/run_bounded.py", "--stdout", str(tmp_path / "o"),
             "--stderr", str(tmp_path / "e"), "--", sys.executable, "-c", "print(1)"],
            cwd=BASE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        assert proc.returncode == 2


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
        assert run_bounded.EXIT_INCOMPLETE_EVENTS == 6

    def test_terminal_success_requires_final_stop(self, tmp_path: Path) -> None:
        import run_bounded

        def write(name: str, records) -> Path:
            path = tmp_path / name
            path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            return path

        assistant = lambda stop: {"type": "message_end", "message": {"role": "assistant", "stopReason": stop}}

        assert run_bounded.has_terminal_success(write("ok.jsonl", [assistant("toolUse"), assistant("stop")]))
        assert not run_bounded.has_terminal_success(write("empty.jsonl", []))
        assert not run_bounded.has_terminal_success(write("incomplete.jsonl", [assistant("toolUse")]))
        garbled = tmp_path / "garbled.jsonl"
        garbled.write_text("not json {{{", encoding="utf-8")
        assert not run_bounded.has_terminal_success(garbled)
        assert not run_bounded.has_terminal_success(tmp_path / "missing.jsonl")
