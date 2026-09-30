"""Offline tests for the issue-30 audio probe and bridge (no network, no credentials).

Run from the repository root:

    uv run pytest agentic/audio-probe/tests -v

Tests that need Node or the installed Pi package skip themselves when the
dependency is missing (they are never silently "passed").
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]  # agentic/audio-probe
sys.path.insert(0, str(BASE / "probe"))

NODE = shutil.which("node")


def run_node(*args: str) -> subprocess.CompletedProcess:
    assert NODE, "node not available"
    return subprocess.run(
        [NODE, *args],
        cwd=BASE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


# ---------------------------------------------------------------------------
# fixture generator (pure Python)
# ---------------------------------------------------------------------------

class TestFixtureGenerator:
    def test_generates_valid_wav(self, tmp_path: Path) -> None:
        import make_fixture

        out = tmp_path / "fixture-a.wav"
        make_fixture.write_wav(out, make_fixture.FIXTURES["fixture-a.wav"])

        import wave

        with wave.open(str(out), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == make_fixture.RATE
            assert wf.getnframes() > 0

    def test_ground_truth_pitch_direction(self, tmp_path: Path) -> None:
        """The fixture's audible content matches the construction spec:
        3 distinct tones, ascending (or descending) in pitch."""
        import struct

        import make_fixture

        for name, freqs in make_fixture.FIXTURES.items():
            out = tmp_path / name
            make_fixture.write_wav(out, freqs)

            import wave

            with wave.open(str(out), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
            samples = struct.unpack(f"<{len(frames) // 2}h", frames)

            tone_len = int(make_fixture.RATE * make_fixture.TONE_SECONDS)
            gap_len = int(make_fixture.RATE * make_fixture.GAP_SECONDS)
            step = tone_len + gap_len

            estimated = []
            for i in range(len(freqs)):
                segment = samples[i * step : i * step + tone_len]
                zero_crossings = sum(
                    1 for a, b in zip(segment, segment[1:]) if (a >= 0) != (b >= 0)
                )
                estimated.append(zero_crossings / (2 * make_fixture.TONE_SECONDS))

            for got, want in zip(estimated, freqs):
                assert abs(got - want) < 30, f"{name}: estimated {got} vs {want}"

            ascending = estimated[0] < estimated[-1]
            expected_ascending = freqs[0] < freqs[-1]
            assert ascending == expected_ascending


# ---------------------------------------------------------------------------
# bridge guard logic (via Node harness)
# ---------------------------------------------------------------------------

class TestBridgeGuard:
    def test_guard_harness(self) -> None:
        if NODE is None:
            pytest.skip("node not available")
        proc = run_node("tests/node_guard_harness.mjs")
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# Pi payload shape / native-path probes (require installed Pi package)
# ---------------------------------------------------------------------------

class TestPiProbes:
    def test_payload_shape_has_mime_and_bytes(self) -> None:
        if NODE is None:
            pytest.skip("node not available")
        proc = run_node("probe/payload_shape.mjs", "--json")
        if proc.returncode == 2:
            pytest.skip(f"pi installation not found: {proc.stderr.strip()}")
        assert proc.returncode == 0, proc.stderr

        data = json.loads(proc.stdout)
        assert data["model"]["id"] == "gemini-3.8-flash"
        assert data["model"]["input"] == ["text", "image"], "catalog declares no audio modality"

        user_parts = data["cases"]["user"]["requestContents"][0]["parts"]
        inline = [p for p in user_parts if "inlineData" in p]
        assert inline, "bridge payload must contain inlineData"
        assert inline[0]["inlineData"]["mimeType"] == "audio/wav", "MIME must reach the API"

    def test_payload_shape_redacts_base64(self) -> None:
        if NODE is None:
            pytest.skip("node not available")
        proc = run_node("probe/payload_shape.mjs", "--json")
        if proc.returncode == 2:
            pytest.skip("pi installation not found")
        assert "base64 redacted" in proc.stdout
        data = json.loads(proc.stdout)
        inline_data = data["cases"]["user"]["requestContents"][0]["parts"][1]["inlineData"]["data"]
        assert inline_data.startswith("<base64 redacted")

    def test_native_paths_reject_audio(self, tmp_path: Path) -> None:
        if NODE is None:
            pytest.skip("node not available")
        import make_fixture

        fixture = tmp_path / "fixture-a.wav"
        make_fixture.write_wav(fixture, make_fixture.FIXTURES["fixture-a.wav"])

        proc = run_node("probe/native_paths_probe.mjs", "--file", str(fixture), "--json")
        if proc.returncode == 2:
            pytest.skip("pi installation not found")
        assert proc.returncode == 0, proc.stderr

        data = json.loads(proc.stdout)
        # Every documented native input path runs processImage(); audio must fail.
        assert data["processImage"]["ok"] is False
        assert "omitted" in data["processImage"]["message"].lower()
        # @file therefore falls back to the TEXT branch: mojibake, not audio.
        assert data["detectImageMime"]["fromBuffer"] is None
        assert data["atfileTextBranch"]["decodedAs"] == "utf-8"


# ---------------------------------------------------------------------------
# evidence redaction (never leak credentials / raw audio into reports)
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_redacts_base64_and_tokens(self) -> None:
        import extract_evidence

        raw = "data: " + "A" * 200 + " key=sk-abc123def456ghi Bearer eyJhbGciOi.payload"
        out = extract_evidence.redact(raw)
        assert "A" * 80 not in out
        assert "<BASE64 redacted: 200 chars>" in out
        assert "sk-abc123def456ghi" not in out
        assert "<REDACTED-TOKEN>" in out

    def test_redacts_home_path(self) -> None:
        import extract_evidence

        home = str(Path.home())
        out = extract_evidence.redact(f"reading {home}/secret/notes.txt")
        assert home not in out
        assert "<HOME>" in out
