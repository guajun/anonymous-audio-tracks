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


class TestExtensionMock:
    """Extension-level offline tests against the REAL bridge extension
    (mock Pi registry/context; review item 2)."""

    def test_extension_harness(self) -> None:
        if NODE is None:
            pytest.skip("node not available")
        proc = run_node("tests/node_extension_harness.mjs")
        assert proc.returncode == 0, proc.stderr
        for marker in (
            "OK happy attach",
            "OK multiple attachments + one-shot injection",
            "OK model restriction",
            "OK failed inputs",
            "OK lifecycle clear boundaries",
            "OK no transcript persistence",
        ):
            assert marker in proc.stdout, f"missing marker: {marker}\n{proc.stdout}"


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

    def test_redacts_before_truncation_boundary(self) -> None:
        """Secrets near the truncation cutoff must not leak a prefix
        (review item 4: redact the full text, truncate afterwards)."""
        import extract_evidence

        raw = "p" * 560 + " " + "sk-" + "b" * 40 + " " + "q" * 30
        out = extract_evidence.format_record(raw, 600)
        assert "sk-" not in out
        assert "bbbb" not in out

        # truncation still applies to long sanitized text
        assert extract_evidence.format_record("lorem ipsum " * 80, 600).endswith("…[truncated]")

        # even when the cutoff lands inside a redaction marker, no secret chars survive
        raw2 = "r" * 300 + " sk-" + "e" * 60 + " " + "u" * 400
        out2 = extract_evidence.format_record(raw2, 30)
        assert "sk-" not in out2
        assert "eeee" not in out2
        assert extract_evidence.find_unsanitized(out2) == []

        raw_b64 = "z" * 590 + "C" * 200 + "tail"
        out_b64 = extract_evidence.format_record(raw_b64, 600)
        assert "CCCC" not in out_b64

    def test_json_escaped_windows_paths(self) -> None:
        import extract_evidence

        raw = '{"path":"F:\\\\LED\\\\agentic-worktrees\\\\issue-30\\\\agentic\\\\x.wav"}'
        out = extract_evidence.redact(raw)
        assert "F:" not in out
        assert "<REPO>" in out

    def test_out_of_home_paths_are_scrubbed(self) -> None:
        import extract_evidence

        out = extract_evidence.redact("see D:\\other\\place\\f.txt for details")
        assert "D:" not in out
        assert "<PATH>" in out

    def test_repo_root_is_repo_root(self) -> None:
        import extract_evidence

        root = extract_evidence.repo_root()
        assert (root / "pyproject.toml").exists(), "repo root must contain pyproject.toml"
        assert root.name != "agentic", "parents[2] would be the agentic dir"
        out = extract_evidence.redact(str(root / "agentic" / "x.wav"))
        assert out.startswith("<REPO>")

    def test_main_end_to_end_no_leak(self, tmp_path: Path) -> None:
        import extract_evidence

        raw_file = tmp_path / "raw.txt"
        raw_file.write_text("p" * 590 + "sk-" + "c" * 40 + " tail " + str(Path.home()), encoding="utf-8")
        out = extract_evidence.format_record(raw_file.read_text(encoding="utf-8"), 600)
        assert extract_evidence.find_unsanitized(out) == []
        assert "sk-" not in out
        assert str(Path.home()) not in out
