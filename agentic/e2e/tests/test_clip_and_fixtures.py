"""tests/test_clip_and_fixtures.py — 受控 clip 与自生成 fixture（issue #33，全离线）。

覆盖：clip 裁剪元数据/实际 hash/登记幂等与冲突守卫/桥接预算；fixture 的 mock 标注与真值；
DSP helper 在 mock fixture 上的检出（证明方法可用，但**不**外推到真实音乐准确率）。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from common import sha256_file, write_wav16
from dsp_onset import detect_onsets, estimate_tempo
import make_clip
import make_mock_fixture


def _mk_source(tmp_path, seconds: float = 6.0, sr: int = 8000) -> Path:
    src = tmp_path / "source-mock.wav"
    n = int(seconds * sr)
    samples = [0.5 * math.sin(2 * math.pi * 220 * i / sr) for i in range(n)]
    write_wav16(src, samples, sr, channels=1)
    return src


def _mk_ws(tmp_path) -> Path:
    ws = tmp_path / "ws"
    (ws / ".pi").mkdir(parents=True)
    return ws


class TestMakeClip:
    def test_clip_metadata_and_hash(self, tmp_path):
        src = _mk_source(tmp_path)
        ws = _mk_ws(tmp_path)
        clip = make_clip.build_clip(src, 1.0, 2.5, ws, "e2e-clip-test.wav", 4 * 1024 * 1024)
        assert clip["duration_s"] == 2.5
        assert clip["sample_rate"] == 8000
        assert clip["channels"] == 1
        assert clip["origin"]["offset_s"] == 1.0
        assert clip["origin"]["sha256"] == sha256_file(src)
        clip_path = ws / "audio" / "inputs" / "e2e-clip-test.wav"
        assert clip_path.is_file()
        assert clip["sha256"] == sha256_file(clip_path)          # actual hash
        assert clip["bytes"] == clip_path.stat().st_size
        # 登记到 #31 的音频 manifest
        manifest = json.loads((ws / "audio" / "inputs" / "manifest.json").read_text(encoding="utf-8"))
        entry = next(e for e in manifest["entries"] if e["name"] == "e2e-clip-test.wav")
        assert entry["sha256"] == clip["sha256"]
        assert clip["registration"] == "ok"

    def test_clip_is_idempotent_and_rejects_conflicts(self, tmp_path):
        src = _mk_source(tmp_path)
        ws = _mk_ws(tmp_path)
        make_clip.build_clip(src, 0.0, 2.0, ws, "e2e-clip-test.wav", 4 * 1024 * 1024)
        again = make_clip.build_clip(src, 0.0, 2.0, ws, "e2e-clip-test.wav", 4 * 1024 * 1024)
        assert again["name"] == "e2e-clip-test.wav"               # 幂等：复用已有 clip
        other = tmp_path / "other-mock.wav"
        write_wav16(other, [0.1] * 8000, 8000, channels=1)
        with pytest.raises(RuntimeError, match="E_HASH_CONFLICT"):
            make_clip.build_clip(other, 0.0, 1.0, ws, "e2e-clip-test.wav", 4 * 1024 * 1024)

    def test_bridge_size_budget(self, tmp_path):
        src = _mk_source(tmp_path, seconds=6.0)
        ws = _mk_ws(tmp_path)
        with pytest.raises(RuntimeError, match="E_BRIDGE_SIZE"):
            make_clip.build_clip(src, 0.0, 6.0, ws, "e2e-clip-big.wav", max_bytes=1024)
        assert not (ws / "audio" / "inputs" / "e2e-clip-big.wav").exists()  # 超预算即清理

    def test_clip_range_guard(self, tmp_path):
        src = _mk_source(tmp_path)
        ws = _mk_ws(tmp_path)
        with pytest.raises(ValueError, match="E_CLIP_RANGE"):
            make_clip.build_clip(src, 99.0, 2.0, ws, "e2e-clip-x.wav", 4 * 1024 * 1024)

    def test_cli_exit_codes(self, tmp_path):
        src = _mk_source(tmp_path)
        ws = _mk_ws(tmp_path)
        assert make_clip.main(["--source", str(src), "--offset", "0.5", "--duration", "2.0",
                               "--workspace", str(ws), "--name", "cli-clip.wav"]) == 0
        assert make_clip.main(["--source", str(tmp_path / "missing.wav"), "--offset", "0",
                               "--duration", "1", "--workspace", str(ws)]) == 3
        assert make_clip.main(["--source", str(src), "--offset", "-1",
                               "--duration", "1", "--workspace", str(ws)]) == 2


class TestFixtureMockBoundary:
    def test_fixture_is_labelled_mock(self, fixture_audio):
        gt = fixture_audio["gt"]
        assert gt["kind"] == "mock"
        assert gt["schema"] == "agentic-e2e-fixture-gt/v1"
        assert gt["bpm"] == 120.0
        assert [s["id"] for s in gt["sources"]] == ["gt-kick", "gt-pluck", "gt-pad"]
        assert gt["audio"]["sha256"] == sha256_file(fixture_audio["wav"])

    def test_fixture_sources_are_disjoint_in_time(self, fixture_audio):
        gt = fixture_audio["gt"]
        kicks = gt["sources"][0]["onset_seconds"]
        plucks = gt["sources"][1]["onset_seconds"]
        assert all(abs(k - p) > 0.2 for k in kicks for p in plucks)  # 复音但 onset 不重合，便于核对


class TestDspHelperOnMockFixture:
    """DSP helper 在 mock fixture 上可检出真值（方法冒烟）；不代表真实音乐准确率。"""

    def test_detects_kick_onsets(self, fixture_audio):
        from common import read_mono
        samples, sr = read_mono(fixture_audio["wav"])
        got = detect_onsets(samples, sr, band="low")
        gt = fixture_audio["gt"]["sources"][0]["onset_seconds"]
        matched = [t for t in gt
                   if any(abs(o["onset_seconds"] - t) <= 0.08 for o in got)]
        assert len(matched) >= 11, f"kick 真值 {gt} vs 检出 {[o['onset_seconds'] for o in got]}"

    def test_detects_pluck_onsets(self, fixture_audio):
        from common import read_mono
        samples, sr = read_mono(fixture_audio["wav"])
        got = detect_onsets(samples, sr, band="high")
        gt = fixture_audio["gt"]["sources"][1]["onset_seconds"]
        matched = [t for t in gt
                   if any(abs(o["onset_seconds"] - t) <= 0.08 for o in got)]
        assert len(matched) >= 11, f"pluck 真值 {gt} vs 检出 {[o['onset_seconds'] for o in got]}"

    def test_tempo_estimate(self, fixture_audio):
        from common import read_mono
        samples, sr = read_mono(fixture_audio["wav"])
        tempo = estimate_tempo(detect_onsets(samples, sr))
        assert tempo["source"] == "dsp"
        assert tempo["bpm"] is not None and abs(tempo["bpm"] - 120.0) < 5.0

    def test_tempo_unknown_when_too_few_onsets(self):
        tempo = estimate_tempo([{"onset_seconds": 1.0}, {"onset_seconds": 2.0}])
        assert tempo["bpm"] is None and tempo["source"] == "unknown" and tempo["confidence"] == 0.0


class TestFixtureGenerationDeterminism:
    def test_same_seed_same_bytes(self, tmp_path):
        a = make_mock_fixture.write_fixture(tmp_path / "a", seed=7)
        b = make_mock_fixture.write_fixture(tmp_path / "b", seed=7)
        assert a["audio"]["sha256"] == b["audio"]["sha256"]
        c = make_mock_fixture.write_fixture(tmp_path / "c", seed=8)
        assert c["audio"]["sha256"] != a["audio"]["sha256"]
