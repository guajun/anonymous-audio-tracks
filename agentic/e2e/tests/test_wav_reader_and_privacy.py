"""tests/test_wav_reader_and_privacy.py — R1 回归（issue #33，全离线）。

覆盖 review 复现项：
* WAV 解码按**实际编码 tag**（float32 不再被当 int32；合成值精确回读）；
  PCM16/24/32 + IEEE float32 + WAVE_FORMAT_EXTENSIBLE fixture；不受支持/损坏格式受控拒绝；
* spotcheck 逐 stem 采样率（48k stem 不再被 44.1k mix 的 sr 索引）与 ``--sample 1`` 不除零；
* 隐私扫描抓**自由文本/JSON 转义里的绝对路径**（Windows/UNC/POSIX），相对引用与 URL 不误伤；
* 写前路径守卫（run id / clip 名 / realpath 包含）与 clip 小数参数幂等。
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

import common
import make_clip
import run_e2e
import spotcheck
from common import WavError, read_mono, wav_info, write_wav, write_wav16


def _wav_bytes(tag: int, bits: int, channels: int, sr: int, frames: list[bytes]) -> bytes:
    """手工构造 WAV（含非标准 tag/位深，用于拒绝路径）。"""
    block = channels * (bits // 8)
    payload = b"".join(frames)
    fmt = struct.pack("<HHIIHH", tag, channels, sr, sr * block, block, bits)
    out = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    out += b"fmt " + struct.pack("<I", len(fmt)) + fmt
    out += b"data" + struct.pack("<I", len(payload)) + payload
    return out


class TestWavReaderFormats:
    def test_float32_roundtrip_exact(self, tmp_path):
        """R1 复现项：float32 [0, .25, -.5, 1] 必须精确回读（旧版当 int32 解出 0/.488/-.508/.496）。"""
        path = tmp_path / "f32.wav"
        vals = [0.0, 0.25, -0.5, 1.0]
        write_wav(path, vals, 8000, fmt="float32")
        info = wav_info(path)
        assert info["tag"] == 3 and info["bits_per_sample"] == 32
        samples, sr = read_mono(path)
        assert sr == 8000
        assert samples == pytest.approx(vals)

    def test_pcm16_pcm24_pcm32_roundtrip(self, tmp_path):
        vals = [0.0, 0.25, -0.5, 0.75]
        for fmt, tag, bits in (("pcm16", 1, 16), ("pcm24", 1, 24), ("pcm32", 1, 32)):
            path = tmp_path / f"{fmt}.wav"
            write_wav(path, vals, 44100, fmt=fmt)
            info = wav_info(path)
            assert info["tag"] == tag and info["bits_per_sample"] == bits
            samples, sr = read_mono(path)
            assert sr == 44100
            assert samples == pytest.approx(vals, abs=1e-6)

    def test_stereo_mean(self, tmp_path):
        path = tmp_path / "st.wav"
        # 交替 L/R：帧均值 = (0.5 + -0.5)/2 = 0
        write_wav16(path, [0.5, -0.5, 0.25, -0.25], 8000, channels=2)
        samples, sr = read_mono(path)
        assert sr == 8000
        assert samples == pytest.approx([0.0, 0.0], abs=1e-3)

    def test_wave_format_extensible_resolves_subformat(self, tmp_path):
        """0xFFFE 容器 + float32 子格式 → tag 解析为 3 并按 float 解码。"""
        frames = [struct.pack("<f", v) for v in (0.5, -0.25)]
        fmt = struct.pack("<HHIIHH", 0xFFFE, 1, 8000, 8000 * 4, 4, 32)
        fmt += struct.pack("<HHI", 22, 32, 3)          # cbSize/validBits/mask
        fmt += struct.pack("<H", 3) + bytes(14)  # 子格式 GUID 头 2 字节=3（仅前 2 字节被解析）
        payload = b"".join(frames)
        out = b"RIFF" + struct.pack("<I", 4 + (8 + len(fmt)) + (8 + len(payload))) + b"WAVE"
        out += b"fmt " + struct.pack("<I", len(fmt)) + fmt
        out += b"data" + struct.pack("<I", len(payload)) + payload
        path = tmp_path / "ext.wav"
        path.write_bytes(out)
        info = wav_info(path)
        assert info["container_tag"] == 0xFFFE and info["tag"] == 3
        samples, _ = read_mono(path)
        assert samples == pytest.approx([0.5, -0.25])

    def test_unsupported_codec_rejected(self, tmp_path):
        path = tmp_path / "adpcm.wav"
        path.write_bytes(_wav_bytes(tag=2, bits=4, channels=1, sr=8000,
                                    frames=[b"\x00\x00"] * 4))
        with pytest.raises(WavError, match="E_WAV_CODEC"):
            wav_info(path)
        with pytest.raises(WavError, match="E_WAV_CODEC"):
            read_mono(path)

    def test_corrupt_block_align_rejected(self, tmp_path):
        path = tmp_path / "badalign.wav"
        path.write_bytes(_wav_bytes(tag=1, bits=16, channels=2, sr=8000,
                                    frames=[b"\x00\x00\x00\x00"] * 4).replace(
            struct.pack("<H", 4), struct.pack("<H", 3), 1))   # block_align 与 channels×bits/8 不符
        with pytest.raises(WavError, match="E_WAV_FORMAT"):
            wav_info(path)

    def test_empty_data_rejected(self, tmp_path):
        path = tmp_path / "empty.wav"
        path.write_bytes(_wav_bytes(tag=1, bits=16, channels=1, sr=8000, frames=[]))
        with pytest.raises(WavError, match="E_WAV_FORMAT"):
            wav_info(path)


class TestSpotcheckPerStemRate:
    def test_stem_windows_use_stem_sample_rate(self, tmp_path):
        """48k stem 不能被 44.1k mix 的 sr 索引：构造只在正确取窗时才出现能量上升的 stem。"""
        import math
        sr_mix, sr_stem = 44100, 48000
        mix = [0.0] * int(2.0 * sr_mix)
        write_wav16(tmp_path / "mix.wav", mix, sr_mix)
        # stem：0.5s 处起 0.2s 的正弦爆发
        stem = [0.0] * int(2.0 * sr_stem)
        for i in range(int(0.5 * sr_stem), int(0.7 * sr_stem)):
            stem[i] = 0.8 * math.sin(2 * math.pi * 440 * i / sr_stem)
        write_wav(tmp_path / "stem.wav", stem, sr_stem, fmt="float32")
        doc = {"instruments": [{"id": "inst-1", "label": "x",
                                "events": [{"id": "e-1", "onset_seconds": 0.5}],
                                "stem": {"filename": "stem.wav", "sha256": "0" * 64}},
                               {"id": "inst-2", "label": "y",
                                "events": [{"id": "e-2", "onset_seconds": 0.5}],
                                "stem": {"filename": "stem2.wav", "sha256": "0" * 64}}]}
        write_wav(tmp_path / "stem2.wav", [0.0] * int(2.0 * sr_stem), sr_stem, fmt="float32")
        stats = spotcheck.stats(tmp_path / "mix.wav", doc, tmp_path, kind="mock", sample=1)
        assert stats["cross_stem_energy"], "应产出跨轴能量代理"
        entry = stats["cross_stem_energy"][0]
        assert entry["own_stem_sample_rate"] == 48000
        assert entry["proxy_only"] is True
        # 正确取窗：onset 后能量显著高于 onset 前 → 比值远大于 1
        assert entry["cross_stem_energy_ratio"] > 5.0

    def test_sample_one_no_division_by_zero(self, tmp_path):
        write_wav16(tmp_path / "mix.wav", [0.0] * 8000, 8000)
        events = [{"id": f"e-{i}", "onset_seconds": i * 0.1, "source": "dsp", "method": "m"}
                  for i in range(5)]
        doc = {"instruments": [{"id": "inst-1", "label": "x", "events": events}]}
        for sample in (1, 2, 5, 99):
            stats = spotcheck.stats(tmp_path / "mix.wav", doc, None, kind="mock", sample=sample)
            assert stats["method"]["sampled_events"] >= 1
        assert stats["method"]["sample_policy"] in ("all", "first-only", "evenly-spaced:5")

    def test_no_isolation_wording(self, tmp_path):
        write_wav16(tmp_path / "mix.wav", [0.1] * 8000, 8000)
        doc = {"instruments": [{"id": "inst-1", "label": "x",
                                "events": [{"id": "e-1", "onset_seconds": 0.2}]}]}
        stats = spotcheck.stats(tmp_path / "mix.wav", doc, None, kind="mock", sample=1)
        blob = json.dumps(stats, ensure_ascii=False)
        assert "isolation_ratio" not in blob          # 旧的“隔离度”字段已废弃
        assert any("不是测得的分离度" in item for item in stats["limitations"])


class TestPrivacyAbsolutePaths:
    def test_windows_absolute_in_free_text(self):
        # 动态拼接：tracked 文件不得含真实个人路径布局字面量（toolbox hygiene 契约）
        drive = "Q" + ":"
        fake = drive + "/" + "synthetic" + "/" + "private" + "/" + "clip.wav"
        assert "Windows 绝对路径" in common.find_private(f"limitations: Loaded {fake}")

    def test_unc_and_posix_paths(self):
        unc = "\\\\" + "synthetic" + "\\\\" + "share" + "\\\\clip.wav"
        posix = "/" + "home" + "/" + "someone" + "/" + "clip.wav"
        assert "UNC 路径" in common.find_private(unc)
        assert "POSIX 绝对路径" in common.find_private(posix)

    def test_escaped_json_absolute_path(self):
        home = "C" + ":" + "\\\\" + "Users" + "\\\\" + "someone" + "\\\\" + "x.wav"
        doc = {"limitations": [f"loaded {home}"]}
        assert common.find_private(json.dumps(doc, ensure_ascii=False))

    def test_relative_refs_and_urls_are_clean(self):
        doc = {
            "audio": {"filename": "e2e-clip-001.wav"},
            "stem": {"filename": "stems/drums/target.wav"},
            "provenance": {"steps": [
                {"tool": "audio-toolbox.sam pin dfbc40a9541f (sam c603de8794cc)", "source": "sam"},
                {"tool": "https://github.com/guajun/agentic-audio-toolbox", "source": "bridge"},
                {"tool": "outputs/e2e/e2e-real-20261001-02/scripts/detect_events.py", "source": "dsp"},
            ]},
            "tempo": {"bpm": None, "source": "unknown", "confidence": 0},
        }
        assert common.find_private(json.dumps(doc, ensure_ascii=False)) == []


class TestWriteTimePathGuards:
    def test_run_id_whitelist(self):
        from common import is_safe_leaf_name, is_safe_run_id
        ok, _ = is_safe_run_id("e2e-real-20261001-01")
        assert ok
        for bad in ("../evil", "a/b", "a\\b", "", ".hidden", "x" * 65, "CON", "ends.", "中文id"):
            assert not is_safe_run_id(bad)[0], bad
        assert is_safe_leaf_name("e2e-clip-001.wav")[0]
        for bad in ("../escape.wav", "sub/x.wav", "C:/x.wav", "..\\x.wav"):
            assert not is_safe_leaf_name(bad)[0], bad

    def test_make_clip_rejects_escape_name_before_write(self, tmp_path):
        src = tmp_path / "src.wav"
        write_wav16(src, [0.1] * 8000, 8000)
        ws = tmp_path / "ws"
        (ws / ".pi").mkdir(parents=True)
        with pytest.raises(ValueError, match="E_NAME"):
            make_clip.build_clip(src, 0.0, 1.0, ws, "../escape.wav", 4 * 1024 * 1024)
        assert not (tmp_path / "escape.wav").exists()          # 什么都没写到 audio 根外
        assert not (ws / "audio").exists() or not any((ws / "audio").rglob("*.wav"))

    def test_run_e2e_cli_rejects_bad_ids(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        from helpers import make_clip_fixture
        make_clip_fixture(ws)
        base = [str(run_e2e.HERE / "run_e2e.py"), "--workspace", str(ws), "--dry-run"]
        import subprocess
        for bad in ("../evil", "a/b"):
            proc = subprocess.run([sys.executable, *base, "--run-id", bad],
                                  capture_output=True, text=True, encoding="utf-8", errors="replace")
            assert proc.returncode == 2, proc.stderr
        proc = subprocess.run([sys.executable, *base, "--clip-name", "../x.wav"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert proc.returncode == 2, proc.stderr

    def test_fresh_run_refuses_existing_evidence_dirs(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        from helpers import make_clip_fixture
        clip = make_clip_fixture(ws)
        run_dir = ws / "outputs" / "e2e" / "e2e-run-1"
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text("{}", encoding="utf-8")
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("p", encoding="utf-8")
        rc = run_e2e.main(["--workspace", str(ws), "--run-id", "e2e-run-1",
                           "--clip-name", clip["name"], "--prompt-file", str(prompt)])
        assert rc == 3                                  # 拒绝覆盖既有证据

    def test_fractional_clip_request_is_idempotent(self, tmp_path):
        src = tmp_path / "src.wav"
        write_wav16(src, [0.1] * 8000, 8000)     # 1s @ 8k
        ws = tmp_path / "ws"
        (ws / ".pi").mkdir(parents=True)
        first = make_clip.build_clip(src, 0.10003, 0.40003, ws, "fx.wav", 4 * 1024 * 1024)
        second = make_clip.build_clip(src, 0.10003, 0.40003, ws, "fx.wav", 4 * 1024 * 1024)
        assert first["sha256"] == second["sha256"]
        assert first["origin"]["frame_window"] == [800, 4000]
        assert first["requested"] == {"offset_s": 0.10003, "duration_s": 0.40003}
        assert first["actual"]["start_frame"] == 800
        # 不同帧窗口才冲突
        with pytest.raises(RuntimeError, match="E_HASH_CONFLICT"):
            make_clip.build_clip(src, 0.2, 0.3, ws, "fx.wav", 4 * 1024 * 1024)

    def test_nonfinite_and_zero_frame_and_budget_guards(self, tmp_path):
        src = tmp_path / "src.wav"
        write_wav16(src, [0.1] * 8000, 8000)
        ws = tmp_path / "ws"
        (ws / ".pi").mkdir(parents=True)
        for bad_offset, bad_dur in ((float("nan"), 1.0), (0.0, float("inf"))):
            with pytest.raises(ValueError):
                make_clip.build_clip(src, bad_offset, bad_dur, ws, "g.wav", 4 * 1024 * 1024)
        with pytest.raises(RuntimeError, match="E_CLIP_EMPTY"):
            make_clip.build_clip(src, 0.0, 1e-9, ws, "g2.wav", 4 * 1024 * 1024)
        with pytest.raises(ValueError, match="E_BRIDGE_BUDGET"):
            make_clip.build_clip(src, 0.0, 1.0, ws, "g3.wav", 0)
        assert not (ws / "audio" / "inputs" / "g.wav").exists()
