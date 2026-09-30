"""tests/helpers.py — 离线测试用的文档构造器（issue #33）。

只构造**测试数据**（mock/自生成），不接触真实音频/模型；真实运行结果由 run_e2e 产生。
"""
from __future__ import annotations

import json
from pathlib import Path

from common import sha256_file, write_wav16


def make_clip_fixture(root: Path, name: str = "e2e-clip-001.wav",
                      duration_s: float = 4.0, sample_rate: int = 8000,
                      offset_s: float = 2.0) -> dict:
    """在 root/audio/inputs 造一个 clip 文件 + clip 描述 dict（mock 音频）。"""
    audio_root = root / "audio" / "inputs"
    audio_root.mkdir(parents=True, exist_ok=True)
    wav = audio_root / name
    samples = [0.2 * ((i // 40) % 2) for i in range(int(duration_s * sample_rate))]
    write_wav16(wav, samples, sample_rate, channels=1)
    clip = {
        "schema": "agentic-e2e-clip/v1",
        "name": name,
        "sha256": sha256_file(wav),
        "bytes": wav.stat().st_size,
        "duration_s": duration_s,
        "sample_rate": sample_rate,
        "channels": 1,
        "timeline": "t=0 是 clip 开头 = 源文件 offset_s 处；所有 onset 用 clip 时轴（秒）",
        "origin": {
            "file": "source-mock.wav", "sha256": "0" * 64, "bytes": 1,
            "duration_s": duration_s + offset_s, "sample_rate": sample_rate,
            "channels": 1, "offset_s": offset_s, "clipped_duration_s": duration_s,
        },
    }
    clip_path = root / "local" / "e2e" / f"clip-{name}.json"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_text(json.dumps(clip, ensure_ascii=False, indent=2), encoding="utf-8")
    return clip


def make_stem(run_dir: Path, rel: str = "stems/drums/target.wav", size: int = 2048) -> dict:
    """在 run 目录内造一个 stem 文件，返回 {filename, sha256}。"""
    path = run_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytes((i * 7 + 13) % 256 for i in range(size))
    path.write_bytes(payload)
    return {"filename": rel, "sha256": sha256_file(path)}


def build_result(clip: dict, *, kind: str = "real", stem: dict | None = None,
                 events: list | None = None, provenance_steps: list | None = None,
                 description: str | None = None) -> dict:
    """构造协议文档；kind=real 走真实来源链（bridge/llm/sam/dsp + pins），kind=mock 全 mock。"""
    if events is None:
        events = [
            {"id": "inst-1-ev-1", "onset_seconds": 1.0, "source": "dsp", "method": "spectral-flux"},
            {"id": "inst-1-ev-2", "onset_seconds": 2.5, "source": "dsp", "method": "spectral-flux"},
        ]
    if kind == "real":
        instrument_source = "sam"
        desc = description or "SAM 目标轨假设 + DSP onset 检测（乐器标签是假设，不是真值）"
        steps = provenance_steps or [
            {"tool": "audio-bridge audio_attach", "source": "bridge",
             "note": "Pi 原生音频不支持（#30），音频经 bridge 一次性注入 google/gemini-3.8-flash"},
            {"tool": "gemini-3.8-flash listening", "source": "llm", "note": "声音层/乐器假设"},
            {"tool": "audio-toolbox.sam pin dfbc40a9541f (sam c603de8794cc)", "source": "sam",
             "note": "sam separate 真实分离（目标/残差两路）"},
            {"tool": "e2e dsp_onset energy-flux", "source": "dsp", "note": "onset/BPM 估计"},
        ]
        limitations = ["真实音乐无精确真值：onset/乐器/tempo 都是估计，不承诺准确率"]
        tempo = {"bpm": 120.0, "source": "dsp", "confidence": 0.5}
    else:
        instrument_source = "mock"
        desc = description or "自生成合成 fixture（mock）：真值由构造定义"
        steps = provenance_steps or [
            {"tool": "aat-e2e-fixtures", "source": "mock", "note": "自生成纯合成 fixture，无真实音频"},
        ]
        limitations = ["纯合成 mock 示例，不代表任何模型/方法精度"]
        tempo = {"bpm": 120.0, "source": "mock", "confidence": 1.0}

    instrument = {
        "id": "inst-1",
        "label": "drums" if kind == "real" else "kick",
        "description": desc,
        "source": instrument_source,
        "confidence": 0.6 if kind == "real" else 1.0,
        "events": events,
    }
    if stem:
        instrument["stem"] = dict(stem)
    return {
        "schema_version": "agentic-audio-tracks/v1",
        "audio": {
            "filename": clip["name"],
            "sha256": clip["sha256"],
            "duration_seconds": clip["duration_s"],
            "sample_rate": clip["sample_rate"],
        },
        "tempo": tempo,
        "instruments": [instrument],
        "provenance": {"steps": steps},
        "limitations": limitations,
    }


def write_result(run_dir: Path, doc: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "result.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
