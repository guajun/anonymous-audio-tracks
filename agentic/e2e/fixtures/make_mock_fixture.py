#!/usr/bin/env python3
"""make_mock_fixture.py — 自生成受控复音 fixture + 真值（issue #33，纯标准库、离线）。

**真实 / mock 边界（冻结口径）**：本脚本产生的一切都是 **mock**（自生成合成音频，
真值由构造定义），只用于离线自动测试与方法冒烟；报告与 run manifest 中一律标
``kind="mock"`` / ``source="mock"``，**绝不**当作真实音乐结果，也不据此声称准确率。

fixture 结构（12 s，44 100 Hz，单声道，三个叠加声源 = 复音）：

| 来源 | 说明 | 真值 onset（秒） |
|---|---|---|
| `kick`  | 低频衰减脉冲（打击） | 0.0, 1.0, …, 11.0（120 BPM 四分音符） |
| `pluck` | 中高频短音（拨弦类） | 0.5, 1.5, …, 11.5 |
| `pad`   | 持续和声垫 | 0.0 / 4.0 / 8.0（长音，onset 记起点） |

用法::

    python agentic/e2e/fixtures/make_mock_fixture.py --out <dir> [--name mock-poly-12s] [--seed 7]

输出：``<out>/<name>.wav`` + ``<out>/<name>.gt.json``（真值，schema `agentic-e2e-fixture-gt/v1`）。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E2E_DIR = HERE.parent
sys.path.insert(0, str(E2E_DIR / "harness"))
from common import sha256_file, wav_info, write_wav16  # noqa: E402

GT_SCHEMA = "agentic-e2e-fixture-gt/v1"
SR = 44100
DURATION_S = 12.0
KICK_ONSETS = [float(t) for t in range(12)]            # 0.0 .. 11.0
PLUCK_ONSETS = [t + 0.5 for t in range(12)]            # 0.5 .. 11.5
PAD_ONSETS = [0.0, 4.0, 8.0]
BPM = 120.0


def _exp_decay(n: int, sr: int, tau: float) -> list[float]:
    return [math.exp(-i / (sr * tau)) for i in range(n)]


def synthesize(name: str = "mock-poly-12s", seed: int = 7) -> tuple[list[float], dict]:
    """合成复音 fixture 样本与真值 dict。"""
    rng = random.Random(seed)
    n = int(SR * DURATION_S)
    mix = [0.0] * n

    # kick：80 Hz 正弦 + 噪声瞬态，tau=0.08s
    kick_len = int(SR * 0.25)
    decay = _exp_decay(kick_len, SR, 0.08)
    kick_wave = [decay[i] * (0.9 * math.sin(2 * math.pi * 80 * i / SR)
                             + 0.1 * (rng.random() * 2 - 1)) for i in range(kick_len)]
    for t in KICK_ONSETS:
        start = int(t * SR)
        for i, v in enumerate(kick_wave):
            if start + i < n:
                mix[start + i] += 0.55 * v

    # pluck：1.2 kHz 附近正弦（带轻微 FM），tau=0.12s
    pluck_len = int(SR * 0.30)
    decay = _exp_decay(pluck_len, SR, 0.12)
    for k, t in enumerate(PLUCK_ONSETS):
        f = 880.0 * (1.0 + 0.25 * ((k % 4) - 1.5) / 3.0)   # 小幅音高变化
        start = int(t * SR)
        for i in range(pluck_len):
            if start + i < n:
                env = decay[i]
                ph = 2 * math.pi * f * i / SR
                mix[start + i] += 0.32 * env * (math.sin(ph) + 0.3 * math.sin(2 * ph))

    # pad：三音和声垫，慢起慢收
    chord = [220.0, 277.18, 329.63]
    for t in PAD_ONSETS:
        start = int(t * SR)
        length = int(SR * 4.0)
        for i in range(length):
            if start + i < n:
                env = min(1.0, i / (SR * 0.4)) * min(1.0, (length - i) / (SR * 0.6))
                acc = 0.0
                for f in chord:
                    acc += math.sin(2 * math.pi * f * i / SR)
                mix[start + i] += 0.10 * env * acc / len(chord)

    peak = max(abs(v) for v in mix) or 1.0
    mix = [0.8 * v / peak for v in mix]

    gt = {
        "schema": GT_SCHEMA,
        "kind": "mock",
        "name": name,
        "note": "自生成合成 fixture：真值由构造定义，不代表任何真实音乐或模型精度",
        "duration_s": DURATION_S,
        "sample_rate": SR,
        "channels": 1,
        "bpm": BPM,
        "sources": [
            {"id": "gt-kick", "label": "kick", "onset_seconds": KICK_ONSETS,
             "band_hint": "low", "description": "80Hz 衰减脉冲 + 噪声瞬态"},
            {"id": "gt-pluck", "label": "pluck", "onset_seconds": PLUCK_ONSETS,
             "band_hint": "high", "description": "短衰减正弦（拨弦类）"},
            {"id": "gt-pad", "label": "pad", "onset_seconds": PAD_ONSETS,
             "band_hint": "mid", "description": "4 秒三音和声垫（onset 记起点）"},
        ],
    }
    return mix, gt


def write_fixture(out_dir: Path, name: str = "mock-poly-12s", seed: int = 7) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{name}.wav"
    gt_path = out_dir / f"{name}.gt.json"
    mix, gt = synthesize(name, seed)
    write_wav16(wav_path, mix, SR, channels=1)
    info = wav_info(wav_path)
    gt["audio"] = {
        "filename": wav_path.name,
        "sha256": sha256_file(wav_path),
        "bytes": wav_path.stat().st_size,
        "duration_s": round(info["duration_s"], 6),
        "sample_rate": info["sample_rate"],
    }
    gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return gt


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="make_mock_fixture.py", description="自生成 mock 复音 fixture（离线）")
    p.add_argument("--out", required=True, help="输出目录（生成物，不入 git）")
    p.add_argument("--name", default="mock-poly-12s")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    gt = write_fixture(Path(args.out), args.name, args.seed)
    print(json.dumps({"wav": gt["audio"]["filename"], "sha256": gt["audio"]["sha256"],
                      "kind": "mock", "sources": [s["id"] for s in gt["sources"]],
                      "bpm": gt["bpm"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
