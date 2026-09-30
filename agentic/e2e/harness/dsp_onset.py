#!/usr/bin/env python3
"""dsp_onset.py — 可选 DSP 辅助脚本（issue #33，纯标准库、离线）。

**定位**：这是给 Agent 的 *可选* helper（任务允许预提供 helper，但自主选择、调用与参数
必须留在 Agent 的真实工具调用 trace 里）。它不是结果来源：任何用它（或自写脚本）检出的
事件在 result 里必须标 ``source="dsp"`` + ``method=<检测方法名>``。

功能：

* ``onset`` —— 三频带能量谱通量 onset 检测（秒，clip 时轴）；
* ``tempo`` —— 基于 onset 间隔直方图的 BPM 估计（不确定就低置信度/unknown）。

不做：分离（那是 SAM 的事）、量化到节拍、填 pitch/duration（无证据不填）。

用法::

    python agentic/e2e/harness/dsp_onset.py onset --audio <clip.wav> [--band all|low|mid|high] \
        [--threshold 1.0] [--min-gap 0.12] [--json]
    python agentic/e2e/harness/dsp_onset.py tempo --audio <clip.wav> [--json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from common import read_mono, wav_info  # noqa: E402

FRAME = 1024
HOP = 256
BANDS = {"low": (0, 250.0), "mid": (250.0, 2000.0), "high": (2000.0, None)}


def _one_pole_lowpass(samples: list[float], sr: int, fc: float) -> list[float]:
    a = 1.0 - math.exp(-2.0 * math.pi * fc / sr)
    out = []
    y = 0.0
    for x in samples:
        y += a * (x - y)
        out.append(y)
    return out


def band_energies(samples: list[float], sr: int) -> dict[str, list[float]]:
    """逐帧（FRAME/HOP）三频带能量。"""
    low = _one_pole_lowpass(samples, sr, 250.0)
    lp2k = _one_pole_lowpass(samples, sr, 2000.0)
    series = {
        "low": low,
        "mid": [m - l for m, l in zip(lp2k, low)],
        "high": [x - m for x, m in zip(samples, lp2k)],
    }
    energies: dict[str, list[float]] = {}
    n = len(samples)
    frames = max(0, (n - FRAME) // HOP + 1)
    for name, sig in series.items():
        acc = []
        for f in range(frames):
            start = f * HOP
            e = 0.0
            for i in range(start, start + FRAME):
                e += sig[i] * sig[i]
            acc.append(e / FRAME)
        energies[name] = acc
    return energies


def detect_onsets(samples: list[float], sr: int, band: str = "all",
                  threshold: float = 1.0, min_gap: float = 0.12) -> list[dict]:
    """能量谱通量 onset 检测。返回 [{onset_seconds, strength, band}]。"""
    energies = band_energies(samples, sr)
    names = list(BANDS) if band == "all" else [band]
    flux = []
    frames = len(energies[names[0]])
    for f in range(frames):
        total = 0.0
        for name in names:
            e = energies[name]
            if f == 0:
                # 边界：音频开头的瞬态也算一次上升（把首帧能量当作从静音起）
                total += e[0]
            else:
                d = e[f] - e[f - 1]
                if d > 0:
                    total += d
        flux.append(total)
    if not flux:
        return []
    ordered = sorted(flux)
    median = ordered[len(ordered) // 2]
    mean = sum(flux) / len(flux)
    var = sum((x - mean) ** 2 for x in flux) / len(flux)
    std = math.sqrt(var)
    global_thr = median + threshold * (std + 1e-12)

    onsets: list[dict] = []
    last_t = -1e9
    for f in range(0, max(1, frames - 1)):
        v = flux[f]
        if v < global_thr:
            continue
        left_ok = (f == 0) or (v >= flux[f - 1])
        right_ok = (f + 1 < frames and v >= flux[f + 1]) or (f + 1 >= frames)
        if not (left_ok and right_ok):
            continue
        t = (f * HOP + FRAME / 2) / sr
        if t - last_t < min_gap:
            continue
        last_t = t
        onsets.append({
            "onset_seconds": round(t, 4),
            "strength": round(v / (std + 1e-12), 3),
            "band": band,
        })
    return onsets


def estimate_tempo(onsets: list[dict], min_bpm: float = 60.0, max_bpm: float = 200.0) -> dict:
    """基于相邻 onset 间隔直方图的 BPM 估计（保守：不确定就降置信度）。"""
    times = sorted(o["onset_seconds"] for o in onsets)
    if len(times) < 4:
        return {"bpm": None, "source": "unknown", "confidence": 0.0,
                "note": "onset 数量不足，无法估计 BPM"}
    intervals = [b - a for a, b in zip(times, times[1:]) if 0.12 <= b - a <= 2.5]
    if not intervals:
        return {"bpm": None, "source": "unknown", "confidence": 0.0,
                "note": "无可用 inter-onset 间隔"}
    bins: dict[int, int] = {}
    for iv in intervals:
        bins[round(iv * 1000)] = bins.get(round(iv * 1000), 0) + 1
    best_ms, best_n = max(bins.items(), key=lambda kv: (kv[1], -kv[0]))
    # 把最佳间隔折算到 [min_bpm, max_bpm]
    bpm = 60.0 / (best_ms / 1000.0)
    while bpm > max_bpm:
        bpm /= 2.0
    while bpm < min_bpm:
        bpm *= 2.0
    confidence = min(0.9, best_n / max(1, len(intervals)))
    return {
        "bpm": round(bpm, 2),
        "source": "dsp",
        "confidence": round(confidence, 2),
        "note": f"IOI 直方图峰值 {best_ms}ms ×{best_n}/{len(intervals)}（倍频折算到 "
                f"[{min_bpm:.0f},{max_bpm:.0f}] BPM；仅供辅助网格，不量化事件）",
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="dsp_onset.py", description="DSP onset / tempo helper（可选）")
    sub = p.add_subparsers(dest="cmd", required=True)
    po = sub.add_parser("onset")
    po.add_argument("--audio", required=True)
    po.add_argument("--band", default="all", choices=("all", "low", "mid", "high"))
    po.add_argument("--threshold", type=float, default=1.0)
    po.add_argument("--min-gap", type=float, default=0.12)
    po.add_argument("--json", action="store_true")
    pt = sub.add_parser("tempo")
    pt.add_argument("--audio", required=True)
    pt.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    path = Path(args.audio)
    if not path.is_file():
        print(f"[fail] E_AUDIO_NOT_FOUND: {path}", file=sys.stderr)
        return 2
    try:
        samples, sr = read_mono(path)
    except Exception as exc:
        print(f"[fail] E_AUDIO_READ: {exc}", file=sys.stderr)
        return 2
    if args.cmd == "onset":
        onsets = detect_onsets(samples, sr, band=args.band,
                               threshold=args.threshold, min_gap=args.min_gap)
        payload = {
            "schema": "agentic-e2e-dsp-onsets/v1",
            "method": "energy-flux",
            "audio": path.name,
            "duration_s": round(len(samples) / sr, 6),
            "sample_rate": sr,
            "band": args.band,
            "params": {"frame": FRAME, "hop": HOP, "threshold": args.threshold,
                       "min_gap_s": args.min_gap},
            "count": len(onsets),
            "events": onsets,
        }
    else:
        onsets = detect_onsets(samples, sr)
        payload = {
            "schema": "agentic-e2e-dsp-tempo/v1",
            "method": "ioi-histogram",
            "audio": path.name,
            "duration_s": round(wav_info(path)["duration_s"], 6),
            **estimate_tempo(onsets),
        }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for ev in payload.get("events", []):
            print(f"{ev['onset_seconds']:.3f}\t{ev['strength']}")
        if "bpm" in payload:
            print(f"bpm={payload['bpm']} confidence={payload['confidence']} source={payload['source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
