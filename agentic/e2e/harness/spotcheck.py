#!/usr/bin/env python3
"""spotcheck.py — onset overlay 与抽查统计（issue #33，纯标准库、离线）。

用途（对应验收第 5 条）：

* ``overlay`` —— 在 clip 上叠加 onset 处的短促 click，生成 ``overlay.wav`` 供人类试听；
* ``stats``   —— 对若干 onset 时间点做**抽查**统计（能量上升比、**跨轴能量代理**（own/other
  能量比，不等于分离度/串音率）、独立检测器与 result 的差异计数），用于记录漏检/误检迹象。

**口径（冻结）**：这些都是抽查与迹象，**不是**总体准确率；真实音乐没有精确真值，
报告不得把本输出写成 accuracy，也不得把跨轴能量比写成“测得的串音/泄露”。真实/mock 由
``--kind`` 显式声明并写进输出。各 stem 用自己的采样率按秒取窗（SAM 输出常见 48 kHz，
mix 常见 44.1 kHz），绝不拿 mix 的采样率去索引 stem。

用法::

    python agentic/e2e/harness/spotcheck.py overlay --result R.json --audio A.wav --out overlay.wav
    python agentic/e2e/harness/spotcheck.py stats --result R.json --audio A.wav \
        [--run-dir <outputs/e2e/<id>>] [--kind real|mock] [--sample 5] [--out stats.json]
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
from common import read_json, read_mono, write_wav16  # noqa: E402
from dsp_onset import detect_onsets  # noqa: E402

EPS = 1e-9


def collect_events(doc: dict) -> list[dict]:
    events = []
    for inst in doc.get("instruments") or []:
        for ev in inst.get("events") or []:
            events.append({
                "id": ev.get("id"),
                "instrument": inst.get("id"),
                "label": inst.get("label"),
                "onset_seconds": ev.get("onset_seconds"),
                "stem": (inst.get("stem") or {}).get("filename"),
            })
    events.sort(key=lambda e: (e["onset_seconds"] or 0.0, str(e["id"])))
    return events


def rms(samples: list[float], sr: int, start_s: float, end_s: float) -> float:
    a = max(0, int(start_s * sr))
    b = min(len(samples), int(end_s * sr))
    if b <= a:
        return 0.0
    acc = 0.0
    for i in range(a, b):
        acc += samples[i] * samples[i]
    return math.sqrt(acc / (b - a))


def build_overlay(audio: Path, doc: dict, out: Path, gain: float = 0.5,
                  click_s: float = 0.06, click_hz: float = 1200.0) -> dict:
    samples, sr = read_mono(audio)
    events = collect_events(doc)
    overlay = [s * gain for s in samples]
    click_len = int(sr * click_s)
    click = [math.sin(2 * math.pi * click_hz * i / sr) * math.exp(-i / (sr * 0.01))
             for i in range(click_len)]
    for ev in events:
        t = ev["onset_seconds"]
        if t is None:
            continue
        start = int(t * sr)
        for i, v in enumerate(click):
            if 0 <= start + i < len(overlay):
                overlay[start + i] += 0.35 * v
    write_wav16(out, overlay, sr, channels=1)
    return {"schema": "agentic-e2e-overlay/v2", "audio": audio.name, "out": out.name,
            "sample_rate": sr, "onset_clicks": len(events), "click_hz": click_hz,
            "kind": "spotcheck-aid",
            "note": "click 仅标注 result 中的 onset 位置，供人类试听核对；不代表真值"}


def _pick(events: list[dict], sample: int) -> tuple[list[dict], str]:
    """确定性抽样：sample<=1 取首个事件；否则等距取至多 sample 个（首尾各留一席）。"""
    if sample <= 1:
        return (events[:1], "first-only")
    if len(events) > sample:
        idx = sorted({round(i * (len(events) - 1) / (sample - 1)) for i in range(sample)})
        return ([events[i] for i in idx], f"evenly-spaced:{len(idx)}")
    return (list(events), "all")


def stats(audio: Path, doc: dict, run_dir: Path | None, kind: str,
          sample: int, method: str = "energy-flux") -> dict:
    samples, sr = read_mono(audio)
    events = collect_events(doc)
    duration = len(samples) / sr

    picked, sample_policy = _pick(events, max(1, int(sample)))

    per_event = []
    for ev in picked:
        t = float(ev["onset_seconds"])
        post = rms(samples, sr, t, min(duration, t + 0.12))
        pre = rms(samples, sr, max(0.0, t - 0.25), max(0.0, t - 0.05))
        per_event.append({
            "id": ev["id"],
            "instrument": ev["instrument"],
            "label": ev["label"],
            "onset_seconds": t,
            "rms_post": round(post, 6),
            "rms_pre": round(pre, 6),
            "rise_ratio": round(post / (pre + EPS), 3),
            "energy_rise_seen": post > pre * 1.2,
        })

    # stem 跨轴能量代理（需要 run 目录里的实际 stem 文件）
    # 注意：每个 stem 用自己的采样率（SAM 输出常见 48 kHz，mix 可能是 44.1 kHz），
    # 窗口一律以**秒**换算到各自采样率索引——绝不拿 mix 的 sr 去索引 stem。
    cross_stem = []
    if run_dir:
        stems: dict[str, tuple[list[float], int]] = {}
        for inst in doc.get("instruments") or []:
            stem = (inst.get("stem") or {}).get("filename")
            if not stem:
                continue
            path = Path(run_dir) / stem
            if path.is_file():
                try:
                    stems[inst.get("id")] = read_mono(path)   # (samples, 自己的 sr)
                except Exception:
                    continue
        for ev in picked[:sample]:
            inst_id = ev["instrument"]
            if inst_id not in stems or len(stems) < 2:
                continue
            t = float(ev["onset_seconds"])
            own_sig, own_sr = stems[inst_id]
            own = rms(own_sig, own_sr, t - 0.05, t + 0.15)
            others = []
            for oid, (sig, sig_sr) in stems.items():
                if oid != inst_id:
                    others.append(rms(sig, sig_sr, t - 0.05, t + 0.15))
            other = sum(others) / len(others) if others else 0.0
            cross_stem.append({
                "id": ev["id"], "instrument": inst_id, "onset_seconds": t,
                "own_stem_rms": round(own, 6), "other_stems_rms": round(other, 6),
                "own_stem_sample_rate": own_sr,
                "cross_stem_energy_ratio": round(own / (other + EPS), 3),
                "proxy_only": True,
            })

    # 独立检测器 vs result 差异计数（漏检/误检迹象，非准确率）
    mismatch = []
    detector = detect_onsets(samples, sr)
    det_times = [d["onset_seconds"] for d in detector]
    reported = [float(e["onset_seconds"]) for e in events if e["onset_seconds"] is not None]
    matched_det = set()
    for t in reported:
        hit = min(range(len(det_times)), key=lambda i: abs(det_times[i] - t)) if det_times else None
        if hit is not None and abs(det_times[hit] - t) <= 0.1:
            matched_det.add(hit)
    possible_missed = [round(det_times[i], 3) for i in range(len(det_times)) if i not in matched_det]
    matched_rep = set()
    for i in matched_det:
        for j, t in enumerate(reported):
            if abs(det_times[i] - t) <= 0.1:
                matched_rep.add(j)
    possible_false = [round(reported[j], 3) for j in range(len(reported)) if j not in matched_rep]

    return {
        "schema": "agentic-e2e-spotcheck/v2",
        "kind": kind,
        "evidence_class": "dsp-spotcheck",
        "audio": {"filename": audio.name, "duration_s": round(duration, 6), "sample_rate": sr},
        "method": {"detector": method, "tolerance_s": 0.1, "sampled_events": len(picked),
                   "total_events": len(events), "sample_policy": sample_policy},
        "per_event": per_event,
        "cross_stem_energy": cross_stem,
        "detector_diff": {
            "detector_peaks": len(det_times),
            "reported_events": len(reported),
            "possible_missed": possible_missed[:20],
            "possible_false": possible_false[:20],
        },
        "limitations": [
            "抽查与迹象统计，不是总体准确率；真实音乐无精确真值",
            "rise_ratio 依赖窗口与阈值，弱 onset 可能比值接近 1",
            "cross_stem_energy_ratio 只是**跨轴能量代理**（own/other 能量比），不是测得的分离度/串音率：",
            "  多声源本就同时发声时比值≈1 并不能证明泄露；也未控制各 stem 增益差异",
            "各 stem 用自己的采样率按秒取窗（SAM 输出常见 48 kHz），但同秒窗口内仍含真实混叠",
            "detector_diff 用同族 DSP 方法交叉比对，共享偏差会被漏掉",
        ],
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="spotcheck.py", description="onset overlay / 抽查统计")
    sub = p.add_subparsers(dest="cmd", required=True)
    po = sub.add_parser("overlay")
    po.add_argument("--result", required=True)
    po.add_argument("--audio", required=True)
    po.add_argument("--out", required=True)
    po.add_argument("--gain", type=float, default=0.5)
    ps = sub.add_parser("stats")
    ps.add_argument("--result", required=True)
    ps.add_argument("--audio", required=True)
    ps.add_argument("--run-dir")
    ps.add_argument("--kind", default="real", choices=("real", "mock"))
    ps.add_argument("--sample", type=int, default=5)
    ps.add_argument("--out")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    doc = read_json(Path(args.result))
    audio = Path(args.audio)
    if not audio.is_file():
        print(f"[fail] E_AUDIO_NOT_FOUND: {audio}", file=sys.stderr)
        return 2
    if args.cmd == "overlay":
        report = build_overlay(audio, doc, Path(args.out), gain=args.gain)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    report = stats(audio, doc, Path(args.run_dir) if args.run_dir else None,
                   kind=args.kind, sample=max(1, int(args.sample)))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    print(json.dumps({
        "kind": report["kind"],
        "sampled": report["method"]["sampled_events"],
        "energy_rise_seen": sum(1 for e in report["per_event"] if e["energy_rise_seen"]),
        "possible_missed": len(report["detector_diff"]["possible_missed"]),
        "possible_false": len(report["detector_diff"]["possible_false"]),
        "note": "抽查统计，非总体准确率",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
