#!/usr/bin/env python3
"""stem_map.py — 唯一 stem 文件名映射（issue #33 R1；不改 schema、不动原 Agent 产物）。

背景：网页 #34 的 stem UI 按 basename 选文件时，SAM 多份 ``target.wav`` 会互相碰撞。
本工具**只做加法**：把每个来源的 stem 复制成唯一文件名副本（``stems-unique/``）并写
``stem-map.json``（含真实 hash / 相对路径 / 采样率 / 时长），供 #34 按 hash/身份消歧；
**不修改**原 Agent 的 result.json、stems/ 原件与任何原始证据（时轴不变）。

用法::

    python agentic/e2e/harness/stem_map.py --run-dir <WS>/outputs/e2e/<RUN_ID> [--out stems-unique]

退出码：0 成功 / 1 有 hash 不符等问题 / 2 用法。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from common import read_json, sha256_file, wav_info, write_json  # noqa: E402

STEM_MAP_SCHEMA = "agentic-e2e-stem-map/v1"


def build_stem_map(run_dir: Path, out_name: str = "stems-unique") -> dict:
    run_dir = Path(run_dir)
    result = read_json(run_dir / "result.json")
    out_dir = run_dir / out_name
    entries = []
    problems = []
    for inst in result.get("instruments") or []:
        stem = inst.get("stem") or {}
        rel = stem.get("filename")
        if not rel:
            continue
        src = run_dir / rel
        if not src.is_file():
            problems.append(f"{inst.get('id')}: 缺文件 {rel}")
            continue
        actual = sha256_file(src)
        if actual != stem.get("sha256"):
            problems.append(f"{inst.get('id')}: hash 不符（实际 {actual[:12]}… vs result {str(stem.get('sha256'))[:12]}…）")
            continue
        unique_name = f"stem-{inst.get('id')}-{src.name}"
        dst = out_dir / unique_name
        out_dir.mkdir(parents=True, exist_ok=True)
        if not dst.is_file():
            shutil.copy2(src, dst)
        elif sha256_file(dst) != actual:
            problems.append(f"{inst.get('id')}: 唯一副本已存在且内容不同：{unique_name}（不覆盖）")
            continue
        try:
            info = wav_info(dst)
            rate, duration = info["sample_rate"], round(info["duration_s"], 6)
        except Exception:  # noqa: BLE001
            rate, duration = None, None
        entries.append({
            "instrument_id": inst.get("id"),
            "label": inst.get("label"),
            "original_rel": rel,
            "unique_rel": f"{out_name}/{unique_name}",
            "sha256": actual,
            "bytes": src.stat().st_size,
            "sample_rate": rate,
            "duration_s": duration,
        })
    stem_map = {
        "schema": STEM_MAP_SCHEMA,
        "run_dir": run_dir.name,
        "note": "唯一 stem 文件名副本（加法产物）；原件与 result.json 未改动；时轴不变",
        "timeline": result.get("audio", {}).get("duration_seconds"),
        "entries": entries,
        "problems": problems,
        "ok": not problems,
    }
    write_json(run_dir / "stem-map.json", stem_map)
    return stem_map


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="stem_map.py", description="唯一 stem 文件名映射（issue #33）")
    p.add_argument("--run-dir", required=True, help="outputs/e2e/<RUN_ID>")
    p.add_argument("--out", default="stems-unique", help="唯一副本目录名（相对 run 目录）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    if not (run_dir / "result.json").is_file():
        print(f"[fail] E_NO_RESULT: {run_dir / 'result.json'}", file=sys.stderr)
        return 2
    stem_map = build_stem_map(run_dir, args.out)
    print(json.dumps({"ok": stem_map["ok"], "entries": len(stem_map["entries"]),
                      "problems": stem_map["problems"]}, ensure_ascii=False, indent=2))
    return 0 if stem_map["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
