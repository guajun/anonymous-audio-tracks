"""Build a case-specific prompt from the reusable Gemini template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"
HISTORY = ROOT / "history"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=HISTORY / "target-breeze.json")
    parser.add_argument("--template", type=Path, default=PROMPTS / "prompt-general-v2.txt")
    parser.add_argument("--output", type=Path, default=PROMPTS / "prompt-breeze-v2.txt")
    args = parser.parse_args()
    case = json.loads(args.case.read_text(encoding="utf-8"))
    context = f"""本次任务参数：
- full_clip 时长 {case['duration_s']} 秒，来自原曲 {case['clip_start_in_original_s']} 秒起。
- reference 是 full_clip 的 {case['reference_start_in_clip_s']}–{case['reference_end_in_clip_s']} 秒裁切。
- 目标在 full_clip 的 {case['target_audible_at_clip_s']} 秒附近清晰可听，对应 reference 的 {round(case['target_audible_at_clip_s'] - case['reference_start_in_clip_s'], 6)} 秒。
- 用户定位线索：{case['user_observation']}
仅根据本次实际附带的音频分析；不假设额外提供了分离音频。"""
    text = args.template.read_text(encoding="utf-8").replace("{{TARGET_CONTEXT}}", context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
