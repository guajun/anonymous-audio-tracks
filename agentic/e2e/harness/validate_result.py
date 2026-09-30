#!/usr/bin/env python3
"""validate_result.py — 冻结 schema 校验 + e2e 一致性校验（issue #33）。

两层校验，缺一不可：

1. **冻结协议校验**（#32）：调用 ``agentic/schema`` 的 ``validate_file``（结构 + 语义，
   错误带 JSON Pointer），任何 schema 失败都直接判 FAIL；
2. **e2e 一致性校验**（本目录，比 schema 更严）：
   - ``clip_meta``      result 的 audio 名称/sha256/时长/采样率必须与受控 clip 实际值一致
     （sha256 现算文件字节，即 *actual hash*）；
   - ``timeline``       clip 时轴统一（t=0 = clip 开头），事件不得越界，audio 时轴一致；
   - ``event_provenance`` 每个事件必须显式 ``source`` + ``method``；DSP 检出的 onset 必须
     ``source="dsp"``（不允许继承乐器标签的 LLM/SAM 来源）；
   - ``provenance_pins``（real 级）模型 / 工具 / SAM 上游 pin 齐全，来源链含 llm|bridge + sam + dsp；
   - ``no_native_audio_claim``（real 级）Pi 原生音频是 unsupported，音频输入必须标 bridge；
   - ``privacy``        绝对路径 / key 形态 / 长 base64 一律拒绝（公开安全）；
   - ``stem_files``     stem 引用必须存在且 sha256 与实际文件一致（在 --run-dir/--audio-root 下）。

真实 / mock 边界：``--level real`` 要求真实来源链与 pin；``--level fixture`` 用于自生成
受控 fixture（mock），两者输出都带 ``level`` 字段，报告里不得混用。

用法::

    python agentic/e2e/harness/validate_result.py --result <result.json> --level real \
        [--clip <clip.json>] [--audio-root <dir>] [--run-dir <dir>] [--json]

退出码：0 全过 / 1 有 FAIL / 2 用法或文件错误（与 #32 CLI 口径一致）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E2E_DIR = HERE.parent
AGENTIC_DIR = E2E_DIR.parent
SCHEMA_DIR = AGENTIC_DIR / "schema"
for _p in (str(HERE), str(SCHEMA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    SCHEMA_VERSION, find_private, is_safe_rel_path, read_json,
    resolve_within, sha256_file,
)

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
LEVELS = ("real", "fixture")


class Checker:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append({"name": name, "pass": bool(ok), "detail": detail})

    @property
    def ok(self) -> bool:
        return all(c["pass"] for c in self.checks)


def frozen_schema_check(result_path: Path, checker: Checker) -> dict:
    try:
        from agentic_schema import validate_file  # type: ignore
    except Exception as exc:  # pragma: no cover - 环境缺失
        checker.add("schema", False, f"无法加载冻结校验器 agentic_schema：{exc.__class__.__name__}")
        return {"ok": False, "engine": None, "issues": []}
    report = validate_file(result_path)
    issues = [
        {"layer": i.layer, "code": i.code, "pointer": i.pointer, "message": i.message}
        for i in report.issues
    ]
    checker.add(
        "schema",
        report.ok,
        f"冻结校验 {SCHEMA_VERSION}（engine={report.engine}）：{len(issues)} 个问题"
        if not report.ok else f"冻结校验通过（engine={report.engine}）",
    )
    return {"ok": report.ok, "engine": report.engine, "issues": issues}


def check_clip_meta(doc: dict, clip: dict | None, audio_root: Path | None, checker: Checker) -> None:
    audio = doc.get("audio") or {}
    name = audio.get("filename")
    ok_name, why = is_safe_rel_path(str(name))
    checker.add("audio_filename_safe", ok_name,
                f"audio.filename 安全相对路径（{why}）：{name!r}")
    if clip:
        same_name = name == clip.get("name")
        checker.add("clip_meta", same_name,
                    f"audio.filename={name!r} 对 clip={clip.get('name')!r}")
        if same_name:
            checks = [
                ("sha256", audio.get("sha256"), clip.get("sha256")),
                ("sample_rate", audio.get("sample_rate"), clip.get("sample_rate")),
            ]
            for key, got, want in checks:
                checker.add(f"clip_{key}", got == want, f"result={got} clip={want}")
            dur_ok = abs(float(audio.get("duration_seconds", -1)) - float(clip.get("duration_s", -2))) <= 1e-6
            checker.add("clip_duration_seconds", dur_ok,
                        f"result={audio.get('duration_seconds')} clip={clip.get('duration_s')}")
    if audio_root and ok_name:
        try:
            path = resolve_within(audio_root, str(name))
        except ValueError as exc:
            checker.add("audio_file", False, str(exc))
            return
        if not path.is_file():
            checker.add("audio_file", False, f"音频文件不存在：{name}")
            return
        actual = sha256_file(path)
        checker.add("audio_file", actual == audio.get("sha256"),
                    f"实际文件 hash {actual[:16]}… vs result {str(audio.get('sha256'))[:16]}…（actual hash）")


def check_timeline(doc: dict, checker: Checker) -> None:
    audio = doc.get("audio") or {}
    duration = audio.get("duration_seconds")
    bad = []
    count = 0
    for inst in doc.get("instruments") or []:
        for ev in inst.get("events") or []:
            count += 1
            onset = ev.get("onset_seconds")
            if not isinstance(onset, (int, float)) or isinstance(onset, bool) \
                    or onset < 0 or onset > duration:
                bad.append(f"{ev.get('id')}:{onset}")
    checker.add("timeline", not bad,
                f"{count} 个事件全部落在 [0, {duration}] 秒（clip 时轴 t=0=clip 开头）"
                if not bad else f"越界事件：{bad[:5]}")


DSP_METHOD_HINTS = re.compile(r"flux|energy|spectral|peak|zcr|onset|wavelet|envelope|autocorr|percuss", re.I)


def check_event_provenance(doc: dict, level: str, checker: Checker) -> None:
    missing, mislabeled, dsp_count, total = [], [], 0, 0
    for inst in doc.get("instruments") or []:
        for ev in inst.get("events") or []:
            total += 1
            src, method = ev.get("source"), ev.get("method")
            if not src or not method:
                missing.append(str(ev.get("id")))
                continue
            if src == "dsp":
                dsp_count += 1
            elif DSP_METHOD_HINTS.search(str(method)):
                # DSP 检测法不得继承乐器/LLM 分类来源（来源错误）
                mislabeled.append(f"{ev.get('id')}:{src}/{method}")
    ok = not missing and not mislabeled and (level != "real" or dsp_count >= 1)
    detail = f"{total} 事件均显式 source+method，dsp 事件 {dsp_count} 个"
    if missing:
        detail = f"缺显式 source/method 的事件：{missing[:5]}"
    elif mislabeled:
        detail = f"DSP 检测法却标了非 dsp 来源（不得继承 LLM/SAM 分类）：{mislabeled[:5]}"
    elif level == "real" and dsp_count == 0:
        detail = "real 级要求至少一个 source='dsp' 的 DSP 检出事件"
    checker.add("event_provenance", ok, detail)


def check_real_pins(doc: dict, checker: Checker) -> None:
    prov = doc.get("provenance") or {}
    blob = json.dumps(prov, ensure_ascii=False)
    steps = prov.get("steps") or []
    sources = {s.get("source") for s in steps if isinstance(s, dict)}
    model_ok = "gemini-3.8-flash" in blob
    toolbox_ok = any(tok in blob for tok in ("audio-toolbox", "sam-audio", "dfbc40a9"))
    sam_ok = any(tok in blob for tok in ("c603de8", "run_inference", "sam-audio", "audio-toolbox"))
    chain_ok = ("sam" in sources and "dsp" in sources and ({"llm", "bridge"} & sources))
    checker.add("provenance_pins", model_ok and toolbox_ok and sam_ok,
                f"模型 pin={'有' if model_ok else '缺'} gemini-3.8-flash；工具 pin={'有' if toolbox_ok else '缺'}；"
                f"SAM 来源={'有' if sam_ok else '缺'}")
    checker.add("provenance_chain", chain_ok,
                f"来源链 steps.source 含 llm|bridge + sam + dsp：{sorted(s for s in sources if s)}")
    native_claims = [s.get("tool") for s in steps
                     if isinstance(s, dict) and s.get("source") == "native"]
    checker.add("no_native_audio_claim", not native_claims,
                "Pi 原生音频 unsupported，音频输入未冒充 native（应标 bridge）"
                if not native_claims else f"出现 source='native' 的步骤：{native_claims[:5]}")


def check_privacy(doc: dict, checker: Checker) -> None:
    text = json.dumps(doc, ensure_ascii=False)
    findings = find_private(text)
    checker.add("privacy", not findings,
                "未发现绝对路径/key 形态/长 base64" if not findings else f"隐私扫描命中：{findings}")


def check_stems(doc: dict, run_dir: Path | None, checker: Checker) -> None:
    stems = [(inst.get("id"), inst.get("stem")) for inst in doc.get("instruments") or []
             if isinstance(inst, dict) and inst.get("stem")]
    if not stems:
        checker.add("stem_files", True, "无 stem 引用（可选字段）")
        return
    if not run_dir:
        checker.add("stem_files", True, f"{len(stems)} 个 stem 引用（未提供 --run-dir，跳过文件核对）")
        return
    problems = []
    for inst_id, stem in stems:
        name = stem.get("filename")
        ok, why = is_safe_rel_path(str(name))
        if not ok:
            problems.append(f"{inst_id}: {why}")
            continue
        try:
            path = resolve_within(run_dir, str(name))
        except ValueError as exc:
            problems.append(f"{inst_id}: {exc}")
            continue
        if not path.is_file():
            problems.append(f"{inst_id}: 文件不存在 {name}")
            continue
        actual = sha256_file(path)
        if actual != stem.get("sha256"):
            problems.append(f"{inst_id}: hash 不符（实际 {actual[:12]}… vs {str(stem.get('sha256'))[:12]}…）")
    checker.add("stem_files", not problems,
                f"{len(stems)} 个 stem 文件存在且 hash 一致（相对 run 目录解析）"
                if not problems else f"stem 问题：{problems[:5]}")


def validate(result_path: Path, level: str = "real", clip_path: Path | None = None,
             audio_root: Path | None = None, run_dir: Path | None = None) -> dict:
    checker = Checker()
    try:
        doc = read_json(result_path)
    except Exception as exc:
        return {"schema": "agentic-e2e-result-validation/v1", "result": str(result_path.name),
                "level": level, "ok": False,
                "checks": [{"name": "parse", "pass": False,
                            "detail": f"result JSON 无法解析：{exc.__class__.__name__}"}]}
    clip = read_json(clip_path) if clip_path else None
    schema = frozen_schema_check(result_path, checker)
    if schema["ok"]:
        check_clip_meta(doc, clip, audio_root, checker)
        check_timeline(doc, checker)
        check_event_provenance(doc, level, checker)
        if level == "real":
            check_real_pins(doc, checker)
        check_privacy(doc, checker)
        check_stems(doc, run_dir, checker)
    return {
        "schema": "agentic-e2e-result-validation/v1",
        "result": result_path.name,
        "level": level,
        "ok": checker.ok,
        "schema_validation": schema,
        "checks": checker.checks,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="validate_result.py", description="冻结 schema + e2e 一致性校验")
    p.add_argument("--result", required=True, help="result.json 路径")
    p.add_argument("--level", default="real", choices=LEVELS, help="real=真实运行 / fixture=自生成 mock")
    p.add_argument("--clip", help="clip 描述 JSON（make_clip 输出）")
    p.add_argument("--audio-root", help="受控音频根目录（audio/inputs）")
    p.add_argument("--run-dir", help="run 目录（outputs/e2e/<id>，用于 stem 文件核对）")
    p.add_argument("--json", action="store_true", help="机器可读输出")
    p.add_argument("--out", help="把校验 JSON 写到该文件")
    return p.parse_args(argv)


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code not in (0, None) else EXIT_OK
    result = Path(args.result)
    if not result.is_file():
        print(f"[fail] E_NO_RESULT: 找不到 {result}", file=sys.stderr)
        return EXIT_USAGE
    report = validate(
        result,
        level=args.level,
        clip_path=Path(args.clip) if args.clip else None,
        audio_root=Path(args.audio_root) if args.audio_root else None,
        run_dir=Path(args.run_dir) if args.run_dir else None,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        state = "OK" if report["ok"] else "FAIL"
        print(f"{state} {report['result']} (level={report['level']})")
        for c in report["checks"]:
            print(f"  [{'pass' if c['pass'] else 'FAIL'}] {c['name']}: {c['detail']}")
        for issue in report.get("schema_validation", {}).get("issues", [])[:20]:
            print(f"    schema: {issue['pointer']} [{issue['layer']}/{issue['code']}] {issue['message']}")
    return EXIT_OK if report["ok"] else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
