#!/usr/bin/env python3
"""run_e2e.py — 真实端到端 harness（issue #33）：扮演用户 → Pi/Gemini → SAM → onset JSON。

职责边界（冻结）：

* harness **只做用户侧**：准备受控 clip、拼任务 prompt、按预算启动真实 Pi run（hard timeout）、
  采集 trace/用量、跑冻结校验、写 run manifest 与 latest 指针；
* harness **绝不生成 result.json、不预先固定任何分离/onset**——那必须由 Agent 自主完成
  （音频 hypothesis → 读 sam-audio SKILL → 真实 SAM 分离 → 自写 DSP → 产出 JSON）。
  任何"手工 pipeline + LLM 润色"都违背本 issue 验收；
* 真实 / mock 分离：`--kind real`（默认，真实 Gemini + bridge + GPU SAM）与离线 fixture/mock
  从不混用；mock 只在 tests/ 里出现，run manifest 记 `kind`。

失败语义（与 agentic/audio-probe/probe/run_bounded.py 一致，原样传播）：
``2`` 用法 / ``3`` 命令失败或无法启动 / ``4`` 超时（杀进程）/ ``5`` provider error /
``6`` 事件流空/乱码/不完整；``1`` = 运行完成但验收判据未过（result 缺失或校验失败）。
**blocked 不写成 success。**

用法::

    python agentic/e2e/harness/run_e2e.py --workspace <WS> [--run-id e2e-...] [--dry-run]
    python agentic/e2e/harness/run_e2e.py --workspace <WS> --verify --run-id e2e-...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
E2E_DIR = HERE.parent
AGENTIC_DIR = E2E_DIR.parent
REPO_ROOT = AGENTIC_DIR.parent
PROBE_DIR = AGENTIC_DIR / "audio-probe" / "probe"
VALIDATE_PY = AGENTIC_DIR / "schema" / "validate.py"
SEMANTIC_RULES = AGENTIC_DIR / "schema" / "schema" / "semantic_rules.json"
SCHEMA_README = AGENTIC_DIR / "schema" / "README.md"
PROMPT_TEMPLATE = HERE / "prompt_template.md"
RUN_BOUNDED = PROBE_DIR / "run_bounded.py"
for _p in (str(HERE), str(PROBE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    AUDIO_ROOT_REL, BRIDGE_SHA256, E2E_OUTPUTS_REL, IMPL_MODEL, LATEST_NAME,
    RESEARCH_MODEL, SAM_UPSTREAM_COMMIT, TOOLBOX_PIN,
    count_real_separations, extract_trace, make_redactor, read_json,
    sha256_file, write_json,
)

EXIT_OK, EXIT_ACCEPTANCE, EXIT_USAGE = 0, 1, 2
MANIFEST_SCHEMA = "agentic-e2e-run-manifest/v1"


# --------------------------------------------------------------------- prompt


def build_prompt(run_id: str, clip: dict, ws: Path, sam_separations: int,
                 sam_timeout: int, max_fixes: int) -> str:
    template = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    return template.format(
        run_id=run_id,
        clip_name=clip["name"],
        duration_s=clip["duration_s"],
        sample_rate=clip["sample_rate"],
        channels=clip["channels"],
        max_separations=sam_separations,
        sam_timeout=sam_timeout,
        max_fixes=max_fixes,
        dsp_helper=str(E2E_DIR / "harness" / "dsp_onset.py"),
        schema_readme=str(SCHEMA_README),
        semantic_rules=str(SEMANTIC_RULES),
        validate_py=str(VALIDATE_PY),
    )


# --------------------------------------------------------------------- 前置


def run_quiet(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout, shell=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 3, f"{exc.__class__.__name__}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def gpu_snapshot() -> dict:
    rc, out = run_quiet(["nvidia-smi",
                         "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                         "--format=csv,noheader"])
    if rc != 0:
        return {"available": False, "detail": "nvidia-smi 不可用"}
    parts = [p.strip() for p in out.strip().splitlines()[:1]]
    fields = [p.strip() for p in parts[0].split(",")] if parts else []
    return {
        "available": True,
        "name": fields[0] if len(fields) > 0 else "",
        "memory_total": fields[1] if len(fields) > 1 else "",
        "memory_used_before": fields[2] if len(fields) > 2 else "",
        "memory_free_before": fields[3] if len(fields) > 3 else "",
        "driver": fields[4] if len(fields) > 4 else "",
    }


def preflight(ws: Path, clip: dict, clip_json: Path) -> list[dict]:
    """启动前检查（每项 {stage, status, detail}）；返回 stages。"""
    stages: list[dict] = []
    if not (ws / ".pi").is_dir():
        stages.append({"stage": "preflight.workspace", "status": "fail",
                       "detail": f"不是已部署 workspace（缺 .pi/）：{ws}"})
        return stages
    stages.append({"stage": "preflight.workspace", "status": "ok", "detail": "workspace 布局存在"})

    clip_path = ws / AUDIO_ROOT_REL / clip["name"]
    if not clip_path.is_file():
        stages.append({"stage": "preflight.clip", "status": "fail",
                       "detail": f"clip 不存在：audio/inputs/{clip['name']}（先运行 make_clip.py）"})
        return stages
    actual = sha256_file(clip_path)
    if actual != clip.get("sha256"):
        stages.append({"stage": "preflight.clip", "status": "fail",
                       "detail": f"clip hash 漂移：实际 {actual[:12]}… vs 记录 {str(clip.get('sha256'))[:12]}…"})
        return stages
    stages.append({"stage": "preflight.clip", "status": "ok",
                   "detail": f"clip 存在且 hash 一致（{clip['bytes']} bytes，{clip['duration_s']}s）"})

    if not VALIDATE_PY.is_file() or not SCHEMA_README.is_file():
        stages.append({"stage": "preflight.schema", "status": "fail",
                       "detail": "冻结 schema/validator 缺失（agentic/schema）"})
        return stages
    stages.append({"stage": "preflight.schema", "status": "ok", "detail": "冻结 schema + validator 就绪"})

    if not (ws / "local" / "config.json").is_file():
        stages.append({"stage": "preflight.sam", "status": "fail",
                       "detail": "local/config.json 缺失（SAM 路径未配置）"})
    else:
        stages.append({"stage": "preflight.sam", "status": "ok", "detail": "SAM 路径配置存在（local/config.json）"})

    try:
        import pi_launcher  # type: ignore
        pi_argv = pi_launcher.resolve_command(["pi", "--version"])
        rc, out = run_quiet(pi_argv)
        version = out.strip().splitlines()[0] if out.strip() else ""
        stages.append({"stage": "preflight.pi", "status": "ok" if rc == 0 else "fail",
                       "detail": f"pi 版本：{version}"})
    except Exception as exc:  # noqa: BLE001
        stages.append({"stage": "preflight.pi", "status": "fail",
                       "detail": f"pi 启动器不可用：{exc.__class__.__name__}"})

    try:
        import pi_launcher  # type: ignore
        rc, out = run_quiet(pi_launcher.resolve_command(
            ["pi", "auth", "check", "--provider", "google", "--json"]))
        status = False
        if rc == 0:
            try:
                payload = json.loads(out.strip().splitlines()[-1])
                status = payload.get("status") == "ready"
            except (ValueError, IndexError):
                status = False
        stages.append({"stage": "preflight.auth", "status": "ok" if (rc == 0 and status) else "fail",
                       "detail": "google provider 认证就绪（状态 only，未打印 key）" if status
                       else f"认证状态未知/未就绪（rc={rc}）"})
    except Exception as exc:  # noqa: BLE001
        stages.append({"stage": "preflight.auth", "status": "fail",
                       "detail": f"认证检查失败：{exc.__class__.__name__}"})

    stages.append({"stage": "preflight.gpu", "status": "ok",
                   "detail": json.dumps(gpu_snapshot(), ensure_ascii=False)})
    return stages


# --------------------------------------------------------------------- manifest


def collect_sam_reports(run_dir: Path) -> list[dict]:
    reports = []
    for report_path in sorted(Path(run_dir).glob("stems/*/report.json")) + \
            sorted(Path(run_dir).glob("stems/*/*/report.json")):
        try:
            rep = read_json(report_path)
        except Exception:  # noqa: BLE001
            continue
        reports.append({
            "run_dir": report_path.parent.name,
            "description": rep.get("description"),
            "anchors": rep.get("anchors"),
            "device": rep.get("device"),
            "dtype": rep.get("dtype"),
            "sample_rate": rep.get("sample_rate"),
            "elapsed_s": rep.get("elapsed_s"),
            "outputs": [Path(p).name for p in (rep.get("outputs") or [])],
        })
    return reports


def build_run_manifest(run_id: str, ws: Path, clip: dict, trace: dict | None,
                       stages: list[dict], validation: dict | None, started: float,
                       params: dict, gpu: dict, prompt_sha: str, rc: int) -> dict:
    run_dir = ws / E2E_OUTPUTS_REL / run_id
    result_path = run_dir / "result.json"
    result_sha = sha256_file(result_path) if result_path.is_file() else None
    separations = count_real_separations(trace or {})
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "issue": "guajun/anonymous-audio-tracks#33",
        "run_id": run_id,
        "kind": "real",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": {
            "music_agent": RESEARCH_MODEL,
            "impl_worker": IMPL_MODEL,
            "via": "pi 0.87.1 项目设置 + project trust（运行命令不传 --model）",
            "assistant_models_seen": (trace or {}).get("models", []),
        },
        "clip": {
            "name": clip["name"], "sha256": clip["sha256"], "bytes": clip["bytes"],
            "duration_s": clip["duration_s"], "sample_rate": clip["sample_rate"],
            "channels": clip["channels"],
            "origin_offset_s": clip["origin"]["offset_s"],
            "origin_sha256": clip["origin"]["sha256"],
            "timeline": clip["timeline"],
        },
        "audio_pathway": {
            "kind": "bridge",
            "tool": "audio_attach",
            "bridge_sha256": BRIDGE_SHA256,
            "native": "unsupported（issue #30 实测；本次音频输入一律 bridge，不冒充 native）",
        },
        "tools": {
            "pi": "0.87.1",
            "sam_skill": "sam-audio",
            "sam_toolbox_pin": TOOLBOX_PIN,
            "sam_upstream_commit": SAM_UPSTREAM_COMMIT,
            "validator": "agentic/schema/validate.py（issue #32 冻结）",
        },
        "params": params,
        "gpu": gpu,
        "timings": {"wall_s": round(time.time() - started, 3),
                    "sam_reports": collect_sam_reports(run_dir)},
        "usage": (trace or {}).get("usage"),
        "trace": {
            "tool_calls_total": (trace or {}).get("tool_calls_total", 0),
            "tool_calls_by_tool": (trace or {}).get("tool_calls_by_tool", {}),
            "turns": (trace or {}).get("turns", 0),
            "audio_attach_calls": [c.get("args_summary") for c in (trace or {}).get("tool_calls", [])
                                   if c.get("tool") == "audio_attach"],
            "sam_separations": [s.get("command") for s in separations],
            "sam_separations_real": len(separations),
            "provider_failures": (trace or {}).get("provider_failures", []),
            "note": "完整事件流在 local/e2e/<run-id>/events.jsonl（本地保留，不入 git）",
        },
        "stages": stages,
        "outputs": {
            "result": f"outputs/e2e/{run_id}/result.json",
            "result_sha256": result_sha,
            "validation": f"outputs/e2e/{run_id}/validation.json",
            "notes": f"outputs/e2e/{run_id}/notes.md",
            "stems": sorted(str(p.relative_to(run_dir)).replace("\\", "/")
                            for p in run_dir.glob("stems/**/target.wav")) if run_dir.is_dir() else [],
            "spotcheck_dir": f"outputs/e2e/{run_id}/spotcheck",
        },
        "validation": {"ok": (validation or {}).get("ok"),
                       "checks": [c["name"] + ":" + ("pass" if c["pass"] else "FAIL")
                                  for c in (validation or {}).get("checks", [])]},
        "runner_exit": rc,
        "limitations": [
            "真实音乐无精确真值：onset/乐器/tempo 都是假设与估计，不承诺准确率",
            "抽查（spotcheck）是能量/交叉检测迹象，不是总体 accuracy",
            "SAM 分离产物是目标/残差两路假设，可能含串音",
            "usage/cost 是 Pi 记账近似值，非账单真值",
        ],
        "repro": {
            "commands": [
                "python agentic/e2e/harness/make_clip.py --source <AUDIO_WAV> --offset <OFFSET_S> --duration <CLIP_S> --workspace <WS> --name <CLIP_NAME>",
                "python agentic/e2e/harness/run_e2e.py --workspace <WS> --run-id <RUN_ID> --clip-name <CLIP_NAME>",
                "python agentic/e2e/harness/validate_result.py --result <WS>/outputs/e2e/<RUN_ID>/result.json --level real --clip <WS>/local/e2e/clip-<CLIP_NAME>.json --audio-root <WS>/audio/inputs --run-dir <WS>/outputs/e2e/<RUN_ID>",
                "python agentic/e2e/harness/spotcheck.py stats --result <WS>/outputs/e2e/<RUN_ID>/result.json --audio <WS>/audio/inputs/<CLIP_NAME> --run-dir <WS>/outputs/e2e/<RUN_ID> --kind real",
            ],
            "prompt_sha256": prompt_sha,
        },
        "artifacts_local": {
            "prompt": f"local/e2e/{run_id}/prompt.txt",
            "events": f"local/e2e/{run_id}/events.jsonl",
            "stderr": f"local/e2e/{run_id}/stderr.txt",
            "trace_summary": f"local/e2e/{run_id}/trace-summary.json",
            "session_dir": "sessions",
        },
    }
    return manifest


# --------------------------------------------------------------------- main


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="run_e2e.py", description="真实端到端 harness（issue #33）")
    p.add_argument("--workspace", required=True, help="已部署的 Pi workspace 根目录")
    p.add_argument("--run-id", default=None, help="run id（默认 e2e-real-<UTC 时间戳>）")
    p.add_argument("--clip-name", default="e2e-clip-001.wav", help="audio/inputs 内的 clip 文件名")
    p.add_argument("--clip-json", default=None, help="clip 描述（默认 <WS>/local/e2e/clip-<clip-name>.json）")
    p.add_argument("--timeout", default="2400", help="Pi run 硬超时秒数（有限正数）")
    p.add_argument("--sam-separations", type=int, default=3, help="真实 SAM 分离次数上限")
    p.add_argument("--sam-timeout", type=int, default=900, help="单次 SAM 调用 --timeout")
    p.add_argument("--max-fixes", type=int, default=2, help="校验修复循环上限")
    p.add_argument("--prompt-file", default=None, help="覆盖默认 prompt 模板实例化结果（followup 用）")
    p.add_argument("--dry-run", action="store_true", help="只打印 argv/环境，不启动（零 API）")
    p.add_argument("--verify", action="store_true", help="对既有 run 重跑校验 + 重建 manifest")
    p.add_argument("--redact", action="store_true", help="输出脱敏（公开粘贴用）")
    p.add_argument("--json", action="store_true", help="JSON 摘要输出")
    return p.parse_args(argv)


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code not in (0, None) else EXIT_OK

    ws = Path(args.workspace).expanduser().resolve()
    run_id = args.run_id or ("e2e-real-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    run_dir = ws / E2E_OUTPUTS_REL / run_id
    local_dir = ws / "local" / "e2e" / run_id
    clip_json = Path(args.clip_json) if args.clip_json else ws / "local" / "e2e" / f"clip-{args.clip_name}.json"
    redact = make_redactor(ws, REPO_ROOT)

    try:
        timeout = float(args.timeout)
        if not (timeout > 0) or timeout == float("inf"):
            raise ValueError
    except ValueError:
        print("[fail] E_TIMEOUT: --timeout 必须是有限正数秒", file=sys.stderr)
        return EXIT_USAGE

    if not clip_json.is_file():
        print(f"[fail] E_CLIP: clip 描述不存在：{clip_json}\n"
              f"  先运行 make_clip.py 生成受控 clip 并登记", file=sys.stderr)
        return EXIT_USAGE
    clip = read_json(clip_json)

    params = {
        "pi_timeout_s": timeout,
        "sam_separations_max": args.sam_separations,
        "sam_timeout_s": args.sam_timeout,
        "validator_fixes_max": args.max_fixes,
        "model_flag_passed": False,
        "approve": True,
    }

    prompt = args.prompt_file and Path(args.prompt_file).read_text(encoding="utf-8") or \
        build_prompt(run_id, clip, ws, args.sam_separations, args.sam_timeout, args.max_fixes)
    prompt_sha = sha256_bytes(prompt)

    import pi_launcher  # type: ignore
    pi_argv = pi_launcher.resolve_command([
        "pi", "--approve", "--print", "--mode", "json",
        "--session-dir", str(ws / "sessions"), "--name", run_id,
        "--", prompt,
    ])
    env = dict(os.environ)
    env["PI_AUDIO_BRIDGE_ROOT"] = str(ws / AUDIO_ROOT_REL)

    if args.dry_run:
        printable = list(pi_argv)
        printable[-1] = "<PROMPT>"
        payload = {
            "argv": [redact(a) if args.redact else a for a in printable],
            "env": {"PI_AUDIO_BRIDGE_ROOT": redact(env["PI_AUDIO_BRIDGE_ROOT"]) if args.redact
                    else env["PI_AUDIO_BRIDGE_ROOT"]},
            "cwd": redact(str(ws)) if args.redact else str(ws),
            "uses_model_flag": any(a == "--model" or a.startswith("--model") for a in pi_argv),
            "timeout_s": timeout,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK

    # ---- verify 模式：对既有 run 重跑校验 + 重建 manifest -------------------
    if args.verify:
        return verify_run(ws, run_id, clip, redact, args.json)

    started = time.time()
    local_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    stages = preflight(ws, clip, clip_json)
    gpu = gpu_snapshot()
    if any(s["status"] == "fail" for s in stages):
        return finish(ws, run_id, clip, None, stages, None, started, params, gpu,
                      prompt_sha, EXIT_ACCEPTANCE, redact, args.json, blocked=True)

    events = local_dir / "events.jsonl"
    stderr_f = local_dir / "stderr.txt"
    command = [sys.executable, str(RUN_BOUNDED), "--timeout", str(timeout),
               "--stdout", str(events), "--stderr", str(stderr_f), "--events", str(events),
               "--"] + pi_argv
    print(f"[run] run_id={run_id} timeout={timeout}s（事件流：{redact(str(events))}）")
    try:
        proc = subprocess.run(command, cwd=str(ws), env=env, shell=False,
                              timeout=timeout + 120)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        stages.append({"stage": "model_run", "status": "blocked",
                       "detail": "run_bounded 外层超时（已保留事件流/stderr）"})
        return finish(ws, run_id, clip, None, stages, None, started, params, gpu,
                      prompt_sha, 4, redact, args.json, blocked=True)
    except OSError as exc:
        stages.append({"stage": "model_run", "status": "fail", "detail": f"无法启动：{exc.__class__.__name__}"})
        return finish(ws, run_id, clip, None, stages, None, started, params, gpu,
                      prompt_sha, 3, redact, args.json, blocked=True)

    trace = extract_trace(load_events_safe(events))
    write_json(local_dir / "trace-summary.json", {
        "schema": "agentic-e2e-trace-summary/v1", "run_id": run_id, **trace})
    run_status = {0: "ok", 4: "blocked", 5: "fail", 6: "fail"}.get(rc, "fail" if rc else "ok")
    stages.append({"stage": "model_run", "status": run_status,
                   "detail": f"run_bounded exit={rc}；assistant models={trace.get('models')}；"
                             f"工具调用 {trace.get('tool_calls_total')} 次；"
                             f"真实 SAM 分离 {len(count_real_separations(trace))} 次"})

    # ---- result / 校验 / 抽查 --------------------------------------------
    validation = None
    result_path = run_dir / "result.json"
    if result_path.is_file():
        stages.append({"stage": "result", "status": "ok",
                       "detail": f"Agent 产出 result.json（{result_path.stat().st_size} bytes）"})
        validation = validate_existing(ws, run_id, clip, run_dir)
        write_json(run_dir / "validation.json", validation)
        stages.append({"stage": "validate", "status": "ok" if validation["ok"] else "fail",
                       "detail": "; ".join(f"{c['name']}={'pass' if c['pass'] else 'FAIL'}"
                                           for c in validation["checks"])})
        stages.extend(spotcheck_run(ws, run_id, clip, run_dir))
    else:
        stages.append({"stage": "result", "status": "fail",
                       "detail": "未找到 Agent 产出的 outputs/e2e/<run-id>/result.json（见事件流定位失败阶段）"})

    acc_rc = EXIT_OK
    if rc != 0:
        acc_rc = rc                      # 传播 runner 失败码（2/3/4/5/6）
    elif not result_path.is_file():
        acc_rc = EXIT_ACCEPTANCE         # 1 = 运行完成但没产出结果
    elif validation is not None and not validation["ok"]:
        acc_rc = EXIT_ACCEPTANCE
    return finish(ws, run_id, clip, trace, stages, validation, started, params, gpu,
                  prompt_sha, acc_rc, redact, args.json, blocked=(rc in (4, 5, 6)))


def load_events_safe(path: Path) -> list[dict]:
    from common import load_events
    try:
        return load_events(path)
    except OSError:
        return []


def validate_existing(ws: Path, run_id: str, clip: dict, run_dir: Path) -> dict:
    from validate_result import validate
    return validate(
        run_dir / "result.json", level="real",
        clip_path=ws / "local" / "e2e" / f"clip-{clip['name']}.json",
        audio_root=ws / AUDIO_ROOT_REL, run_dir=run_dir)


def spotcheck_run(ws: Path, run_id: str, clip: dict, run_dir: Path) -> list[dict]:
    stages = []
    try:
        import spotcheck
        audio = ws / AUDIO_ROOT_REL / clip["name"]
        doc = read_json(run_dir / "result.json")
        sc_dir = run_dir / "spotcheck"
        sc_dir.mkdir(parents=True, exist_ok=True)
        overlay = spotcheck.build_overlay(audio, doc, sc_dir / "overlay.wav")
        write_json(sc_dir / "overlay.json", overlay)
        stats = spotcheck.stats(audio, doc, run_dir, kind="real", sample=5)
        write_json(sc_dir / "spotcheck.json", stats)
        stages.append({"stage": "spotcheck", "status": "ok",
                       "detail": f"overlay + 抽查统计完成（抽样 {stats['method']['sampled_events']} 个事件；"
                                 f"能量上升 {sum(1 for e in stats['per_event'] if e['energy_rise_seen'])} 个；"
                                 f"疑似漏检 {len(stats['detector_diff']['possible_missed'])} / "
                                 f"疑似误检 {len(stats['detector_diff']['possible_false'])}；抽查非总体准确率）"})
    except Exception as exc:  # noqa: BLE001
        stages.append({"stage": "spotcheck", "status": "fail", "detail": f"抽查失败：{exc.__class__.__name__}: {exc}"})
    return stages


def verify_run(ws: Path, run_id: str, clip: dict, redact, as_json: bool) -> int:
    run_dir = ws / E2E_OUTPUTS_REL / run_id
    if not (run_dir / "result.json").is_file():
        print(f"[fail] E_NO_RESULT: {run_dir / 'result.json'}", file=sys.stderr)
        return EXIT_USAGE
    validation = validate_existing(ws, run_id, clip, run_dir)
    write_json(run_dir / "validation.json", validation)

    # 可重复执行的 manifest 重建：保留原始 stages/params/gpu/timings，只刷新可推导字段
    prior_path = run_dir / "run-manifest.json"
    rebuilt = None
    if prior_path.is_file():
        prior = read_json(prior_path)
        trace_path = ws / "local" / "e2e" / run_id / "trace-summary.json"
        trace = read_json(trace_path) if trace_path.is_file() else None
        rebuilt = build_run_manifest(
            run_id, ws, clip, trace, prior.get("stages", []), validation, time.time(),
            prior.get("params", {}), prior.get("gpu", {}),
            (prior.get("repro") or {}).get("prompt_sha256", ""), prior.get("runner_exit", 0))
        rebuilt["timings"] = prior.get("timings", rebuilt["timings"])
        write_json(prior_path, rebuilt)
    if as_json:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
    else:
        print(f"{'OK' if validation['ok'] else 'FAIL'} {run_id} level=real")
        for c in validation["checks"]:
            print(f"  [{'pass' if c['pass'] else 'FAIL'}] {c['name']}: {redact(c['detail'])}")
        if rebuilt:
            print(f"  manifest rebuilt: 真实 SAM 分离 {rebuilt['trace']['sam_separations_real']} 次；"
                  f"audio_attach {len(rebuilt['trace']['audio_attach_calls'])} 次")
    return EXIT_OK if validation["ok"] else EXIT_ACCEPTANCE


def finish(ws: Path, run_id: str, clip: dict, trace: dict | None, stages: list[dict],
           validation: dict | None, started: float, params: dict, gpu: dict,
           prompt_sha: str, rc: int, redact, as_json: bool, blocked: bool) -> int:
    manifest = build_run_manifest(run_id, ws, clip, trace, stages, validation,
                                  started, params, gpu, prompt_sha, rc)
    run_dir = ws / E2E_OUTPUTS_REL / run_id
    write_json(run_dir / "run-manifest.json", manifest)
    # latest 指针策略（冻结）：outputs/e2e/LATEST.txt = 最近一次 run id
    latest = ws / E2E_OUTPUTS_REL / LATEST_NAME
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(run_id + "\n", encoding="utf-8")

    summary = {
        "run_id": run_id,
        "result": "blocked" if blocked else ("pass" if rc == 0 else f"failed({rc})"),
        "runner_exit": rc,
        "stages": [f"{s['stage']}={s['status']}" for s in stages],
        "validation_ok": (validation or {}).get("ok"),
        "manifest": redact(str(run_dir / "run-manifest.json")),
    }
    if as_json:
        print(redact(json.dumps(summary, ensure_ascii=False, indent=2)))
    else:
        print(f"e2e {summary['result']} run_id={run_id}")
        for s in stages:
            print(f"  [{s['status']:>7}] {s['stage']}: {redact(s['detail'])}")
        print(f"  manifest: {summary['manifest']}")
    return rc


def sha256_bytes(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # 可读错误，不抛裸 traceback
        print(f"[fail] E_INTERNAL: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
