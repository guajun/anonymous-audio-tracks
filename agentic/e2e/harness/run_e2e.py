#!/usr/bin/env python3
"""run_e2e.py — 真实端到端 harness（issue #33）：扮演用户 → Pi/Gemini → SAM → onset JSON。

职责边界（冻结）：

* harness **只做用户侧**：准备受控 clip、拼任务 prompt、按预算启动真实 Pi run（hard timeout）、
  采集 trace/用量、跑冻结校验、写 run manifest 与 latest 指针；
* harness **绝不生成 result.json、不预先固定任何分离/onset**——那必须由 Agent 自主完成
  （音频 hypothesis → 读 sam-audio SKILL → 真实 SAM 分离 → 自写 DSP → 产出 JSON）。
  任何"手工 pipeline + LLM 润色"都违背本 issue 验收；
* 真实 / mock 分离：真实运行（真实 Gemini + bridge + GPU SAM）与离线 fixture/mock 从不混用；
  mock 只在 tests/ 里出现，run manifest 记 `kind`。

真实通过判据（R1 修订）：**文件/schema 校验只算“数据校验”**；一次“真实 pass”还必须由
**观测到的执行证据**门控（execution gate，见 `execution_gate()`）：观测到的模型/提供方、
成功的音频附件（MIME+字节）、成功的真·SAM 分离（非 dry-run/非失败）、预算合规、必要 stage
非失败、原 runner 成功。`--verify` 不会把失败的原始 run 或缺失的执行证据洗成 pass。

失败语义（与 agentic/audio-probe/probe/run_bounded.py 一致，原样传播）：
``2`` 用法 / ``3`` 命令失败或无法启动 / ``4`` 超时（杀进程）/ ``5`` provider error /
``6`` 事件流空/乱码/不完整；``1`` = 运行完成但验收判据未过（result 缺失/校验失败/执行门未过）。
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
    classify_separations, count_real_separations, ensure_contained, extract_trace,
    is_safe_leaf_name, is_safe_run_id, make_redactor, read_json,
    sha256_file, write_json,
)

EXIT_OK, EXIT_ACCEPTANCE, EXIT_USAGE = 0, 1, 2
EXIT_GUARD = 3
MANIFEST_SCHEMA = "agentic-e2e-run-manifest/v2"
HARNESS_FILES = ("common.py", "make_clip.py", "run_e2e.py", "validate_result.py",
                 "spotcheck.py", "dsp_onset.py", "prompt_template.md")
REQUIRED_STAGES = ("model_run", "result", "validate")


# --------------------------------------------------------------------- prompt


def build_prompt(run_id: str, clip: dict, ws: Path, sam_separations: int,
                 sam_timeout: int, max_fixes: int) -> str:
    ok, why = is_safe_run_id(run_id)
    if not ok:
        raise ValueError(f"E_RUN_ID: {why}: {run_id!r}")
    ok, why = is_safe_leaf_name(str(clip.get("name", "")))
    if not ok:
        raise ValueError(f"E_NAME: clip 名不安全（{why}）：{clip.get('name')!r}")
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


# --------------------------------------------------- 代码/产物可归属性（R1）


def code_provenance(run_dir: Path, prior_code: dict | None = None) -> dict:
    """记录代码归属：git revision（当前）、harness 组件 hash、Agent DSP 脚本 hash。

    run 当时的 revision 若未记录则保持 ``unknown``（**不伪造**）；verification 时的
    revision/文件 hash 只证明“校验代码版本”，两者分开标注。"""
    def _git(*args: str) -> str:
        rc, out = run_quiet(["git", "-C", str(REPO_ROOT), *args])
        return out.strip() if rc == 0 else "unknown"

    revision = _git("rev-parse", "HEAD")
    dirty_out = _git("status", "--porcelain")
    dirty = None if dirty_out == "unknown" else bool(dirty_out)
    harness_hashes = {}
    for name in HARNESS_FILES:
        path = HERE / name
        if path.is_file():
            harness_hashes[f"harness/{name}"] = sha256_file(path)
    dsp_hashes = {}
    for path in sorted((run_dir / "scripts").glob("*.py")) if (run_dir / "scripts").is_dir() else []:
        dsp_hashes[f"scripts/{path.name}"] = sha256_file(path)
    return {
        "git_revision_at_run": (prior_code or {}).get("git_revision_at_run", "unknown（run 时未记录）"),
        "git_revision_at_verification": revision,
        "git_dirty_at_verification": dirty,
        "harness_files_sha256": harness_hashes,
        "agent_dsp_scripts_sha256": dsp_hashes,
        "note": "revision/hash 证明校验与 harness 代码归属；run 时代码未记录的部分保持 unknown",
    }


# ------------------------------------------------------- 执行证据门（R1）


def execution_gate(trace: dict | None, params: dict, run_dir: Path, local_dir: Path,
                   stages: list[dict], prior: dict | None = None) -> dict:
    """观测执行证据门：真实 pass 的必要条件（不是文档自述）。

    每项 {name, pass, detail}；任一失败即整体失败。"""
    checks: list[dict] = []
    trace = trace or {}

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    models = trace.get("models") or []
    providers = trace.get("providers") or []
    add("observed_model", models == ["gemini-3.8-flash"] and (not providers or providers == ["google"]),
        f"观测 assistant 消息 model={models} provider={providers}（要求仅 google/gemini-3.8-flash）")

    attach_calls = [c for c in trace.get("tool_calls", []) if c.get("tool") == "audio_attach"]
    attach_ok = [c for c in attach_calls
                 if not c.get("isError") and "mime" in c.get("result_head", "")
                 and "bytes" in c.get("result_head", "")]
    add("attachment_observed", bool(attach_ok),
        f"audio_attach 成功（MIME+字节） {len(attach_ok)}/{len(attach_calls)} 次"
        if attach_calls else "未观测到 audio_attach 调用")

    buckets = classify_separations(trace)
    add("sam_separation_observed", bool(buckets["real_success"]),
        f"真·SAM 分离成功 {len(buckets['real_success'])} 次"
        f"（dry-run {len(buckets['dry_run'])}、失败 {len(buckets['real_failed'])} 分开计数）")
    reports = list(Path(run_dir).glob("stems/*/report.json")) + list(Path(run_dir).glob("stems/*/*/report.json"))
    add("sam_artifacts_present", bool(reports),
        f"分离产物 report.json ×{len(reports)}（观测到的执行产物，不是文本自述）")

    max_sep = int(params.get("sam_separations_max", 0) or 0)
    add("budget_compliance",
        len(buckets["real_success"]) <= max_sep if max_sep > 0 else False,
        f"真实分离 {len(buckets['real_success'])} ≤ 预算 {max_sep}")

    bad_stages = [s for s in stages if s.get("stage", "").split(".")[0] in REQUIRED_STAGES
                  and s.get("status") not in ("ok", "skip")]
    add("required_stages_nonfailed", not bad_stages,
        "必要 stage（model_run/result/validate）均非失败" if not bad_stages
        else f"失败 stage：{[(s.get('stage'), s.get('status')) for s in bad_stages]}")

    events = local_dir / "events.jsonl"
    add("execution_evidence_present", events.is_file() and trace.get("tool_calls_total", 0) > 0,
        f"事件流 {events.name} 存在，工具调用 {trace.get('tool_calls_total', 0)} 次")

    runner_exit = (prior or {}).get("runner_exit", trace.get("runner_exit"))
    if prior is not None:
        add("original_runner_success", runner_exit == 0,
            f"原始 runner exit={runner_exit}（--verify 不得把失败 run 洗成 pass）")
    else:
        add("original_runner_success", runner_exit == 0,
            f"本次 runner exit={runner_exit}")

    return {"schema": "agentic-e2e-execution-gate/v1",
            "kind": "observed-execution",
            "ok": all(c["pass"] for c in checks),
            "checks": checks,
            "note": "执行证据门：证明真实执行；文件/schema 校验只是数据校验，不证明执行"}


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
                       params: dict, gpu: dict, prompt_sha: str, rc: int,
                       prior: dict | None = None, execution: dict | None = None) -> dict:
    run_dir = ws / E2E_OUTPUTS_REL / run_id
    local_dir = ws / "local" / "e2e" / run_id
    result_path = run_dir / "result.json"
    result_sha = sha256_file(result_path) if result_path.is_file() else None
    buckets = classify_separations(trace or {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    created_at = (prior or {}).get("created_at", now)   # 不改写原始 run 时间戳
    verifications = list((prior or {}).get("verification_runs", []))

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "issue": "guajun/anonymous-audio-tracks#33",
        "run_id": run_id,
        "kind": "real",
        "created_at": created_at,
        "models": {
            "music_agent": RESEARCH_MODEL,
            "impl_worker": IMPL_MODEL,
            "via": "pi 0.87.1 项目设置 + project trust（运行命令不传 --model）",
            "assistant_models_seen": (trace or {}).get("models", []),
            "assistant_providers_seen": (trace or {}).get("providers", []),
        },
        "clip": {
            "name": clip["name"], "sha256": clip["sha256"], "bytes": clip["bytes"],
            "duration_s": clip["duration_s"], "sample_rate": clip["sample_rate"],
            "channels": clip["channels"],
            "requested": clip.get("requested"),
            "actual": clip.get("actual"),
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
        "code": code_provenance(run_dir, (prior or {}).get("code")),
        "params": params,
        "gpu": gpu,
        "timings": (prior or {}).get("timings") or {
            "wall_s": round(time.time() - started, 3),
            "sam_reports": collect_sam_reports(run_dir),
        },
        "usage": (trace or {}).get("usage"),
        "trace": {
            "tool_calls_total": (trace or {}).get("tool_calls_total", 0),
            "tool_calls_by_tool": (trace or {}).get("tool_calls_by_tool", {}),
            "turns": (trace or {}).get("turns", 0),
            "audio_attach_calls": [c.get("args_summary") for c in (trace or {}).get("tool_calls", [])
                                   if c.get("tool") == "audio_attach"],
            "audio_attach_success": sum(1 for c in (trace or {}).get("tool_calls", [])
                                        if c.get("tool") == "audio_attach" and not c.get("isError")),
            "sam_separations": [s.get("command") for s in buckets["real_success"]],
            "sam_separations_real_success": len(buckets["real_success"]),
            "sam_separations_real_failed": len(buckets["real_failed"]),
            "sam_separations_dry_run": len(buckets["dry_run"]),
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
        "execution_gate": {
            "ok": (execution or {}).get("ok"),
            "checks": [c["name"] + ":" + ("pass" if c["pass"] else "FAIL")
                       for c in (execution or {}).get("checks", [])],
            "note": "观测执行证据门；文件/schema 校验只是数据校验",
        },
        "runner_exit": (prior or {}).get("runner_exit", rc),
        "verification_runs": verifications,
        "limitations": [
            "真实音乐无精确真值：onset/乐器/tempo 都是假设与估计，不承诺准确率",
            "抽查（spotcheck）是能量/同族交叉检测迹象，不是总体 accuracy",
            "跨轴能量代理（own/other 能量比）不是测得的分离度/串音率",
            "SAM 分离产物是目标/残差两路假设，可能含串音",
            "usage/cost 是 Pi 记账近似值，非账单真值（含 cacheRead/cacheWrite 口径）",
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
    p.add_argument("--run-id", default=None, help="run id（默认 e2e-real-<UTC 时间戳>；白名单字符）")
    p.add_argument("--clip-name", default="e2e-clip-001.wav", help="audio/inputs 内的 clip 文件名（单段）")
    p.add_argument("--clip-json", default=None, help="clip 描述（默认 <WS>/local/e2e/clip-<clip-name>.json）")
    p.add_argument("--timeout", default="2400", help="Pi run 硬超时秒数（有限正数）")
    p.add_argument("--sam-separations", type=int, default=3, help="真实 SAM 分离次数上限")
    p.add_argument("--sam-timeout", type=int, default=900, help="单次 SAM 调用 --timeout")
    p.add_argument("--max-fixes", type=int, default=2, help="校验修复循环上限")
    p.add_argument("--prompt-file", default=None, help="覆盖默认 prompt 模板实例化结果（followup 用）")
    p.add_argument("--dry-run", action="store_true", help="只打印 argv/环境，不启动、不写任何文件（零 API）")
    p.add_argument("--verify", action="store_true", help="对既有 run 重跑校验/执行门 + 更新 manifest（不改原时间戳）")
    p.add_argument("--redact", action="store_true", help="输出脱敏（公开粘贴用）")
    p.add_argument("--json", action="store_true", help="JSON 摘要输出")
    return p.parse_args(argv)


def load_events_safe(path: Path) -> list[dict]:
    from common import load_events
    try:
        return load_events(path)
    except OSError:
        return []


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code not in (0, None) else EXIT_OK

    ws = Path(args.workspace).expanduser().resolve()
    run_id = args.run_id or ("e2e-real-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    redact = make_redactor(ws, REPO_ROOT)

    # 写前守卫（R1）：run id / clip 名白名单 + 数值预算
    ok, why = is_safe_run_id(run_id)
    if not ok:
        print(f"[fail] E_RUN_ID: {why}: {run_id!r}", file=sys.stderr)
        return EXIT_USAGE
    ok, why = is_safe_leaf_name(args.clip_name)
    if not ok:
        print(f"[fail] E_NAME: clip 名不安全（{why}）：{args.clip_name!r}", file=sys.stderr)
        return EXIT_USAGE
    try:
        timeout = float(args.timeout)
        if not (timeout > 0) or timeout == float("inf"):
            raise ValueError
        if not (args.sam_separations > 0) or not (args.sam_timeout > 0) \
                or not (0 <= args.max_fixes <= 5):
            raise ValueError
    except (ValueError, TypeError):
        print("[fail] E_BUDGET: --timeout/--sam-separations/--sam-timeout 需为正有限数，"
              "--max-fixes ∈ [0,5]", file=sys.stderr)
        return EXIT_USAGE

    run_dir = ensure_contained(ws / E2E_OUTPUTS_REL, run_id)
    local_dir = ensure_contained(ws / "local" / "e2e", run_id)
    clip_json = Path(args.clip_json) if args.clip_json else ws / "local" / "e2e" / f"clip-{args.clip_name}.json"

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

    try:
        prompt = args.prompt_file and Path(args.prompt_file).read_text(encoding="utf-8") or \
            build_prompt(run_id, clip, ws, args.sam_separations, args.sam_timeout, args.max_fixes)
    except ValueError as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return EXIT_USAGE
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

    if args.verify:
        return verify_run(ws, run_id, clip, redact, args.json)

    # ---- 新 run：拒绝复用旧证据目录（R1 所有权/不可变）--------------------
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"[fail] E_RUN_EXISTS: run 目录已有产物（拒绝覆盖既有证据）：{redact(str(run_dir))}\n"
              f"  复核旧 run 用 --verify；新运行请换 --run-id", file=sys.stderr)
        return EXIT_GUARD
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"[fail] E_RUN_EXISTS: 本地证据目录已有文件（拒绝覆盖事件流/prompt）："
              f"{redact(str(local_dir))}\n  复核旧 run 用 --verify；新运行请换 --run-id", file=sys.stderr)
        return EXIT_GUARD

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
    trace["runner_exit"] = rc
    write_json(local_dir / "trace-summary.json", {
        "schema": "agentic-e2e-trace-summary/v2", "run_id": run_id, **trace})
    buckets = classify_separations(trace)
    run_status = {0: "ok", 4: "blocked", 5: "fail", 6: "fail"}.get(rc, "fail" if rc else "ok")
    stages.append({"stage": "model_run", "status": run_status,
                   "detail": f"run_bounded exit={rc}；assistant models={trace.get('models')} "
                             f"providers={trace.get('providers')}；工具调用 {trace.get('tool_calls_total')} 次；"
                             f"真·SAM 分离成功 {len(buckets['real_success'])} 次"
                             f"（dry-run {len(buckets['dry_run'])}、失败 {len(buckets['real_failed'])}）"})

    # ---- result / 数据校验 / 执行门 / 抽查 --------------------------------
    validation = None
    result_path = run_dir / "result.json"
    if result_path.is_file():
        stages.append({"stage": "result", "status": "ok",
                       "detail": f"Agent 产出 result.json（{result_path.stat().st_size} bytes）"})
        validation = validate_existing(ws, run_id, clip, run_dir)
        stages.append({"stage": "validate", "status": "ok" if validation["ok"] else "fail",
                       "detail": "; ".join(f"{c['name']}={'pass' if c['pass'] else 'FAIL'}"
                                           for c in validation["checks"])
                       + "（数据校验；不证明执行）"})
        stages.extend(spotcheck_run(ws, run_id, clip, run_dir))
    else:
        stages.append({"stage": "result", "status": "fail",
                       "detail": "未找到 Agent 产出的 outputs/e2e/<run-id>/result.json（见事件流定位失败阶段）"})

    execution = execution_gate(trace, params, run_dir, local_dir, stages)
    stages.append({"stage": "execution_gate", "status": "ok" if execution["ok"] else "fail",
                   "detail": ", ".join(f"{c['name']}={'pass' if c['pass'] else 'FAIL'}"
                                        for c in execution["checks"])})
    if validation is not None:
        validation["execution"] = execution
        validation["ok"] = bool(validation["ok"] and execution["ok"])
        write_json(run_dir / "validation.json", validation)

    acc_rc = EXIT_OK
    if rc != 0:
        acc_rc = rc                      # 传播 runner 失败码（2/3/4/5/6）
    elif not result_path.is_file():
        acc_rc = EXIT_ACCEPTANCE
    elif validation is not None and not validation["ok"]:
        acc_rc = EXIT_ACCEPTANCE
    elif not execution["ok"]:
        acc_rc = EXIT_ACCEPTANCE
    return finish(ws, run_id, clip, trace, stages, validation, started, params, gpu,
                  prompt_sha, acc_rc, redact, args.json, blocked=(rc in (4, 5, 6)),
                  execution=execution)


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
                                 f"疑似误检 {len(stats['detector_diff']['possible_false'])}；"
                                 f"跨轴能量比仅是代理，抽查非总体准确率）"})
    except Exception as exc:  # noqa: BLE001
        stages.append({"stage": "spotcheck", "status": "fail", "detail": f"抽查失败：{exc.__class__.__name__}: {exc}"})
    return stages


def verify_run(ws: Path, run_id: str, clip: dict, redact, as_json: bool) -> int:
    """对既有 run 复核（离线）：数据校验 + 执行证据门；不改原时间戳，追加 verification 记录。"""
    run_dir = ws / E2E_OUTPUTS_REL / run_id
    local_dir = ws / "local" / "e2e" / run_id
    if not (run_dir / "result.json").is_file():
        print(f"[fail] E_NO_RESULT: {run_dir / 'result.json'}", file=sys.stderr)
        return EXIT_USAGE
    prior_path = run_dir / "run-manifest.json"
    prior = read_json(prior_path) if prior_path.is_file() else None
    trace_path = local_dir / "trace-summary.json"
    # 优先从**原始事件流**重抽取（权威证据，含 provider/cache 口径）；事件流缺失才用旧摘要
    if (local_dir / "events.jsonl").is_file():
        trace = extract_trace(load_events_safe(local_dir / "events.jsonl"))
    elif trace_path.is_file():
        trace = read_json(trace_path)
    else:
        trace = None
    if prior is not None:
        trace = dict(trace or {})
        trace.setdefault("runner_exit", prior.get("runner_exit"))

    validation = validate_existing(ws, run_id, clip, run_dir)
    execution = execution_gate(trace, (prior or {}).get("params", {}), run_dir, local_dir,
                               (prior or {}).get("stages", []), prior=prior if prior else None)
    validation["execution"] = execution
    validation["ok"] = bool(validation["ok"] and execution["ok"])
    write_json(run_dir / "validation.json", validation)

    manifest = None
    if prior is not None:
        manifest = build_run_manifest(
            run_id, ws, clip, trace, prior.get("stages", []), validation, time.time(),
            prior.get("params", {}), prior.get("gpu", {}),
            (prior.get("repro") or {}).get("prompt_sha256", ""), prior.get("runner_exit", 0),
            prior=prior, execution=execution)
        manifest["verification_runs"] = list(prior.get("verification_runs", [])) + [{
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kind": "offline-reverify",
            "data_validation_ok": validation.get("ok"),
            "execution_gate_ok": execution.get("ok"),
            "note": "离线复核（无新 API/GPU）；原 run 时间戳/usage/stages 保持原样",
        }]
        write_json(prior_path, manifest)

    if as_json:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
    else:
        print(f"{'OK' if validation['ok'] else 'FAIL'} {run_id} level=real")
        for c in validation["checks"]:
            print(f"  [{'pass' if c['pass'] else 'FAIL'}] {c['name']}: {redact(c['detail'])}")
        print("  执行证据门：")
        for c in execution["checks"]:
            print(f"  [{'pass' if c['pass'] else 'FAIL'}] {c['name']}: {redact(c['detail'])}")
        if manifest:
            print(f"  manifest 更新（原 created_at={manifest['created_at']} 保持不变）："
                  f"真·SAM 分离成功 {manifest['trace']['sam_separations_real_success']} 次 / "
                  f"失败 {manifest['trace']['sam_separations_real_failed']} / "
                  f"dry-run {manifest['trace']['sam_separations_dry_run']}")
    return EXIT_OK if validation["ok"] else EXIT_ACCEPTANCE


def finish(ws: Path, run_id: str, clip: dict, trace: dict | None, stages: list[dict],
           validation: dict | None, started: float, params: dict, gpu: dict,
           prompt_sha: str, rc: int, redact, as_json: bool, blocked: bool,
           execution: dict | None = None) -> int:
    run_dir = ws / E2E_OUTPUTS_REL / run_id
    prior = read_json(run_dir / "run-manifest.json") if (run_dir / "run-manifest.json").is_file() else None
    manifest = build_run_manifest(run_id, ws, clip, trace, stages, validation,
                                  started, params, gpu, prompt_sha, rc,
                                  prior=prior, execution=execution)
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
        "execution_gate_ok": (execution or {}).get("ok"),
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
