#!/usr/bin/env python3
"""smoke.py — REAL Pi smoke test for the deployed workspace (issue #31).

This is the REAL-model path (costs a little API): one bounded Pi run that
proves, in order,

  1. project trust works   — the run uses `pi --approve` with NO `--model`/
     `--provider` flags; every assistant message in the events must carry the
     model id `gemini-3.8-flash`, i.e. the trust-gated project
     `.pi/settings.json` supplied the fixed research model (the global default
     on the test machine is a different provider/model);
  2. skill discovery works — Pi's system prompt advertises the `sam-audio`
     skill (`skills` section) and the agent reports it;
  3. SAM CLI is callable   — the agent really runs
     `python .pi/skills/sam-audio/scripts/audio_toolbox.py --help` and the tool
     result contains the CLI usage text;
  4. the audio bridge tool is registered with its real error contract —
     the agent calls `audio_attach {"path": "smoke-missing.wav"}` and gets a
     FAILED tool result `E_AUDIO_NOT_FOUND` (no audio bytes involved).

Failure semantics (mirrors agentic/audio-probe/probe/run_bounded.py, reused):
  2 usage, 3 command failed/could not start, 4 timeout (process killed),
  5 provider error/aborted in the event stream, 6 empty/garbled/incomplete
  event stream. Artifacts (events/stderr/summary) are always kept under
  <workspace>/local/smoke/ (local only, never committed).

The event stream IS pi's stdout (`--mode json`), so the same file is handed to
run_bounded as both --stdout and --events.
"""
from __future__ import annotations

# Windows 控制台/管道默认 GBK 会把中文日志写成乱码；统一强制 UTF-8 输出。
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEMPLATE_DIR.parents[1]
MANIFEST_PATH = TEMPLATE_DIR / "manifest.json"
PROBE_DIR = REPO_ROOT / "agentic" / "audio-probe" / "probe"
RUN_BOUNDED = PROBE_DIR / "run_bounded.py"

sys.path.insert(0, str(PROBE_DIR))

SMOKE_PROMPT = """Automated smoke test for this workspace. Do EXACTLY these steps and nothing else:
1. Report the exact names of all skills currently available to you (from your skill list). Do NOT open any SKILL.md file.
2. Run this shell command and quote its first output line:
   python .pi/skills/sam-audio/scripts/audio_toolbox.py --help
3. Call the tool audio_attach with arguments {"path": "smoke-missing.wav"} and quote the exact error code from the failed result.
4. Finish with exactly this one line (values substituted, single line):
SMOKE-DONE skills=<comma-separated skill names> help=<first help line> err=<error code>
"""

EXPECTED_MODEL = "gemini-3.8-flash"


class SmokeError(RuntimeError):
    def __init__(self, code: str, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


def build_argv() -> list[str]:
    """Pi argv. NO --model/--provider: project settings must supply the pinned
    model (that is what the smoke verifies). NO --extension: the bridge must be
    loaded through project settings (trust-gated)."""
    try:
        import pi_launcher  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SmokeError("E_LAUNCHER", f"无法加载 pi_launcher：{exc.__class__.__name__}", 3)
    return pi_launcher.resolve_command([
        "pi", "--approve", "--mode", "json", "--no-session", "--", SMOKE_PROMPT,
    ])


def parse_events(events_path: Path) -> dict:
    """Extract exactly the facts the criteria need from the JSONL event stream."""
    facts: dict = {
        "models": [],            # model ids of assistant messages, in order
        "skills_section": "",    # system-prompt skills section (discovery evidence)
        "tool_results": [],      # (toolName, isError, result text)
        "final_text": "",        # last assistant text
        "usage": [],             # distinct assistant usage objects
    }
    seen_usage: set[str] = set()
    try:
        lines = events_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return facts

    def texts(node) -> str:
        return " ".join(c.get("text", "") for c in (node or [])
                        if isinstance(c, dict) and c.get("type") == "text")

    def walk(node) -> None:
        if isinstance(node, dict):
            message = node.get("message")
            if isinstance(message, dict):
                role = message.get("role")
                if role == "assistant":
                    if message.get("model"):
                        facts["models"].append(message.get("model"))
                    text = texts(message.get("content"))
                    if text.strip():
                        facts["final_text"] = text
                    usage = message.get("usage")
                    if isinstance(usage, dict):
                        key = json.dumps(usage, sort_keys=True)
                        if key not in seen_usage:
                            seen_usage.add(key)
                            facts["usage"].append(usage)
                elif role == "system":
                    sections = message.get("sections") or {}
                    if isinstance(sections, dict) and sections.get("skills"):
                        facts["skills_section"] = str(sections.get("skills"))
            if node.get("type") == "tool_execution_end":
                result = node.get("result") or {}
                facts["tool_results"].append(
                    (str(node.get("toolName")), bool(node.get("isError")), texts(result.get("content"))))
            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            walk(json.loads(line))
        except json.JSONDecodeError:
            continue
    return facts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="smoke.py", description="真实 Pi smoke（issue #31；会产生少量 API 费用）")
    parser.add_argument("--workspace", default=str(REPO_ROOT / ".local" / "agentic" / "workspace"))
    parser.add_argument("--timeout", default="300", help="硬超时秒数（有限正数；超时杀进程并退出 4）")
    parser.add_argument("--print-argv", action="store_true", help="只打印解析后的 native argv 与环境，不运行（离线）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 摘要（本地证据）")
    parser.add_argument("--redact", action="store_true", help="绝对路径替换为占位符")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code not in (0, None) else 0
    ws = Path(args.workspace).expanduser().resolve()
    try:
        timeout = float(args.timeout)
        if not (timeout > 0) or timeout == float("inf"):
            raise ValueError
    except ValueError:
        print("[fail] E_TIMEOUT: --timeout 必须是有限正数秒", file=sys.stderr)
        return 2

    try:
        pi_argv = build_argv()
    except SmokeError as exc:
        print(f"[fail] {exc.code}: {exc}", file=sys.stderr)
        return exc.exit_code

    if args.print_argv:
        printable = list(pi_argv)
        printable[-1] = "<SMOKE_PROMPT>"
        print(json.dumps({
            "argv": printable,
            "uses_model_flag": any(a == "--model" or a.startswith("--model") for a in pi_argv),
            "uses_approve": "--approve" in pi_argv,
            "env": {"PI_AUDIO_BRIDGE_ROOT": "<WORKSPACE>/audio/inputs" if args.redact
                    else str(ws / "audio" / "inputs")},
        }, ensure_ascii=False, indent=2))
        return 0

    smoke_dir = ws / "local" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    events = smoke_dir / "events.jsonl"
    stderr_f = smoke_dir / "stderr.txt"

    env = dict(os.environ)
    env["PI_AUDIO_BRIDGE_ROOT"] = str(ws / "audio" / "inputs")
    # pi --mode json 的事件流就是 stdout：同一文件交给 run_bounded 作为 --stdout/--events。
    command = [sys.executable, str(RUN_BOUNDED), "--timeout", str(timeout),
               "--stdout", str(events), "--stderr", str(stderr_f), "--events", str(events),
               "--"] + pi_argv
    try:
        proc = subprocess.run(command, cwd=str(ws), env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", shell=False, timeout=timeout + 60)
    except subprocess.TimeoutExpired:
        print("[fail] E_TIMEOUT: runner 超时（已保留产物）", file=sys.stderr)
        return 4
    except OSError as exc:
        print(f"[fail] E_LAUNCH: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 3

    rc = proc.returncode
    facts = parse_events(events)

    def tool_hit(name: str, error: bool | None, needle: str) -> bool:
        return any(t == name and (error is None or e is error) and needle in text
                   for t, e, text in facts["tool_results"])

    summary: dict = {"schema": "workspace-smoke-summary/v1", "run_bounded_exit": rc, "criteria": {}}

    def criterion(name: str, passed: bool, detail: str) -> None:
        summary["criteria"][name] = {"pass": bool(passed), "detail": detail}

    models_ok = bool(facts["models"]) and set(facts["models"]) == {EXPECTED_MODEL}
    criterion("model", models_ok,
              f"全部 assistant 消息 model={sorted(set(facts['models']))}（未传 --model，来自项目 settings + project trust）"
              if models_ok else f"模型不符：{sorted(set(facts['models']))}（要求仅 {EXPECTED_MODEL}）")
    skill_ok = "sam-audio" in facts["skills_section"] and "sam-audio" in facts["final_text"]
    criterion("skill", skill_ok,
              "system prompt skills 段含 sam-audio，且 agent 报告该 skill"
              if skill_ok else "system skills 段或 agent 回复中未见 sam-audio")
    help_ok = tool_hit("bash", False, "audio-toolbox")
    criterion("sam_cli", help_ok,
              "bash 工具真实执行 audio_toolbox.py --help，结果含 usage: audio-toolbox"
              if help_ok else "未见 bash 工具执行 audio_toolbox.py --help 的真实输出")
    bridge_ok = tool_hit("audio_attach", True, "E_AUDIO_NOT_FOUND")
    criterion("bridge_tool", bridge_ok,
              "audio_attach 已注册且失败返回 E_AUDIO_NOT_FOUND（失败即抛，契约真实）"
              if bridge_ok else "未见 audio_attach 失败调用或 E_AUDIO_NOT_FOUND")
    final_ok = "SMOKE-DONE" in facts["final_text"]
    criterion("final_line", final_ok,
              "终态回复包含 SMOKE-DONE 标记" if final_ok else "终态回复缺少 SMOKE-DONE 标记")

    usage = {
        "turns_with_usage": len(facts["usage"]),
        "totalTokens": sum(int(u.get("totalTokens") or 0) for u in facts["usage"]),
        "cost": round(sum(float((u.get("cost") or {}).get("total") or 0) for u in facts["usage"]), 6),
        "note": "Pi 记账近似值，非账单真值",
    }
    summary["usage"] = usage
    summary["result"] = "pass" if (rc == 0 and all(c["pass"] for c in summary["criteria"].values())) else "fail"
    if rc == 4:
        summary["result"] = "blocked-timeout"
    summary["artifacts"] = {
        "events": f"<WORKSPACE>/local/smoke/{events.name}",
        "stderr": f"<WORKSPACE>/local/smoke/{stderr_f.name}",
        "note": "完整事件流仅本地保留（local/smoke/，ignored）；公开材料只贴脱敏摘要",
    }
    (smoke_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def redact(text: str) -> str:
        return text.replace(str(ws), "<WORKSPACE>") if args.redact else text

    if args.json:
        print(redact(json.dumps(summary, ensure_ascii=False, indent=2)))
    else:
        print(f"smoke result: {summary['result']} (run_bounded exit {rc})")
        for name, item in summary["criteria"].items():
            print(f"  [{'pass' if item['pass'] else 'FAIL'}] {name}: {redact(item['detail'])}")
        print(f"  usage: totalTokens={usage['totalTokens']} cost={usage['cost']}（{usage['note']}）")
        print(f"  artifacts: {redact(str(smoke_dir))}")
    return 0 if summary["result"] == "pass" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # readable error, never a bare traceback
        print(f"[fail] E_INTERNAL: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
