"""tests/test_harness_offline.py — harness 侧离线回归（issue #33，0 API 0 GPU）。

覆盖：事件流 trace 提取（工具链/用量/模型）、真实 SAM 分离计数（区分 dry-run 与真跑）、
prompt 模板实例化、`run_e2e --dry-run`（零 API，只打印 argv）、脱敏与隐私扫描、
run manifest / latest 指针策略、spotcheck（overlay + 抽查统计）。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

import common
import run_e2e
import spotcheck
from helpers import build_result, make_clip_fixture


# ------------------------------------------------------------------ 事件流


def _events():
    return [
        {"type": "session", "version": 1},
        {"type": "turn_start"},
        {"type": "message_end", "message": {"role": "assistant", "model": "gemini-3.8-flash",
                                            "stopReason": "toolUse", "usage": {"input": 100, "output": 20,
                                                                               "totalTokens": 120,
                                                                               "cost": {"total": 0.001}},
                                            "content": [{"type": "toolCall", "name": "audio_attach",
                                                         "arguments": {"path": "e2e-clip-001.wav"}}]}},
        {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "audio_attach",
         "args": {"path": "e2e-clip-001.wav"}},
        {"type": "tool_execution_end", "toolCallId": "c1", "toolName": "audio_attach", "isError": False,
         "result": {"content": [{"type": "text", "text": '{"attached": true, "mime": "audio/wav"}'}]}},
        {"type": "tool_execution_start", "toolCallId": "c2", "toolName": "bash",
         "args": {"command": "python audio_toolbox.py sam separate --audio x.wav --description drums "
                             "--output-dir outputs/e2e/run-1/stems/drums --timeout 900"}},
        {"type": "tool_execution_end", "toolCallId": "c2", "toolName": "bash", "isError": False,
         "result": {"content": [{"type": "text", "text": '{"ok": true, "action": "sam separate"}'}]}},
        {"type": "tool_execution_start", "toolCallId": "c3", "toolName": "bash",
         "args": {"command": "python audio_toolbox.py sam separate --audio x.wav --description melodic "
                             "--dry-run"}},
        {"type": "tool_execution_end", "toolCallId": "c3", "toolName": "bash", "isError": False,
         "result": {"content": [{"type": "text", "text": '{"ok": true, "action": "plan"}'}]}},
        {"type": "message_end", "message": {"role": "assistant", "model": "gemini-3.8-flash",
                                            "stopReason": "stop", "usage": {"input": 500, "output": 80,
                                                                            "totalTokens": 580,
                                                                            "cost": {"total": 0.004}},
                                            "content": [{"type": "text", "text": "E2E-DONE ok"}]}},
    ]


class TestTraceExtraction:
    def test_trace_summary(self):
        trace = common.extract_trace(_events())
        assert trace["models"] == ["gemini-3.8-flash"]
        assert trace["tool_calls_by_tool"] == {"audio_attach": 1, "bash": 2}
        assert trace["usage"]["totalTokens"] == 700
        assert trace["usage"]["cost_total"] == pytest.approx(0.005)
        assert "E2E-DONE" in trace["final_text_tail"]
        assert trace["last_assistant_stop"] == "stop"
        assert trace["provider_failures"] == []

    def test_real_separations_exclude_dry_run(self):
        trace = common.extract_trace(_events())
        real = common.count_real_separations(trace)
        assert len(real) == 1
        assert "--dry-run" not in real[0]["command"]

    def test_real_separations_trust_result_action_over_truncated_command(self):
        """长命令可能被截断；工具结果里的 action 字段是 dry-run / 真跑的权威标记。"""
        trace = {"tool_calls": [
            {"tool": "bash", "args_summary": "python audio_toolbox.py sam separate " + "x" * 400,
             "result_head": '{"action": "sam.dry-run", "ok": true}', "isError": False},
            {"tool": "bash", "args_summary": "python audio_toolbox.py sam separate " + "y" * 400,
             "result_head": '{"action": "sam.separate", "ok": true}', "isError": False},
        ]}
        real = common.count_real_separations(trace)
        assert len(real) == 1
        assert "sam.separate" in real[0]["result_head"]

    def test_provider_failure_detection(self):
        events = [{"type": "message_end", "message": {"role": "assistant", "stopReason": "error",
                                                      "errorMessage": "quota", "content": []}}]
        trace = common.extract_trace(events)
        assert trace["provider_failures"]


class TestPrompt:
    def test_prompt_fills_all_placeholders(self, tmp_path):
        clip = make_clip_fixture(tmp_path)
        prompt = run_e2e.build_prompt("e2e-run-1", clip, tmp_path, 3, 900, 2)
        assert "e2e-run-1" in prompt
        assert clip["name"] in prompt
        assert str(run_e2e.VALIDATE_PY) in prompt
        assert str(run_e2e.SEMANTIC_RULES) in prompt
        assert "google/gemini-3.8-flash" in prompt
        assert "source=\"dsp\"" in prompt            # DSP 来源标注要求
        assert "E2E-DONE" in prompt
        assert '{"path"' in prompt                   # audio_attach 用法（转义括号已展开）
        for token in ("{run_id}", "{clip_name}", "{validate_py}", "{max_separations}"):
            assert token not in prompt


class TestDryRun:
    def test_dry_run_prints_argv_without_launch(self, tmp_path):
        try:
            import pi_launcher
            pi_launcher.resolve_command(["pi", "--version"])
        except Exception:  # pragma: no cover - 本机未装 pi 时跳过
            pytest.skip("pi 启动器不可用")
        ws = tmp_path / "ws"
        ws.mkdir()
        make_clip_fixture(ws)
        proc = subprocess.run(
            [sys.executable, str(run_e2e.HERE / "run_e2e.py"), "--workspace", str(ws),
             "--run-id", "e2e-dry", "--dry-run", "--json"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["uses_model_flag"] is False     # 模型来自项目设置，不传 --model
        assert "--approve" in payload["argv"]
        assert payload["env"]["PI_AUDIO_BRIDGE_ROOT"].replace("\\", "/").endswith("audio/inputs")
        assert not (ws / "outputs").exists()            # dry-run 不产生任何产物


class TestRedaction:
    def test_redactor_and_secret_scan(self):
        redact = common.make_redactor(Path("C:/work/ws"))
        text = r"C:\work\ws\audio\inputs\a.wav and C:\Users\me\private\x.wav"
        out = redact(text)
        assert "C:\\work\\ws" not in out
        assert "private" not in out.split("private")[-1] or "<PATH>" in out
        assert common.find_private("api_key = AIzaSyFakeFakeFakeFakeFake123456")
        assert common.find_private("x" * 300)           # 长 base64 形态
        assert not common.find_private("onset_seconds: 1.5，source=dsp")

    def test_safe_rel_path(self):
        assert common.is_safe_rel_path("stems/drums/target.wav")[0]
        for bad in ("C:/x.wav", "../x.wav", "/abs/x.wav", "a/../b.wav", "a//b.wav", "con.wav"):
            assert not common.is_safe_rel_path(bad)[0], bad


class TestManifestAndLatest:
    def test_manifest_records_bridge_not_native_and_latest_pointer(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        clip = make_clip_fixture(ws)
        started = time.time() - 5
        trace = common.extract_trace(_events())
        stages = [{"stage": "model_run", "status": "ok", "detail": "run_bounded exit=0"}]
        validation = {"ok": True, "checks": [{"name": "schema", "pass": True, "detail": ""}]}
        params = {"pi_timeout_s": 2400, "sam_separations_max": 3, "sam_timeout_s": 900}
        rc = run_e2e.finish(ws, "e2e-run-1", clip, trace, stages, validation, started,
                            params, {"available": True}, "ab" * 32, 0,
                            (lambda s: s), as_json=True, blocked=False)
        assert rc == 0
        manifest = json.loads((ws / "outputs" / "e2e" / "e2e-run-1" / "run-manifest.json")
                              .read_text(encoding="utf-8"))
        assert manifest["schema"] == "agentic-e2e-run-manifest/v1"
        assert manifest["kind"] == "real"
        assert manifest["audio_pathway"]["kind"] == "bridge"
        assert "unsupported" in manifest["audio_pathway"]["native"]
        assert manifest["models"]["music_agent"] == "google/gemini-3.8-flash"
        assert manifest["models"]["impl_worker"] == "openrouter/xiaomi/mimo-v2.6-pro"
        assert manifest["tools"]["sam_toolbox_pin"] == common.TOOLBOX_PIN
        assert manifest["clip"]["sha256"] == clip["sha256"]
        assert manifest["trace"]["sam_separations_real"] == 1
        assert all("<" in cmd and ">" in cmd for cmd in manifest["repro"]["commands"])
        assert not common.find_private(json.dumps(manifest, ensure_ascii=False))
        # latest 指针策略（冻结）：outputs/e2e/LATEST.txt = 最近 run id
        assert (ws / "outputs" / "e2e" / "LATEST.txt").read_text(encoding="utf-8").strip() == "e2e-run-1"

    def test_finish_marks_blocked_runs(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        clip = make_clip_fixture(ws)
        run_e2e.finish(ws, "e2e-run-2", clip, None,
                       [{"stage": "model_run", "status": "blocked", "detail": "timeout"}],
                       None, time.time(), {}, {}, "cd" * 32, 4, (lambda s: s),
                       as_json=True, blocked=True)
        manifest = json.loads((ws / "outputs" / "e2e" / "e2e-run-2" / "run-manifest.json")
                              .read_text(encoding="utf-8"))
        assert manifest["runner_exit"] == 4
        assert manifest["validation"]["ok"] is None      # 不写成 success


class TestSpotcheck:
    def test_overlay_and_stats_on_mock_fixture(self, fixture_audio, tmp_path):
        gt = fixture_audio["gt"]
        clip = {"name": gt["audio"]["filename"], "sha256": gt["audio"]["sha256"],
                "duration_s": gt["audio"]["duration_s"], "sample_rate": gt["audio"]["sample_rate"]}
        events = [{"id": f"inst-1-ev-{i+1}", "onset_seconds": t, "source": "dsp", "method": "energy-flux"}
                  for i, t in enumerate(gt["sources"][0]["onset_seconds"])]
        doc = build_result(clip, kind="mock", events=events)
        overlay = spotcheck.build_overlay(fixture_audio["wav"], doc,
                                          tmp_path / "overlay.wav")
        assert (tmp_path / "overlay.wav").is_file()
        assert overlay["onset_clicks"] == len(events)
        stats = spotcheck.stats(fixture_audio["wav"], doc, None, kind="mock", sample=4)
        assert stats["kind"] == "mock"
        assert stats["evidence_class"] == "dsp-spotcheck"
        assert len(stats["per_event"]) == 4
        assert any(e["energy_rise_seen"] for e in stats["per_event"])
        assert any("不是总体准确率" in item for item in stats["limitations"])
