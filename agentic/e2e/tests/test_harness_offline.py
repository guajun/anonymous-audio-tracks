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
import stem_map
from helpers import build_result, make_clip_fixture, make_stem, write_result


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
         "result": {"content": [{"type": "text", "text":
                                 '{"attached": true, "mime": "audio/wav", "bytes": 2822444, "sha256": "ab"}'}]}},
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
        assert trace["usage"]["cacheRead"] == 0
        assert "E2E-DONE" in trace["final_text_tail"]
        assert trace["last_assistant_stop"] == "stop"
        assert trace["provider_failures"] == []

    def test_real_separations_exclude_dry_run(self):
        trace = common.extract_trace(_events())
        real = common.count_real_separations(trace)
        assert len(real) == 1
        assert "--dry-run" not in real[0]["command"]

    def test_failed_separations_counted_separately(self):
        """失败的 SAM 调用不得计入成功（attempted/failed/success 分开）。"""
        trace = {"tool_calls": [
            {"tool": "bash", "args_summary": "python audio_toolbox.py sam separate --audio a.wav",
             "result_head": '{"action": "sam.separate"}', "isError": True},
            {"tool": "bash", "args_summary": "python audio_toolbox.py sam separate --audio a.wav",
             "result_head": '{"action": "sam.separate"}', "isError": False},
        ]}
        buckets = common.classify_separations(trace)
        assert len(buckets["real_success"]) == 1
        assert len(buckets["real_failed"]) == 1
        assert len(common.count_real_separations(trace)) == 1

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
        # 路径字面量动态拼接：tracked 文件不出现真实个人路径布局（toolbox hygiene 契约）
        ws_root = "C" + ":" + "\\work\\ws"
        home_file = "C" + ":" + "\\" + "Users" + "\\someone\\private\\x.wav"
        redact = common.make_redactor(Path(ws_root))
        out = redact(f"{ws_root}\\audio\\inputs\\a.wav and {home_file}")
        assert ws_root not in out
        assert "<PATH>" in out
        assert "Users" not in out
        assert common.find_private("api_key = AIzaSyFakeFakeFakeFakeFake123456")
        assert common.find_private("x" * 300)           # 长 base64 形态
        assert common.find_private(f"Loaded {home_file}")   # 自由文本绝对路径（R1）
        assert not common.find_private("onset_seconds: 1.5，source=dsp")
        assert not common.find_private("see https://github.com/guajun/x and stems/drums/target.wav")

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
        assert manifest["schema"] == "agentic-e2e-run-manifest/v2"
        assert manifest["kind"] == "real"
        assert manifest["audio_pathway"]["kind"] == "bridge"
        assert "unsupported" in manifest["audio_pathway"]["native"]
        assert manifest["models"]["music_agent"] == "google/gemini-3.8-flash"
        assert manifest["models"]["impl_worker"] == "openrouter/xiaomi/mimo-v2.6-pro"
        assert manifest["tools"]["sam_toolbox_pin"] == common.TOOLBOX_PIN
        assert manifest["clip"]["sha256"] == clip["sha256"]
        assert manifest["trace"]["sam_separations_real_success"] == 1
        assert manifest["trace"]["sam_separations_real_failed"] == 0
        assert manifest["trace"]["sam_separations_dry_run"] == 1
        # 代码可归属性（R1）：harness 组件 hash + git revision（run 时未知则保持 unknown）
        assert manifest["code"]["harness_files_sha256"]
        assert "unknown" in str(manifest["code"]["git_revision_at_run"])
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
        assert "isolation_ratio" not in json.dumps(stats)   # 旧“隔离度”口径已废弃


class TestStemMap:
    """唯一 stem 文件名映射（R1：只做加法，不改原 Agent 产物/不改 schema）。"""

    def test_unique_names_hashes_and_originals_untouched(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stem = make_stem(run_dir, rel="stems/drums/target.wav")
        write_result(run_dir, build_result(make_clip_fixture(tmp_path), stem=stem))
        original_bytes = (run_dir / "stems" / "drums" / "target.wav").read_bytes()
        result_before = (run_dir / "result.json").read_bytes()
        mapping = stem_map.build_stem_map(run_dir)
        assert mapping["ok"], mapping["problems"]
        entry = mapping["entries"][0]
        assert entry["unique_rel"] == "stems-unique/stem-inst-1-target.wav"
        assert entry["sha256"] == stem["sha256"]
        assert (run_dir / entry["unique_rel"]).is_file()
        assert (run_dir / entry["original_rel"]).read_bytes() == original_bytes   # 原件未动
        assert (run_dir / "result.json").read_bytes() == result_before            # result 未动
        # 幂等：重复执行不冲突、不重复改变
        again = stem_map.build_stem_map(run_dir)
        assert again["entries"] == mapping["entries"]

    def test_hash_mismatch_is_reported_not_hidden(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stem = make_stem(run_dir, rel="stems/drums/target.wav")
        write_result(run_dir, build_result(make_clip_fixture(tmp_path),
                                           stem=dict(stem, sha256="0" * 64)))
        mapping = stem_map.build_stem_map(run_dir)
        assert not mapping["ok"]
        assert "hash 不符" in mapping["problems"][0]
        assert not (run_dir / "stems-unique").exists()      # 不产生伪造副本


class TestExecutionGate:
    """真实 pass 必须由观测执行证据门控（R1）：文档自述不算数。"""

    def _env(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        clip = make_clip_fixture(ws)
        run_dir = ws / "outputs" / "e2e" / "run-1"
        run_dir.mkdir(parents=True)
        local_dir = ws / "local" / "e2e" / "run-1"
        local_dir.mkdir(parents=True)
        (local_dir / "events.jsonl").write_text("{}", encoding="utf-8")
        return ws, clip, run_dir, local_dir

    def test_pass_with_observed_evidence(self, tmp_path):
        ws, clip, run_dir, local_dir = self._env(tmp_path)
        (run_dir / "stems" / "drums").mkdir(parents=True)
        (run_dir / "stems" / "drums" / "report.json").write_text("{}", encoding="utf-8")
        trace = common.extract_trace(_events())
        trace["runner_exit"] = 0
        gate = run_e2e.execution_gate(
            trace, {"sam_separations_max": 3}, run_dir, local_dir,
            [{"stage": "model_run", "status": "ok"}, {"stage": "result", "status": "ok"},
             {"stage": "validate", "status": "ok"}])
        assert gate["ok"], json.dumps(gate, ensure_ascii=False)
        assert gate["kind"] == "observed-execution"

    def test_missing_model_evidence_fails(self, tmp_path):
        ws, clip, run_dir, local_dir = self._env(tmp_path)
        trace = common.extract_trace(_events())
        trace["models"] = ["some-other-model"]
        gate = run_e2e.execution_gate(trace, {"sam_separations_max": 3}, run_dir, local_dir, [])
        assert not gate["ok"]
        assert not next(c for c in gate["checks"] if c["name"] == "observed_model")["pass"]

    def test_failed_sam_and_missing_artifacts_fail(self, tmp_path):
        ws, clip, run_dir, local_dir = self._env(tmp_path)
        trace = {"tool_calls": [
            {"tool": "bash", "args_summary": "python audio_toolbox.py sam separate --audio a.wav",
             "result_head": '{"action": "sam.separate"}', "isError": True}],
            "tool_calls_total": 1, "models": ["gemini-3.8-flash"], "runner_exit": 0}
        gate = run_e2e.execution_gate(trace, {"sam_separations_max": 3}, run_dir, local_dir,
                                      [{"stage": "model_run", "status": "ok"},
                                       {"stage": "result", "status": "ok"},
                                       {"stage": "validate", "status": "ok"}])
        assert not gate["ok"]
        assert not next(c for c in gate["checks"] if c["name"] == "sam_separation_observed")["pass"]
        assert not next(c for c in gate["checks"] if c["name"] == "sam_artifacts_present")["pass"]

    def test_budget_violation_fails(self, tmp_path):
        ws, clip, run_dir, local_dir = self._env(tmp_path)
        (run_dir / "stems" / "a").mkdir(parents=True)
        (run_dir / "stems" / "a" / "report.json").write_text("{}", encoding="utf-8")
        trace = common.extract_trace(_events())
        trace["runner_exit"] = 0
        gate = run_e2e.execution_gate(trace, {"sam_separations_max": 0}, run_dir, local_dir,
                                      [{"stage": "model_run", "status": "ok"},
                                       {"stage": "result", "status": "ok"},
                                       {"stage": "validate", "status": "ok"}])
        assert not gate["ok"]
        assert not next(c for c in gate["checks"] if c["name"] == "budget_compliance")["pass"]

    def test_verify_cannot_laundry_failed_runner(self, tmp_path):
        ws, clip, run_dir, local_dir = self._env(tmp_path)
        (run_dir / "stems" / "a").mkdir(parents=True)
        (run_dir / "stems" / "a" / "report.json").write_text("{}", encoding="utf-8")
        trace = common.extract_trace(_events())
        trace["runner_exit"] = 5                          # 原 run 是 provider-error 失败
        gate = run_e2e.execution_gate(
            trace, {"sam_separations_max": 3}, run_dir, local_dir,
            [{"stage": "model_run", "status": "ok"}, {"stage": "result", "status": "ok"},
             {"stage": "validate", "status": "ok"}],
            prior={"runner_exit": 5})
        assert not gate["ok"]
        assert not next(c for c in gate["checks"] if c["name"] == "original_runner_success")["pass"]

    def test_verify_preserves_run_timestamp_and_appends_verification(self, tmp_path):
        ws, clip, run_dir, local_dir = self._env(tmp_path)
        (run_dir / "stems" / "a").mkdir(parents=True)
        (run_dir / "stems" / "a" / "report.json").write_text("{}", encoding="utf-8")
        write_result(run_dir, build_result(clip))
        # 与真实流程一致：事件流落盘，verify 从事件流/trace-summary 复原 trace
        (local_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in _events()), encoding="utf-8")
        trace = common.extract_trace(_events())
        trace["runner_exit"] = 0
        run_e2e.finish(ws, "run-1", clip, trace,
                       [{"stage": "model_run", "status": "ok"}, {"stage": "result", "status": "ok"},
                        {"stage": "validate", "status": "ok"}],
                       {"ok": True, "checks": []}, time.time(), {"sam_separations_max": 3}, {},
                       "ab" * 32, 0, (lambda s: s), as_json=True, blocked=False)
        before = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
        assert before["verification_runs"] == []
        rc = run_e2e.verify_run(ws, "run-1", clip, (lambda s: s), as_json=False)
        assert rc == 0
        after = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
        assert after["created_at"] == before["created_at"]         # 原 run 时间戳不变
        assert after["timings"] == before["timings"]               # 原耗时记录不变
        assert len(after["verification_runs"]) == 1
        assert after["verification_runs"][0]["kind"] == "offline-reverify"
        assert after["verification_runs"][0]["at"] >= before["created_at"]
