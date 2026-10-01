"""tests/test_residual_flow.py — 残差逐层剥离离线回归（issue #42，0 API 0 GPU）。

覆盖：prompt 强制逐层 residual（禁止重复原混音）/ 复用 drums、链规则（每级 input==前级 raw
residual）、**错误原混音/监听代理/trace 偷换输入拒绝**（不能误报 sequential）、时间轴（每 stem
自身采样率、等长、禁 trim/shift）、停止/预算规则、监听副本（raw/proxy hash、转换参数、等长、
run 派生目录）、基线复用完整性（hash/事件/用户反馈）与**基线不可变**、路径/隐私、
run_e2e 集成（残差流 gate 只加不减、`--profile residual --dry-run`、verify CLI）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import common
import residual_chain
import run_e2e
from common import sha256_file, wav_info, write_wav
from helpers import make_clip_fixture

DURATION_S = 4.0
SR = 8000
RUN_ID = "e2e-rflow-1"
BASE_ID = "e2e-base-1"


# --------------------------------------------------------------- 构造工具


def _samples(n: int, seed: int = 1) -> list[float]:
    return [0.3 * ((i // 40) % 2) * (1.0 if (i // 200) % 2 == 0 else 0.5) + 0.001 * seed
            for i in range(n)]


def _write_wav32(path: Path, seconds: float = DURATION_S, sr: int = SR, seed: int = 1) -> Path:
    write_wav(path, _samples(int(seconds * sr), seed), sr, channels=1, fmt="float32")
    return path


def _events_doc(inst_id: str, count: int = 3) -> list[dict]:
    return [{"id": f"{inst_id}-ev-{i+1}", "onset_seconds": round(0.5 * i, 3),
             "source": "dsp", "method": "energy-flux"} for i in range(count)]


def _result_doc(clip: dict, instruments: list[dict]) -> dict:
    return {
        "schema_version": "agentic-audio-tracks/v1",
        "audio": {"filename": clip["name"], "sha256": clip["sha256"],
                  "duration_seconds": clip["duration_s"], "sample_rate": clip["sample_rate"]},
        "tempo": {"bpm": None, "source": "unknown", "confidence": 0},
        "instruments": instruments,
        "provenance": {"steps": [
            {"tool": "audio-bridge audio_attach", "source": "bridge", "note": "bridge 注入"},
            {"tool": "gemini-3.8-flash listening", "source": "llm", "note": "假设"},
            {"tool": "audio-toolbox.sam pin dfbc40a9541f (sam c603de8794cc)", "source": "sam", "note": "分离"},
            {"tool": "dsp energy-flux", "source": "dsp", "note": "onset"},
        ]},
        "limitations": ["测试文档（mock），不代表精度"],
    }


class Env:
    """残差剥离场景：基线 run（drums）+ 新 run（复用 drums + 2 级剥离 = 预算 2/2 后停止）。"""

    def __init__(self, tmp_path: Path):
        self.ws = tmp_path / "ws"
        self.ws.mkdir()
        self.clip = make_clip_fixture(self.ws, duration_s=DURATION_S, sample_rate=SR)
        self.clip_path = self.ws / "audio" / "inputs" / self.clip["name"]

        # ---- 基线 run：stems/drums/{target,residual}.wav（float32 @SR，等长）----
        self.base_dir = self.ws / "outputs" / "e2e" / BASE_ID
        bdir = self.base_dir / "stems" / "drums"
        bdir.mkdir(parents=True)
        self.base_target = _write_wav32(bdir / "target.wav", seed=1)
        self.base_residual = _write_wav32(bdir / "residual.wav", seed=2)
        (bdir / "request.json").write_text(json.dumps(
            {"audio": str(self.clip_path), "description": "drums", "duration_s": DURATION_S}), encoding="utf-8")
        (bdir / "report.json").write_text(json.dumps(
            {"description": "drums", "outputs": [str(self.base_target), str(self.base_residual)],
             "sample_rate": SR}), encoding="utf-8")
        self.drums_events = _events_doc("inst-drums", 3)
        self.base_result = _result_doc(self.clip, [{
            "id": "inst-drums", "label": "drums", "description": "drums 假设", "source": "sam",
            "confidence": 0.8, "events": self.drums_events,
            "stem": {"filename": "stems/drums/target.wav", "sha256": sha256_file(self.base_target)}}])
        (self.base_dir / "result.json").write_text(
            json.dumps(self.base_result, ensure_ascii=False), encoding="utf-8")

        # ---- 新 run：复用 drums 逐字节副本 + 2 级真实分离（layer2 / layer3）----
        self.run_dir = self.ws / "outputs" / "e2e" / RUN_ID
        rdir = self.run_dir / "stems" / "drums-reused"
        rdir.mkdir(parents=True)
        self.copy_target = rdir / "target.wav"
        self.copy_residual = rdir / "residual.wav"
        self.copy_target.write_bytes(self.base_target.read_bytes())
        self.copy_residual.write_bytes(self.base_residual.read_bytes())

        self.layer_dirs = {}
        self.sam_inputs = {}
        for name in ("layer2", "layer3"):
            d = self.run_dir / "stems" / name
            d.mkdir(parents=True)
            self.layer_dirs[name] = d
        self.l2_target = _write_wav32(self.layer_dirs["layer2"] / "target.wav", seed=3)
        self.l2_residual = _write_wav32(self.layer_dirs["layer2"] / "residual.wav", seed=4)
        self.l3_target = _write_wav32(self.layer_dirs["layer3"] / "target.wav", seed=5)
        self.l3_residual = _write_wav32(self.layer_dirs["layer3"] / "residual.wav", seed=6)
        # SAM 输入链：layer2 ← R1（基线 drums residual）；layer3 ← R2（layer2 residual）
        self.sam_inputs["layer2"] = self.base_residual
        self.sam_inputs["layer3"] = self.l2_residual
        for name, desc in (("layer2", "low layer"), ("layer3", "pad layer")):
            (self.layer_dirs[name] / "request.json").write_text(json.dumps(
                {"audio": str(self.sam_inputs[name]), "description": desc,
                 "duration_s": DURATION_S}), encoding="utf-8")
            (self.layer_dirs[name] / "report.json").write_text(json.dumps(
                {"description": desc,
                 "outputs": [str(self.layer_dirs[name] / "target.wav"),
                             str(self.layer_dirs[name] / "residual.wav")],
                 "sample_rate": SR}), encoding="utf-8")

        # ---- 监听副本（R1/R2/R3，PCM16，run 派生目录）----
        self.listen_dir = f"{RUN_ID}-listen"
        self.proxies = {}
        self.raw_of = {"r1": self.base_residual, "r2": self.l2_residual, "r3": self.l3_residual}
        self.raw_rel_of = {"r1": ("baseline", "stems/drums/residual.wav"),
                           "r2": ("run", "stems/layer2/residual.wav"),
                           "r3": ("run", "stems/layer3/residual.wav")}
        for key in ("r1", "r2", "r3"):
            out = self.ws / "audio" / "inputs" / self.listen_dir / f"{key}-listen.wav"
            raw_root, raw_rel = self.raw_rel_of[key]
            self.proxies[key] = residual_chain.make_listen_proxy(
                self.raw_of[key], out, self.ws / "audio" / "inputs", RUN_ID,
                codec="pcm16", raw_root=raw_root, raw_rel=raw_rel)

        self.chain = self._default_chain()
        self.trace = self._default_trace()

    # -- sidecar ----------------------------------------------------------
    def _file_entry(self, root: str, rel: str, path: Path) -> dict:
        return {"root": root, "rel": rel, "sha256": sha256_file(path),
                "sample_rate": SR, "duration_s": DURATION_S}

    def _default_chain(self) -> dict:
        facts = residual_chain.drums_baseline_facts(self.base_dir)
        clip_sha = self.clip["sha256"]
        return {
            "schema": residual_chain.CHAIN_SCHEMA,
            "run_id": RUN_ID,
            "clip": {"filename": self.clip["name"], "sha256": clip_sha,
                     "duration_s": self.clip["duration_s"], "sample_rate": self.clip["sample_rate"],
                     "timeline": "t=0 = clip 开头；同一零点/秒轴"},
            "stages": [
                {
                    "index": 0, "layer": "drums", "description": "drums（复用基线已认可步骤）",
                    "sam": {"performed": False, "reused": True, "description": "drums"},
                    "input": {"kind": "mix", "root": "audio", "rel": self.clip["name"],
                              "sha256": clip_sha},
                    "target": self._file_entry("run", "stems/drums-reused/target.wav", self.copy_target),
                    "residual": self._file_entry("run", "stems/drums-reused/residual.wav", self.copy_residual),
                    "request": {"root": "baseline", "rel": "stems/drums/request.json",
                                "sha256": sha256_file(self.base_dir / "stems" / "drums" / "request.json")},
                    "reused": {
                        "baseline_run": BASE_ID,
                        "baseline_result_sha256": sha256_file(self.base_dir / "result.json"),
                        "baseline_target": {"root": "baseline", "rel": "stems/drums/target.wav",
                                            "sha256": sha256_file(self.base_target)},
                        "baseline_residual": {"root": "baseline", "rel": "stems/drums/residual.wav",
                                              "sha256": sha256_file(self.base_residual)},
                        "events_instrument": "inst-drums",
                        "events_count": facts["events_count"],
                        "events_sha256": facts["events_sha256"],
                        "user_feedback": "issue #42 局部定性：drums 是准的（不推导 target 干净/residual 真值）",
                    },
                    "listen": {"method": "audio_attach", "path": f"{self.listen_dir}/r1-listen.wav",
                               "proxy": self.proxies["r1"], "note": "听 R1 → 选下一层"},
                    "next_choice": "听 R1：剩下低音线条与铺底；下一层描述 low layer",
                },
                {
                    "index": 1, "layer": "low layer", "description": "R1 里的低音线条",
                    "sam": {"performed": True, "reused": False, "description": "low layer"},
                    "input": {"kind": "residual", "root": "baseline", "rel": "stems/drums/residual.wav",
                              "sha256": sha256_file(self.base_residual)},
                    "target": self._file_entry("run", "stems/layer2/target.wav", self.l2_target),
                    "residual": self._file_entry("run", "stems/layer2/residual.wav", self.l2_residual),
                    "request": {"root": "run", "rel": "stems/layer2/request.json",
                                "sha256": sha256_file(self.layer_dirs["layer2"] / "request.json")},
                    "listen": {"method": "audio_attach", "path": f"{self.listen_dir}/r2-listen.wav",
                               "proxy": self.proxies["r2"], "note": "听 R2 → 再剥"},
                    "next_choice": "听 R2：剩铺底长音；下一层描述 pad layer",
                },
                {
                    "index": 2, "layer": "pad layer", "description": "R2 里的铺底长音",
                    "sam": {"performed": True, "reused": False, "description": "pad layer"},
                    "input": {"kind": "residual", "root": "run", "rel": "stems/layer2/residual.wav",
                              "sha256": sha256_file(self.l2_residual)},
                    "target": self._file_entry("run", "stems/layer3/target.wav", self.l3_target),
                    "residual": self._file_entry("run", "stems/layer3/residual.wav", self.l3_residual),
                    "request": {"root": "run", "rel": "stems/layer3/request.json",
                                "sha256": sha256_file(self.layer_dirs["layer3"] / "request.json")},
                    "listen": {"method": "audio_attach", "path": f"{self.listen_dir}/r3-listen.wav",
                               "proxy": self.proxies["r3"], "note": "听 R3 → 停"},
                },
            ],
            "stop": {
                "reason": "budget",
                "evidence": "R3 听感仍有残余，但真实 SAM 预算 2/2 已用尽",
                "residual_left": self._file_entry("run", "stems/layer3/residual.wav", self.l3_residual),
                "limitations": "残余未剥离（未知声层）；residual 是模型估计，误差会累积",
            },
            "limitations": [
                "residual/target 是模型估计，逐层误差会累积，不是真值，不承诺更准",
                "drums 复用基于用户局部定性反馈",
            ],
        }

    def _events(self, *, late_listen: bool = False, batch_same_turn: bool = False) -> list[dict]:
        """构造事件流：默认每工具调用前一个模型轮次（turn_start）。

        * ``late_listen``：R1 挂载晚于 layer2 分离（late-listen 反例）；
        * ``batch_same_turn``：R1 挂载与 layer2 分离同轮并行（无中间模型轮次反例）。
        """
        attach = json.dumps({"attached": True, "mime": "audio/wav", "bytes": 1000,
                             "sha256": "ab", "queueDepth": 1, "queuedBytes": 1000})
        events: list[dict] = []
        counter = {"n": 0}

        def add_attach(path: str) -> None:
            counter["n"] += 1
            cid = f"a{counter['n']}"
            events.append({"type": "tool_execution_start", "toolCallId": cid,
                           "toolName": "audio_attach", "args": {"path": path}})
            events.append({"type": "tool_execution_end", "toolCallId": cid,
                           "toolName": "audio_attach", "isError": False,
                           "result": {"content": [{"type": "text", "text": attach}]}})

        def add_sam(name: str, dry: bool = False) -> None:
            counter["n"] += 1
            cid = f"s{counter['n']}"
            wrapper = json.dumps({
                "schema": "audio-toolbox.sam/v1", "tool": "audio-toolbox", "ok": True,
                "action": "sam.dry-run" if dry else "sam.separate", "exit_code": 0,
                "run_dir": f"outputs/e2e/{RUN_ID}/stems/{name}",
                "outputs": [] if dry else [f"stems/{name}/target.wav", f"stems/{name}/residual.wav"],
                "report": {"description": name}})
            cmd = (f'python audio_toolbox.py sam separate --audio "{self.sam_inputs[name]}" '
                   f'--description "{name}" --output-dir outputs/e2e/{RUN_ID}/stems/{name} '
                   f'--timeout 900' + (" --dry-run" if dry else ""))
            events.append({"type": "tool_execution_start", "toolCallId": cid, "toolName": "bash",
                           "args": {"command": cmd}})
            events.append({"type": "tool_execution_end", "toolCallId": cid, "toolName": "bash",
                           "isError": False,
                           "result": {"content": [{"type": "text", "text": wrapper}]}})

        def turn() -> None:
            events.append({"type": "turn_start"})

        r1 = f"{self.listen_dir}/r1-listen.wav"
        r2 = f"{self.listen_dir}/r2-listen.wav"
        r3 = f"{self.listen_dir}/r3-listen.wav"
        turn()
        add_attach(self.clip["name"])
        if batch_same_turn:
            turn()
            add_attach(r1)                 # 同轮内并行：无中间模型轮次
            add_sam("layer2", dry=True)
            add_sam("layer2", dry=False)
        elif late_listen:
            turn()
            add_sam("layer2", dry=True)
            add_sam("layer2", dry=False)   # 先分离
            turn()
            add_attach(r1)                 # 后听（late listen）
        else:
            turn()
            add_attach(r1)
            turn()
            add_sam("layer2", dry=True)
            turn()
            add_sam("layer2", dry=False)
        turn()
        add_attach(r2)
        turn()
        add_sam("layer3", dry=True)
        turn()
        add_sam("layer3", dry=False)
        turn()
        add_attach(r3)
        events.append(
            {"type": "message_end", "message": {"role": "assistant", "model": "gemini-3.8-flash",
                                                "provider": "google", "stopReason": "stop",
                                                "usage": {"input": 10, "output": 5, "totalTokens": 15},
                                                "content": [{"type": "text", "text": "E2E-DONE ok"}]}})
        return events

    def _default_trace(self) -> dict:
        return common.extract_trace(self._events()) | {"runner_exit": 0}

    # -- 核对 --------------------------------------------------------------
    def verify(self, *, doc="chain", **kw) -> dict:
        """doc="chain" 用默认 sidecar；doc=None 模拟 sidecar 缺失。"""
        if doc == "chain":
            self.write_chain()
            doc = self.chain
        roots = {"audio": self.ws / "audio" / "inputs", "run": self.run_dir,
                 "baseline": self.base_dir}
        result_path = kw.pop("result_path", None)
        if result_path is None:
            result_path = self.write_result()
        return residual_chain.verify_chain(
            doc, run_dir=self.run_dir, clip=self.clip, roots=roots, run_id=RUN_ID,
            trace=kw.pop("trace", self.trace), ws=self.ws, budget=kw.pop("budget", 2),
            result_path=result_path, **kw)

    def write_chain(self) -> Path:
        path = self.run_dir / "stage-chain.json"
        path.write_text(json.dumps(self.chain, ensure_ascii=False), encoding="utf-8")
        return path

    def write_result(self) -> Path:
        doc = _result_doc(self.clip, [
            {"id": "inst-drums", "label": "drums", "description": "复用基线", "source": "sam",
             "confidence": 0.8, "events": self.drums_events,
             "stem": {"filename": "stems/drums-reused/target.wav",
                      "sha256": sha256_file(self.copy_target)}},
            {"id": "inst-low", "label": "low layer", "description": "R1 听感", "source": "sam",
             "confidence": 0.5, "events": _events_doc("inst-low", 2),
             "stem": {"filename": "stems/layer2/target.wav", "sha256": sha256_file(self.l2_target)}},
            {"id": "inst-pad", "label": "pad layer", "description": "R2 听感", "source": "sam",
             "confidence": 0.4, "events": _events_doc("inst-pad", 1),
             "stem": {"filename": "stems/layer3/target.wav", "sha256": sha256_file(self.l3_target)}},
        ])
        path = self.run_dir / "result.json"
        path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        return path

    def baseline_digest(self) -> dict:
        out = {}
        for p in sorted(self.base_dir.rglob("*")):
            if p.is_file():
                out[str(p.relative_to(self.base_dir)).replace("\\", "/")] = sha256_file(p)
        return out


def failed_names(report: dict) -> list[str]:
    return [c["name"] for c in report["checks"] if not c["pass"]]


def make_other_clip(env: "Env") -> Path:
    """同长度（4.0 s）但不同内容的音频（“同长不同源”反例用）。"""
    other = env.ws / "audio" / "inputs" / "other-clip.wav"
    write_wav(other, [0.5 * s for s in _samples(int(DURATION_S * SR), seed=9)], SR, channels=1, fmt="pcm16")
    return other


@pytest.fixture()
def env(tmp_path):
    return Env(tmp_path)


# --------------------------------------------------------------- prompt


class TestResidualPrompt:
    def test_prompt_forces_residual_workflow(self, env):
        prompt = run_e2e.build_residual_prompt(RUN_ID, env.clip, env.ws, BASE_ID, 2, 900, 2)
        assert "残差逐层剥离" in prompt and "residual-first sequential peeling" in prompt
        assert "--audio` 必须指向 R1 的 raw residual" in prompt
        assert "绝不允许" in prompt and env.clip["name"] in prompt       # 禁止偷换原混音
        assert "监听代理" in prompt                                       # 禁止 proxy 当 SAM 输入
        assert "不强制 bass/synth" in prompt and "不强制填满层数" in prompt
        for reason in residual_chain.STOP_REASONS:
            assert reason in prompt
        assert "≤ 2 次" in prompt and "--timeout 900" in prompt          # 预算明确
        assert residual_chain.CHAIN_SCHEMA in prompt                      # sidecar schema
        assert "stage-chain.json" in prompt
        assert "局部定性" in prompt                                       # drum 反馈口径
        assert "误差会累积" in prompt and "不承诺" in prompt
        assert "E2E-DONE" in prompt and "chain=outputs/e2e/" in prompt
        # 基线复用事实已填入（真实 hash）
        assert sha256_file(env.base_target) in prompt
        assert sha256_file(env.base_residual) in prompt
        assert residual_chain.canonical_events_sha256(env.drums_events) in prompt
        assert BASE_ID in prompt
        for token in ("{run_id}", "{clip_name}", "{validate_py}", "{max_separations}",
                      "{drums_target_sha}", "{baseline_run_dir}"):
            assert token not in prompt

    def test_prompt_reuses_baseline_and_listening_copy_rules(self, env):
        prompt = run_e2e.build_residual_prompt(RUN_ID, env.clip, env.ws, BASE_ID, 2, 900, 2)
        assert f"{RUN_ID}-listen" in prompt                   # 本 run 派生监听目录
        assert "make-proxy" in prompt                         # raw/proxy hash + 转换参数 + 等长
        assert "逐字节复制" in prompt and "stems/drums-reused" in prompt
        assert "不得改动" in prompt                            # 基线不可变

    def test_prompt_rejects_bad_baseline(self, env):
        with pytest.raises(ValueError):
            run_e2e.build_residual_prompt(RUN_ID, env.clip, env.ws, RUN_ID, 2, 900, 2)
        with pytest.raises(ValueError):
            run_e2e.build_residual_prompt(RUN_ID, env.clip, env.ws, "no-such-run", 2, 900, 2)
        with pytest.raises(ValueError):
            run_e2e.build_residual_prompt("../evil", env.clip, env.ws, BASE_ID, 2, 900, 2)

    def test_same_source_baseline_accepted(self, env):
        """同源基线照常接受（评审口径：当前真实基线也由既有 run 复核覆盖）。"""
        assert residual_chain.check_baseline_clip(env.base_dir, env.clip) is None
        assert run_e2e.build_residual_prompt(RUN_ID, env.clip, env.ws, BASE_ID, 2, 900, 2)


# --------------------------------------------------------------- 链规则


class TestChainRules:
    def test_valid_chain_passes(self, env):
        report = env.verify()
        assert report["ok"], json.dumps(failed_names(report))
        names = [c["name"] for c in report["checks"]]
        for expect in ("sidecar_wellformed", "chain_input_prev_residual", "no_mix_swap",
                       "request_proves_input", "trace_input_chain", "residual_listen_observed",
                       "listen_before_next_separation", "stop_policy",
                       "residual_left_is_final_residual", "sam_budget", "reused_baseline_integrity",
                       "baseline_same_clip", "reused_events_carried_into_result", "sidecar_privacy"):
            assert expect in names

    def test_wrong_original_mix_rejected(self, env):
        """中间级偷换原混音（= 旧 baseline 的独立分离）→ 链断 + no_mix_swap，不能误报 sequential。"""
        env.chain["stages"][1]["input"] = {"kind": "mix", "root": "audio", "rel": env.clip["name"],
                                           "sha256": env.clip["sha256"]}
        report = env.verify()
        assert not report["ok"]
        assert "chain_input_prev_residual" in failed_names(report)
        assert "no_mix_swap" in failed_names(report)

    def test_independent_three_way_separation_not_sequential(self, env):
        """全部从原混音分离（旧 baseline 形态）绝不能通过链校验。"""
        for stage in env.chain["stages"]:
            stage["input"] = {"kind": "mix", "root": "audio", "rel": env.clip["name"],
                              "sha256": env.clip["sha256"]}
        report = env.verify()
        assert not report["ok"]
        assert "chain_input_prev_residual" in failed_names(report)
        assert "no_mix_swap" in failed_names(report)

    def test_listen_proxy_as_sam_input_rejected(self, env):
        """PCM16 监听代理当 SAM 输入 → hash != 前级 raw residual → 链断。"""
        env.chain["stages"][2]["input"] = {
            "kind": "residual", "root": "audio", "rel": f"{env.listen_dir}/r2-listen.wav",
            "sha256": sha256_file(env.ws / "audio" / "inputs" / env.listen_dir / "r2-listen.wav")}
        report = env.verify()
        assert not report["ok"]
        assert "chain_input_prev_residual" in failed_names(report)

    def test_request_swap_rejected(self, env):
        """sidecar 声明 input=R1，但真实 request.json 的 audio 是原混音 → 偷换被识破。"""
        req = env.layer_dirs["layer2"] / "request.json"
        req.write_text(json.dumps({"audio": str(env.clip_path), "description": "low layer"}),
                       encoding="utf-8")
        env.chain["stages"][1]["request"]["sha256"] = sha256_file(req)
        report = env.verify()
        assert not report["ok"]
        assert "request_proves_input" in failed_names(report)

    def test_trace_swap_rejected(self, env):
        """trace 里 SAM --audio 指向原混音 → trace 链核对失败。"""
        env.sam_inputs["layer3"] = env.clip_path
        env.trace = env._default_trace()
        report = env.verify()
        assert not report["ok"]
        assert "trace_input_chain" in failed_names(report)

    def test_declared_hash_mismatch_rejected(self, env):
        env.chain["stages"][1]["target"]["sha256"] = "0" * 64
        report = env.verify()
        assert not report["ok"]
        assert "artifacts_exist_hash_match" in failed_names(report)

    def test_late_listen_rejected(self, env):
        """先分离后听 R1（late listen）→ 不能声称听后自主选择下一层。"""
        env.trace = common.extract_trace(env._events(late_listen=True)) | {"runner_exit": 0}
        report = env.verify()
        assert not report["ok"]
        assert "listen_before_next_separation" in failed_names(report)

    def test_same_turn_batching_rejected(self, env):
        """同轮并行（听与分之间无模型轮次）→ 不能声称依据新声音选择。"""
        env.trace = common.extract_trace(env._events(batch_same_turn=True)) | {"runner_exit": 0}
        report = env.verify()
        assert not report["ok"]
        assert "listen_before_next_separation" in failed_names(report)


# --------------------------------------------------------------- 时间轴


class TestTimeline:
    def test_per_stem_own_sample_rate_ok(self, env):
        """不同采样率（每 stem 自身 sr）但等长 → 合规；不得用 clip 时钟索引。"""
        path = env.layer_dirs["layer2"] / "target.wav"
        _write_wav32(path, seconds=DURATION_S, sr=12000)
        env.chain["stages"][1]["target"].update(sha256=sha256_file(path), sample_rate=12000)
        report = env.verify()
        assert report["ok"], json.dumps(failed_names(report))

    def test_trim_rejected(self, env):
        path = env.layer_dirs["layer2"] / "residual.wav"
        _write_wav32(path, seconds=DURATION_S - 1.0, sr=SR)     # 3s ≠ clip 4s（trim）
        env.chain["stages"][1]["residual"].update(sha256=sha256_file(path), duration_s=DURATION_S - 1.0)
        report = env.verify()
        assert not report["ok"]
        assert "artifacts_exist_hash_match" in failed_names(report)

    def test_wrong_sample_rate_record_rejected(self, env):
        env.chain["stages"][1]["target"]["sample_rate"] = 44100
        report = env.verify()
        assert not report["ok"]
        assert "artifacts_exist_hash_match" in failed_names(report)


# --------------------------------------------------------------- 停止/预算


class TestStopAndBudget:
    def test_missing_stop_rejected(self, env):
        env.chain["stop"] = {}
        report = env.verify()
        assert not report["ok"]
        assert "stop_policy" in failed_names(report)

    def test_bad_stop_reason_rejected(self, env):
        env.chain["stop"]["reason"] = "party"
        report = env.verify()
        assert not report["ok"]
        assert "stop_policy" in failed_names(report)

    def test_budget_stop_must_exhaust_budget_and_mention_residue(self, env):
        env.chain["stop"]["limitations"] = "done"
        env.chain["limitations"] = ["done"]
        report = env.verify()
        assert not report["ok"]     # 未提 residual/误差，且未写残余
        assert "stop_policy" in failed_names(report)

        env.chain["stop"]["limitations"] = "残余未剥离，residual 是模型估计"
        env.chain["limitations"] = ["residual 误差会累积"]
        report = env.verify(budget=1)      # performed=2 > budget=1
        assert not report["ok"]
        assert "sam_budget" in failed_names(report)

    def test_budget_stop_requires_exhausted_attempts(self, env):
        """声称“预算到达”但实际没用完预算（含失败尝试）→ 不合规。"""
        report = env.verify(budget=3)      # 只做了 2 级就声称预算用尽
        assert not report["ok"]
        assert "stop_policy" in failed_names(report)

    def test_no_identifiable_layer_stop_ok(self, env):
        env.chain["stop"] = {
            "reason": "no-identifiable-layer",
            "evidence": "R3 试听只剩模糊尾音，无可辨认声层",
            "residual_left": {"root": "run", "rel": "stems/layer3/residual.wav",
                              "sha256": sha256_file(env.l3_residual), "sample_rate": SR,
                              "duration_s": DURATION_S},
            "limitations": "残余为不可辨认成分；residual 是模型估计，误差会累积",
        }
        # 非 budget 停止时最后一级也必须有听音记录（默认已有 r3 listen）
        report = env.verify()
        assert report["ok"], json.dumps(failed_names(report))

    def test_last_listen_required_unless_budget(self, env):
        env.chain["stages"][2]["listen"] = {}
        env.chain["stop"]["reason"] = "uncertain"
        env.chain["stop"]["evidence"] = "R3 听感不确定"
        report = env.verify()
        assert not report["ok"]
        assert "residual_listen_observed" in failed_names(report)

    def test_residual_left_must_be_final_residual(self, env):
        """stop.residual_left 必须就是最后一级 residual（不得省略/不得写原混音或更早 residual）。"""
        env.chain["stop"].pop("residual_left")                       # 省略
        report = env.verify()
        assert not report["ok"]
        assert "residual_left_is_final_residual" in failed_names(report)

        env.chain["stop"]["residual_left"] = {                       # 写成原混音
            "root": "audio", "rel": env.clip["name"], "sha256": env.clip["sha256"],
            "sample_rate": SR, "duration_s": DURATION_S}
        report = env.verify()
        assert not report["ok"]
        assert "residual_left_is_final_residual" in failed_names(report)

        env.chain["stop"]["residual_left"] = {                       # 写成更早级 residual
            "root": "run", "rel": "stems/layer2/residual.wav",
            "sha256": sha256_file(env.l2_residual), "sample_rate": SR, "duration_s": DURATION_S}
        report = env.verify()
        assert not report["ok"]
        assert "residual_left_is_final_residual" in failed_names(report)


# --------------------------------------------------------------- 监听副本


class TestListenProxy:
    def test_make_proxy_record_and_codec(self, env):
        rec = env.proxies["r1"]
        assert rec["codec"] == "pcm16" and rec["convert"]
        assert rec["raw_sha256"] == sha256_file(env.base_residual)
        assert rec["proxy_sha256"] == sha256_file(env.ws / "audio" / "inputs" /
                                                  env.listen_dir / "r1-listen.wav")
        assert rec["frames"] == wav_info(env.base_residual)["frames"]       # 等长
        info = wav_info(env.ws / "audio" / "inputs" / env.listen_dir / "r1-listen.wav")
        assert (info["tag"], info["bits_per_sample"]) == (1, 16)            # PCM16 编码

    def test_make_proxy_copy_keeps_bytes(self, env):
        out = env.ws / "audio" / "inputs" / env.listen_dir / "r1-copy.wav"
        rec = residual_chain.make_listen_proxy(env.base_residual, out,
                                               env.ws / "audio" / "inputs", RUN_ID, codec="copy")
        assert rec["proxy_sha256"] == rec["raw_sha256"]
        assert "none" in rec["convert"]

    def test_make_proxy_rejects_outside_or_foreign_dir(self, env):
        with pytest.raises(ValueError):
            residual_chain.make_listen_proxy(
                env.base_residual, env.ws / "outputs" / "elsewhere" / "x.wav",
                env.ws / "audio" / "inputs", RUN_ID)
        with pytest.raises(ValueError):
            residual_chain.make_listen_proxy(
                env.base_residual, env.ws / "audio" / "inputs" / "other-run" / "x.wav",
                env.ws / "audio" / "inputs", RUN_ID)
        with pytest.raises(ValueError):
            residual_chain.make_listen_proxy(
                env.base_residual, env.ws / "audio" / "inputs" / env.listen_dir / "x.wav",
                env.ws / "audio" / "inputs", "../evil")

    def test_sibling_prefix_namespace_rejected(self, env):
        """精确命名空间：`run-1` 不得写入 `run-10-listen/`（startswith 反例，评审复现）。"""
        out = env.ws / "audio" / "inputs" / "run-10-listen" / "x.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"USER_DATA")
        with pytest.raises(ValueError):
            residual_chain.make_listen_proxy(env.base_residual, out,
                                             env.ws / "audio" / "inputs", "run-1")
        assert out.read_bytes() == b"USER_DATA"          # 兄弟 run 的文件未被动过
        ok, why = residual_chain.check_listen_rel("run-10-listen/x.wav", "run-1")
        assert not ok and "精确命名空间" in why

    def test_existing_file_never_overwritten(self, env):
        out = env.ws / "audio" / "inputs" / env.listen_dir / "r9-listen.wav"
        out.write_bytes(b"USER_DATA")
        with pytest.raises(ValueError):
            residual_chain.make_listen_proxy(env.base_residual, out,
                                             env.ws / "audio" / "inputs", RUN_ID)
        assert out.read_bytes() == b"USER_DATA"          # 既有听音/证据文件保留

    def test_idempotent_same_raw_out(self, env):
        out = env.ws / "audio" / "inputs" / env.listen_dir / "r7-listen.wav"
        rec1 = residual_chain.make_listen_proxy(env.base_residual, out,
                                                env.ws / "audio" / "inputs", RUN_ID)
        rec2 = residual_chain.make_listen_proxy(env.base_residual, out,
                                                env.ws / "audio" / "inputs", RUN_ID)
        assert rec1 == rec2                              # 逐字节相同 → 幂等放行

    def test_raw_equals_out_rejected(self, env):
        raw_in = env.ws / "audio" / "inputs" / env.listen_dir / "same.wav"
        raw_in.write_bytes(env.base_residual.read_bytes())
        with pytest.raises(ValueError):
            residual_chain.make_listen_proxy(raw_in, raw_in,
                                             env.ws / "audio" / "inputs", RUN_ID)

    def test_proxy_dir_rule(self, env):
        env.chain["stages"][0]["listen"]["path"] = "fixture-a.wav"      # 非 run 派生目录
        env.chain["stages"][0]["listen"]["proxy"]["proxy_rel"] = "fixture-a.wav"
        report = env.verify()
        assert not report["ok"]
        assert "residual_listen_observed" in failed_names(report)

    def test_proxy_length_rule(self, env):
        short_path = env.ws / "audio" / "inputs" / env.listen_dir / "r1-short.wav"
        write_wav(short_path, _samples(int((DURATION_S - 1.0) * SR)), SR, channels=1, fmt="pcm16")
        env.chain["stages"][0]["listen"]["proxy"].update(
            proxy_rel=f"{env.listen_dir}/r1-short.wav", proxy_sha256=sha256_file(short_path),
            duration_s=DURATION_S - 1.0, frames=wav_info(short_path)["frames"])
        report = env.verify()
        assert not report["ok"]                     # 等长/同 sr 规则（禁 trim/shift）
        assert "residual_listen_observed" in failed_names(report)

    def test_proxy_codec_mismatch_rejected(self, env):
        out = env.ws / "audio" / "inputs" / env.listen_dir / "r1-float.wav"
        out.write_bytes(env.base_residual.read_bytes())                 # 实际是 float32 副本
        env.chain["stages"][0]["listen"]["proxy"].update(
            proxy_rel=f"{env.listen_dir}/r1-float.wav", proxy_sha256=sha256_file(out),
            codec="pcm16")                                              # 声明 pcm16 但非 PCM16
        report = env.verify()
        assert not report["ok"]
        assert "residual_listen_observed" in failed_names(report)

    def test_attach_evidence_required(self, env):
        """sidecar 写了 listen 但 trace 没有真实挂载 → 不算自主残差听音。"""
        env.trace["tool_calls"] = [c for c in env.trace["tool_calls"]
                                   if not c["args_summary"].startswith(env.listen_dir)]
        report = env.verify()
        assert not report["ok"]
        assert "residual_listen_observed" in failed_names(report)

    def test_residual_attach_calls_only_counts_residual_copies(self, env):
        calls = residual_chain.residual_attach_calls(env.trace, env.clip["name"], RUN_ID)
        assert {c["path"] for c in calls} == {f"{env.listen_dir}/r1-listen.wav",
                                              f"{env.listen_dir}/r2-listen.wav",
                                              f"{env.listen_dir}/r3-listen.wav"}


# --------------------------------------------------------------- 基线复用/不可变


class TestBaselineReuse:
    def test_baseline_files_unchanged_after_verify(self, env):
        before = env.baseline_digest()
        report = env.verify()
        assert report["ok"]
        assert env.baseline_digest() == before          # 核对只读，基线不可变

    def test_reuse_hash_mismatch_rejected(self, env):
        env.chain["stages"][0]["reused"]["baseline_target"]["sha256"] = "0" * 64
        report = env.verify()
        assert not report["ok"]
        assert "reused_baseline_integrity" in failed_names(report)

    def test_reuse_copy_must_be_identical(self, env):
        env.chain["stages"][0]["target"]["sha256"] = "0" * 64      # 副本 ≠ 基线原件
        report = env.verify()
        assert not report["ok"]
        assert "reused_baseline_integrity" in failed_names(report)

    def test_reused_events_must_carry_into_result(self, env):
        result_path = env.write_result()
        doc = json.loads(result_path.read_text(encoding="utf-8"))
        doc["instruments"][0]["events"] = doc["instruments"][0]["events"][:2]   # 丢事件
        result_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        report = env.verify(result_path=result_path)
        assert not report["ok"]
        assert "reused_events_carried_into_result" in failed_names(report)

    def test_reused_events_must_match_baseline(self, env):
        env.chain["stages"][0]["reused"]["events_count"] = 36
        report = env.verify()
        assert not report["ok"]
        assert "reused_baseline_integrity" in failed_names(report)

    def test_user_feedback_recorded(self, env):
        env.chain["stages"][0]["reused"]["user_feedback"] = ""
        report = env.verify()
        assert not report["ok"]
        assert "reused_baseline_integrity" in failed_names(report)

    def test_baseline_wrong_clip_same_duration_rejected(self, env):
        """同长度但不同源的基线 → 写前拒 + 链核对拒（不得继承另一段音频的 drums）。"""
        other = make_other_clip(env)
        base_result_path = env.base_dir / "result.json"
        base_result = json.loads(base_result_path.read_text(encoding="utf-8"))
        base_result["audio"]["sha256"] = sha256_file(other)          # 同 duration，不同源
        base_result_path.write_text(json.dumps(base_result, ensure_ascii=False), encoding="utf-8")
        req = env.base_dir / "stems" / "drums" / "request.json"
        req.write_text(json.dumps({"audio": str(other), "description": "drums",
                                   "duration_s": DURATION_S}), encoding="utf-8")
        # 写前拒（prompt 构建 = 启动/复用/复制之前）
        with pytest.raises(ValueError) as exc:
            run_e2e.build_residual_prompt(RUN_ID, env.clip, env.ws, BASE_ID, 2, 900, 2)
        assert "E_BASELINE_CLIP" in str(exc.value)
        # 链核对也拒（sidecar 自述不算数）
        env.chain["stages"][0]["reused"]["baseline_result_sha256"] = sha256_file(base_result_path)
        env.chain["stages"][0]["request"]["sha256"] = sha256_file(req)
        report = env.verify()
        assert not report["ok"]
        assert "baseline_same_clip" in failed_names(report)


# --------------------------------------------------------------- 路径/隐私/结构


class TestPathsAndPrivacy:
    def test_unsafe_rel_rejected(self, env):
        env.chain["stages"][1]["target"]["rel"] = "../escape.wav"
        report = env.verify()
        assert not report["ok"]
        assert "artifacts_exist_hash_match" in failed_names(report)

    def test_sidecar_absolute_path_rejected(self, env):
        env.chain["stages"][1]["input"]["rel"] = "C" + ":" + "\\\\" + "private" + "\\\\" + "raw.wav"
        report = env.verify()
        assert not report["ok"]
        assert "sidecar_privacy" in failed_names(report)

    def test_missing_sidecar_fails(self, env):
        report = env.verify(doc=None, trace=env.trace)
        assert not report["ok"]
        assert "sidecar_present" in failed_names(report)

    def test_example_fixture_is_wellformed(self):
        example = json.loads((run_e2e.E2E_DIR / "fixtures" / "stage-chain.example.json")
                             .read_text(encoding="utf-8"))
        assert example["schema"] == residual_chain.CHAIN_SCHEMA
        assert [s["index"] for s in example["stages"]] == list(range(len(example["stages"])))
        assert all(isinstance(s["sam"]["performed"], bool) for s in example["stages"])
        assert example["stages"][1]["input"]["sha256"] == example["stages"][0]["residual"]["sha256"]


# --------------------------------------------------------------- run_e2e 集成


class TestRunE2EIntegration:
    OK_STAGES = [{"stage": "model_run", "status": "ok"}, {"stage": "result", "status": "ok"},
                 {"stage": "validate", "status": "ok"}, {"stage": "spotcheck", "status": "ok"}]

    def _mk(self, env):
        env.write_chain()
        env.write_result()
        local_dir = env.ws / "local" / "e2e" / RUN_ID
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "events.jsonl").write_text("{}", encoding="utf-8")
        return local_dir

    def test_residual_gate_checks_added_not_relaxed(self, env):
        local_dir = self._mk(env)
        chain = env.verify()
        params = {"sam_separations_max": 2, "flow": run_e2e.FLOW_RESIDUAL}
        stages = self.OK_STAGES + [{"stage": "residual_chain", "status": "ok"}]
        gate = run_e2e.execution_gate(env.trace, params, env.run_dir, local_dir, stages,
                                      chain=chain, clip_name=env.clip["name"])
        assert gate["ok"], json.dumps(gate, ensure_ascii=False)
        names = [c["name"] for c in gate["checks"]]
        assert len(names) == 12                       # 既有 9 项 + 残差流 3 项（只加不减）
        for extra in ("residual_chain_recorded", "residual_chain_verified",
                      "residual_listen_attach_observed"):
            assert extra in names

    def test_residual_gate_without_chain_or_listen_fails(self, env):
        local_dir = self._mk(env)
        params = {"sam_separations_max": 2, "flow": run_e2e.FLOW_RESIDUAL}
        stages = self.OK_STAGES + [{"stage": "residual_chain", "status": "fail"}]
        trace = dict(env.trace)
        trace["tool_calls"] = [c for c in env.trace["tool_calls"]
                               if not c["args_summary"].startswith(env.listen_dir)]
        gate = run_e2e.execution_gate(trace, params, env.run_dir, local_dir, stages,
                                      chain=None, clip_name=env.clip["name"])
        assert not gate["ok"]
        failed = [c["name"] for c in gate["checks"] if not c["pass"]]
        for name in ("residual_chain_recorded", "residual_chain_verified",
                     "residual_listen_attach_observed", "required_stages_present_nonfailed"):
            assert name in failed, name

    def test_baseline_gate_shape_unchanged(self, env):
        local_dir = self._mk(env)
        gate = run_e2e.execution_gate(env.trace, {"sam_separations_max": 2}, env.run_dir,
                                      local_dir, self.OK_STAGES)
        assert len(gate["checks"]) == 9               # 非残差流保持原 9 项

    def test_chain_verify_run_writes_sidecar_verify(self, env):
        env.write_chain()
        env.write_result()
        report = run_e2e.chain_verify_run(env.ws, RUN_ID, env.clip, BASE_ID, env.trace, 2,
                                          result_path=env.run_dir / "result.json")
        assert report["ok"], json.dumps(failed_names(report))
        saved = json.loads((env.run_dir / "stage-chain-verify.json").read_text(encoding="utf-8"))
        assert saved["schema"] == residual_chain.CHAIN_VERIFY_SCHEMA
        assert saved["ok"] is True

    def test_dry_run_profile_residual(self, env):
        try:
            import pi_launcher  # type: ignore
            pi_launcher.resolve_command(["pi", "--version"])
        except Exception:  # pragma: no cover
            pytest.skip("pi 启动器不可用")
        proc = subprocess.run(
            [sys.executable, str(run_e2e.HERE / "run_e2e.py"), "--workspace", str(env.ws),
             "--run-id", f"{RUN_ID}-dry", "--dry-run", "--json", "--profile", "residual",
             "--baseline-run", BASE_ID, "--sam-separations", "2"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["flow"] == run_e2e.FLOW_RESIDUAL
        assert payload["baseline_run"] == BASE_ID
        assert not (env.ws / "outputs" / "e2e" / f"{RUN_ID}-dry").exists()

    def test_dry_run_residual_requires_baseline(self, env):
        try:
            import pi_launcher  # type: ignore
            pi_launcher.resolve_command(["pi", "--version"])
        except Exception:  # pragma: no cover
            pytest.skip("pi 启动器不可用")
        proc = subprocess.run(
            [sys.executable, str(run_e2e.HERE / "run_e2e.py"), "--workspace", str(env.ws),
             "--run-id", f"{RUN_ID}-dry2", "--dry-run", "--json", "--profile", "residual"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert proc.returncode == 2                  # E_BASELINE 用法错误
        proc = subprocess.run(
            [sys.executable, str(run_e2e.HERE / "run_e2e.py"), "--workspace", str(env.ws),
             "--run-id", BASE_ID, "--dry-run", "--json", "--profile", "residual",
             "--baseline-run", BASE_ID],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert proc.returncode == 2                  # 不得把新 run 写进基线 id

    def test_verify_cli_exit_codes(self, env):
        chain = env.write_chain()
        env.write_result()
        cmd = [sys.executable, str(run_e2e.HERE / "residual_chain.py"), "verify",
               "--chain", str(chain), "--run-dir", str(env.run_dir),
               "--clip-json", str(env.ws / "local" / "e2e" / f"clip-{env.clip['name']}.json"),
               "--audio-root", str(env.ws / "audio" / "inputs"),
               "--baseline-run-dir", str(env.base_dir), "--ws", str(env.ws),
               "--budget", "2", "--result", str(env.run_dir / "result.json"), "--json"]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert json.loads(proc.stdout)["ok"] is True
        env.chain["stages"][1]["input"]["sha256"] = env.clip["sha256"]
        env.write_chain()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        assert proc.returncode == 1                  # 错误原混音 → 非零退出
