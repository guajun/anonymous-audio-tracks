# -*- coding: utf-8 -*-
"""Offline tests for agentic/workspace-template (issue #31).

Everything here is OFFLINE (mock/local): no network, no `gh skill install`, no
Pi API calls, no GPU. The REAL verification (Windows launch, project trust,
Gemini skill discovery, SAM help/dry-run) is a separate, documented evidence
set — see agentic/workspace-template/reports/30-deploy-verification.md.

Run:  uv run pytest agentic/workspace-template/tests -v
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TEMPLATE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TEMPLATE_DIR.parents[1]
BOOTSTRAP = TEMPLATE_DIR / "bootstrap.py"
DOCTOR = TEMPLATE_DIR / "doctor.py"
SMOKE = TEMPLATE_DIR / "smoke.py"
MANIFEST = json.loads((TEMPLATE_DIR / "manifest.json").read_text(encoding="utf-8"))

sys.path.insert(0, str(TEMPLATE_DIR))
import bootstrap as bootstrap_mod  # noqa: E402  共享 canonical LF hash 策略


def run_script(script: Path, args: list[str], env: dict | None = None, cwd: Path | None = None):
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, str(script)] + args,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600, shell=False, env=merged, cwd=str(cwd) if cwd else None,
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha256(repo: Path, rel: str) -> str:
    """sha256 of the committed blob (canonical LF) — what an LF checkout has."""
    blob = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{rel}"],
                          capture_output=True, timeout=60).stdout
    assert blob, f"git show HEAD:{rel} returned nothing"
    return hashlib.sha256(blob).hexdigest()


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """A deployed workspace (no skill install, with fixtures) in a path that
    contains a space — Windows argv/quoting regression."""
    ws = tmp_path / "ws dir"
    result = run_script(BOOTSTRAP, ["--workspace", str(ws), "--fixture", "--skip-skill"])
    assert result.returncode == 0, result.stderr + result.stdout
    return ws


# --------------------------------------------------------------------------
# frozen interface pins
# --------------------------------------------------------------------------

def test_manifest_bridge_pins_match_repo_files():
    for spec in MANIFEST["bridge"]["files"]:
        source = REPO_ROOT / spec["source"]
        assert source.is_file(), f"frozen bridge source missing: {spec['source']}"
        # canonical LF hash（与 bootstrap/doctor 同一策略）必须命中 pin
        assert bootstrap_mod.canonical_sha256(source) == spec["sha256"], (
            f"{spec['source']} drifted from manifest pin (interface is frozen in issue #30 README §9)"
        )


def test_manifest_pins_are_git_blob_canonical_lf():
    """Regression (review 1): pins must match the git blob (canonical LF), not
    Windows CRLF working-tree bytes, so an LF checkout/bootstrap also works."""
    for spec in MANIFEST["bridge"]["files"]:
        assert git_blob_sha256(REPO_ROOT, spec["source"]) == spec["sha256"], (
            f"pin for {spec['source']} is not the canonical LF git-blob hash"
        )


def test_bridge_deploy_writes_canonical_lf_bytes(workspace: Path):
    for spec in MANIFEST["bridge"]["files"]:
        data = (workspace / spec["install_path"]).read_bytes()
        assert b"\r\n" not in data, "deployed bridge bytes must be canonical LF (deterministic)"
        assert hashlib.sha256(data).hexdigest() == spec["sha256"]


def test_manifest_pins_match_toolbox_manifest():
    toolbox = json.loads((REPO_ROOT / "agentic" / "toolbox" / "manifest.json").read_text(encoding="utf-8"))
    assert MANIFEST["toolbox"]["pin"] == toolbox["skill"]["install"]["pin"]
    assert MANIFEST["sam_source"]["commit"] == toolbox["sam_source"]["commit"]
    assert MANIFEST["bridge"]["frozen_in"].startswith("agentic/audio-probe/README.md")


def test_template_settings_pin_research_model():
    settings = json.loads((TEMPLATE_DIR / "template" / ".pi" / "settings.json").read_text(encoding="utf-8"))
    assert settings["defaultProvider"] == MANIFEST["model"]["provider"] == "google"
    assert settings["defaultModel"] == MANIFEST["model"]["id"] == "gemini-3.8-flash"
    assert "extensions/audio-bridge.ts" in settings["extensions"]
    assert settings["sessionDir"] == "sessions"


def test_run_scripts_use_process_scoped_trust_and_bridge_root():
    ps1 = (TEMPLATE_DIR / "template" / "run.ps1").read_bytes()
    assert ps1[:3] == b"\xef\xbb\xbf", "run.ps1 must be UTF-8 with BOM (PowerShell 5.1 ANSI fallback bug)"
    ps1_text = ps1.decode("utf-8-sig")
    assert all(ord(c) < 128 for c in ps1_text), "run.ps1 must stay ASCII-only (PS 5.1 BOM-less line-swallow bug)"
    assert "--approve" in ps1_text and "PI_AUDIO_BRIDGE_ROOT" in ps1_text
    assert "trust.json" not in ps1_text.replace("never writes ~/.pi/agent/trust.json", "")
    sh_text = (TEMPLATE_DIR / "template" / "run.sh").read_text(encoding="utf-8")
    assert "--approve" in sh_text and "PI_AUDIO_BRIDGE_ROOT" in sh_text


def test_template_gitignore_covers_sensitive_dirs():
    gitignore = (TEMPLATE_DIR / "template" / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("audio/", "outputs/", "local/", "sessions/", ".pi/skills/", "*.wav"):
        assert pattern in gitignore


# --------------------------------------------------------------------------
# bootstrap: deploy / idempotency / no-overwrite / conflicts
# --------------------------------------------------------------------------

def test_bootstrap_deploys_layout_and_audio_manifest(workspace: Path):
    for rel in ("AGENTS.md", ".gitignore", ".pi/settings.json", ".pi/extensions/audio-bridge.ts",
                ".pi/extensions/audio_guard.mjs", "run.ps1", "run.sh",
                "local/config.json", "local/.bootstrap.json",
                "audio/inputs/manifest.json", "outputs", "sessions"):
        assert (workspace / rel).exists(), f"missing {rel}"
    for spec in MANIFEST["bridge"]["files"]:
        assert sha256_file(workspace / spec["install_path"]) == spec["sha256"]
    manifest = json.loads((workspace / "audio" / "inputs" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "workspace-audio-manifest/v1"
    names = [e["name"] for e in manifest["entries"]]
    assert names == ["fixture-a.wav", "fixture-b.wav"]
    for entry in manifest["entries"]:
        assert set(entry) >= {"name", "sha256", "bytes", "format", "duration_s",
                              "sample_rate", "channels", "probe", "source"}
        assert entry["duration_s"] == pytest.approx(1.5, abs=0.01)  # 3 tones + gaps by construction
        assert entry["sample_rate"] == 8000
        assert entry["source"] == "fixture"
        assert entry["sha256"] == sha256_file(workspace / "audio" / "inputs" / entry["name"])


def test_bootstrap_second_run_is_idempotent(workspace: Path):
    before = {str(p.relative_to(workspace)): sha256_file(p)
              for p in workspace.rglob("*") if p.is_file()}
    result = run_script(BOOTSTRAP, ["--workspace", str(workspace), "--fixture", "--skip-skill"])
    assert result.returncode == 0, result.stderr
    after = {str(p.relative_to(workspace)): sha256_file(p)
             for p in workspace.rglob("*") if p.is_file()}
    changed = {rel for rel in set(before) | set(after) if before.get(rel) != after.get(rel)}
    assert changed == {str(Path("local") / "bootstrap-report.json")}, \
        f"re-run must not modify user files: {changed}"
    assert "[fail]" not in result.stdout


def test_bootstrap_never_overwrites_user_config_or_audio(workspace: Path):
    (workspace / "AGENTS.md").write_text("user-edited agents\n", encoding="utf-8")
    settings = workspace / ".pi" / "settings.json"
    settings.write_text(json.dumps({"defaultProvider": "google", "defaultModel": "gemini-3.8-flash",
                                    "user_sentinel": True}), encoding="utf-8")
    config = workspace / "local" / "config.json"
    config.write_text(json.dumps({"sam_root": "C:/user/kept"}), encoding="utf-8")
    result = run_script(BOOTSTRAP, ["--workspace", str(workspace), "--fixture", "--skip-skill",
                                    "--sam-root", "C:/other/root"])
    assert result.returncode == 0, result.stderr
    assert (workspace / "AGENTS.md").read_text(encoding="utf-8") == "user-edited agents\n"
    assert json.loads(settings.read_text(encoding="utf-8"))["user_sentinel"] is True
    assert json.loads(config.read_text(encoding="utf-8"))["sam_root"] == "C:/user/kept"
    assert "[kept]" in result.stdout


def test_bootstrap_audio_hash_conflict_fails_without_overwrite(workspace: Path):
    target = workspace / "audio" / "inputs" / "fixture-a.wav"
    original = target.read_bytes()
    # 同名（basename）不同内容的导入必须显式失败，且绝不覆盖已有音频
    clashing = workspace / "clashing" / "fixture-a.wav"
    clashing.parent.mkdir()
    clashing.write_bytes(b"\x09\x09\x09\x09" * 500)
    result = run_script(BOOTSTRAP, ["--workspace", str(workspace), "--skip-skill", "--audio", str(clashing)])
    assert result.returncode == 3, result.stdout + result.stderr
    assert "E_HASH_CONFLICT" in result.stderr
    assert target.read_bytes() == original, "conflicting import must never overwrite user audio"
    # 同名同内容（重跑）应当幂等
    same = workspace / "same" / "fixture-a.wav"
    same.parent.mkdir()
    same.write_bytes(original)
    result = run_script(BOOTSTRAP, ["--workspace", str(workspace), "--skip-skill", "--audio", str(same)])
    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == original


def test_bootstrap_refuses_unmanaged_nonempty_dir(tmp_path: Path):
    ws = tmp_path / "occupied"
    ws.mkdir()
    (ws / "user-file.txt").write_text("keep", encoding="utf-8")
    result = run_script(BOOTSTRAP, ["--workspace", str(ws), "--skip-skill"])
    assert result.returncode == 3
    assert "E_WS_UNMANAGED" in result.stderr
    assert (ws / "user-file.txt").read_text(encoding="utf-8") == "keep"


def test_bootstrap_missing_audio_is_usage_error(workspace: Path):
    result = run_script(BOOTSTRAP, ["--workspace", str(workspace), "--skip-skill",
                                    "--audio", str(workspace / "nope.wav")])
    assert result.returncode == 2
    assert "E_AUDIO_NOT_FOUND" in result.stderr
    assert "Traceback" not in result.stderr


# --------------------------------------------------------------------------
# doctor: readable failures (no stack traces), JSON contract
# --------------------------------------------------------------------------

def _doctor_json(workspace: Path, extra: list[str] | None = None, env: dict | None = None):
    result = run_script(DOCTOR, ["--workspace", str(workspace), "--json"] + (extra or []), env=env)
    return result, json.loads(result.stdout)


def test_doctor_ok_on_deployed_workspace_without_sam_config(workspace: Path):
    result, data = _doctor_json(workspace)
    by_id = {c["id"]: c for c in data["checks"]}
    assert data["schema"] == "workspace-doctor/v1"
    assert by_id["workspace.settings"]["status"] == "ok"
    assert by_id["bridge.pin"]["status"] == "ok"
    assert by_id["skill.install"]["status"] == "fail"       # --skip-skill: not installed
    assert by_id["sam.config"]["status"] == "fail"          # placeholder config
    assert by_id["gpu.inference"]["status"] == "blocked"    # deferred to #33
    assert result.returncode == 1
    assert "Traceback" not in result.stderr


def test_doctor_missing_toolchain_is_readable_not_a_crash(workspace: Path, tmp_path: Path):
    empty = tmp_path / "empty-path"
    empty.mkdir()
    env = {"PATH": str(empty), "PI_NODE": "", "PI_INSTALL_DIR": "", "PI_CODING_AGENT_DIR": str(empty)}
    result, data = _doctor_json(workspace, env=env)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    by_id = {c["id"]: c for c in data["checks"]}
    for check_id in ("pi.cli", "node.cli", "ffmpeg.detect", "sam.config"):
        assert by_id[check_id]["status"] in ("fail", "warn"), check_id
        assert by_id[check_id].get("fix"), f"{check_id} must carry a readable fix hint"


def test_doctor_detects_model_drift_in_settings(workspace: Path):
    settings = workspace / ".pi" / "settings.json"
    data = json.loads(settings.read_text(encoding="utf-8"))
    data["defaultModel"] = "some-other-model"
    settings.write_text(json.dumps(data), encoding="utf-8")
    result, payload = _doctor_json(workspace)
    by_id = {c["id"]: c for c in payload["checks"]}
    assert by_id["workspace.settings"]["status"] == "fail"
    assert "defaultModel" in by_id["workspace.settings"]["detail"]


def test_doctor_detects_bridge_drift(workspace: Path):
    bridge = workspace / ".pi" / "extensions" / "audio-bridge.ts"
    bridge.write_text(bridge.read_text(encoding="utf-8") + "\n// drift\n", encoding="utf-8")
    result, payload = _doctor_json(workspace)
    by_id = {c["id"]: c for c in payload["checks"]}
    assert by_id["bridge.pin"]["status"] == "fail"
    assert "pin" in by_id["bridge.pin"]["detail"]


def test_doctor_collision_never_deletes_foreign_file(workspace: Path, monkeypatch):
    """Regression (review round 2 repro): when the exclusive create FAILS with
    EEXIST (mocked os.urandom forces a name collision), the pre-existing file
    with the probe name must survive with content and existence unchanged."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("doctor_mod", DOCTOR)
    doctor_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doctor_mod)
    monkeypatch.setattr(doctor_mod.os, "urandom", lambda n: b"\x00" * n)  # 强制碰撞名
    name = f".doctor-write-probe-{os.getpid()}-00000000"
    sentinel = workspace / "outputs" / name
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("USER_DATA", encoding="utf-8")
    doctor = doctor_mod.Doctor(workspace, deep=False, deep_audio=None, redact=True)
    doctor.check_outputs()
    assert sentinel.exists(), "EEXIST collision must never delete the pre-existing file"
    assert sentinel.read_text(encoding="utf-8") == "USER_DATA", "collision must never modify it"
    check = [c for c in doctor.checks if c["id"] == "outputs.writable"][0]
    assert check["status"] == "fail"  # 碰撞如实报告为探测失败，不假报成功
    assert not any(p.name.startswith(".doctor-write-probe-") and p.name != name
                   for p in sentinel.parent.iterdir())


def test_doctor_cleans_up_its_own_probe(workspace: Path):
    """Regression (review round 2): a successful run must clean up its OWN
    probe (and only that one)."""
    result = run_script(DOCTOR, ["--workspace", str(workspace), "--json"])
    assert result.returncode == 1  # skill/sam 未配置仍 fail，不影响本断言
    leftovers = [p.name for p in (workspace / "outputs").iterdir()
                 if p.name.startswith(".doctor-write-probe-")]
    assert leftovers == [], f"doctor must remove its own probe, left: {leftovers}"


def test_doctor_hints_only_advertise_real_commands(workspace: Path):
    """Regression (review 4): no fix hint may reference flags that do not exist."""
    result = run_script(DOCTOR, ["--workspace", str(workspace), "--json"])
    for bogus in ("--rehash-audio", "--verify-models"):
        assert bogus not in result.stdout, f"doctor must not advertise nonexistent {bogus}"
    payload = json.loads(result.stdout)
    for check in payload["checks"]:
        assert "Traceback" not in check.get("detail", "")


def test_doctor_redacts_absolute_paths(workspace: Path):
    result, payload = _doctor_json(workspace, extra=["--redact"])
    assert str(workspace) not in result.stdout
    assert "<WORKSPACE>" in result.stdout


# --------------------------------------------------------------------------
# smoke: offline surface (argv construction only; the real run is documented)
# --------------------------------------------------------------------------

def test_smoke_print_argv_uses_settings_and_process_trust(workspace: Path):
    result = run_script(SMOKE, ["--workspace", str(workspace), "--print-argv", "--redact"])
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["uses_approve"] is True
    assert data["uses_model_flag"] is False, "model must come from trust-gated project settings"
    assert data["argv"][-1] == "<SMOKE_PROMPT>"
    assert "--no-session" in data["argv"] and "--mode" in data["argv"]
    assert data["env"]["PI_AUDIO_BRIDGE_ROOT"] == "<WORKSPACE>/audio/inputs"


def test_smoke_rejects_bad_timeout(workspace: Path):
    result = run_script(SMOKE, ["--workspace", str(workspace), "--timeout", "0"])
    assert result.returncode == 2
    assert "E_TIMEOUT" in result.stderr


# --------------------------------------------------------------------------
# review round 1 regressions: outputs sentinel, redaction, exit codes, skill pin
# --------------------------------------------------------------------------

def test_doctor_preserves_existing_outputs_files(workspace: Path):
    """Regression (review 2): the writability probe must never delete or touch
    pre-existing outputs/ files (e.g. a user's .write-test sentinel)."""
    sentinel = workspace / "outputs" / ".write-test"
    sentinel.write_text("USER_DATA", encoding="utf-8")
    before = {p.name: sha256_file(p) for p in (workspace / "outputs").iterdir() if p.is_file()}
    result = run_script(DOCTOR, ["--workspace", str(workspace), "--json"])
    assert result.returncode == 1  # skill/sam 未配置仍 fail，但不影响本断言
    assert sentinel.exists(), "doctor must not delete pre-existing outputs/.write-test"
    assert sentinel.read_text(encoding="utf-8") == "USER_DATA"
    after = {p.name: sha256_file(p) for p in (workspace / "outputs").iterdir() if p.is_file()}
    assert before == after, "doctor must not create/delete/modify anything else in outputs/"
    payload = json.loads(result.stdout)
    by_id = {c["id"]: c for c in payload["checks"]}
    assert by_id["outputs.writable"]["status"] == "ok"


def _load_smoke_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("smoke_mod", SMOKE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GOOD_EVENTS = "\n".join([
    json.dumps({"type": "message_start", "message": {"role": "system", "content": "",
                "sections": {"skills": "- sam-audio: Run local SAM Audio separation"}}}),
    json.dumps({"type": "tool_execution_end", "toolName": "bash", "isError": False,
                "result": {"content": [{"type": "text", "text": "usage: audio-toolbox [-h]"}]}}),
    json.dumps({"type": "tool_execution_end", "toolName": "audio_attach", "isError": True,
                "result": {"content": [{"type": "text", "text": "E_AUDIO_NOT_FOUND: audio file not found"}]}}),
    json.dumps({"type": "message_end", "message": {"role": "assistant", "model": "gemini-3.8-flash",
                "stopReason": "stop", "content": [{"type": "text",
                "text": "SMOKE-DONE skills=sam-audio help=usage: audio-toolbox err=E_AUDIO_NOT_FOUND"}],
                "usage": {"totalTokens": 100, "cost": {"total": 0.001}}}}),
])


def _fake_runner(monkeypatch, module, rc: int, events_text: str = GOOD_EVENTS):
    """Fake run_bounded: writes the given event stream and returns `rc`."""
    def fake_run(command, **kwargs):
        out_path = Path(command[command.index("--stdout") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(events_text, encoding="utf-8")
        Path(command[command.index("--stderr") + 1]).write_text("", encoding="utf-8")
        class Result:
            returncode = rc
            stdout = ""
            stderr = ""
        return Result()
    monkeypatch.setattr(module, "build_argv", lambda: ["pi-fake"])
    monkeypatch.setattr(module.subprocess, "run", fake_run)


def test_smoke_propagates_runner_failure_codes(workspace: Path, monkeypatch):
    """Regression (review 5): documented run_bounded codes 3/4/5/6 propagate
    unchanged instead of being flattened to 0/1."""
    module = _load_smoke_module()
    for rc in (3, 4, 5, 6):
        _fake_runner(monkeypatch, module, rc)
        assert module.main(["--workspace", str(workspace)]) == rc, f"runner rc {rc} must propagate"


def test_smoke_acceptance_failure_is_distinct_code(workspace: Path, monkeypatch):
    module = _load_smoke_module()
    broken = GOOD_EVENTS.replace("SMOKE-DONE", "NOT-DONE")
    _fake_runner(monkeypatch, module, 0, events_text=broken)
    assert module.main(["--workspace", str(workspace)]) == 1  # 运行完成但判据未过
    _fake_runner(monkeypatch, module, 0)
    assert module.main(["--workspace", str(workspace)]) == 0


def test_smoke_redacts_all_argv_and_output(tmp_path: Path, monkeypatch, capsys):
    """Regression (review 3): --redact must hide home/repo/out-of-home tool
    paths (with spaces) in argv and every printed field."""
    module = _load_smoke_module()
    ws = tmp_path / "ws dir"
    ws.mkdir()
    home_tool = Path.home() / "my tools" / "node.exe"
    repo_tool = REPO_ROOT / "dist" / "cli.js"
    outside_tool = Path("D:/tools outside/pi bin/pi.exe")
    monkeypatch.setattr(module, "build_argv", lambda: [str(home_tool), str(repo_tool),
                                                        str(outside_tool), "--", "<PROMPT>"])
    code = module.main(["--workspace", str(ws), "--print-argv", "--redact"])
    assert code == 0
    out = capsys.readouterr().out
    for secret in (str(Path.home()), str(REPO_ROOT), "D:/tools outside", "my tools", "pi bin"):
        assert secret not in out, f"redaction leaked: {secret}"
    assert "<PATH>/node.exe" in out and "<PATH>/cli.js" in out and "<PATH>/pi.exe" in out


def _fake_skill(workspace: Path, pinned: str | None = "dfbc40a9541f686207b65b93b1332bb505654261",
                drop_metadata: bool = False) -> Path:
    skill = workspace / ".pi" / "skills" / "sam-audio"
    (skill / "references").mkdir(parents=True, exist_ok=True)
    (skill / "scripts").mkdir(parents=True, exist_ok=True)
    meta = "" if drop_metadata else (
        f"    github-pinned: {pinned}\n"
        "    github-repo: https://github.com/guajun/agentic-audio-toolbox\n"
    )
    (skill / "SKILL.md").write_text(
        "---\nmetadata:\n" + meta + "name: sam-audio\n---\n# body\n", encoding="utf-8")
    (skill / "references" / "cli-reference.md").write_text("# cli\n", encoding="utf-8")
    (skill / "scripts" / "audio_toolbox.py").write_text("print('x')\n", encoding="utf-8")
    return skill


def test_skill_wrong_pin_fails_without_overwrite(workspace: Path):
    """Regression (review 6): an existing skill with wrong/missing pin metadata
    must fail loudly and never be silently accepted, overwritten or reinstalled."""
    skill = _fake_skill(workspace, pinned="0000000000000000000000000000000000000000")
    before = {p.name: p.read_bytes() for p in skill.rglob("*") if p.is_file()}
    result = run_script(BOOTSTRAP, ["--workspace", str(workspace)])
    assert result.returncode == 3, result.stdout + result.stderr
    assert "E_SKILL_PIN" in result.stderr
    after = {p.name: p.read_bytes() for p in skill.rglob("*") if p.is_file()}
    assert before == after, "bootstrap must not modify user skill files on pin mismatch"
    # doctor 同样显式失败
    result, payload = _doctor_json(workspace)
    by_id = {c["id"]: c for c in payload["checks"]}
    assert by_id["skill.install"]["status"] == "fail"
    assert "pin" in by_id["skill.install"]["detail"].lower()


def test_skill_missing_pin_metadata_fails(workspace: Path):
    skill = _fake_skill(workspace, drop_metadata=True)
    result = run_script(BOOTSTRAP, ["--workspace", str(workspace)])
    assert result.returncode == 3
    assert "E_SKILL_PIN" in result.stderr
    assert (skill / "SKILL.md").read_text(encoding="utf-8").startswith("---")


def test_skill_content_drift_fails(workspace: Path):
    skill = _fake_skill(workspace)  # metadata 正确但内容 hash 不符
    result = run_script(BOOTSTRAP, ["--workspace", str(workspace)])
    assert result.returncode == 3
    assert "E_SKILL_PIN" in result.stderr
    assert "hash" in result.stderr
