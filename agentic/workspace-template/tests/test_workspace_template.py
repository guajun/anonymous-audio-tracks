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
        assert sha256_file(source) == spec["sha256"], (
            f"{spec['source']} drifted from manifest pin (interface is frozen in issue #30 README §9)"
        )


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
