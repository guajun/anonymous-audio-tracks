#!/usr/bin/env python3
"""doctor.py — diagnose a deployed Pi audio-analysis workspace (issue #31).

Read-only diagnostics, with ONE documented exception: outputs/ writability is
probed with a unique exclusively-created temporary file which is deleted right
away — existing files (e.g. a user's outputs/.write-test) are never touched.
Never prints or reads API keys (`pi auth check` only), never loads a model,
never uses the GPU (GPU inference is issue #33), never downloads anything.
Missing key / weights / FFmpeg produce READABLE findings with a fix hint
pointing at commands that actually exist — not a stack trace.

Checks (status: ok | warn | fail | blocked):
  workspace.layout / workspace.settings / bridge.pin / skill.install /
  audio.manifest / outputs.writable / node.cli / pi.cli / pi.auth.google /
  gh.cli / sam.config / sam.entry / sam.weights / ffmpeg.detect /
  gpu.inference (blocked, #33)

`--deep` additionally runs the REAL toolbox CLI against the local SAM checkout:
  sam.check-environment  (`sam check-environment`, no torch/GPU/network)
  sam.dry-run            (`sam separate --dry-run`, reads real audio duration,
                          validates anchors/FFmpeg/model paths, no model load)
  sam.verify-models      (upstream `scripts/verify_models.py`: real byte+SHA-256
                          integrity against model-manifest.json)

Weight checks in the default run are LAYOUT/size checks only (not SHA-256
integrity); full integrity is the upstream verify command above.

FFmpeg detection is a FILE/PATH level check only. It does NOT prove
TorchCodec/GPU inference readiness; that distinction is kept explicit in the
output (`ffmpeg.detect` vs `gpu.inference`).

Exit codes: 0 no failures (warnings allowed) / 1 failures present / 2 usage.
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
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEMPLATE_DIR.parents[1]
MANIFEST_PATH = TEMPLATE_DIR / "manifest.json"
PROBE_DIR = REPO_ROOT / "agentic" / "audio-probe" / "probe"

sys.path.insert(0, str(PROBE_DIR))
sys.path.insert(0, str(TEMPLATE_DIR))

# 共享工具（与 bootstrap 同一套 canonical LF hash 策略与 skill pin 校验）
from bootstrap import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    sha256_bytes,
    verify_skill_install,
)

OK, WARN, FAIL, BLOCKED = "ok", "warn", "fail", "blocked"


def run_cmd(argv: list[str], timeout: int = 60, cwd: Path | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, shell=False, cwd=str(cwd) if cwd else None)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError:
        return 127, "", f"command not found: {argv[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except OSError as exc:
        return 126, "", f"{exc.__class__.__name__}: {exc}"


class Doctor:
    def __init__(self, ws: Path, deep: bool, deep_audio: Path | None, redact: bool) -> None:
        self.ws = ws
        self.deep = deep
        self.deep_audio = deep_audio
        self.redact = redact
        self.checks: list[dict] = []
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def add(self, check_id: str, status: str, detail: str, fix: str = "") -> None:
        entry = {"id": check_id, "status": status, "detail": detail}
        if fix:
            entry["fix"] = fix
        self.checks.append(entry)

    # ---------- workspace ----------
    def check_layout(self) -> None:
        missing = [rel for rel in ("audio/inputs", "outputs", "local", ".pi/extensions", ".pi/skills")
                   if not (self.ws / rel).is_dir()]
        if not self.ws.is_dir():
            self.add("workspace.layout", FAIL, f"workspace 不存在：{self.ws}",
                     "运行 agentic/workspace-template/bootstrap.py --workspace <WS> 部署")
        elif missing:
            self.add("workspace.layout", FAIL, f"缺少目录：{', '.join(missing)}",
                     "重跑 bootstrap（幂等，不覆盖已有文件）")
        else:
            self.add("workspace.layout", OK, f"目录齐全：{self.ws}")

    def check_settings(self) -> None:
        path = self.ws / ".pi" / "settings.json"
        expected_model = self.manifest["model"]
        if not path.is_file():
            self.add("workspace.settings", FAIL, ".pi/settings.json 缺失", "重跑 bootstrap")
            return
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self.add("workspace.settings", FAIL, f".pi/settings.json 不是合法 JSON：{exc}", "修复 JSON 语法")
            return
        problems = []
        if settings.get("defaultProvider") != expected_model["provider"]:
            problems.append(f"defaultProvider={settings.get('defaultProvider')!r}")
        if settings.get("defaultModel") != expected_model["id"]:
            problems.append(f"defaultModel={settings.get('defaultModel')!r}")
        if "extensions/audio-bridge.ts" not in (settings.get("extensions") or []):
            problems.append("extensions 未包含 extensions/audio-bridge.ts")
        if problems:
            self.add("workspace.settings", FAIL,
                     f"项目设置与冻结模型不一致：{'; '.join(problems)}（要求 {expected_model['provider']}/{expected_model['id']}）",
                     "恢复模板设置；模型由 issue #30 冻结，不要改")
        else:
            self.add("workspace.settings", OK,
                     f"模型固定 {expected_model['provider']}/{expected_model['id']}；桥接扩展列入 extensions（需 project trust 加载）")

    def check_bridge(self) -> None:
        problems = []
        for spec in self.manifest["bridge"]["files"]:
            dest = self.ws / spec["install_path"]
            pinned = spec["sha256"]
            if not dest.is_file():
                problems.append(f"{spec['install_path']} 缺失")
                continue
            actual = canonical_sha256(dest)  # canonical LF（与 bootstrap/测试同一策略）
            if actual != pinned:
                problems.append(f"{Path(spec['install_path']).name} 与冻结 pin 不一致（pin {pinned[:12]}…, 实际 {actual[:12]}…）")
        if problems:
            self.add("bridge.pin", FAIL, "; ".join(problems),
                     "接口在 issue #30 README §9 冻结；bootstrap --force-bridge 可显式恢复 pin 版本（hash 为 canonical LF，换行符差异不算漂移）")
        else:
            self.add("bridge.pin", OK, "桥接 2 个文件均为冻结 pin（canonical LF hash：audio-bridge.ts + audio_guard.mjs；bridge ≠ Pi 原生音频）")

    def check_skill(self) -> None:
        expected = [self.ws / rel for rel in self.manifest["toolbox"]["installed_files"]]
        missing = [p for p in expected if not p.is_file()]
        if missing:
            self.add("skill.install", FAIL,
                     f"sam-audio skill 不完整，缺少：{', '.join(str(p.relative_to(self.ws)) for p in missing)}",
                     f"运行：{self.manifest['toolbox']['install_command']}")
            return
        problems, notes = verify_skill_install(self.ws)
        if problems:
            self.add("skill.install", FAIL,
                     "skill 存在但 pin/内容校验失败（不覆盖、不重装）：" + "; ".join(problems),
                     "人工确认来源后删除 .pi/skills/sam-audio，再运行 manifest.json 中的 install_command 重新安装（--pin 固定）")
            return
        self.add("skill.install", OK,
                 f"sam-audio skill pin 校验通过（@{self.manifest['toolbox']['pin'][:12]}…）：{notes[0]}")

    def check_audio_manifest(self) -> None:
        path = self.ws / "audio" / "inputs" / "manifest.json"
        if not path.is_file():
            self.add("audio.manifest", WARN, "audio/inputs/manifest.json 不存在（尚无导入音频）",
                     "bootstrap --audio <FILE> 或 --fixture 导入")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = data["entries"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            self.add("audio.manifest", FAIL, f"manifest.json 损坏：{exc.__class__.__name__}",
                     "人工确认后删除 audio/inputs/manifest.json，再用 bootstrap --audio <FILE> 重新登记（不覆盖音频本体）")
            return
        problems = []
        for entry in entries:
            file = self.ws / "audio" / "inputs" / entry.get("name", "")
            if not file.is_file():
                problems.append(f"{entry.get('name')}: 文件缺失")
            elif sha256_file(file) != entry.get("sha256"):
                problems.append(f"{entry.get('name')}: sha256 与 manifest 不符（文件被改动）")
        if problems:
            self.add("audio.manifest", FAIL, "; ".join(problems),
                     "不要覆盖用户音频；人工确认后删除 manifest.json 中对应条目并用 bootstrap --audio <FILE> 重新登记，或恢复文件原内容")
        else:
            self.add("audio.manifest", OK, f"{len(entries)} 条音频记录，hash 全部一致")

    def check_outputs(self) -> None:
        """outputs/ writability via a UNIQUE exclusively-created probe file.
        Only that probe is removed; every pre-existing file (including a user's
        outputs/.write-test) is preserved untouched."""
        out = self.ws / "outputs"
        probe = out / f".doctor-write-probe-{os.getpid()}-{os.urandom(4).hex()}"
        try:
            out.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(probe), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(b"doctor write probe")
        except OSError as exc:
            try:
                probe.unlink()
            except OSError:
                pass
            self.add("outputs.writable", FAIL, f"outputs/ 不可写：{exc.__class__.__name__}", "检查权限/磁盘")
            return
        try:
            probe.unlink()  # 只删除本探测文件（唯一独占创建）
        except OSError:
            pass
        self.add("outputs.writable", OK, "outputs/ 可写（唯一独占临时探测文件已删除；现存文件一律不动；结果约定 outputs/result.json，schema 归 issue #32）")

    # ---------- toolchain ----------
    def check_node_pi(self) -> None:
        node = shutil.which("node")
        if not node:
            self.add("node.cli", FAIL, "PATH 中找不到 node（pi 需要 Node >= 22.19）", "安装 Node 或设置 PI_NODE")
        else:
            code, out, _ = run_cmd([node, "--version"], timeout=30)
            self.add("node.cli", OK if code == 0 else FAIL,
                     f"node {out.strip() or '(no output)'}" if code == 0 else "node 无法执行", "")
        try:
            import pi_launcher  # type: ignore
        except Exception as exc:
            self.add("pi.cli", FAIL, f"无法加载 pi_launcher（{exc.__class__.__name__}）",
                     "在仓库内运行 doctor（需要 agentic/audio-probe/probe/pi_launcher.py）")
            return
        try:
            argv = pi_launcher.resolve_command(["pi", "--version"])
        except pi_launcher.LauncherError as exc:
            self.add("pi.cli", FAIL, f"找不到 Pi 安装：{exc}", "安装 Pi 0.87.x 或设置 PI_INSTALL_DIR")
            return
        code, out, err = run_cmd(argv, timeout=60)
        if code != 0:
            self.add("pi.cli", FAIL, f"pi --version 失败（exit {code}）：{(err or out).strip()[:160]}", "检查 Pi 安装")
        else:
            version = out.strip().splitlines()[0] if out.strip() else "?"
            status = OK if version.startswith("0.87") else WARN
            self.add("pi.cli", status, f"pi {version}（native argv：node + CLI entry）",
                     "" if status == OK else "本模板按 Pi 0.87.1 验证；其他版本需重新验证接口")

    def check_auth(self) -> None:
        try:
            import pi_launcher  # type: ignore
            argv = pi_launcher.resolve_command(["pi", "auth", "check", "--provider", "google", "--json"])
        except Exception as exc:
            self.add("pi.auth.google", FAIL, f"无法解析 pi：{exc.__class__.__name__}", "先修复 pi.cli")
            return
        code, out, err = run_cmd(argv, timeout=60)
        status_text = ""
        try:
            status_text = (json.loads(out.strip().splitlines()[-1]) or {}).get("status", "")
        except (json.JSONDecodeError, IndexError):
            status_text = ""
        if code == 0 and status_text == "ready":
            # 只看状态；绝不调用 print-api-key / print-bearer-token。
            self.add("pi.auth.google", OK, "Pi google provider 认证就绪（只读取状态，不读取/打印 key）")
        elif status_text in ("not_ready", "invalid"):
            self.add("pi.auth.google", FAIL,
                     f"Pi google 认证状态：{status_text}（无 key / key 无效）",
                     "用 Pi 自身认证流程配置（如 pi auth / 交互式登录）；本工具不读取、不打印、不保存 key")
        else:
            self.add("pi.auth.google", FAIL, f"auth check 无法解析（exit {code}）：{(err or out).strip()[:160]}",
                     "手动运行 `pi auth check --provider google --json` 查看")

    def check_gh(self) -> None:
        gh = shutil.which("gh")
        if not gh:
            self.add("gh.cli", WARN, "PATH 中找不到 gh（重新安装 skill 时需要）", "安装 GitHub CLI（skill 已装则不影响运行）")
            return
        code, out, _ = run_cmd([gh, "skill", "list"], timeout=60, cwd=self.ws)
        listed = any(line.split()[:1] == ["sam-audio"] and "project" in line for line in out.splitlines())
        if code == 0 and listed:
            self.add("gh.cli", OK, "gh skill list 含 sam-audio (pi / project)")
        elif code == 0:
            self.add("gh.cli", WARN, "gh skill list 未列出 sam-audio（skill 可能装在别处）", "在 workspace 目录运行 install_command")
        else:
            self.add("gh.cli", WARN, f"gh skill list 失败（exit {code}）", "gh 需登录/更新（skill 为 preview）")

    # ---------- SAM ----------
    def _sam_config(self) -> tuple[Path | None, Path | None, str]:
        path = self.ws / "local" / "config.json"
        if not path.is_file():
            return None, None, "local/config.json 不存在"
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return None, None, f"local/config.json 不是合法 JSON：{exc}"
        def resolve(value: str | None) -> Path | None:
            if not value or value.startswith("<"):
                return None
            return Path(value).expanduser()
        root = resolve(cfg.get("sam_root"))
        python = resolve(cfg.get("sam_python"))
        if root is None or python is None:
            return root, python, "local/config.json 未配置 sam_root / sam_python（仍为占位符）"
        return root, python, "local/config.json 已配置"

    def check_sam(self) -> None:
        root, python, detail = self._sam_config()
        if root is None or python is None:
            self.add("sam.config", FAIL, detail,
                     "在 local/config.json 填写 sam_root/sam_python（本机迁移的 SAM checkout）；不自动下载")
            return
        self.add("sam.config", OK, f"SAM checkout 已配置（{root.name}）")
        entry = root / "scripts" / "run_inference.py"
        if entry.is_file():
            self.add("sam.entry", OK, "SAM 入口 scripts/run_inference.py 存在")
        else:
            self.add("sam.entry", FAIL, f"SAM 入口缺失：{entry}", "检查 sam_root 是否指向 SAM checkout 根目录")
        missing, present = [], 0
        for rel in self.manifest["sam_source"]["weights"]["local_layout"]:
            file = root / rel
            if file.is_file() and file.stat().st_size > 0:
                present += 1
            else:
                missing.append(rel)
        if missing:
            self.add("sam.weights", FAIL,
                     f"权重缺失：{', '.join(missing)}（权重不自动下载、不分发）",
                     "按上游 model-manifest.json 手动补齐，再运行 `<SAM_PYTHON> <SAM_ROOT>/scripts/verify_models.py` 校验（字节+SHA-256）")
        else:
            self.add("sam.weights", OK,
                     f"权重布局+非空检查通过（{present} 个文件；**非** SHA-256 完整性校验）；字节/SHA-256 校验请运行上游 `scripts/verify_models.py`（doctor --deep 会真实运行）")
        if not python.is_file():
            self.add("sam.python", FAIL, f"sam_python 不存在：{python}", "指向 SAM checkout 的 .venv 解释器")
        else:
            self.add("sam.python", OK, "SAM 解释器存在（真实分离用；dry-run 不加载模型）")

    def check_ffmpeg(self) -> None:
        # 文件级探测 ≠ TorchCodec / GPU 推理就绪（后者留给 #33）。
        cfg_ffmpeg = ""
        cfg_path = self.ws / "local" / "config.json"
        if cfg_path.is_file():
            try:
                cfg_ffmpeg = (json.loads(cfg_path.read_text(encoding="utf-8")) or {}).get("ffmpeg") or ""
            except json.JSONDecodeError:
                cfg_ffmpeg = ""
        ffmpeg = cfg_ffmpeg if cfg_ffmpeg and Path(cfg_ffmpeg).is_file() else shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg and ffprobe:
            self.add("ffmpeg.detect", OK,
                     "ffmpeg/ffprobe 可执行文件存在（文件级探测；不代表 TorchCodec/GPU 推理可用）")
        elif ffmpeg:
            self.add("ffmpeg.detect", WARN, "ffmpeg 存在但缺 ffprobe（非 wav 元数据探测会退回 null）", "安装完整 FFmpeg 4–8 full-shared 构建")
        else:
            self.add("ffmpeg.detect", FAIL,
                     "未找到 ffmpeg（Windows 需 FFmpeg 4–8 full-shared 构建在 PATH；static 构建缺共享 DLL 不够）",
                     "安装 FFmpeg 或在 local/config.json 的 ffmpeg 字段给出路径")

    def check_gpu(self) -> None:
        self.add("gpu.inference", BLOCKED,
                 "GPU/CUDA 推理验证不在本项范围（issue #33）；本工具从不加载 torch/模型",
                 "见 issue #33 的端到端任务")

    def check_deep(self) -> None:
        root, python, detail = self._sam_config()
        toolbox = self.ws / ".pi" / "skills" / "sam-audio" / "scripts" / "audio_toolbox.py"
        if root is None or python is None or not toolbox.is_file():
            self.add("sam.check-environment", FAIL, f"前置不满足（{detail}）", "先修复 sam.config / skill.install")
            return
        argv = [sys.executable, str(toolbox), "sam", "check-environment",
                "--sam-root", str(root), "--python", str(python)]
        code, out, err = run_cmd(argv, timeout=300, cwd=self.ws)
        payload = self._parse_json(out)
        if code == 0 and payload and payload.get("ok"):
            failed = payload.get("failed_checks") or []
            self.add("sam.check-environment", OK if not failed else FAIL,
                     f"audio-toolbox sam check-environment 通过（{len(payload.get('checks', []))} 项）"
                     if not failed else f"check-environment 失败项：{', '.join(map(str, failed))}",
                     "" if not failed else "见 error/failed_checks 明细")
        else:
            error = (payload or {}).get("error", {})
            self.add("sam.check-environment", FAIL,
                     f"sam check-environment 失败（exit {code}）：{error.get('code', '?')} {error.get('message', '') or (err or out).strip()[:160]}",
                     "按错误码修复（E_ENVIRONMENT=路径/解释器/权重）")
        audio = self.deep_audio
        if audio is None:
            entries = []
            manifest_path = self.ws / "audio" / "inputs" / "manifest.json"
            if manifest_path.is_file():
                try:
                    entries = json.loads(manifest_path.read_text(encoding="utf-8")).get("entries", [])
                except json.JSONDecodeError:
                    entries = []
            if entries:
                audio = self.ws / "audio" / "inputs" / entries[0]["name"]
        if audio is None or not Path(audio).is_file():
            self.add("sam.dry-run", FAIL, "没有可用的输入音频（--audio 或 audio/inputs/）",
                     "bootstrap --fixture 生成测试音频后重试")
            return
        argv = [sys.executable, str(toolbox), "sam", "separate",
                "--audio", str(audio), "--description", "melodic sound", "--dry-run",
                "--sam-root", str(root), "--python", str(python)]
        code, out, err = run_cmd(argv, timeout=600, cwd=self.ws)
        payload = self._parse_json(out)
        if code == 0 and payload and payload.get("ok"):
            plan = payload.get("plan") or {}
            self.add("sam.dry-run", OK,
                     f"真实 dry-run 通过（audio={Path(str(audio)).name}, duration_s={plan.get('duration_s')}, "
                     f"device={plan.get('device')}；未加载模型、未用 GPU）")
        else:
            error = (payload or {}).get("error", {})
            self.add("sam.dry-run", FAIL,
                     f"sam dry-run 失败（exit {code}）：{error.get('code', '?')} {error.get('message', '') or (err or out).strip()[:160]}",
                     "常见：E_ENVIRONMENT（FFmpeg/权重/解释器）、E_AUDIO_NOT_FOUND、E_ANCHOR_INVALID")

    def check_verify_models(self) -> None:
        """Real upstream integrity check: scripts/verify_models.py compares byte
        size + SHA-256 against model-manifest.json. This is the only weight
        integrity claim we make; the default layout check does not claim it."""
        root, python, detail = self._sam_config()
        script = (root / "scripts" / "verify_models.py") if root else None
        if root is None or python is None or script is None or not script.is_file() or not python.is_file():
            self.add("sam.verify-models", FAIL, f"前置不满足（{detail}）", "先修复 sam.config / sam.entry")
            return
        code, out, err = run_cmd([str(python), str(script)], timeout=900, cwd=root)
        tail = (out or err).strip()[-200:]
        if code == 0 and "All model files verified" in (out or ""):
            self.add("sam.verify-models", OK,
                     "上游 scripts/verify_models.py 真实校验通过（字节+SHA-256 对照 model-manifest.json）")
        else:
            self.add("sam.verify-models", FAIL,
                     f"权重完整性校验失败（exit {code}）：{tail}",
                     "按输出补齐/修复权重（不自动下载）；校验命令：`<SAM_PYTHON> <SAM_ROOT>/scripts/verify_models.py`")

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Parse the first complete JSON object found in `text` (stdout may carry a
        pretty-printed multi-line JSON document plus trailing noise)."""
        text = (text or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        for index in (m.start() for m in re.finditer(re.escape("{"), text)):
            try:
                data = json.loads(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return None

    def redact_text(self, text: str) -> str:
        replacements = [(str(self.ws), "<WORKSPACE>")]
        root, python, _ = self._sam_config()
        if root:
            replacements.append((str(root), "<SAM_ROOT>"))
        if python:
            replacements.append((str(python), "<SAM_PYTHON>"))
        home = str(Path.home())
        # 本地样本文件名也不进公开材料
        manifest_path = self.ws / "audio" / "inputs" / "manifest.json"
        if manifest_path.is_file():
            try:
                for entry in json.loads(manifest_path.read_text(encoding="utf-8")).get("entries", []):
                    if entry.get("name"):
                        replacements.append((str(entry["name"]), "<AUDIO>"))
            except (OSError, json.JSONDecodeError):
                pass
        for old, new in replacements:
            text = text.replace(old, new)
        return text.replace(home, "<HOME>")

    def run(self) -> int:
        self.check_layout()
        self.check_settings()
        self.check_bridge()
        self.check_skill()
        self.check_audio_manifest()
        self.check_outputs()
        self.check_node_pi()
        self.check_auth()
        self.check_gh()
        self.check_sam()
        self.check_ffmpeg()
        self.check_gpu()
        if self.deep:
            self.check_deep()
            self.check_verify_models()
        return 0 if not any(c["status"] == FAIL for c in self.checks) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="doctor.py", description="workspace 诊断（issue #31，只读）")
    parser.add_argument("--workspace", default=str(REPO_ROOT / ".local" / "agentic" / "workspace"))
    parser.add_argument("--json", action="store_true", help="输出 JSON（schema workspace-doctor/v1）")
    parser.add_argument("--deep", action="store_true", help="追加真实 sam check-environment / dry-run（无 GPU、无模型加载）")
    parser.add_argument("--audio", default="", help="--deep dry-run 用的音频（默认 audio/inputs 第一条）")
    parser.add_argument("--redact", action="store_true", help="把绝对路径替换为占位符（用于公开粘贴）")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code not in (0, None) else 0
    ws = Path(args.workspace).expanduser().resolve()
    doctor = Doctor(ws, args.deep, Path(args.audio).expanduser().resolve() if args.audio else None, args.redact)
    code = doctor.run()
    if args.json:
        print(json.dumps({
            "schema": "workspace-doctor/v1",
            "workspace": "<WORKSPACE>" if args.redact else str(ws),
            "checks": [
                {**c, "detail": doctor.redact_text(c["detail"]) if args.redact else c["detail"],
                 **({"fix": doctor.redact_text(c["fix"])} if args.redact and c.get("fix") else {})}
                for c in doctor.checks
            ],
            "summary": {
                "ok": sum(1 for c in doctor.checks if c["status"] == OK),
                "warn": sum(1 for c in doctor.checks if c["status"] == WARN),
                "fail": sum(1 for c in doctor.checks if c["status"] == FAIL),
                "blocked": sum(1 for c in doctor.checks if c["status"] == BLOCKED),
            },
        }, ensure_ascii=False, indent=2))
    else:
        for check in doctor.checks:
            line = f"[{check['status']:>7}] {check['id']}: {check['detail']}"
            if args.redact:
                line = doctor.redact_text(line)
            print(line)
            if check.get("fix") and check["status"] in (FAIL, WARN):
                fix = doctor.redact_text(check["fix"]) if args.redact else check["fix"]
                print(f"          fix: {fix}")
        fails = sum(1 for c in doctor.checks if c["status"] == FAIL)
        print(f"\n结论：{fails} 项失败（blocked 项留给 issue #33，不计失败）")
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # readable error, never a bare traceback
        print(f"[fail] E_INTERNAL: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
