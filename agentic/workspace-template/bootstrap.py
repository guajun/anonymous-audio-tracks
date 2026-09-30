#!/usr/bin/env python3
"""bootstrap.py — deploy a real Pi audio-analysis workspace (issue #31).

What this does (issue #31, scope: workspace-template/bootstrap/doctor/tests/docs):
  * creates a workspace directory (default: <repo>/.local/agentic/workspace,
    a gitignored local directory) as a real Pi project:
      AGENTS.md, .pi/settings.json, .pi/extensions/audio-bridge.ts,
      .pi/skills/sam-audio (installed via `gh skill install ... --pin`),
      audio/inputs/ (+ manifest.json), outputs/, local/, sessions/, run scripts
  * imports audio (`--audio FILE`, repeatable) or generates deterministic
    fixtures (`--fixture`), recording sha256 / duration / sample rate
  * writes local machine config (`local/config.json`) for the SAM checkout

Safety rules enforced here (they are acceptance criteria, not preferences):
  * IDEMPOTENT: a second run never overwrites user-edited files, user audio or
    user config. Template files are written only when missing; a differing file
    is KEPT and reported (`[kept]`).
  * HASH CONFLICTS FAIL: importing audio whose target name already exists with
    a different sha256 is an error (exit 3), never a silent overwrite.
  * The bridge extension is a frozen interface (issue #30 README §9): its
    sha256 is pinned in manifest.json. A differing copy is an error unless
    --force-bridge is given explicitly.
  * No global state is touched: no ~/.pi/agent settings, no auth, no
    ~/.pi/agent/trust.json. Project trust is granted per process by the run
    scripts via `pi --approve`.
  * No API keys, audio bytes, model weights or absolute paths are written into
    anything tracked by git (the workspace itself is a local ignored directory;
    its .gitignore additionally ignores audio/, outputs/, local/, sessions/).

Exit codes:
  0 success (including idempotent re-runs)
  2 usage error
  3 guard/conflict (managed dir conflict, hash conflict, pin mismatch)
  4 external dependency failure (git/gh missing or `gh skill install` failed)
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
import math
import os
import shutil
import struct
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEMPLATE_DIR.parents[1]
MANIFEST = TEMPLATE_DIR / "manifest.json"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CONFLICT = 3
EXIT_EXTERNAL = 4

AUDIO_MANIFEST_SCHEMA = "workspace-audio-manifest/v1"

# Deterministic fixture ground truth (same construction as
# agentic/audio-probe/probe/make_fixture.py, issue #30):
#   fixture-a.wav : 3 tones, ascending pitch (440, 660, 880 Hz)
#   fixture-b.wav : 3 tones, descending pitch (880, 660, 440 Hz)
FIXTURE_RATE = 8000
FIXTURE_TONE_SECONDS = 0.30
FIXTURE_GAP_SECONDS = 0.20
FIXTURE_AMPLITUDE = 12000
FIXTURES = {
    "fixture-a.wav": [440.0, 660.0, 880.0],
    "fixture-b.wav": [880.0, 660.0, 440.0],
}

# 模板文件：(模板内相对路径, 部署后相对路径)。config.example.json 放在模板根而
# 不是 template/local/ 下，否则会被模板自带 .gitignore 的 `local/` 规则忽略。
TEMPLATE_FILES = [
    ("AGENTS.md", "AGENTS.md"),
    (".gitignore", ".gitignore"),
    (os.path.join(".pi", "settings.json"), os.path.join(".pi", "settings.json")),
    ("run.ps1", "run.ps1"),
    ("run.sh", "run.sh"),
    ("config.example.json", os.path.join("local", "config.example.json")),
]


class BootstrapError(RuntimeError):
    def __init__(self, code: str, message: str, exit_code: int = EXIT_CONFLICT) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
        channels = handle.getnchannels()
    return {
        "duration_s": round(frames / rate, 6) if rate else None,
        "sample_rate": rate,
        "channels": channels,
        "probe": "wave",
    }


def ffprobe_info(path: Path, ffprobe: str | None) -> dict | None:
    if not ffprobe:
        return None
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration:stream=sample_rate,channels",
             "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60, shell=False,
        )
        data = json.loads(proc.stdout or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    if proc.returncode != 0:
        return None
    streams = data.get("streams") or [{}]
    duration = data.get("format", {}).get("duration")
    try:
        duration_s = round(float(duration), 6) if duration is not None else None
    except (TypeError, ValueError):
        duration_s = None
    try:
        sample_rate = int(streams[0].get("sample_rate")) if streams[0].get("sample_rate") else None
    except (TypeError, ValueError):
        sample_rate = None
    try:
        channels = int(streams[0].get("channels")) if streams[0].get("channels") else None
    except (TypeError, ValueError):
        channels = None
    return {"duration_s": duration_s, "sample_rate": sample_rate, "channels": channels, "probe": "ffprobe"}


def audio_info(path: Path) -> dict:
    """Metadata for the manifest: sha256 / bytes / duration / sample rate.

    WAV is parsed with the standard library; other containers use ffprobe when
    available. Missing metadata is recorded as null with probe="unavailable" —
    never guessed.
    """
    info: dict = {
        "name": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "format": path.suffix.lower().lstrip(".") or None,
    }
    parsed: dict | None = None
    if info["format"] == "wav":
        try:
            parsed = wav_info(path)
        except (wave.Error, EOFError, OSError):
            parsed = None
    if parsed is None:
        parsed = ffprobe_info(path, shutil.which("ffprobe"))
    if parsed is None:
        parsed = {"duration_s": None, "sample_rate": None, "channels": None, "probe": "unavailable"}
    info.update(parsed)
    return info


def render_fixture(freqs: list[float]) -> bytes:
    samples: list[int] = []
    for freq in freqs:
        n_tone = int(FIXTURE_RATE * FIXTURE_TONE_SECONDS)
        samples.extend(
            int(FIXTURE_AMPLITUDE * math.sin(2.0 * math.pi * freq * i / FIXTURE_RATE))
            for i in range(n_tone)
        )
        samples.extend([0] * int(FIXTURE_RATE * FIXTURE_GAP_SECONDS))
    return struct.pack(f"<{len(samples)}h", *samples)


def write_fixture(path: Path, freqs: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = render_fixture(freqs)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(FIXTURE_RATE)
        handle.writeframes(frames)


class Bootstrap:
    def __init__(self, workspace: Path, args: argparse.Namespace) -> None:
        self.ws = workspace
        self.args = args
        self.steps: list[dict] = []

    def log(self, status: str, message: str) -> None:
        print(f"[{status}] {message}")
        self.steps.append({"status": status, "message": message})

    def preflight(self) -> None:
        if sys.version_info < (3, 11):
            raise BootstrapError("E_PYTHON", "需要 Python 3.11+（当前版本过旧）", EXIT_EXTERNAL)
        if not shutil.which("git"):
            raise BootstrapError("E_GIT_MISSING", "未找到 git：gh skill 的 project 安装依赖项目 git 根目录，请先安装 Git", EXIT_EXTERNAL)
        if not self.args.skip_skill and not shutil.which("gh"):
            raise BootstrapError(
                "E_GH_MISSING",
                "未找到 gh（GitHub CLI）：安装 skill 需要它。安装后重试，或用 --skip-skill 跳过（之后需手动安装 skill）",
                EXIT_EXTERNAL,
            )
        self.log("ok", f"preflight (python {sys.version.split()[0]}, git{', gh' if shutil.which('gh') else ''})")

    def prepare_dirs(self) -> None:
        ws = self.ws
        marker = ws / "local" / ".bootstrap.json"
        if ws.exists() and any(ws.iterdir()) and not marker.exists():
            raise BootstrapError(
                "E_WS_UNMANAGED",
                f"目标目录非空且不是本工具管理的 workspace（缺少 local/.bootstrap.json）：{ws}\n"
                "  请换一个空目录或人工确认后处理，避免覆盖用户文件。",
            )
        for rel in ("audio/inputs", "outputs/runs", "local", "sessions", ".pi/extensions", ".pi/skills"):
            (ws / rel).mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text(json.dumps({
                "schema": "workspace-bootstrap-marker/v1",
                "created_at": utc_now(),
                "tool": "agentic/workspace-template/bootstrap.py",
                "manifest": json.loads(MANIFEST.read_text(encoding="utf-8")).get("schema"),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.log("ok", f"workspace layout: {ws}")

    def ensure_git_repo(self) -> None:
        if (self.ws / ".git").exists():
            self.log("skip", "git 仓库已存在（gh skill project scope 需要 git 根目录）")
            return
        proc = subprocess.run(["git", "init", "-q"], cwd=str(self.ws), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", shell=False)
        if proc.returncode != 0:
            raise BootstrapError("E_GIT_INIT", f"git init 失败：{(proc.stderr or proc.stdout).strip()[:200]}", EXIT_EXTERNAL)
        self.log("ok", "git init（本地仓库，仅用于 gh skill project 安装与 ignore 防护；不要 push）")

    def copy_template_files(self) -> None:
        for src_rel, dest_rel in TEMPLATE_FILES:
            src = TEMPLATE_DIR / "template" / src_rel
            dest = self.ws / dest_rel
            if not src.is_file():
                raise BootstrapError("E_TEMPLATE_MISSING", f"模板文件缺失：{src}", EXIT_EXTERNAL)
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                self.log("ok", f"写入 {dest_rel}")
            else:
                same = sha256_file(dest) == sha256_file(src)
                if same:
                    self.log("skip", f"{dest_rel} 已存在且与模板一致")
                else:
                    self.log("kept", f"{dest_rel} 已存在且被用户修改过 —— 不覆盖（如需新版请人工合并）")

    def install_bridge(self) -> None:
        """Deploy the frozen bridge files (issue #30 interface = BOTH files).
        A differing copy is a conflict: never silently overwritten."""
        manifest = read_manifest()
        for spec in manifest["bridge"]["files"]:
            pinned = spec["sha256"]
            source = REPO_ROOT / spec["source"]
            if not source.is_file():
                raise BootstrapError("E_BRIDGE_SOURCE", f"桥接文件缺失：{source}", EXIT_EXTERNAL)
            source_hash = sha256_file(source)
            if source_hash != pinned:
                raise BootstrapError(
                    "E_BRIDGE_PIN",
                    f"{source.name} 内容与 manifest.json 冻结 pin 不一致（pin {pinned[:12]}…, 实际 {source_hash[:12]}…）。\n"
                    "  桥接接口在 issue #30 README §9 冻结；请先确认接口变更是否被 review。",
                )
            dest = self.ws / spec["install_path"]
            if dest.exists():
                dest_hash = sha256_file(dest)
                if dest_hash == pinned:
                    self.log("skip", f"{spec['install_path']} 已存在且与 pin 一致")
                    continue
                if not self.args.force_bridge:
                    raise BootstrapError(
                        "E_BRIDGE_CONFLICT",
                        f"{spec['install_path']} 与冻结 pin 不一致（已有 {dest_hash[:12]}…）。\n"
                        "  不静默覆盖；确认后用 --force-bridge 显式替换。",
                    )
                shutil.copyfile(source, dest)
                self.log("ok", f"{spec['install_path']} 已用 --force-bridge 替换为 pin 版本")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            self.log("ok", f"{spec['install_path']}（sha256 {pinned[:12]}…，issue #30 冻结接口）")

    def install_skill(self) -> None:
        skill_dir = self.ws / ".pi" / "skills" / "sam-audio"
        manifest = read_manifest()
        expected = [self.ws / rel for rel in manifest["toolbox"]["installed_files"]]
        if (skill_dir / "SKILL.md").is_file():
            missing = [p for p in expected if not p.is_file()]
            if missing:
                raise BootstrapError(
                    "E_SKILL_BROKEN",
                    f"sam-audio skill 已安装但缺文件：{', '.join(str(p.relative_to(self.ws)) for p in missing)}\n"
                    "  请删除 .pi/skills/sam-audio 后重跑 bootstrap（显式重装，不做静默修补）。",
                )
            self.log("skip", "sam-audio skill 已安装（pin 安装不做覆盖更新）")
            return
        if self.args.skip_skill:
            self.log("kept", "跳过 skill 安装（--skip-skill）；之后请手动运行 manifest.json 中的 install_command")
            return
        command = manifest["toolbox"]["install_command"].split()
        proc = subprocess.run(command, cwd=str(self.ws), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", shell=False, timeout=300)
        tail = ((proc.stdout or "") + (proc.stderr or "")).strip()[-400:]
        if proc.returncode != 0 or not (skill_dir / "SKILL.md").is_file():
            raise BootstrapError(
                "E_SKILL_INSTALL",
                f"`gh skill install` 失败（exit {proc.returncode}）。输出尾部：\n{tail}\n"
                "  常见原因：gh 未登录（gh auth login）、无网络、或 gh 版本过旧（skill 为 preview）。",
                EXIT_EXTERNAL,
            )
        self.log("ok", f"gh skill install sam-audio @{manifest['toolbox']['pin'][:12]}… -> .pi/skills/sam-audio")

    def write_local_config(self) -> None:
        dest = self.ws / "local" / "config.json"
        example = TEMPLATE_DIR / "template" / "config.example.json"
        wanted = {"sam_root": self.args.sam_root or "", "sam_python": self.args.sam_python or "", "ffmpeg": self.args.ffmpeg or ""}
        if dest.exists() and not self.args.update_config:
            self.log("kept", "local/config.json 已存在 —— 不覆盖（改用 --update-config 显式更新）")
            return
        if dest.exists() and self.args.update_config:
            try:
                current = json.loads(dest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = {}
            for key, value in wanted.items():
                if value:
                    current[key] = value
            dest.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.log("ok", "local/config.json 已按 --update-config 更新")
            return
        config = json.loads(example.read_text(encoding="utf-8"))
        for key, value in wanted.items():
            if value:
                config[key] = value
        dest.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.log("ok", "local/config.json 创建（本机路径只留在本地，不提交）")

    def _load_audio_manifest(self) -> dict:
        path = self.ws / "audio" / "inputs" / "manifest.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("schema") == AUDIO_MANIFEST_SCHEMA and isinstance(data.get("entries"), list):
                    return data
            except (OSError, json.JSONDecodeError):
                pass
            raise BootstrapError("E_AUDIO_MANIFEST", f"audio/inputs/manifest.json 损坏或 schema 不符：{path}")
        return {"schema": AUDIO_MANIFEST_SCHEMA, "entries": []}

    def import_audio(self, sources: list[Path]) -> None:
        manifest = self._load_audio_manifest()
        entries = {e["name"]: e for e in manifest["entries"]}
        inputs = self.ws / "audio" / "inputs"
        for src in sources:
            if not src.is_file():
                raise BootstrapError("E_AUDIO_NOT_FOUND", f"输入音频不存在：{src}", EXIT_USAGE)
            dest = inputs / src.name
            new_info = audio_info(src)
            if dest.exists():
                existing_hash = sha256_file(dest)
                if existing_hash == new_info["sha256"]:
                    self.log("skip", f"audio/inputs/{src.name} 已存在且 hash 一致（幂等）")
                else:
                    raise BootstrapError(
                        "E_HASH_CONFLICT",
                        f"audio/inputs/{src.name} 已存在但内容不同（existing {existing_hash[:12]}…, incoming {new_info['sha256'][:12]}…）。\n"
                        "  不覆盖用户音频；请换文件名导入或人工确认后处理。",
                    )
            else:
                shutil.copyfile(src, dest)
                self.log("ok", f"导入 audio/inputs/{src.name}（{new_info['bytes']} bytes, {new_info['probe']}）")
            record = dict(new_info)
            record["source"] = "fixture" if src.resolve() in self._fixture_paths() else "import"
            entries[src.name] = record
        manifest["entries"] = sorted(entries.values(), key=lambda e: e["name"])
        (inputs / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.log("ok", f"audio/inputs/manifest.json 更新（{len(manifest['entries'])} 条：sha256/时长/采样率）")

    def _fixture_paths(self) -> list[Path]:
        return [(self.ws / "audio" / "inputs" / name).resolve() for name in FIXTURES]

    def generate_fixtures(self) -> list[Path]:
        inputs = self.ws / "audio" / "inputs"
        created: list[Path] = []
        for name, freqs in FIXTURES.items():
            dest = inputs / name
            if dest.exists():
                self.log("skip", f"audio/inputs/{name} 已存在 —— 不覆盖（fixture 也遵守不覆盖规则）")
            else:
                write_fixture(dest, freqs)
                self.log("ok", f"生成 fixture audio/inputs/{name}（真值：{'-'.join(str(int(f)) for f in freqs)} Hz）")
            created.append(dest)
        return created

    def write_report(self) -> None:
        report = {
            "schema": "workspace-bootstrap-report/v1",
            "generated_at": utc_now(),
            "workspace": str(self.ws),
            "steps": self.steps,
            "note": "本地证据（含本机绝对路径），不提交 git；公开文档只用占位符。",
        }
        (self.ws / "local" / "bootstrap-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def run(self) -> int:
        self.preflight()
        self.prepare_dirs()
        self.ensure_git_repo()
        self.copy_template_files()
        self.install_bridge()
        self.install_skill()
        self.write_local_config()
        fixtures: list[Path] = []
        if self.args.fixture:
            fixtures = self.generate_fixtures()
        sources = [Path(p) for p in self.args.audio] + fixtures
        if sources:
            self.import_audio(sources)
        else:
            self.log("kept", "未提供 --audio/--fixture：audio/inputs/ 保持为空（可随时重跑导入）")
        self.write_report()
        print("\n下一步：")
        print(f"  1) 检查 local/config.json（SAM checkout 路径；只留本地）")
        print(f"  2) python agentic/workspace-template/doctor.py --workspace \"{self.ws}\"")
        print(f"  3) 启动 Pi：run.ps1（PowerShell）或 run.sh（git-bash），均自带 --approve（进程级 project trust）")
        return EXIT_OK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bootstrap.py", description="部署 Pi 音频分析 workspace（issue #31）")
    parser.add_argument("--workspace", default=str(REPO_ROOT / ".local" / "agentic" / "workspace"),
                        help="workspace 目录（默认 <repo>/.local/agentic/workspace，本地忽略目录）")
    parser.add_argument("--audio", action="append", default=[], metavar="FILE",
                        help="导入音频（可重复）；同名不同 hash 会显式失败")
    parser.add_argument("--fixture", action="store_true", help="生成自有确定性 WAV fixture（fixture-a/b.wav）")
    parser.add_argument("--sam-root", default="", help="SAM checkout 根目录（写入 local/config.json）")
    parser.add_argument("--sam-python", default="", help="SAM 环境解释器（写入 local/config.json）")
    parser.add_argument("--ffmpeg", default="", help="ffmpeg/ffprobe 可执行文件路径（可选，写入 local/config.json）")
    parser.add_argument("--update-config", action="store_true", help="显式更新已存在的 local/config.json")
    parser.add_argument("--force-bridge", action="store_true", help="显式替换不一致的 audio-bridge.ts（默认拒绝覆盖）")
    parser.add_argument("--skip-skill", action="store_true", help="跳过 gh skill 安装（离线/测试用）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code not in (0, None) else EXIT_OK
    workspace = Path(args.workspace).expanduser().resolve()
    try:
        return Bootstrap(workspace, args).run()
    except BootstrapError as exc:
        print(f"[fail] {exc.code}: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # readable error, never a bare traceback
        print(f"[fail] E_INTERNAL: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return EXIT_CONFLICT


if __name__ == "__main__":
    sys.exit(main())
