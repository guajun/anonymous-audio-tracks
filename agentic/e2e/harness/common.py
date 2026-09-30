#!/usr/bin/env python3
"""common.py — agentic/e2e 共享工具（issue #33）。

纯标准库、离线、不联网、不接触 API key。功能分四类：

1. 音频读写（WAV RIFF 手工解析：PCM 16/24/32-bit 与 float32，torchcodec 产物可读）；
2. 事件流解析（Pi ``--mode json`` JSONL → 工具调用 trace / 用量 / 终态文本）；
3. 路径安全与包含检查（复用 #32 E_PATH 的安全相对路径语义 + 解析后仍在根目录内）；
4. 隐私扫描与脱敏（绝对路径 / 盘符 / UNC / key 形态 / 长 base64 → 标记或占位符）。

真实 / mock 边界：本文件不产生任何音频分析结论，只搬运与校验数据；fixture/mock
标注由 fixtures/make_mock_fixture.py 与 validate_result.py 显式记录，绝不与真实运行混淆。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

# Windows 控制台/管道默认 GBK 会把中文写成乱码；统一强制 UTF-8 输出。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SCHEMA_VERSION = "agentic-audio-tracks/v1"

# 冻结 pin（与 agentic/workspace-template/manifest.json、agentic/toolbox/manifest.json 一致）
TOOLBOX_PIN = "dfbc40a9541f686207b65b93b1332bb505654261"
SAM_UPSTREAM_COMMIT = "c603de8794cc16880dc01be0f1e868f6c2845417"
BRIDGE_SHA256 = "dac9abe552cf47c5963b98273f8588e5ca1f47475c939619b5ef2ad94bd833de"
RESEARCH_MODEL = "google/gemini-3.8-flash"      # 真正音乐 Agent（issue #30 冻结）
IMPL_MODEL = "openrouter/xiaomi/mimo-v2.6-pro"  # 实施 worker（issue #33 执行约定）

AUDIO_ROOT_REL = Path("audio") / "inputs"   # 桥接受控根目录（= PI_AUDIO_BRIDGE_ROOT）
E2E_OUTPUTS_REL = Path("outputs") / "e2e"
LATEST_NAME = "LATEST.txt"


# ---------------------------------------------------------------- 音频读写


class WavError(ValueError):
    """WAV 解析失败（受控错误信息，不抛裸 traceback）。"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wav_info(path: Path) -> dict:
    """解析 WAV 头（不依赖 wave 模块对 float/非 PCM 的限制）。

    返回 {format, channels, sample_rate, bits_per_sample, frames, duration_s}。
    """
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavError(f"E_WAV_FORMAT: 不是 RIFF/WAVE 文件：{path.name}")
    pos = 12
    fmt = None
    nframes = None
    data_bytes = 0
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            if len(body) < 16:
                raise WavError(f"E_WAV_FORMAT: fmt chunk 太短：{path.name}")
            tag, channels, sr, _brate, block_align, bits = struct.unpack_from("<HHIIHH", body, 0)
            if tag == 0xFFFE and len(body) >= 26:  # WAVE_FORMAT_EXTENSIBLE：取子格式前 2 字节
                tag = struct.unpack_from("<H", body, 24)[0]
            fmt = {"tag": tag, "channels": channels, "sample_rate": sr,
                   "block_align": block_align, "bits_per_sample": bits}
        elif cid == b"data":
            data_bytes = len(body)
        pos += 8 + size + (size % 2)
    if fmt is None:
        raise WavError(f"E_WAV_FORMAT: 缺少 fmt chunk：{path.name}")
    if fmt["block_align"] <= 0:
        raise WavError(f"E_WAV_FORMAT: 非法 block_align：{path.name}")
    nframes = data_bytes // fmt["block_align"]
    if nframes <= 0:
        raise WavError(f"E_WAV_FORMAT: 没有采样数据：{path.name}")
    return {
        "format": "wav",
        "channels": fmt["channels"],
        "sample_rate": fmt["sample_rate"],
        "bits_per_sample": fmt["bits_per_sample"],
        "frames": nframes,
        "duration_s": nframes / fmt["sample_rate"],
    }


def read_mono(path: Path) -> tuple[list[float], int]:
    """读取 WAV 为单声道 float 样本（多声道取均值）。返回 (samples, sample_rate)。"""
    path = Path(path)
    info = wav_info(path)
    data = path.read_bytes()
    # 定位 data chunk
    pos = 12
    body = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if cid == b"data":
            body = data[pos + 8:pos + 8 + size]
            break
        pos += 8 + size + (size % 2)
    if body is None:
        raise WavError(f"E_WAV_FORMAT: 缺少 data chunk：{path.name}")

    tag, ch, bits = info.get("tag", 1), info["channels"], info["bits_per_sample"]
    frames = info["frames"]
    out: list[float] = []
    if bits == 16:
        vals = struct.unpack_from(f"<{frames * ch}h", body, 0)
        scale = 32768.0
    elif bits == 32 and tag == 3:
        vals = struct.unpack_from(f"<{frames * ch}f", body, 0)
        scale = 1.0
    elif bits == 32:
        vals = struct.unpack_from(f"<{frames * ch}i", body, 0)
        scale = 2147483648.0
    elif bits == 24:
        raw = body[: frames * ch * 3]
        vals = []
        for i in range(0, len(raw) - 2, 3):
            b0, b1, b2 = raw[i], raw[i + 1], raw[i + 2]
            v = b0 | (b1 << 8) | (b2 << 16)
            if v >= 1 << 23:
                v -= 1 << 24
            vals.append(v)
        scale = float(1 << 23)
    else:
        raise WavError(f"E_WAV_BITS: 不支持的位深 {bits}：{path.name}")
    if ch == 1:
        out = [v / scale for v in vals]
    else:
        for i in range(frames):
            acc = 0.0
            for c in range(ch):
                acc += vals[i * ch + c]
            out.append(acc / (ch * scale))
    return out, info["sample_rate"]


def write_wav16(path: Path, samples: list[float], sample_rate: int, channels: int = 1) -> None:
    """写 16-bit PCM WAV（用于 fixture / overlay；样本按 [-1,1] 限幅）。"""
    import array

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ints = array.array("h")
    for s in samples:
        v = int(max(-1.0, min(1.0, s)) * 32767)
        ints.append(v)
    payload = ints.tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                    sample_rate * channels * 2, channels * 2, 16)
    header += b"data" + struct.pack("<I", len(payload))
    path.write_bytes(header + payload)


# ---------------------------------------------------------- 路径安全 / 包含


_UNSAFE_CHARS = set('<>:"|?*\\')
_RESERVED = {"CON", "NUL", "PRN", "AUX", "COM1", "COM2", "COM3", "COM4", "COM5",
             "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5"}


def is_safe_rel_path(rel: str) -> tuple[bool, str]:
    """#32 E_PATH 语义的安全相对路径判定（与 schema 校验一致的子集）。"""
    if not isinstance(rel, str) or not rel:
        return False, "路径为空"
    if len(rel) > 512:
        return False, "路径超长"
    if any(ord(ch) < 0x20 for ch in rel):
        return False, "含控制字符"
    if "\\" in rel:
        return False, "含反斜杠"
    if rel.startswith("/"):
        return False, "绝对路径"
    if re.match(r"^[A-Za-z]:", rel):
        return False, "含盘符"
    if "://" in rel:
        return False, "URL 形式"
    for seg in rel.split("/"):
        if seg in ("", ".", ".."):
            return False, "含空段或 . / .. 段"
        if seg != seg.strip() or seg.endswith("."):
            return False, "段首尾空格或结尾点"
        if seg.split(".")[0].upper() in _RESERVED:
            return False, "Windows 保留设备名"
        if any(ch in _UNSAFE_CHARS for ch in seg):
            return False, "含非法字符"
    return True, "ok"


def resolve_within(base: Path, rel: str) -> Path:
    """把安全相对路径解析到 base 内；越界/不安全即抛 ValueError。"""
    ok, why = is_safe_rel_path(rel)
    if not ok:
        raise ValueError(f"E_PATH: {why}: {rel!r}")
    base = Path(base).resolve()
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise ValueError(f"E_PATH: 解析后越出根目录：{rel!r}") from None
    return target


# ------------------------------------------------------------ 脱敏 / 隐私

_ABS_WIN = re.compile(r'[A-Za-z]:[\\/][^\s"\'<>|]*')
_ABS_POSIX = re.compile(r'(?<![\w:.-])/(?:[^/\s"\'<>|]+/)+[^/\s"\'<>|]*')
_SECRET_PATTERNS = [
    (re.compile(r"AIza[0-9A-Za-z_-]{10,}"), "google-api-key 形态"),
    (re.compile(r"(?i)(api[_-]?key|apikey|secret|token|passwd|password)\s*[:=]\s*\S+"), "凭据赋值形态"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{10,}"), "Bearer token 形态"),
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "sk- key 形态"),
    (re.compile(r"[A-Za-z0-9+/]{200,}={0,2}"), "长 base64（疑似音频/权重字节）"),
]


def make_redactor(*roots: Path | str):
    """返回脱敏函数：给定根路径 → 占位符；其余绝对路径只留末段。"""
    mapping = []
    for r in roots:
        if r:
            mapping.append((str(r), None))
    placeholders = ["<WORKSPACE>", "<HOME>", "<REPO>", "<SAM_ROOT>", "<PATH>"]

    def assign(idx: int) -> str:
        return placeholders[idx] if idx < len(placeholders) else f"<ROOT{idx}>"

    fixed = [(p, assign(i)) for i, (p, _) in enumerate(mapping)]
    home = str(Path.home())

    def redact(text: str) -> str:
        text = str(text)
        for path, ph in fixed:
            text = text.replace(path, ph).replace(path.replace("\\", "/"), ph)
        text = text.replace(home, "<HOME>")
        text = _ABS_WIN.sub(lambda m: "<PATH>/" + m.group(0).replace("\\", "/").rstrip("/").split("/")[-1], text)
        text = _ABS_POSIX.sub(lambda m: "<PATH>/" + m.group(0).rstrip("/").split("/")[-1], text)
        return text

    return redact


def find_private(text: str) -> list[str]:
    """隐私扫描：返回发现项（空列表 = 干净）。用于公开前的自动把关。"""
    findings = []
    for pattern, label in _SECRET_PATTERNS:
        if pattern.search(str(text)):
            findings.append(label)
    return findings


# ------------------------------------------------------------ 事件流解析


def load_events(path: Path) -> list[dict]:
    events = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _content_texts(node) -> str:
    return " ".join(c.get("text", "") for c in (node or [])
                   if isinstance(c, dict) and c.get("type") == "text")


def summarize_args(tool: str, args) -> str:
    """把工具参数压成单行摘要（bash 留完整命令行，audio_attach 留 path，其它留键名）。"""
    if not isinstance(args, dict):
        return ""
    if tool == "bash":
        return str(args.get("command", ""))[:2000]
    if tool == "audio_attach":
        return str(args.get("path", ""))
    if tool in ("read", "write", "edit"):
        return str(args.get("path", args.get("file_path", "")))[:200]
    return ",".join(sorted(args.keys()))[:200]


def extract_trace(events: list[dict]) -> dict:
    """从事件流抽取 trace 摘要（工具调用链、用量、模型、终态文本）。"""
    calls: list[dict] = []
    usage_total = {"input": 0, "output": 0, "totalTokens": 0, "cost_total": 0.0}
    models: set[str] = set()
    final_text = ""
    turns = 0
    provider_failures: list[str] = []
    last_assistant_stop = None

    for rec in events:
        rtype = rec.get("type")
        if rtype == "tool_execution_start":
            calls.append({
                "call_id": rec.get("toolCallId"),
                "tool": str(rec.get("toolName")),
                "isError": False,
                "args_summary": summarize_args(str(rec.get("toolName")), rec.get("args")),
                "result_head": "",
            })
        elif rtype == "tool_execution_end":
            entry = next((c for c in calls if c.get("call_id") == rec.get("toolCallId")
                          and c.get("tool") == str(rec.get("toolName"))), None)
            if entry is None:
                entry = {"call_id": rec.get("toolCallId"), "tool": str(rec.get("toolName")),
                         "args_summary": "", "result_head": ""}
                calls.append(entry)
            entry["isError"] = bool(rec.get("isError"))
            entry["result_head"] = _content_texts((rec.get("result") or {}).get("content"))[:300]
        elif rtype == "turn_start":
            turns += 1
        elif rtype == "message_end":
            msg = rec.get("message") or {}
            if msg.get("role") == "assistant":
                if msg.get("model"):
                    models.add(str(msg["model"]))
                last_assistant_stop = msg.get("stopReason")
                if msg.get("stopReason") in ("error", "aborted") or msg.get("errorMessage"):
                    provider_failures.append(
                        f"assistant stopReason={msg.get('stopReason')} {str(msg.get('errorMessage') or '')[:80]}")
                text = _content_texts(msg.get("content"))
                if text.strip():
                    final_text = text
                usage = msg.get("usage") or {}
                usage_total["input"] += int(usage.get("input") or 0)
                usage_total["output"] += int(usage.get("output") or 0)
                usage_total["totalTokens"] += int(usage.get("totalTokens") or 0)
                usage_total["cost_total"] += float(((usage.get("cost") or {}).get("total")) or 0)
        elif rtype == "error":
            provider_failures.append(f"error event: {str(rec.get('message') or rec.get('error') or '')[:120]}")

    by_tool: dict[str, int] = {}
    for c in calls:
        by_tool[c["tool"]] = by_tool.get(c["tool"], 0) + 1
    return {
        "turns": turns,
        "models": sorted(models),
        "tool_calls": calls,
        "tool_calls_total": len(calls),
        "tool_calls_by_tool": by_tool,
        "usage": {
            "input": usage_total["input"],
            "output": usage_total["output"],
            "totalTokens": usage_total["totalTokens"],
            "cost_total": round(usage_total["cost_total"], 6),
            "note": "Pi 记账近似值，非账单真值",
        },
        "final_text_tail": final_text[-2000:],
        "last_assistant_stop": last_assistant_stop,
        "provider_failures": provider_failures,
    }


def count_real_separations(trace: dict) -> list[dict]:
    """从 trace 里挑出**真实** SAM 分离调用。

    判定：命令含 ``audio_toolbox.py ... separate``，且既不是 ``--dry-run``，工具结果也不是
    ``sam.dry-run``（结果里的 action 字段是权威标记，命令文本只是辅助——两者都查，避免长命令
    截断导致 dry-run 被误计为真实分离）。"""
    hits = []
    for call in trace.get("tool_calls", []):
        cmd = call.get("args_summary", "")
        head = call.get("result_head", "")
        if "audio_toolbox.py" not in cmd or "separate" not in cmd:
            continue
        if "--dry-run" in cmd or 'sam.dry-run' in head:
            continue
        hits.append({"command": cmd, "isError": call.get("isError", False),
                     "result_head": head[:200]})
    return hits


# ------------------------------------------------------------ 通用小工具


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
