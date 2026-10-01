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
import os
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


# 支持的 WAV 编码（其余一律拒绝，不猜）：PCM 整型 16/24/32-bit、IEEE float 32/64-bit；
# WAVE_FORMAT_EXTENSIBLE(0xFFFE) 解析子格式后按同一规则。
WAV_PCM, WAV_FLOAT, WAV_EXTENSIBLE = 1, 3, 0xFFFE
_SUPPORTED = {WAV_PCM: (16, 24, 32), WAV_FLOAT: (32, 64)}


def wav_info(path: Path) -> dict:
    """解析 WAV 头（不依赖 wave 模块对 float/非 PCM 的限制）。

    返回 ``{format, container_tag, tag, channels, sample_rate, bits_per_sample,
    block_align, frames, duration_s}``；``tag`` 是**实际编码**（EXTENSIBLE 已解析子格式），
    ``container_tag`` 是原始 fmt tag。不受支持的编码 / 损坏头部受控拒绝（WavError），
    绝不按另一种编码猜读。"""
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavError(f"E_WAV_FORMAT: 不是 RIFF/WAVE 文件：{path.name}")
    pos = 12
    fmt = None
    data_bytes = 0
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if pos + 8 + size > len(data):
            raise WavError(f"E_WAV_FORMAT: chunk 超出文件长度（损坏）：{path.name}")
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            if len(body) < 16:
                raise WavError(f"E_WAV_FORMAT: fmt chunk 太短：{path.name}")
            container_tag, channels, sr, _brate, block_align, bits = struct.unpack_from("<HHIIHH", body, 0)
            tag = container_tag
            if container_tag == WAV_EXTENSIBLE:
                if len(body) < 26:
                    raise WavError(f"E_WAV_FORMAT: EXTENSIBLE fmt 太短：{path.name}")
                tag = struct.unpack_from("<H", body, 24)[0]  # 子格式前 2 字节
            fmt = {"container_tag": container_tag, "tag": tag, "channels": channels,
                   "sample_rate": sr, "block_align": block_align, "bits_per_sample": bits}
        elif cid == b"data":
            data_bytes = len(body)
        pos += 8 + size + (size % 2)
    if fmt is None:
        raise WavError(f"E_WAV_FORMAT: 缺少 fmt chunk：{path.name}")
    tag = fmt["tag"]
    if tag not in _SUPPORTED:
        raise WavError(
            f"E_WAV_CODEC: 不支持的 WAV 编码 tag={tag}（仅 PCM 16/24/32-bit 与 "
            f"IEEE float 32/64-bit）：{path.name}")
    bits = fmt["bits_per_sample"]
    if bits not in _SUPPORTED[tag]:
        raise WavError(f"E_WAV_BITS: tag={tag} 不支持的位深 {bits}：{path.name}")
    channels, sr, block_align = fmt["channels"], fmt["sample_rate"], fmt["block_align"]
    if channels < 1 or sr < 1:
        raise WavError(f"E_WAV_FORMAT: 非法 channels/sample_rate：{path.name}")
    expected_align = channels * (bits // 8)
    if block_align != expected_align:
        raise WavError(
            f"E_WAV_FORMAT: block_align={block_align} 与 channels×bits/8={expected_align} 不符"
            f"（损坏/不受支持）：{path.name}")
    if data_bytes < block_align:
        raise WavError(f"E_WAV_FORMAT: 没有完整采样帧：{path.name}")
    nframes = data_bytes // block_align
    return {
        "format": "wav",
        "container_tag": fmt["container_tag"],
        "tag": tag,
        "channels": channels,
        "sample_rate": sr,
        "bits_per_sample": bits,
        "block_align": block_align,
        "frames": nframes,
        "duration_s": nframes / sr,
    }


def read_mono(path: Path) -> tuple[list[float], int]:
    """读取 WAV 为单声道 float 样本（多声道取均值）。返回 (samples, sample_rate)。

    解码严格按 ``wav_info`` 报告的**实际编码 tag**（PCM 整型 / IEEE float）；
    float32 不会再被当 int32 读（旧版 bug 已修）。不受支持的编码在这里也拒绝。"""
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

    tag, ch, bits = info["tag"], info["channels"], info["bits_per_sample"]
    frames = info["frames"]
    count = frames * ch
    if tag == WAV_FLOAT and bits == 32:
        vals = list(struct.unpack_from(f"<{count}f", body, 0))
        out_vals = [float(v) for v in vals]
    elif tag == WAV_FLOAT and bits == 64:
        vals = list(struct.unpack_from(f"<{count}d", body, 0))
        out_vals = [float(v) for v in vals]
    elif tag == WAV_PCM and bits == 16:
        vals = struct.unpack_from(f"<{count}h", body, 0)
        out_vals = [v / 32768.0 for v in vals]
    elif tag == WAV_PCM and bits == 32:
        vals = struct.unpack_from(f"<{count}i", body, 0)
        out_vals = [v / 2147483648.0 for v in vals]
    elif tag == WAV_PCM and bits == 24:
        raw = body[:count * 3]
        scale = float(1 << 23)
        out_vals = []
        for i in range(0, len(raw) - 2, 3):
            b0, b1, b2 = raw[i], raw[i + 1], raw[i + 2]
            v = b0 | (b1 << 8) | (b2 << 16)
            if v >= 1 << 23:
                v -= 1 << 24
            out_vals.append(v / scale)
    else:  # pragma: no cover - wav_info 已拒
        raise WavError(f"E_WAV_CODEC: 不支持的组合 tag={tag} bits={bits}：{path.name}")

    if ch == 1:
        return out_vals, info["sample_rate"]
    out = []
    for i in range(frames):
        acc = 0.0
        for c in range(ch):
            acc += out_vals[i * ch + c]
        out.append(acc / ch)
    return out, info["sample_rate"]


def write_wav(path: Path, samples: list[float], sample_rate: int, channels: int = 1,
              fmt: str = "pcm16") -> None:
    """写 WAV（测试 fixture 用）：fmt ∈ {pcm16, pcm24, pcm32, float32}。样本按 [-1,1] 处理
    （float32 不裁幅，保留原始值）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pcm16":
        tag, bits = WAV_PCM, 16
        import array
        arr = array.array("h", (max(-32768, min(32767, int(s * 32768))) for s in samples))
        payload = arr.tobytes()
    elif fmt == "pcm32":
        tag, bits = WAV_PCM, 32
        payload = struct.pack(f"<{len(samples)}i",
                              *(max(-2147483648, min(2147483647, int(s * 2147483648)))
                                for s in samples))
    elif fmt == "pcm24":
        tag, bits = WAV_PCM, 24
        chunks = []
        for s in samples:
            v = max(-(1 << 23), min((1 << 23) - 1, int(s * (1 << 23))))
            chunks.append(struct.pack("<i", v)[:3])
        payload = b"".join(chunks)
    elif fmt == "float32":
        tag, bits = WAV_FLOAT, 32
        payload = struct.pack(f"<{len(samples)}f", *samples)
    else:
        raise ValueError(f"E_WAV_CODEC: write_wav 不支持 fmt={fmt}")
    block_align = channels * (bits // 8)
    header = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, tag, channels, sample_rate,
                                    sample_rate * block_align, block_align, bits)
    header += b"data" + struct.pack("<I", len(payload))
    path.write_bytes(header + payload)


def write_wav16(path: Path, samples: list[float], sample_rate: int, channels: int = 1) -> None:
    """写 16-bit PCM WAV（唯一实现是 ``write_wav(fmt="pcm16")``；此处只是显式兼容包装）。"""
    write_wav(path, samples, sample_rate, channels=channels, fmt="pcm16")


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


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}(?![\s\S])")


def is_safe_run_id(value: str) -> tuple[bool, str]:
    """run id 白名单（防路径拼接/模板注入）：字母数字开头，仅 ``[A-Za-z0-9._-]``，≤64 字符。"""
    if not isinstance(value, str) or not _SAFE_ID.match(value):
        return False, "run id 必须匹配 ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
    if value.endswith("."):
        return False, "run id 不得以点结尾"
    if value.split(".")[0].upper() in _RESERVED:
        return False, "Windows 保留设备名"
    return True, "ok"


def is_safe_leaf_name(value: str) -> tuple[bool, str]:
    """单段文件名（不含路径分隔符），用于 clip 名等写入 audio 根的文件。"""
    ok, why = is_safe_rel_path(value)
    if not ok:
        return ok, why
    if "/" in value:
        return False, "必须是单段文件名（不含 /）"
    return True, "ok"


def ensure_contained(base: Path, name: str) -> Path:
    """写前守卫：解析 base/name 的 realpath 并确认仍在 base 内；否则抛 ValueError。"""
    base_real = Path(base).resolve()
    target = Path(base_real / name)
    resolved = Path(os.path.realpath(target))
    try:
        resolved.relative_to(base_real)
    except ValueError:
        raise ValueError(f"E_PATH: 解析后越出根目录：{name!r}") from None
    return resolved


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
# 文件系统绝对路径（自由文本 / JSON 转义都要能抓）：Windows 盘符、UNC、常见 POSIX 绝对根。
# 负向后顾排除 URL scheme（`http://…` 的 `p:/`）与标识符内部；相对引用（stems/x.wav）与 URL 不命中。
_ABS_PATH_PATTERNS = [
    (re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'<>|]*"), "Windows 绝对路径"),
    (re.compile(r"\\\\[A-Za-z0-9._$-]+[\\/]"), "UNC 路径"),
    # POSIX：常见一级根（含 /tmp /etc /var 等）或 ≥3 段的绝对路径；
    # 后顾排除 scheme（http://）、上级引用（../）、双斜杠（//）、占位符续接（<WS>/…）与词内；
    # 相对引用与 URL 不命中
    (re.compile(r"(?<![A-Za-z0-9_:.>])/"
                r"(?:home|Users|root|tmp|etc|var|usr|opt|mnt|media|srv|export|private|workspace|data|storage|proc)/"
                r"[^\s\"'<>|]*"), "POSIX 绝对路径"),
    (re.compile(r"(?<![A-Za-z0-9_:/.>])/(?:[^/\s\"'<>|]+/){2,}[^/\s\"'<>|]*"),
     "POSIX 绝对路径（多段）"),
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
    """隐私扫描：返回发现项（空列表 = 干净）。用于公开前的自动把关。

    覆盖：key/token 形态、长 base64、**文件系统绝对路径**（Windows 盘符 / UNC / POSIX 绝对根，
    含 JSON 转义后的 `C:\\…` 形态）；合法的相对引用（`stems/x.wav`）与 URL（`https://…`）不命中。"""
    findings = []
    for pattern, label in _SECRET_PATTERNS + _ABS_PATH_PATTERNS:
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


WRAPPER_SCHEMA = "audio-toolbox.sam/v1"


def parse_first_json(text: str, required_key: str | None = None) -> dict | None:
    """解析文本中第一个 JSON 对象（可要求含指定键）；解析不到返回 None。

    用于在**截断诊断文本之前**提取结构化成功载荷（wrapper 恰好向 stdout 打印一个 JSON）。"""
    text = str(text)
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and (required_key is None or required_key in obj):
            return obj
    except (ValueError, TypeError):
        pass
    dec = json.JSONDecoder()
    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            return None
        try:
            obj, _ = dec.raw_decode(text[start:])
        except ValueError:
            obj = None
        if isinstance(obj, dict) and (required_key is None or required_key in obj):
            return obj
        idx = start + 1


def call_facts(tool: str, command: str, full_text: str, is_error: bool) -> dict:
    """结构化成功事实（在截断前解析；不只信工具 isError）。

    * ``audio_attach``：必须是可解析 JSON 载荷且 ``attached`` 真、``mime`` 为 ``audio/*``、
      ``bytes`` 为 >0 整数、工具未报错 → ``attach_success=True``（纯文本“mime bytes”不算）；
    * SAM wrapper：必须是含 ``schema="audio-toolbox.sam/v1"`` 的 JSON 载荷，``action`` 精确为
      ``sam.separate`` / ``sam.dry-run``，``ok=true`` 且 ``exit_code=0`` 且工具未报错才算成功；
      shell 用 ``;`` / ``echo`` / ``|| true`` 掩盖的 wrapper 失败**不**算成功。"""
    facts: dict = {}
    if tool == "audio_attach":
        payload = parse_first_json(full_text)
        mime = (payload or {}).get("mime")
        nbytes = (payload or {}).get("bytes")
        ok = (not is_error and payload is not None and bool(payload.get("attached"))
              and isinstance(mime, str) and mime.startswith("audio/")
              and isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes > 0)
        facts["attach_success"] = ok
        facts["attach_mime"] = mime if isinstance(mime, str) else None
        facts["attach_bytes"] = nbytes if isinstance(nbytes, int) and not isinstance(nbytes, bool) else None
        if payload is None:
            facts["attach_error"] = "非结构化载荷（结果文本无法解析为 JSON）"
    elif tool == "bash" and "audio_toolbox.py" in command and "separate" in command:
        payload = parse_first_json(full_text, required_key="schema")
        if payload is not None and payload.get("schema") == WRAPPER_SCHEMA:
            run_dir = payload.get("run_dir")
            outputs = payload.get("outputs") or []
            facts.update({
                "sam_wrapper_schema": True,
                "sam_action": payload.get("action"),
                "sam_ok": payload.get("ok"),
                "sam_exit_code": payload.get("exit_code"),
                "sam_run_dir_name": str(Path(str(run_dir)).name) if run_dir else None,
                "sam_output_names": [Path(str(p)).name for p in outputs][:8],
                "sam_report_present": bool(payload.get("report")),
            })
        else:
            facts["sam_wrapper_schema"] = False
    return facts


def extract_trace(events: list[dict]) -> dict:
    """从事件流抽取 trace 摘要（工具调用链、用量、模型、终态文本）。"""
    calls: list[dict] = []
    usage_total = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                   "totalTokens": 0, "cost_total": 0.0}
    models: set[str] = set()
    providers: set[str] = set()
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
            full_text = _content_texts((rec.get("result") or {}).get("content"))
            entry["isError"] = bool(rec.get("isError"))
            entry["result_head"] = full_text[:300]
            # 结构化成功事实：在截断诊断文本之前从完整结果提取（wrapper/attach 载荷）
            entry["facts"] = call_facts(entry.get("tool", ""), entry.get("args_summary", ""),
                                        full_text, entry["isError"])
        elif rtype == "turn_start":
            turns += 1
        elif rtype == "message_end":
            msg = rec.get("message") or {}
            if msg.get("role") == "assistant":
                if msg.get("model"):
                    models.add(str(msg["model"]))
                if msg.get("provider"):
                    providers.add(str(msg["provider"]))
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
                usage_total["cacheRead"] += int(usage.get("cacheRead") or 0)
                usage_total["cacheWrite"] += int(usage.get("cacheWrite") or 0)
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
        "providers": sorted(providers),
        "tool_calls": calls,
        "tool_calls_total": len(calls),
        "tool_calls_by_tool": by_tool,
        "usage": {
            "input": usage_total["input"],
            "output": usage_total["output"],
            "cacheRead": usage_total["cacheRead"],
            "cacheWrite": usage_total["cacheWrite"],
            "totalTokens": usage_total["totalTokens"],
            "cost_total": round(usage_total["cost_total"], 6),
            "reconcile": "input + output + cacheRead + cacheWrite == totalTokens（逐 assistant 消息累计）",
            "note": "Pi 记账近似值，非账单真值",
        },
        "final_text_tail": final_text[-2000:],
        "last_assistant_stop": last_assistant_stop,
        "provider_failures": provider_failures,
    }


def classify_separations(trace: dict) -> dict:
    """把 SAM separate 相关调用分成四类：dry_run / real_success / real_failed / unknown。

    判定**只信截断前解析出的结构化载荷**（``call_facts``）：wrapper 载荷必须含
    ``schema="audio-toolbox.sam/v1"``、``action`` 精确、``ok=true``、``exit_code=0`` 且工具未报错
    才算 ``real_success``；shell 掩盖的失败 / ok=false / exit≠0 → ``real_failed``；无结构化载荷 →
    ``unknown``（不计成功）。`--dry-run` / ``sam.dry-run`` → ``dry_run``。"""
    buckets: dict[str, list[dict]] = {"dry_run": [], "real_success": [], "real_failed": [], "unknown": []}
    for call in trace.get("tool_calls", []):
        cmd = call.get("args_summary", "")
        head = call.get("result_head", "")
        if "audio_toolbox.py" not in cmd or "separate" not in cmd:
            continue
        facts = call.get("facts") or {}
        entry = {"command": cmd, "isError": call.get("isError", False),
                 "result_head": head[:200], "facts": facts}
        if "--dry-run" in cmd or facts.get("sam_action") == "sam.dry-run" or "sam.dry-run" in head:
            buckets["dry_run"].append(entry)
            continue
        if facts.get("sam_wrapper_schema") is not True:
            buckets["unknown"].append(entry)      # 无结构化 wrapper 载荷 → 不算成功
        elif (facts.get("sam_action") == "sam.separate"
              and facts.get("sam_ok") is True
              and facts.get("sam_exit_code") == 0
              and not call.get("isError", False)):
            buckets["real_success"].append(entry)
        else:
            buckets["real_failed"].append(entry)
    return buckets


def count_real_separations(trace: dict) -> list[dict]:
    """**成功**的真实 SAM 分离调用（结构化载荷证明成功；不含 dry-run/失败/unknown）。"""
    return classify_separations(trace)["real_success"]


# ------------------------------------------------------------ 通用小工具


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
