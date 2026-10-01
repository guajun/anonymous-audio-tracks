#!/usr/bin/env python3
"""residual_chain.py — 残差逐层剥离 sidecar 与链校验（issue #42）。

工作流（``prompt_template_residual.md`` 强制）：原混音 → 优先复用已认可 drums
（target/residual/36 事件，用户局部定性反馈）→ **听 R1** → 自主选下一层 → **SAM ``--audio``
必须是 R1 的 raw residual** → 听 R2 → 再剥或停（近静音 / 只剩伪影 / 无可辨认声层 /
预算到达）。每级在 ``outputs/e2e/<run-id>/stage-chain.json`` 记录 sidecar
（schema ``agentic-e2e-stage-chain/v1``）——**加法 sidecar，不扩展 #32 冻结 result 协议**。

本文件只做**校验 / 派生事实 / 监听副本编码转换**，不生成任何分离 / onset / result 内容：

* ``verify_chain``：把 sidecar 与**真实产物**逐项对照——文件 hash、SAM ``request.json`` 记录的
  实际 input、时间轴（每文件按**自身采样率**的时长 == clip 时长；禁止 trim/shift/量化）、
  监听代理（raw/proxy hash、转换参数、等长、本 run 派生目录）、停止 / 预算规则、基线复用
  完整性（含事件 canonical hash）、trace 里 SAM ``--audio`` 参数链。
  **每一级 input hash 必须 == 前一级 residual 的 raw hash**；任何“各自从原混音独立分离”
  或“PCM16 监听代理当 SAM 输入”都会被判失败（不能误报 sequential）。
* ``make_listen_proxy``：raw residual → 监听副本（PCM16 或逐字节 copy），只做 codec 转换，
  同采样率 / 同声道 / 同帧数，打印 raw/proxy hash + 转换参数记录。
* ``drums_baseline_facts``：从基线 run 派生 drums 复用事实（target/residual/事件 hash）。

事件 canonical hash（复用核对口径）::

    sha256(json.dumps(events, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8"))

用法::

    python residual_chain.py verify --chain <run>/stage-chain.json --run-dir <run> \\
        --clip-json <WS>/local/e2e/clip-<clip>.json --audio-root <WS>/audio/inputs \\
        --baseline-run-dir <WS>/outputs/e2e/<base-run> [--ws <WS>] [--events <events.jsonl>] \\
        [--budget 2] [--result <run>/result.json] [--json]
    python residual_chain.py make-proxy --raw <raw residual.wav> \\
        --out <WS>/audio/inputs/<run-id>-listen/r1-listen.wav --audio-root <WS>/audio/inputs \\
        --run-id <run-id> [--record <run>/listen-proxies.json]
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    WAV_FLOAT, WAV_PCM, WavError, classify_separations, find_private, is_safe_rel_path,
    is_safe_run_id, load_events, read_json, resolve_within, sha256_file, wav_info, write_json,
)

CHAIN_SCHEMA = "agentic-e2e-stage-chain/v1"
CHAIN_VERIFY_SCHEMA = "agentic-e2e-stage-chain-verify/v1"
PROXY_RECORD_SCHEMA = "agentic-e2e-listen-proxies/v1"
STOP_REASONS = ("near-silence", "artifacts-only", "no-identifiable-layer", "uncertain", "budget")
INPUT_KINDS = ("mix", "residual")
FILE_ROOTS = ("audio", "run", "baseline")


# ----------------------------------------------------------------- 小工具


class Checker:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append({"name": name, "pass": bool(ok), "detail": detail})

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c["pass"] for c in self.checks)


def canonical_events_sha256(events: list) -> str:
    """事件数组的稳定 hash（复用核对口径；与 JSON 展示格式无关）。"""
    import hashlib
    payload = json.dumps(events, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _norm(path_like: str) -> str:
    return str(path_like).replace("\\", "/")


def resolve_entry(entry: dict, roots: dict, what: str) -> Path:
    """把 ``{root, rel}`` 文件条目解析到真实路径（安全相对路径 + 含根内）。"""
    if not isinstance(entry, dict):
        raise ValueError(f"E_CHAIN_ENTRY: {what} 不是对象")
    root_name = entry.get("root")
    if root_name not in FILE_ROOTS:
        raise ValueError(f"E_CHAIN_ENTRY: {what}.root 必须是 {FILE_ROOTS} 之一：{root_name!r}")
    base = roots.get(root_name)
    if base is None:
        raise ValueError(f"E_CHAIN_ENTRY: {what} 声明 root={root_name} 但未提供对应根目录")
    rel = entry.get("rel")
    try:
        return resolve_within(Path(base), str(rel))
    except ValueError as exc:
        raise ValueError(f"E_CHAIN_ENTRY: {what}.rel 不安全：{exc}") from None


def wav_facts(path: Path) -> dict:
    """按**实际编码 tag** 读 WAV 头（每文件自身采样率），并算真实 hash。"""
    info = wav_info(path)
    return {
        "sha256": sha256_file(path),
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "frames": info["frames"],
        "duration_s": info["duration_s"],
        "tag": info["tag"],
        "bits_per_sample": info["bits_per_sample"],
    }


def read_frames(path: Path) -> tuple[list[float], int, int]:
    """读 WAV 全部交错样本（保留声道；按实际编码 tag 解码）。返回 (samples, channels, sr)。"""
    info = wav_info(path)
    data = Path(path).read_bytes()
    pos, body = 12, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if cid == b"data":
            body = data[pos + 8:pos + 8 + size]
            break
        pos += 8 + size + (size % 2)
    if body is None:
        raise WavError(f"E_WAV_FORMAT: 缺少 data chunk：{Path(path).name}")
    tag, ch, bits = info["tag"], info["channels"], info["bits_per_sample"]
    count = info["frames"] * ch
    if tag == WAV_FLOAT and bits == 32:
        return [float(v) for v in struct.unpack_from(f"<{count}f", body, 0)], ch, info["sample_rate"]
    if tag == WAV_FLOAT and bits == 64:
        return [float(v) for v in struct.unpack_from(f"<{count}d", body, 0)], ch, info["sample_rate"]
    if tag == WAV_PCM and bits == 16:
        return [v / 32768.0 for v in struct.unpack_from(f"<{count}h", body, 0)], ch, info["sample_rate"]
    if tag == WAV_PCM and bits == 32:
        return [v / 2147483648.0 for v in struct.unpack_from(f"<{count}i", body, 0)], ch, info["sample_rate"]
    if tag == WAV_PCM and bits == 24:
        raw = body[:count * 3]
        scale = float(1 << 23)
        out = []
        for i in range(0, len(raw) - 2, 3):
            v = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16)
            if v >= 1 << 23:
                v -= 1 << 24
            out.append(v / scale)
        return out, ch, info["sample_rate"]
    raise WavError(f"E_WAV_CODEC: 不支持的组合 tag={tag} bits={bits}")


def check_listen_rel(rel: str, run_id: str) -> tuple[bool, str]:
    """监听副本路径规则：安全相对路径 + 首段必须是本 run 派生目录（``<run_id>-listen``）。"""
    ok, why = is_safe_rel_path(str(rel))
    if not ok:
        return False, f"路径不安全（{why}）：{rel!r}"
    first = _norm(str(rel)).split("/")[0]
    if not first.startswith(str(run_id)):
        return False, f"不在本 run 派生目录内（首段应以 run id 开头）：{rel!r}"
    return True, "ok"


# ------------------------------------------------------- 派生事实（真实产物）


def sam_request_facts(run_dir: Path) -> list[dict]:
    """从 ``stems/*/request.json``(+``report.json``) 派生真实 SAM 事实（不信任 sidecar 自述）。"""
    facts = []
    for request_path in sorted(Path(run_dir).glob("stems/*/request.json")):
        entry: dict = {"request_rel": str(request_path.relative_to(Path(run_dir))).replace("\\", "/")}
        try:
            req = read_json(request_path)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"request.json 无法解析（{exc.__class__.__name__}）"
            facts.append(entry)
            continue
        entry["description"] = req.get("description")
        audio = req.get("audio")
        entry["audio_basename"] = Path(str(audio)).name if audio else None
        if audio and Path(str(audio)).is_file():
            entry["audio_sha256"] = sha256_file(Path(str(audio)))
        else:
            entry["error"] = f"request.json 的 audio 文件不可读：{entry['audio_basename']}"
        report_path = request_path.parent / "report.json"
        if report_path.is_file():
            try:
                rep = read_json(report_path)
                entry["outputs"] = [Path(str(p)).name for p in (rep.get("outputs") or [])]
                entry["sample_rate"] = rep.get("sample_rate")
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"report.json 无法解析（{exc.__class__.__name__}）"
        else:
            entry["error"] = "缺少 report.json"
        facts.append(entry)
    return facts


def events_sha_from_result(result: dict, instrument_id: str) -> tuple[int, str] | None:
    for inst in result.get("instruments") or []:
        if isinstance(inst, dict) and inst.get("id") == instrument_id:
            events = inst.get("events") or []
            return len(events), canonical_events_sha256(events)
    return None


def drums_baseline_facts(baseline_run_dir: Path) -> dict | None:
    """基线 run 的 drums 复用事实（真实 hash + 事件 canonical hash）；找不到返回 None。"""
    baseline_run_dir = Path(baseline_run_dir)
    result_path = baseline_run_dir / "result.json"
    if not result_path.is_file():
        return None
    result = read_json(result_path)
    inst = next((i for i in result.get("instruments") or []
                 if isinstance(i, dict) and i.get("label") == "drums"), None)
    if inst is None:
        return None
    stem = inst.get("stem") or {}
    facts = {
        "baseline_run": baseline_run_dir.name,
        "baseline_result_sha256": sha256_file(result_path),
        "events_instrument": inst.get("id"),
        "events_count": len(inst.get("events") or []),
        "events_sha256": canonical_events_sha256(inst.get("events") or []),
        "target_rel": stem.get("filename"),
        "target_sha256": stem.get("sha256"),
        "residual_rel": None,
        "residual_sha256": None,
    }
    stem_dir = Path(str(stem.get("filename") or "")).parent
    residual = baseline_run_dir / stem_dir / "residual.wav"
    if residual.is_file():
        facts["residual_rel"] = str(residual.relative_to(baseline_run_dir)).replace("\\", "/")
        facts["residual_sha256"] = sha256_file(residual)
    if facts["target_rel"] and (baseline_run_dir / facts["target_rel"]).is_file():
        facts["target_sample_rate"] = wav_info(baseline_run_dir / facts["target_rel"])["sample_rate"]
        facts["target_duration_s"] = wav_info(baseline_run_dir / facts["target_rel"])["duration_s"]
    return facts


# ----------------------------------------------------------- 监听副本（代理）


def make_listen_proxy(raw: Path, out: Path, audio_root: Path, run_id: str,
                      codec: str = "pcm16", raw_root: str | None = None,
                      raw_rel: str | None = None) -> dict:
    """raw residual → 监听副本（**只做 codec 转换**）。

    同采样率 / 同声道 / 同帧数（禁止 trim/shift/重采样/量化）。返回可直接写入 sidecar
    ``listen.proxy`` 的扁平记录：``{raw_root, raw_rel, raw_sha256, proxy_rel, proxy_sha256,
    codec, convert, sample_rate, channels, frames, duration_s}``。
    """
    ok, why = is_safe_run_id(run_id)
    if not ok:
        raise ValueError(f"E_RUN_ID: {why}: {run_id!r}")
    raw = Path(raw)
    out = Path(out)
    audio_root = Path(audio_root).resolve()
    rel = out.resolve().relative_to(audio_root).as_posix() if out.resolve().is_relative_to(audio_root) \
        else None
    if rel is None:
        raise ValueError("E_PATH: 监听副本必须位于 audio 根内")
    ok, why = check_listen_rel(rel, run_id)
    if not ok:
        raise ValueError(f"E_PATH: {why}")

    raw_info = wav_facts(raw)
    if codec == "copy":
        payload = raw.read_bytes()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        convert = "none（逐字节副本）"
    elif codec == "pcm16":
        if raw_info["tag"] not in (1, 3) or raw_info["bits_per_sample"] not in (16, 24, 32, 64):
            raise ValueError(f"E_WAV_CODEC: raw residual 编码不受支持："
                             f"tag={raw_info['tag']} bits={raw_info['bits_per_sample']}")
        from common import write_wav
        samples, channels, sr = read_frames(raw)
        if sr != raw_info["sample_rate"] or len(samples) != raw_info["frames"] * channels:
            raise ValueError("E_WAV_FORMAT: 采样率/帧数读取不一致")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_wav(out, samples, sr, channels=channels, fmt="pcm16")
        convert = (f"tag{raw_info['tag']}/bits{raw_info['bits_per_sample']}->pcm16"
                   f"（幅度 [-1,1] 量化，帧数/声道/采样率不变）")
    else:
        raise ValueError(f"E_CODEC: 不支持的代理 codec：{codec!r}（pcm16|copy）")

    proxy_info = wav_facts(out)
    if proxy_info["frames"] != raw_info["frames"] \
            or proxy_info["sample_rate"] != raw_info["sample_rate"] \
            or proxy_info["channels"] != raw_info["channels"]:
        raise ValueError("E_PROXY_LENGTH: 监听副本帧数/采样率/声道必须与 raw residual 相同")
    return {
        "raw_root": raw_root,
        "raw_rel": raw_rel,
        "raw_sha256": raw_info["sha256"],
        "proxy_rel": rel,
        "proxy_sha256": proxy_info["sha256"],
        "codec": codec,
        "convert": convert,
        "sample_rate": proxy_info["sample_rate"],
        "channels": proxy_info["channels"],
        "frames": proxy_info["frames"],
        "duration_s": proxy_info["duration_s"],
    }


def verify_proxy_record(record: dict, *, residual_sha256: str, raw_path: Path, audio_root: Path,
                        run_id: str, clip_duration_s: float) -> tuple[bool, str]:
    """校验 sidecar 的 ``listen.proxy`` 记录（raw/proxy hash、编码、等长、run 派生目录）。"""
    try:
        proxy_rel = record.get("proxy_rel")
        ok, why = check_listen_rel(str(proxy_rel), run_id)
        if not ok:
            return False, why
        proxy_path = resolve_within(Path(audio_root), str(proxy_rel))
        if not proxy_path.is_file():
            return False, f"监听副本不存在：{proxy_rel}"
        if sha256_file(proxy_path) != record.get("proxy_sha256"):
            return False, f"监听副本 hash 不符：{proxy_rel}"
        if record.get("raw_sha256") != residual_sha256:
            return False, "proxy.raw_sha256 不等于本级 residual raw hash"
        if not raw_path.is_file() or sha256_file(raw_path) != residual_sha256:
            return False, "raw residual 文件缺失或 hash 漂移"
        if not str(record.get("convert") or "").strip():
            return False, "缺少转换参数 convert（哪怕是 none 也要写明）"
        for key in ("sample_rate", "channels", "frames", "duration_s"):
            if record.get(key) is None:
                return False, f"proxy 记录缺少 {key}"
        codec = record.get("codec")
        if codec not in ("pcm16", "copy"):
            return False, f"codec 必须是 pcm16|copy：{codec!r}"
        raw_info = wav_facts(raw_path)
        proxy_info = wav_facts(proxy_path)
        if int(record.get("frames")) != raw_info["frames"] \
                or int(record.get("sample_rate")) != raw_info["sample_rate"] \
                or int(record.get("channels")) != raw_info["channels"]:
            return False, "proxy 记录的帧数/采样率/声道与 raw residual 实际不符"
        if abs(float(record.get("duration_s")) - raw_info["duration_s"]) > 0.001:
            return False, "proxy 记录的时长与 raw residual 实际不符"
        if codec == "copy" and proxy_info["sha256"] != raw_info["sha256"]:
            return False, "codec=copy 但副本与 raw residual 字节不同"
        if codec == "pcm16" and (proxy_info["tag"] != 1 or proxy_info["bits_per_sample"] != 16):
            return False, f"codec=pcm16 但实际编码 tag={proxy_info['tag']} bits={proxy_info['bits_per_sample']}"
        if proxy_info["frames"] != raw_info["frames"] or proxy_info["sample_rate"] != raw_info["sample_rate"] \
                or proxy_info["channels"] != raw_info["channels"]:
            return False, ("监听副本必须与 raw residual 等长同采样率同声道"
                           f"（raw {raw_info['frames']}f@{raw_info['sample_rate']} vs "
                           f"proxy {proxy_info['frames']}f@{proxy_info['sample_rate']}）")
        if abs(proxy_info["duration_s"] - clip_duration_s) > 1.0 / proxy_info["sample_rate"] + 1e-6:
            return False, f"监听副本时长 {proxy_info['duration_s']:.4f}s != clip {clip_duration_s}s"
        return True, (f"{proxy_rel}（codec={codec}，{proxy_info['frames']} 帧 @ "
                      f"{proxy_info['sample_rate']} Hz，raw/proxy hash 均已核对）")
    except (ValueError, WavError, OSError) as exc:
        return False, f"{exc}"


def residual_attach_calls(trace: dict | None, clip_name: str, run_id: str) -> list[dict]:
    """trace 里对**残差监听副本**的成功 audio_attach（不是原 clip 的挂载）。"""
    out = []
    for call in (trace or {}).get("tool_calls", []):
        if call.get("tool") != "audio_attach":
            continue
        facts = call.get("facts") or {}
        path = _norm(str(call.get("args_summary") or ""))
        first = path.split("/")[0]
        if path == _norm(str(clip_name)) or not first.startswith(str(run_id)):
            continue
        if facts.get("attach_success") is True:
            out.append({"path": path, "bytes": facts.get("attach_bytes"),
                        "mime": facts.get("attach_mime")})
    return out


# ------------------------------------------------------------ 链校验（核心）


def verify_chain(doc: dict | None, *, run_dir: Path, clip: dict, roots: dict,
                 run_id: str | None = None, baseline_result_sha256: str | None = None,
                 trace: dict | None = None, ws: Path | None = None,
                 budget: int | None = None, result_path: Path | None = None) -> dict:
    """校残差剥离链（sidecar × 真实产物 × trace）。返回 ``agentic-e2e-stage-chain-verify/v1``。

    ``roots`` = ``{"audio": <WS>/audio/inputs, "run": run 目录, "baseline": 基线 run 目录|None}``。
    """
    chk = Checker()
    run_dir = Path(run_dir)
    run_id = run_id or run_dir.name
    clip_sha = str(clip.get("sha256") or "")
    clip_duration = float(clip.get("duration_s") or 0.0)
    problems: list[str] = []

    # 1) sidecar 结构 -------------------------------------------------------
    stages = []
    if not isinstance(doc, dict):
        chk.add("sidecar_present", False,
                f"缺少或无法解析 sidecar stage-chain.json（{CHAIN_SCHEMA}）")
        return _report(chk, run_id, problems)
    chk.add("sidecar_present", True, "sidecar stage-chain.json 存在")
    if doc.get("schema") != CHAIN_SCHEMA:
        chk.add("sidecar_wellformed", False, f"schema 必须是 {CHAIN_SCHEMA}：{doc.get('schema')!r}")
        return _report(chk, run_id, problems)
    stages = doc.get("stages")
    shape_ok = isinstance(stages, list) and len(stages) >= 1
    if shape_ok:
        for i, st in enumerate(stages):
            if not isinstance(st, dict) or st.get("index") != i:
                shape_ok = False
                problems.append(f"stages[{i}].index 必须 == {i}")
                break
            sam = st.get("sam") or {}
            if not isinstance(sam, dict) or not isinstance(sam.get("performed"), bool):
                shape_ok = False
                problems.append(f"stages[{i}].sam.performed 必须是布尔")
                break
            for key in ("layer", "input", "target", "residual"):
                if not st.get(key):
                    shape_ok = False
                    problems.append(f"stages[{i}] 缺少 {key}")
                    break
            if not shape_ok:
                break
    clip_doc = doc.get("clip") or {}
    if not str(clip_doc.get("timeline") or "").strip():
        shape_ok = False
        problems.append("clip.timeline 必须写明共同秒轴（t=0 = clip 开头）")
    chk.add("sidecar_wellformed", shape_ok,
            f"{len(stages) if isinstance(stages, list) else 0} 级 stage，index 连续、字段齐全"
            if shape_ok else f"sidecar 结构问题：{problems[:3]}")
    if not shape_ok:
        return _report(chk, run_id, problems)

    performed = [st for st in stages if (st.get("sam") or {}).get("performed")]

    # 2) 文件条目解析 + hash/时间轴 ------------------------------------------
    def entry_ok(st_index: int, what: str, entry: dict, *, is_wav: bool) -> Path | None:
        try:
            path = resolve_entry(entry, roots, f"stages[{st_index}].{what}")
        except ValueError as exc:
            problems.append(str(exc))
            return None
        if not path.is_file():
            problems.append(f"stages[{st_index}].{what} 文件不存在：{entry.get('rel')}")
            return None
        actual = sha256_file(path)
        if actual != entry.get("sha256"):
            problems.append(f"stages[{st_index}].{what} hash 不符（实际 {actual[:12]}… vs "
                            f"声明 {str(entry.get('sha256'))[:12]}…）")
            return None
        if is_wav:
            try:
                info = wav_facts(path)
            except WavError as exc:
                problems.append(f"stages[{st_index}].{what}：{exc}")
                return None
            if entry.get("sample_rate") != info["sample_rate"]:
                problems.append(f"stages[{st_index}].{what} 采样率记录不符（实际 "
                                f"{info['sample_rate']} vs 声明 {entry.get('sample_rate')}）")
            if abs(float(entry.get("duration_s", -1)) - info["duration_s"]) > 0.001:
                problems.append(f"stages[{st_index}].{what} 时长记录不符（实际 "
                                f"{info['duration_s']:.4f}s vs 声明 {entry.get('duration_s')}）")
            if clip_duration and abs(info["duration_s"] - clip_duration) > 1.0 / info["sample_rate"] + 1e-6:
                problems.append(f"stages[{st_index}].{what} 时长 {info['duration_s']:.4f}s != "
                                f"clip {clip_duration}s（禁止 trim/shift）")
        return path

    for i, st in enumerate(stages):
        entry_ok(i, "target", st.get("target") or {}, is_wav=True)
        entry_ok(i, "residual", st.get("residual") or {}, is_wav=True)
        entry_ok(i, "request", st.get("request") or {}, is_wav=False)
    chk.add("artifacts_exist_hash_match", not problems,
            "各级 target/residual/request 文件存在且 hash/采样率/时长记录一致"
            if not problems else f"产物问题：{problems[:4]}")

    # 3) 链规则：input == 前一级 raw residual；禁止偷换原混音 -----------------
    chain_bad = []
    if stages[0].get("input", {}).get("kind") != "mix" or stages[0].get("input", {}).get("sha256") != clip_sha:
        chain_bad.append("stages[0].input 必须是原混音（kind=mix，sha256=clip hash）")
    for i in range(1, len(stages)):
        prev_sha = str((stages[i - 1].get("residual") or {}).get("sha256") or "")
        cur = stages[i].get("input") or {}
        if cur.get("kind") != "residual":
            chain_bad.append(f"stages[{i}].input.kind 必须是 residual（实际 {cur.get('kind')!r}）")
        if cur.get("sha256") != prev_sha:
            chain_bad.append(f"stages[{i}].input.sha256 必须 == stages[{i-1}].residual.sha256"
                             f"（{str(cur.get('sha256'))[:12]}… vs {prev_sha[:12]}…）")
    chk.add("chain_input_prev_residual", not chain_bad,
            "每一级 input hash == 前一级 residual 的 raw hash（可审计链）"
            if not chain_bad else f"链断裂：{chain_bad[:3]}")

    swap_bad = [f"stages[{i}].input.sha256 == 原 clip hash（偷换原混音）"
                for i in range(1, len(stages))
                if (stages[i].get("input") or {}).get("sha256") == clip_sha]
    chk.add("no_mix_swap", not swap_bad,
            "第 2 级起 input 均非原混音（错误的独立分离不能误报 sequential）"
            if not swap_bad else f"偷换原混音：{swap_bad[:3]}")

    # 4) request.json 证明真实 SAM input ------------------------------------
    req_bad = []
    for i, st in enumerate(performed):
        idx = stages.index(st)
        try:
            req_path = resolve_entry(st.get("request") or {}, roots, f"stage[{idx}].request")
            req_rel = str(req_path.relative_to(Path(roots["run"]).resolve())).replace("\\", "/")
        except (ValueError, OSError) as exc:
            req_bad.append(f"stage[{idx}]: {exc}")
            continue
        facts = next((f for f in sam_request_facts(run_dir)
                      if f.get("request_rel") == req_rel), None)
        if facts is None or facts.get("error"):
            req_bad.append(f"stage[{idx}]: {facts.get('error') if facts else 'request.json 缺失'}")
            continue
        if facts.get("audio_sha256") != (st.get("input") or {}).get("sha256"):
            req_bad.append(f"stage[{idx}]: request.json 实际 input hash "
                           f"{str(facts.get('audio_sha256'))[:12]}… != 声明 {str((st.get('input') or {}).get('sha256'))[:12]}…")
        outputs = set(facts.get("outputs") or [])
        need = {Path(str((st.get("target") or {}).get("rel"))).name,
                Path(str((st.get("residual") or {}).get("rel"))).name}
        if not need <= outputs:
            req_bad.append(f"stage[{idx}]: report.json outputs 未含 {sorted(need - outputs)}")
    chk.add("request_proves_input", not req_bad,
            f"{len(performed)} 个真实分离的 request.json 均证明 input = 声明输入（raw residual）"
            if not req_bad else f"request 证据问题：{req_bad[:3]}")

    # 5) trace 里 SAM --audio 参数链 ---------------------------------------
    trace_bad: list[str] = []
    trace_detail = "未提供 trace（standalone 核对；run 内会用事件流强制）"
    if trace is not None:
        buckets = classify_separations(trace)
        cmds = [e.get("command", "") for e in buckets["real_success"]]
        if len(cmds) != len(performed):
            trace_bad.append(f"真实 SAM 成功调用 {len(cmds)} 次 != sidecar performed {len(performed)} 级")
        audio_re = re.compile(r"--audio\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))")
        for i, (cmd, st) in enumerate(zip(cmds, performed)):
            m = audio_re.search(str(cmd))
            arg = next((g for g in (m.groups() if m else ()) if g), None) if m else None
            if not arg:
                trace_bad.append(f"第 {i+1} 次 SAM 调用未解析到 --audio 参数")
                continue
            try:
                apath = Path(arg)
                apath = apath if apath.is_absolute() else (Path(ws) / apath) if ws else None
                if apath is None or not apath.is_file():
                    trace_bad.append(f"第 {i+1} 次 SAM --audio 文件不可读：{Path(arg).name}")
                    continue
                if sha256_file(apath) != (st.get("input") or {}).get("sha256"):
                    trace_bad.append(f"第 {i+1} 次 SAM --audio（{Path(arg).name}）hash != "
                                     f"声明 input hash（偷换输入？）")
            except (OSError, ValueError) as exc:
                trace_bad.append(f"第 {i+1} 次 SAM --audio 校验失败：{exc}")
        trace_detail = (f"trace 中 {len(cmds)} 次真实 SAM --audio 与 sidecar input 链逐一相符"
                        if not trace_bad else f"trace 链问题：{trace_bad[:3]}")
    chk.add("trace_input_chain", not trace_bad, trace_detail)

    # 6) 监听残差（自主听音）证据 ------------------------------------------
    listen_bad = []
    for i, st in enumerate(stages):
        is_last = i == len(stages) - 1
        stop_reason = str((doc.get("stop") or {}).get("reason") or "")
        required = (not is_last) or stop_reason != "budget"
        listen = st.get("listen") or {}
        if not required and not listen:
            continue
        if not listen:
            listen_bad.append(f"stages[{i}] 缺少 listen（必须先听该级 residual 再决定下一层/停止）")
            continue
        if listen.get("method") != "audio_attach":
            listen_bad.append(f"stages[{i}].listen.method 必须是 audio_attach")
        record = dict(listen.get("proxy") or {})
        record.setdefault("proxy_rel", listen.get("path"))
        try:
            raw_path = resolve_entry(st.get("residual") or {}, roots, f"stages[{i}].residual")
        except ValueError as exc:
            listen_bad.append(str(exc))
            continue
        ok, why = verify_proxy_record(record, residual_sha256=str((st.get("residual") or {}).get("sha256")),
                                      raw_path=raw_path, audio_root=Path(roots["audio"]),
                                      run_id=run_id, clip_duration_s=clip_duration)
        if not ok:
            listen_bad.append(f"stages[{i}].listen.proxy：{why}")
            continue
        if trace is not None:
            attached = [c for c in residual_attach_calls(trace, str(clip.get("name") or ""), run_id)
                        if _norm(str(listen.get("path"))) == c["path"]]
            if not attached:
                listen_bad.append(f"stages[{i}]：trace 中无对 {listen.get('path')} 的成功 audio_attach")
    chk.add("residual_listen_observed", not listen_bad,
            "各级 residual 均有监听副本（raw/proxy hash、转换参数、等长）且 trace 观测到真实挂载"
            if not listen_bad else f"监听证据问题：{listen_bad[:4]}")

    # 7) 停止策略 ----------------------------------------------------------
    stop = doc.get("stop") or {}
    reason = stop.get("reason")
    lim_text = " ".join(str(x) for x in (doc.get("limitations") or []) if isinstance(x, str)) + " " + \
        str(stop.get("limitations") or "")
    stop_bad = []
    if reason not in STOP_REASONS:
        stop_bad.append(f"stop.reason 必须是 {STOP_REASONS} 之一：{reason!r}")
    if not str(stop.get("evidence") or "").strip():
        stop_bad.append("stop.evidence 必须写明停止依据（听感/测量/预算）")
    if not lim_text.strip():
        stop_bad.append("limitations 不能为空（残余/不确定必须如实记录）")
    elif not ("残差" in lim_text or "residual" in lim_text.lower()):
        stop_bad.append("limitations 必须写明 residual 的性质（模型估计、不是真值）")
    if not any(k in lim_text for k in ("误差", "估计", "真值", "不确定")):
        stop_bad.append("limitations 必须写明误差/不确定性（误差会累积、不承诺更准）")
    if reason == "budget" and budget is not None:
        failed_attempts = 0
        if trace is not None:
            _b = classify_separations(trace)
            failed_attempts = len(_b["real_failed"]) + len(_b["unknown"])
        if len(performed) + failed_attempts < budget:
            stop_bad.append(f"stop.reason=budget 时预算必须真实用尽（成功 {len(performed)} 级 + "
                            f"失败/未知尝试 {failed_attempts} 次 < 预算 {budget}）")
    if reason == "budget" and not any(k in lim_text for k in ("残余", "残留", "未剥离", "未辨认", "剩余")):
        stop_bad.append("stop.reason=budget 时 limitations 必须写明残余未剥离部分")
    residual_left = stop.get("residual_left")
    if residual_left:
        left = entry_ok(len(stages), "stop.residual_left", residual_left, is_wav=True)
        if left is None:
            stop_bad.append("stop.residual_left 文件/hash 核对失败")
    chk.add("stop_policy", not stop_bad,
            f"停止原因 {reason!r} 合规，残余与不确定性已如实记录"
            if not stop_bad else f"停止策略问题：{stop_bad[:3]}")

    # 8) 预算 --------------------------------------------------------------
    budget_bad = []
    if budget is not None and len(performed) > budget:
        budget_bad.append(f"真实 SAM 分离 {len(performed)} 级 > 预算 {budget}")
    chk.add("sam_budget", not budget_bad,
            f"真实 SAM 分离 {len(performed)} 级 ≤ 预算 {budget if budget is not None else '未设'}"
            if not budget_bad else f"超预算：{budget_bad[:2]}")

    # 9) 基线复用完整性（drums 保留证据） ----------------------------------
    reuse_bad = []
    reused = [st for st in stages if (st.get("sam") or {}).get("reused")]
    for st in reused:
        idx = stages.index(st)
        block = st.get("reused") or {}
        base_root = roots.get("baseline")
        if base_root is None:
            reuse_bad.append(f"stages[{idx}] 声明 reused 但未提供 --baseline-run-dir")
            continue
        base_result = Path(base_root) / "result.json"
        if not base_result.is_file():
            reuse_bad.append(f"stages[{idx}]: 基线 result.json 不存在")
            continue
        actual_result_sha = sha256_file(base_result)
        if block.get("baseline_result_sha256") != actual_result_sha:
            reuse_bad.append(f"stages[{idx}]: 基线 result.json hash 不符（{actual_result_sha[:12]}… vs "
                             f"{str(block.get('baseline_result_sha256'))[:12]}…）")
        if baseline_result_sha256 and actual_result_sha != baseline_result_sha256:
            reuse_bad.append(f"stages[{idx}]: 基线 result.json 与外部登记 hash 不符（基线被改动？）")
        for what in ("baseline_target", "baseline_residual"):
            try:
                bpath = resolve_entry(block.get(what) or {}, roots, f"stages[{idx}].reused.{what}")
            except ValueError as exc:
                reuse_bad.append(str(exc))
                continue
            if not bpath.is_file() or sha256_file(bpath) != (block.get(what) or {}).get("sha256"):
                reuse_bad.append(f"stages[{idx}].reused.{what} 与基线文件不符（基线不得被覆盖/改动）")
        for what in ("target", "residual"):
            declared = str((st.get(what) or {}).get("sha256") or "")
            base_declared = str((block.get(f"baseline_{what}") or {}).get("sha256") or "")
            if declared != base_declared:
                reuse_bad.append(f"stages[{idx}]: 复用副本 {what} hash != 基线原件 hash（不是逐字节复用）")
        pair = events_sha_from_result(read_json(base_result), str(block.get("events_instrument") or ""))
        if pair is None:
            reuse_bad.append(f"stages[{idx}]: 基线无乐器 {block.get('events_instrument')!r}")
        else:
            count, sha = pair
            if block.get("events_count") != count or block.get("events_sha256") != sha:
                reuse_bad.append(f"stages[{idx}]: 复用事件声明（{block.get('events_count')}/"
                                 f"{str(block.get('events_sha256'))[:12]}…）与基线实际（{count}/{sha[:12]}…）不符")
        if not str(block.get("user_feedback") or "").strip():
            reuse_bad.append(f"stages[{idx}]: 必须记录用户局部反馈的来源（局部定性，不推导干净/真值）")
    chk.add("reused_baseline_integrity", not reuse_bad,
            f"{len(reused)} 个复用级与基线 hash/事件逐一相符（基线只读，未被覆盖）"
            if not reuse_bad else f"基线复用问题：{reuse_bad[:3]}")

    # 10) 复用事件必须原样进入 result.json（drums 保留证据） -----------------
    carry_bad = []
    if result_path is not None and Path(result_path).is_file():
        result = read_json(Path(result_path))
        for st in reused:
            idx = stages.index(st)
            block = st.get("reused") or {}
            inst_id = str(block.get("events_instrument") or "")
            inst = next((i for i in result.get("instruments") or []
                         if isinstance(i, dict) and i.get("id") == inst_id), None)
            if inst is None:
                carry_bad.append(f"stages[{idx}]: result.json 缺少复用乐器 {inst_id!r}")
                continue
            count, sha = len(inst.get("events") or []), canonical_events_sha256(inst.get("events") or [])
            if count != block.get("events_count") or sha != block.get("events_sha256"):
                carry_bad.append(f"stages[{idx}]: result.json 的 {inst_id} 事件（{count}/{sha[:12]}…）"
                                 f"!= 基线复用声明（{block.get('events_count')}/"
                                 f"{str(block.get('events_sha256'))[:12]}…）")
    chk.add("reused_events_carried_into_result", not carry_bad,
            "复用的基线事件已原样进入 result.json（drums 保留）"
            if not carry_bad else f"复用事件问题：{carry_bad[:3]}")

    # 11) sidecar 隐私（相对路径 + hash，无绝对路径/密钥） ------------------
    findings = find_private(json.dumps(doc, ensure_ascii=False, sort_keys=True))
    chk.add("sidecar_privacy", not findings,
            "sidecar 无绝对路径/密钥/字节块（只有相对路径与 hash）"
            if not findings else f"sidecar 含私有内容：{findings[:3]}")

    return _report(chk, run_id, problems)


def _report(chk: Checker, run_id: str, problems: list[str]) -> dict:
    return {
        "schema": CHAIN_VERIFY_SCHEMA,
        "kind": "residual-chain",
        "run_id": run_id,
        "ok": chk.ok,
        "checks": chk.checks,
        "problems": problems,
        "note": "残差链核对：input==前级 raw residual、request/trace 实证、监听代理 hash/等长、"
                "停止/预算规则、基线复用只读；sidecar 不扩展 #32 冻结 result 协议",
    }


# --------------------------------------------------------------------- CLI


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="residual_chain.py",
                               description="残差逐层剥离 sidecar 链校验 / 监听副本（issue #42）")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="校验 stage-chain.json（对照真实产物/trace）")
    v.add_argument("--chain", required=True, help="stage-chain.json 路径")
    v.add_argument("--run-dir", required=True, help="新 run 目录（outputs/e2e/<id>）")
    v.add_argument("--clip-json", required=True, help="clip 描述 JSON")
    v.add_argument("--audio-root", required=True, help="<WS>/audio/inputs（bridge 根）")
    v.add_argument("--baseline-run-dir", default=None, help="基线 run 目录（复用核对；只读）")
    v.add_argument("--baseline-result-sha", default=None, help="外部登记的基线 result.json sha256（可选）")
    v.add_argument("--ws", default=None, help="workspace 根（解析 trace 中相对 --audio）")
    v.add_argument("--events", default=None, help="事件流 events.jsonl（trace 交叉核对）")
    v.add_argument("--result", default=None, help="新 run result.json（复用事件核对）")
    v.add_argument("--budget", type=int, default=None, help="真实 SAM 分离预算")
    v.add_argument("--json", action="store_true", help="JSON 输出")

    m = sub.add_parser("make-proxy", help="raw residual → 监听副本（PCM16/copy，等长）")
    m.add_argument("--raw", required=True, help="raw residual WAV")
    m.add_argument("--out", required=True, help="监听副本路径（audio 根内 <run-id>-listen/…）")
    m.add_argument("--audio-root", required=True, help="<WS>/audio/inputs")
    m.add_argument("--run-id", required=True, help="run id（派生目录守卫）")
    m.add_argument("--codec", default="pcm16", choices=("pcm16", "copy"), help="转换方式")
    m.add_argument("--record", default=None, help="追加/写入记录 JSON（schema listen-proxies/v1）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cmd == "make-proxy":
        try:
            record = make_listen_proxy(Path(args.raw), Path(args.out), Path(args.audio_root),
                                       args.run_id, codec=args.codec)
        except (ValueError, WavError, OSError) as exc:
            print(f"[fail] {exc}", file=sys.stderr)
            return 2
        if args.record:
            rec_path = Path(args.record)
            doc = {"schema": PROXY_RECORD_SCHEMA, "run_id": args.run_id, "proxies": []}
            if rec_path.is_file():
                try:
                    doc = read_json(rec_path)
                except Exception:  # noqa: BLE001
                    pass
            doc.setdefault("proxies", []).append(record)
            write_json(rec_path, doc)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    chain_path = Path(args.chain)
    doc = None
    if chain_path.is_file():
        try:
            doc = read_json(chain_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] stage-chain.json 无法解析：{exc.__class__.__name__}", file=sys.stderr)
            doc = None
    clip = read_json(Path(args.clip_json))
    roots = {
        "audio": Path(args.audio_root),
        "run": Path(args.run_dir),
        "baseline": Path(args.baseline_run_dir) if args.baseline_run_dir else None,
    }
    trace = None
    if args.events:
        try:
            from common import extract_trace
            trace = extract_trace(load_events(Path(args.events)))
        except OSError:
            trace = None
    report = verify_chain(
        doc, run_dir=Path(args.run_dir), clip=clip, roots=roots,
        baseline_result_sha256=args.baseline_result_sha, trace=trace,
        ws=Path(args.ws) if args.ws else None,
        budget=args.budget, result_path=Path(args.result) if args.result else None)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"{'OK' if report['ok'] else 'FAIL'} residual-chain run_id={report['run_id']}")
        for c in report["checks"]:
            print(f"  [{'pass' if c['pass'] else 'FAIL'}] {c['name']}: {c['detail']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # 受控错误，不抛裸 traceback
        print(f"[fail] E_INTERNAL: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
