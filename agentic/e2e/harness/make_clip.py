#!/usr/bin/env python3
"""make_clip.py — 受控 clip 制作（issue #33）。

把一段**真实**样本按 (offset, duration) 裁剪成桥接预算内的 WAV，放进 workspace 的
受控音频根目录 ``audio/inputs/``，并登记到 ``audio/inputs/manifest.json``
（#31 的 workspace-audio-manifest/v1 格式），同时写本地 clip 描述
``local/e2e/clip-<name>.json``（记录源 hash / offset / 时轴约定）。

时轴约定（冻结）：**clip 的 t=0 = 源文件 offset_s 处**；所有 onset 都在 clip 时轴上。
桥接预算：单文件 ≤4 MiB、排队 ≤8 MiB（#30 冻结；本脚本默认拒绝超预算 clip）。

真实 / mock：本脚本只是搬运字节，不做任何分析；clip 的 ``origin`` 记录真实来源，
mock/fixture 由 ``fixtures/make_mock_fixture.py`` 显式标注，二者永不混用。

用法::

    python agentic/e2e/harness/make_clip.py --source <源wav> --offset 4.0 --duration 16.0 \
        --workspace <WS> [--name e2e-clip-001.wav] [--max-bytes 4194304]

退出码：0 成功 / 2 用法 / 3 冲突或守卫（E_HASH_CONFLICT、E_BRIDGE_SIZE）/ 4 IO 或格式错误。
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    AUDIO_ROOT_REL, WavError, read_json, sha256_file, wav_info, write_json,
)

EXIT_OK, EXIT_USAGE, EXIT_GUARD, EXIT_IO = 0, 2, 3, 4
AUDIO_MANIFEST_SCHEMA = "workspace-audio-manifest/v1"
CLIP_SCHEMA = "agentic-e2e-clip/v1"
DEFAULT_MAX_BYTES = 4 * 1024 * 1024  # #30 冻结：单文件 ≤4 MiB


def wav_slice(src: Path, offset_s: float, duration_s: float, dst: Path) -> dict:
    """按帧裁剪 WAV（保持 fmt/声道/采样率不变）。返回 clip 元数据。"""
    data = src.read_bytes()
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavError(f"E_WAV_FORMAT: 不是 RIFF/WAVE：{src.name}")
    fmt_chunk = None
    pos = 12
    data_body = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt_chunk = body
        elif cid == b"data":
            data_body = body
        pos += 8 + size + (size % 2)
    if fmt_chunk is None or data_body is None:
        raise WavError(f"E_WAV_FORMAT: 缺 fmt/data chunk：{src.name}")
    tag, channels, sr, _br, block_align, bits = struct.unpack_from("<HHIIHH", fmt_chunk, 0)
    if block_align <= 0:
        raise WavError(f"E_WAV_FORMAT: 非法 block_align：{src.name}")
    total_frames = len(data_body) // block_align
    total_s = total_frames / sr
    if not (0 <= offset_s < total_s):
        raise ValueError(f"E_CLIP_RANGE: offset {offset_s} 超出源时长 {total_s:.3f}s")
    if duration_s <= 0:
        raise ValueError(f"E_CLIP_RANGE: duration 必须为正，得到 {duration_s}")
    end_s = min(offset_s + duration_s, total_s)
    start_f = int(round(offset_s * sr))
    end_f = int(round(end_s * sr))
    frames = end_f - start_f
    payload = data_body[start_f * block_align:end_f * block_align]
    out = b"RIFF" + struct.pack("<I", 4 + (8 + len(fmt_chunk)) + (8 + len(payload)))
    out += b"WAVE"
    out += b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
    out += b"data" + struct.pack("<I", len(payload)) + payload
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out)
    return {
        "format": "wav",
        "channels": channels,
        "sample_rate": sr,
        "bits_per_sample": bits,
        "frames": frames,
        "duration_s": frames / sr,
        "offset_s": start_f / sr,
    }


def register_audio_entry(ws: Path, name: str, meta: dict, sha: str, nbytes: int) -> str:
    """登记到 audio/inputs/manifest.json（幂等；同名不同内容 → E_HASH_CONFLICT）。"""
    manifest_path = ws / AUDIO_ROOT_REL / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("schema") != AUDIO_MANIFEST_SCHEMA:
            raise RuntimeError(f"E_AUDIO_MANIFEST: schema 不符：{manifest_path}")
        entries = manifest.get("entries") or []
    else:
        entries = []
    for entry in entries:
        if entry.get("name") == name:
            if entry.get("sha256") == sha:
                return "kept"
            raise RuntimeError(
                f"E_HASH_CONFLICT: {name} 已登记但内容不同（manifest {str(entry.get('sha256'))[:12]}… "
                f"vs 新 {sha[:12]}…）；不覆盖，换 clip 名或人工确认后删条目")
    entries.append({
        "name": name,
        "sha256": sha,
        "bytes": nbytes,
        "format": meta["format"],
        "duration_s": round(meta["duration_s"], 6),
        "sample_rate": meta["sample_rate"],
        "channels": meta["channels"],
        "probe": "wave",
        "source": "import",
    })
    entries.sort(key=lambda e: str(e.get("name")))
    write_json(manifest_path, {"schema": AUDIO_MANIFEST_SCHEMA, "entries": entries})
    return "ok"


def build_clip(source: Path, offset_s: float, duration_s: float, ws: Path,
               name: str, max_bytes: int) -> dict:
    """裁剪 + 登记 + 写 clip 描述。返回 clip 描述 dict。"""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"E_SOURCE_NOT_FOUND: {source}")
    audio_root = ws / AUDIO_ROOT_REL
    audio_root.mkdir(parents=True, exist_ok=True)
    dst = audio_root / name

    src_info = wav_info(source)
    src_sha = sha256_file(source)

    if dst.exists():
        dst_sha = sha256_file(dst)
        clip_path = ws / "local" / "e2e" / f"clip-{name}.json"
        existing = read_json(clip_path) if clip_path.exists() else None
        if existing and existing.get("sha256") == dst_sha:
            origin = existing.get("origin") or {}
            same_origin = (origin.get("sha256") == src_sha
                           and abs(float(origin.get("offset_s", -1)) - float(offset_s)) <= 1e-9
                           and abs(float(origin.get("clipped_duration_s", -1))
                                   - float(min(duration_s, src_info["duration_s"] - offset_s))) <= 1e-6)
            if same_origin:
                return existing
            raise RuntimeError(
                f"E_HASH_CONFLICT: {name} 已存在且来自不同源/参数"
                f"（已记录 origin {str(origin.get('sha256'))[:12]}…@{origin.get('offset_s')}s，"
                f"请求 {src_sha[:12]}…@{offset_s}s）；不覆盖，换 --name 或人工确认")
        raise RuntimeError(
            f"E_HASH_CONFLICT: {name} 已存在且内容不同（{dst_sha[:12]}…）；不覆盖，换 --name 或人工确认")

    meta = wav_slice(source, offset_s, duration_s, dst)
    sha = sha256_file(dst)
    nbytes = dst.stat().st_size
    if nbytes > max_bytes:
        dst.unlink()
        raise RuntimeError(
            f"E_BRIDGE_SIZE: clip {nbytes} bytes > 预算 {max_bytes}；缩短 duration 或降采样")

    clip = {
        "schema": CLIP_SCHEMA,
        "name": name,
        "sha256": sha,
        "bytes": nbytes,
        "duration_s": round(meta["duration_s"], 9),
        "sample_rate": meta["sample_rate"],
        "channels": meta["channels"],
        "bits_per_sample": meta["bits_per_sample"],
        "timeline": "t=0 是 clip 开头 = 源文件 offset_s 处；所有 onset 用 clip 时轴（秒）",
        "origin": {
            "file": source.name,
            "sha256": src_sha,
            "bytes": source.stat().st_size,
            "duration_s": round(src_info["duration_s"], 9),
            "sample_rate": src_info["sample_rate"],
            "channels": src_info["channels"],
            "offset_s": round(meta["offset_s"], 9),
            "clipped_duration_s": round(meta["duration_s"], 9),
        },
    }
    status = register_audio_entry(ws, name, meta, sha, nbytes)
    clip["registration"] = status
    write_json(ws / "local" / "e2e" / f"clip-{name}.json", clip)
    return clip


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="make_clip.py", description="受控 clip 制作（issue #33）")
    p.add_argument("--source", required=True, help="真实源 WAV（本地，不上传）")
    p.add_argument("--offset", type=float, required=True, help="源文件起点（秒）")
    p.add_argument("--duration", type=float, required=True, help="clip 时长（秒）")
    p.add_argument("--workspace", required=True, help="已部署的 Pi workspace 根目录")
    p.add_argument("--name", default="e2e-clip-001.wav", help="clip 在 audio/inputs 内的文件名")
    p.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="桥接单文件预算（默认 4 MiB）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code not in (0, None) else EXIT_OK
    if not (args.offset >= 0) or not (args.duration > 0):
        print("[fail] E_CLIP_RANGE: --offset 需 >=0 且 --duration 需 >0（有限数）", file=sys.stderr)
        return EXIT_USAGE
    ws = Path(args.workspace).expanduser().resolve()
    if not (ws / ".pi").is_dir():
        print(f"[fail] E_WS: 不是已部署 workspace（缺 .pi/）：{ws}", file=sys.stderr)
        return EXIT_GUARD
    try:
        clip = build_clip(Path(args.source), args.offset, args.duration, ws,
                          args.name, int(args.max_bytes))
    except FileNotFoundError as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return EXIT_GUARD
    except (RuntimeError, ValueError) as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return EXIT_GUARD
    except (OSError, WavError) as exc:
        print(f"[fail] E_IO: {exc}", file=sys.stderr)
        return EXIT_IO
    print(json.dumps({
        "clip": clip["name"],
        "sha256": clip["sha256"],
        "bytes": clip["bytes"],
        "duration_s": clip["duration_s"],
        "sample_rate": clip["sample_rate"],
        "channels": clip["channels"],
        "origin_offset_s": clip["origin"]["offset_s"],
        "registration": clip["registration"],
    }, ensure_ascii=False, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
