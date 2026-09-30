"""Run one local, offline SAM Audio separation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.toml"
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "config.example.toml"
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_ffmpeg = shutil.which("ffmpeg")
if _ffmpeg and os.name == "nt":
    os.add_dll_directory(str(Path(_ffmpeg).parent))

import numpy as np
import soundfile as sf

from local_model_options import loading_options, model_directories


def _resolve(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_config() -> dict:
    path = CONFIG_PATH if CONFIG_PATH.is_file() else EXAMPLE_CONFIG_PATH
    if not path.is_file():
        return {}
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--description", required=True, help="Lowercase noun/verb phrase, for example 'bowed strings'")
    parser.add_argument("--anchor", action="append", metavar="START,END", help="Positive time span; may be repeated")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--text-encoder-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _parse_anchors(raw: list[str] | None) -> list[list[tuple[str, float, float]]] | None:
    if not raw:
        return None
    spans: list[tuple[str, float, float]] = []
    for item in raw:
        try:
            start, end = (float(value.strip()) for value in item.split(",", 1))
        except ValueError as error:
            raise ValueError(f"Invalid --anchor {item!r}; expected START,END") from error
        if start < 0 or end <= start:
            raise ValueError(f"Invalid --anchor {item!r}; require 0 <= START < END")
        spans.append(("+", start, end))
    return [spans]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _load_config()
    paths_config = config.get("paths", {})
    inference_config = config.get("inference", {})
    audio = _resolve(args.audio)
    model_dir, text_encoder_dir = model_directories(
        _resolve(args.model_dir or paths_config.get("model_dir")) if args.model_dir or paths_config.get("model_dir") else None,
        _resolve(args.text_encoder_dir or paths_config.get("text_encoder_dir")) if args.text_encoder_dir or paths_config.get("text_encoder_dir") else None,
    )
    device_name = args.device or os.getenv("SAM_AUDIO_DEVICE") or inference_config.get("device", "cuda")
    dtype_name = args.dtype or os.getenv("SAM_AUDIO_DTYPE") or inference_config.get("dtype", "auto")
    if dtype_name not in {"auto", "bfloat16", "float32"}:
        _parser().error(f"Unsupported dtype: {dtype_name}")
    try:
        anchors = _parse_anchors(args.anchor)
    except ValueError as error:
        _parser().error(str(error))
    if not audio.is_file():
        _parser().error(f"Audio file not found: {audio}")
    required = [model_dir / "checkpoint.pt", model_dir / "config.json", text_encoder_dir / "config.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        _parser().error("Missing local model file(s): " + ", ".join(missing))

    output_dir = _resolve(args.output_dir) if args.output_dir else PROJECT_ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    plan = {
        "audio": str(audio),
        "description": args.description,
        "anchors": anchors[0] if anchors else [],
        "model_dir": str(model_dir),
        "text_encoder_dir": str(text_encoder_dir),
        "device": device_name,
        "dtype": dtype_name,
        "output_dir": str(output_dir),
        "network": "disabled",
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    import torch
    from sam_audio import SAMAudio, SAMAudioProcessor

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    dtype = torch.bfloat16 if dtype_name == "bfloat16" or (dtype_name == "auto" and device.type == "cuda") else torch.float32
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "request.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    started = time.monotonic()
    print("Loading SAM Audio small from", model_dir, flush=True)
    model = SAMAudio.from_pretrained(str(model_dir), **loading_options(model_dir, text_encoder_dir)).eval()
    # No video is used here; keep the vision encoder on CPU while moving the audio path to the accelerator.
    vision_encoder = model.vision_encoder
    model.vision_encoder = torch.nn.Identity()
    model = model.to(device=device, dtype=dtype)
    model.vision_encoder = vision_encoder
    # Decode each generated stream separately to keep peak VRAM below the 8 GB laptop-GPU budget.
    original_decode = model.audio_codec.decode

    def sequential_decode(features):
        return torch.cat([original_decode(row.unsqueeze(0)) for row in features], dim=0)

    model.audio_codec.decode = sequential_decode
    processor = SAMAudioProcessor.from_pretrained(str(model_dir))
    batch = processor(audios=[str(audio)], descriptions=[args.description], anchors=anchors).to(device)
    autocast_enabled = device.type == "cuda" and dtype == torch.bfloat16
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
        result = model.separate(batch, predict_spans=False, reranking_candidates=1)

    outputs = []
    for name, waveform in (("target.wav", result.target[0]), ("residual.wav", result.residual[0])):
        samples = waveform.detach().float().cpu().numpy().reshape(-1)
        if not np.isfinite(samples).all():
            raise RuntimeError(f"Non-finite samples returned for {name}")
        path = output_dir / name
        sf.write(path, samples, processor.audio_sampling_rate, subtype="FLOAT")
        outputs.append(str(path))
    plan.update({"outputs": outputs, "elapsed_s": time.monotonic() - started, "sample_rate": processor.audio_sampling_rate})
    (output_dir / "report.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
