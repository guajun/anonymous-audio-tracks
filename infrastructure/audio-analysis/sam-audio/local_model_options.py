"""Resolve local SAM Audio checkpoints without contacting a model hub."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "model-cache" / "sam-audio-small"
DEFAULT_T5_DIR = ROOT / "model-cache" / "t5-base"


def _path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


MODEL_DIR = _path(os.getenv("SAM_AUDIO_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
T5_DIR = _path(os.getenv("SAM_AUDIO_T5_DIR", str(DEFAULT_T5_DIR)))


def model_directories(
    model_dir: str | os.PathLike[str] | None = None,
    text_encoder_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path]:
    return (
        _path(model_dir) if model_dir else MODEL_DIR,
        _path(text_encoder_dir) if text_encoder_dir else T5_DIR,
    )


def loading_options(
    model_dir: str | os.PathLike[str] | None = None,
    text_encoder_dir: str | os.PathLike[str] | None = None,
) -> dict:
    """Return explicit offline options for ``SAMAudio.from_pretrained``."""

    resolved_model_dir, resolved_t5_dir = model_directories(model_dir, text_encoder_dir)
    config_path = resolved_model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"SAM Audio config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "local_files_only": True,
        "text_encoder": {**config["text_encoder"], "name": str(resolved_t5_dir)},
        "visual_ranker": None,
        "text_ranker": None,
        "span_predictor": None,
    }
