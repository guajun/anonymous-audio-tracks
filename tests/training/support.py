"""Shared helpers for the training tests (no test collection here)."""

from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aat.training.config import TrainConfig


def smoke_config(
    corpus: Mapping[str, Any],
    out_dir: str | Path,
    *,
    steps: int = 3,
    interval_steps: int | None = None,
    eval_enabled: bool = False,
    eval_splits: tuple[str, ...] = ("val",),
    max_songs: int = 1,
    centers_per_item: int = 4,
    min_center_gap: int = 5,
    groups_per_step: int = 1,
    split: str = "train",
    lr: float = 1e-3,
    fake_seed: int = 20260929,
    slots: int = 8,
    valid_only: bool = True,
    run_seed: int = 20260929,
    dropout: float = 0.0,
    index_path: str | None = None,
    data_root: str | None = None,
) -> TrainConfig:
    payload = {
        "run": {
            "name": "test-run",
            "out_dir": str(out_dir),
            "seed": run_seed,
            "device": "cpu",
        },
        "encoder": {"mode": "fake", "feature_dim": 16, "fake_seed": fake_seed},
        "data": {
            "index": index_path or str(corpus["index_path"]),
            "data_root": data_root or str(corpus["data_root"]),
            "split": split,
            "groups_per_step": groups_per_step,
            "centers_per_item": centers_per_item,
            "min_center_gap": min_center_gap,
            "valid_only": valid_only,
        },
        "head": {"slots": slots, "dropout": dropout},
        "optim": {"steps": steps, "lr": lr},
        "checkpoint": {"interval_steps": interval_steps or steps},
        "eval": {"enabled": eval_enabled, "splits": list(eval_splits), "max_songs": max_songs},
    }
    return TrainConfig.from_dict(payload, origin="test-config")


def read_log(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def copy_corpus(corpus: Mapping[str, Any], target: Path) -> dict[str, Any]:
    """Copy a generated corpus (samples + index) to a scratch root."""

    source_root = Path(corpus["index_path"]).parent
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, target / "corpus", dirs_exist_ok=True)
    index_path = target / "corpus" / Path(corpus["index_path"]).name
    data_root = target / "corpus" / "samples"
    return {"index_path": str(index_path), "data_root": str(data_root)}


def rewrite_wav_different_content(path: Path, *, amplitude: float = 0.5) -> None:
    """Rewrite a PCM16 WAV with the same shape but different samples."""

    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        payload = np.frombuffer(handle.readframes(params.nframes), dtype="<i2")
    changed = np.clip(payload.astype(np.float64) * 0.25 + amplitude * 1000.0 * np.sin(
        np.arange(payload.size, dtype=np.float64) * 0.13
    ), -32768.0, 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setparams(params)
        handle.writeframes(changed.tobytes())


def rehash_manifest_mix(sample_dir: Path, relative: str = "mix.wav") -> None:
    """Update manifest content_sha256 after an intentional same-shape rewrite."""

    import hashlib

    from aat.contracts import SampleManifest

    manifest_path = sample_dir / "manifest.json"
    manifest = SampleManifest.load(manifest_path)
    payload = manifest.to_json_dict()
    payload["content_sha256"][relative] = hashlib.sha256(
        (sample_dir / relative).read_bytes()
    ).hexdigest()
    SampleManifest.from_json_dict(payload).save(manifest_path)
