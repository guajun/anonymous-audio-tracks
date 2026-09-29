"""Sample-directory and CLI helpers for label tests (synthetic only)."""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import sys
import wave
from pathlib import Path

import numpy as np

from aat.contracts import (
    ControlEvent,
    Controls,
    SampleManifest,
    SourceEntry,
    SourceRegistry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "scripts" / "label_sample.py"
RATE = 16000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        channels = 1
        payload = data
    elif data.ndim == 2:
        channels = int(data.shape[1])
        payload = data
    else:
        raise ValueError("expected 1-D or 2-D samples")
    clipped = np.clip(np.round(payload * 32767.0), -32768, 32767).astype("<i2")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(clipped.tobytes())
    return target


def make_sample_dir(
    root: Path,
    *,
    stems: dict[str, np.ndarray],
    sample_rate: int = RATE,
    duration_seconds: float | None = None,
    track_start_seconds: float = 0.0,
    source_ids: list[str] | None = None,
    controls_events: tuple[dict, ...] | list[dict] = (),
    sample_id: str = "synth-0001",
    seed: int = 7,
) -> Path:
    """Write a valid ``rendered`` protocol sample directory."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ids = list(source_ids) if source_ids is not None else list(stems)
    if set(ids) != set(stems):
        raise ValueError("source_ids must match the stems keys")

    first = np.asarray(stems[ids[0]], dtype=np.float64)
    if duration_seconds is None:
        duration_seconds = first.shape[0] / sample_rate
    frames = round(duration_seconds * sample_rate)
    mix = np.zeros(frames, dtype=np.float64)
    for source_id in ids:
        array = np.asarray(stems[source_id], dtype=np.float64)
        mono = array if array.ndim == 1 else array.mean(axis=1)
        mix[: mono.shape[0]] += mono[:frames]
    write_pcm16(root / "mix.wav", mix, sample_rate)

    stem_paths: dict[str, str] = {}
    for source_id in ids:
        relative = f"stems/{source_id}.wav"
        write_pcm16(root / relative, stems[source_id], sample_rate)
        stem_paths[source_id] = relative

    SourceRegistry(
        sources=tuple(
            SourceEntry(source_id=source_id, index=index)
            for index, source_id in enumerate(ids)
        ),
        sample_id=sample_id,
    ).save(root / "sources.json")
    Controls(
        events=tuple(ControlEvent(**event) for event in controls_events),
        sample_id=sample_id,
    ).save(root / "controls.json")

    content_sha256 = {
        "mix.wav": sha256_file(root / "mix.wav"),
        "sources.json": sha256_file(root / "sources.json"),
        "controls.json": sha256_file(root / "controls.json"),
    }
    content_sha256.update(
        {relative: sha256_file(root / relative) for relative in stem_paths.values()}
    )
    SampleManifest(
        sample_id=sample_id,
        stage="rendered",
        seed=seed,
        sample_rate=sample_rate,
        duration_seconds=float(duration_seconds),
        track_start_seconds=float(track_start_seconds),
        groups={
            "composition": "comp-test",
            "preset": ["preset-a"],
            "sample_origin": [],
        },
        mix_path="mix.wav",
        stem_paths=stem_paths,
        sources_path="sources.json",
        controls_path="controls.json",
        versions={"renderer": "test-fixture"},
        content_sha256=content_sha256,
    ).save(root / "manifest.json")
    return root


def update_manifest_hash(root: Path, relative: str) -> None:
    """Recompute one content hash after a test mutates a file."""

    manifest = SampleManifest.load(Path(root) / "manifest.json")
    payload = manifest.to_json_dict()
    payload["content_sha256"][relative] = sha256_file(Path(root) / relative)
    SampleManifest.from_json_dict(payload).save(Path(root) / "manifest.json")


@functools.lru_cache(maxsize=1)
def load_cli():
    spec = importlib.util.spec_from_file_location("aat_label_sample_cli", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
