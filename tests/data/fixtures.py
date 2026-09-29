"""Synthetic labeled-sample fixtures for dataset tests.

Nothing here uses DawDreamer: the fixtures write real protocol documents and
real PCM WAV files, then either run the shared :func:`aat.labels.label_stems`
labeler or accept explicit activity arrays so a test can control masks exactly.
The integration test (``test_integration_render_chain.py``) is the only place
that exercises the real renderer.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aat.contracts import (
    ActivityData,
    ControlEvent,
    Controls,
    SampleManifest,
    SourceEntry,
    SourceRegistry,
    dump_json,
)
from aat.labels import LabelConfig, label_stems
from aat.windowing import center_times, centered_window_bounds, window_sample_count

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_CLI = REPO_ROOT / "scripts" / "build_dataset_index.py"
LABEL_CLI = REPO_ROOT / "scripts" / "label_sample.py"
RATE = 16000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write 1-D or (frames, channels) float audio as 16-bit PCM."""

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


def tone_bursts(
    duration_seconds: float,
    sample_rate: int,
    bursts: Sequence[tuple[float, float]],
    *,
    freq_hz: float = 440.0,
    amplitude: float = 0.6,
) -> np.ndarray:
    """Sum of rectangular sine bursts ``[start, stop)`` on one 1-D buffer."""

    frames = round(duration_seconds * sample_rate)
    t = np.arange(frames, dtype=np.float64) / sample_rate
    buffer = np.zeros(frames, dtype=np.float64)
    for start, stop in bursts:
        mask = (t >= start) & (t < stop)
        buffer[mask] += amplitude * np.sin(2.0 * np.pi * freq_hz * t)[mask]
    return buffer


def default_valid(
    centers: np.ndarray,
    *,
    sample_rate: int,
    duration_seconds: float,
    track_start_seconds: float,
    center_window_seconds: float,
) -> np.ndarray:
    """Mirror the labeler's ``valid`` semantics for manual activity fixtures."""

    frames = round(duration_seconds * sample_rate)
    local = np.asarray(centers, dtype=np.float64) - track_start_seconds
    center_samples = np.floor(local * sample_rate + 0.5).astype(np.int64)
    width = window_sample_count(center_window_seconds, sample_rate)
    valid = np.zeros(center_samples.shape, dtype=bool)
    if center_samples.size:
        starts = np.array(
            [centered_window_bounds(int(center), width)[0] for center in center_samples],
            dtype=np.int64,
        )
        stops = np.array(
            [centered_window_bounds(int(center), width)[1] for center in center_samples],
            dtype=np.int64,
        )
        valid = (starts >= 0) & (stops <= frames)
    return valid


def manual_activity(
    *,
    sample_id: str,
    source_ids: Sequence[str],
    sample_rate: int = RATE,
    duration_seconds: float = 4.0,
    track_start_seconds: float = 0.0,
    hop_seconds: float = 0.02,
    activity: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    label_config: LabelConfig | None = None,
    label_params: Mapping[str, Any] | None = None,
) -> ActivityData:
    """Build an explicit activity document for precise mask/pair tests."""

    config = label_config if label_config is not None else LabelConfig()
    centers = center_times(duration_seconds, hop_seconds, origin_seconds=track_start_seconds)
    rows = int(centers.size)
    ids = tuple(source_ids)
    if activity is None:
        array = np.zeros((rows, len(ids)), dtype=np.float32)
    else:
        array = np.asarray(activity, dtype=np.float32)
        if array.shape != (rows, len(ids)):
            raise ValueError(f"activity shape {array.shape} != {(rows, len(ids))}")
    if valid is None:
        mask = default_valid(
            centers,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            track_start_seconds=track_start_seconds,
            center_window_seconds=config.center_window_seconds,
        )
    else:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != (rows,):
            raise ValueError(f"valid shape {mask.shape} != {(rows,)}")
    if label_params is None:
        label_params = config.to_label_params()
    return ActivityData(
        center_times=centers,
        activity=array,
        valid=mask,
        source_ids=ids,
        sample_rate=sample_rate,
        hop_seconds=hop_seconds,
        sample_id=sample_id,
        label_params=label_params,
    )


def make_sample(
    data_root: Path,
    relative: str,
    *,
    sample_id: str,
    composition: str,
    presets: Sequence[str] = (),
    sample_origins: Sequence[str] = (),
    stems: Mapping[str, np.ndarray],
    sample_rate: int = RATE,
    duration_seconds: float = 4.0,
    track_start_seconds: float = 0.0,
    controls_events: Sequence[Mapping[str, Any]] = (),
    seed: int = 7,
    label_config: LabelConfig | None = None,
    activity: ActivityData | None = None,
    labeled: bool = True,
    channels: int = 1,
) -> Path:
    """Write one protocol sample directory and return it.

    ``stems`` must be 1-D arrays of exactly ``duration_seconds``; the mix is the
    sum of the stems.  With ``labeled=True`` an ``activity.json``/``.npz`` pair
    is produced (shared labeler unless ``activity`` is given) and the manifest
    is advanced to ``stage = "labeled"`` with both digests.
    """

    directory = Path(data_root) / relative
    directory.mkdir(parents=True, exist_ok=True)
    frames = round(duration_seconds * sample_rate)
    ids = list(stems)
    source_ids = tuple(ids)
    stem_arrays: dict[str, np.ndarray] = {}
    for source_id in ids:
        array = np.asarray(stems[source_id], dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"stem {source_id}: expected a 1-D array")
        if array.shape[0] != frames:
            raise ValueError(
                f"stem {source_id}: {array.shape[0]} frames != {frames} expected"
            )
        stem_arrays[source_id] = array

    mix = np.zeros(frames, dtype=np.float64)
    for source_id in ids:
        mix += stem_arrays[source_id]
    if channels < 1:
        raise ValueError("channels must be >= 1")
    if channels == 1:
        write_pcm16(directory / "mix.wav", mix, sample_rate)
    else:
        write_pcm16(directory / "mix.wav", np.repeat(mix[:, None], channels, axis=1), sample_rate)

    stem_paths: dict[str, str] = {}
    for source_id in ids:
        relative_stem = f"stems/{source_id}.wav"
        array = stem_arrays[source_id]
        payload = array if channels == 1 else np.repeat(array[:, None], channels, axis=1)
        write_pcm16(directory / relative_stem, payload, sample_rate)
        stem_paths[source_id] = relative_stem

    SourceRegistry(
        sources=tuple(
            SourceEntry(source_id=source_id, index=index, renderer="fixture")
            for index, source_id in enumerate(ids)
        ),
        sample_id=sample_id,
    ).save(directory / "sources.json")
    Controls(
        events=tuple(ControlEvent(**dict(event)) for event in controls_events),
        sample_id=sample_id,
    ).save(directory / "controls.json")

    content_sha256 = {
        "mix.wav": sha256_file(directory / "mix.wav"),
        "sources.json": sha256_file(directory / "sources.json"),
        "controls.json": sha256_file(directory / "controls.json"),
    }
    for relative_stem in stem_paths.values():
        content_sha256[relative_stem] = sha256_file(directory / relative_stem)

    manifest_payload: dict[str, Any] = {
        "sample_id": sample_id,
        "stage": "rendered",
        "seed": seed,
        "sample_rate": sample_rate,
        "duration_seconds": float(duration_seconds),
        "track_start_seconds": float(track_start_seconds),
        "groups": {
            "composition": composition,
            "preset": list(presets),
            "sample_origin": list(sample_origins),
        },
        "mix_path": "mix.wav",
        "stem_paths": stem_paths,
        "sources_path": "sources.json",
        "controls_path": "controls.json",
        "versions": {"renderer": "fixture"},
        "content_sha256": content_sha256,
    }

    if labeled:
        if activity is None:
            result = label_stems(
                stem_arrays,
                source_ids=source_ids,
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
                track_start_seconds=track_start_seconds,
                config=label_config,
                sample_id=sample_id,
            )
            activity_data = result.activity
        else:
            activity_data = activity
            if tuple(activity_data.source_ids) != source_ids:
                raise ValueError("activity source_ids must match the stems keys")
            if activity_data.sample_id is None:
                activity_data.sample_id = sample_id
        activity_data.save(directory)
        manifest_payload["stage"] = "labeled"
        manifest_payload["activity_metadata_path"] = "activity.json"
        manifest_payload["activity_arrays_path"] = "activity.npz"
        content_sha256["activity.json"] = sha256_file(directory / "activity.json")
        content_sha256["activity.npz"] = sha256_file(directory / "activity.npz")

    SampleManifest.from_json_dict(
        {"schema_version": "0.1.0", "kind": "sample_manifest", **manifest_payload}
    ).save(directory / "manifest.json")
    return directory


def rehash_manifest(sample_dir: Path, *relative_paths: str) -> None:
    """Recompute manifest digests after a test intentionally mutates a file."""

    manifest_path = Path(sample_dir) / "manifest.json"
    manifest = SampleManifest.load(manifest_path)
    payload = manifest.to_json_dict()
    for relative in relative_paths:
        payload["content_sha256"][relative] = sha256_file(Path(sample_dir) / relative)
    SampleManifest.from_json_dict(payload).save(manifest_path)


def rewrite_json(path: Path, mutate) -> None:
    """Load a JSON document, mutate it in place and write it back."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mutate(payload)
    dump_json(path, payload)


def load_cli(path: Path, name: str):
    """Import a ``scripts/*.py`` module under a unique name."""

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
