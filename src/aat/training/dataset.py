"""Training data preparation and dataset fingerprints (issue #8).

This module is deliberately torch-free so that a render-only environment can
run ``scripts/train.py prepare-data`` and ``smoke`` dataset setup without
installing the ML extra:

* :func:`load_dataset_plan` validates a TOML dataset plan and resolves one
  :class:`aat.render.config.RenderConfig` per song (the plan is the same
  DawDreamer graph configuration used by issue #3, including explicit
  ``preset_ref``/``sample_ref`` family labels).
* :func:`make_smoke_dataset` writes a tiny deterministic labeled corpus with
  real protocol documents and real PCM audio, for the CPU fake-encoder smoke
  and for training tests.  It does **not** use DawDreamer; the committed
  evidence for real renders comes from ``prepare-data``.
* :func:`dataset_fingerprint` builds the content digest that a checkpoint
  records so resume can refuse silently replaced data.

Two dataset families matter for leakage safety: every song belongs to one
``composition`` and carries ``preset``/``sample_origin`` asset lists.  The
plan writer below lets songs share an asset label only inside one split
family; the index then keeps the whole connected group in one split.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from aat.contracts import (
    ActivityData,
    ControlEvent,
    Controls,
    SampleManifest,
    SourceEntry,
    SourceRegistry,
)
from aat.data import DEFAULT_RATIOS, DatasetIndex, build_dataset_index
from aat.labels import LabelConfig, label_stems
from aat.render.config import RenderConfig, config_from_dict

from .config import SPLITS, TrainingError, canonical_json, sha256_canonical

#: File name of the generated dataset plan index inside a data root.
SMOKE_INDEX_FILENAME = "index.json"


def _check_keys(table: Mapping[str, Any], allowed: tuple[str, ...], path: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise TrainingError(f"{path}: unknown key(s) {unknown}; allowed keys: {list(allowed)}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingError(f"{path}: expected a non-empty string")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError(f"{path}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise TrainingError(f"{path}: must be finite, got {value!r}")
    if minimum is not None and number < minimum:
        raise TrainingError(f"{path}: must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise TrainingError(f"{path}: must be <= {maximum}, got {number}")
    return number


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingError(f"{path}: expected an integer")
    if minimum is not None and value < minimum:
        raise TrainingError(f"{path}: must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class DatasetPlan:
    """A validated DawDreamer dataset plan (issue #8 training data only)."""

    name: str
    seed: int
    ratios: Mapping[str, float]
    slots: int
    window_seconds: float
    songs: tuple[RenderConfig, ...]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(song.sample_id for song in self.songs)


def load_dataset_plan(path: str | Path) -> DatasetPlan:
    """Parse and validate a TOML dataset plan, one render config per song."""

    source = Path(path)
    if not source.is_file():
        raise TrainingError(f"{source}: dataset plan does not exist")
    try:
        with open(source, "rb") as handle:
            payload = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise TrainingError(f"{source}: invalid TOML: {error}") from error
    _check_keys(payload, ("dataset", "songs"), str(source))
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TrainingError(f"{source}: missing [dataset] table")
    _check_keys(
        dataset,
        ("name", "seed", "sample_rate", "block_size", "ratios", "slots", "window_seconds"),
        f"{source}: [dataset]",
    )
    name = _string(dataset.get("name"), f"{source}: [dataset].name")
    seed = _integer(dataset.get("seed"), f"{source}: [dataset].seed", minimum=0)
    sample_rate = _integer(
        dataset.get("sample_rate", 44100), f"{source}: [dataset].sample_rate", minimum=8000
    )
    block_size = _integer(
        dataset.get("block_size", 512), f"{source}: [dataset].block_size", minimum=1
    )

    ratios_raw = dataset.get("ratios", list(DEFAULT_RATIOS.values()))
    if isinstance(ratios_raw, Mapping):
        ratio_map = {key: _number(value, f"{source}: [dataset].ratios.{key}", minimum=0.0)
                     for key, value in ratios_raw.items()}
    elif isinstance(ratios_raw, list) and len(ratios_raw) == 3:
        ratio_map = {
            split: _number(value, f"{source}: [dataset].ratios[{index}]", minimum=0.0)
            for index, (split, value) in enumerate(zip(SPLITS, ratios_raw))
        }
    else:
        raise TrainingError(
            f"{source}: [dataset].ratios must be a 3-element array or a table with split ratios"
        )
    if set(ratio_map) != set(SPLITS):
        raise TrainingError(f"{source}: [dataset].ratios must name train/val/test exactly")
    if sum(ratio_map.values()) <= 0:
        raise TrainingError(f"{source}: [dataset].ratios must not be all zero")
    slots = _integer(dataset.get("slots", 8), f"{source}: [dataset].slots", minimum=1)
    window_seconds = _number(
        dataset.get("window_seconds", 2.0),
        f"{source}: [dataset].window_seconds",
        minimum=0.0,
    )
    if window_seconds <= 0:
        raise TrainingError(f"{source}: [dataset].window_seconds must be > 0")

    songs_raw = payload.get("songs")
    if not isinstance(songs_raw, list) or not songs_raw:
        raise TrainingError(f"{source}: [[songs]] must be a non-empty array of tables")
    songs: list[RenderConfig] = []
    seen: set[str] = set()
    for index, raw in enumerate(songs_raw):
        if not isinstance(raw, Mapping):
            raise TrainingError(f"{source}: songs[{index}] must be a table")
        song_path = f"{source}: songs[{index}]"
        _check_keys(
            raw,
            ("sample_id", "composition", "seed", "bpm", "duration_seconds", "tail_seconds", "notes", "sources"),
            song_path,
        )
        sample_id = _string(raw.get("sample_id"), f"{song_path}.sample_id")
        if sample_id in seen:
            raise TrainingError(f"{song_path}.sample_id: duplicate sample id {sample_id!r}")
        seen.add(sample_id)
        sources = raw.get("sources")
        if not isinstance(sources, list) or not sources:
            raise TrainingError(f"{song_path}.sources: expected a non-empty array of tables")
        render_table: dict[str, Any] = {
            "sample_id": sample_id,
            "composition": _string(raw.get("composition"), f"{song_path}.composition"),
            "seed": _integer(raw.get("seed", seed + index), f"{song_path}.seed", minimum=0),
            "sample_rate": sample_rate,
            "block_size": block_size,
            "bpm": _number(raw.get("bpm", 100.0), f"{song_path}.bpm", minimum=1.0),
            "duration_seconds": _number(
                raw.get("duration_seconds"),
                f"{song_path}.duration_seconds",
                minimum=0.0,
            ),
            "tail_seconds": _number(
                raw.get("tail_seconds", 1.5), f"{song_path}.tail_seconds", minimum=0.0
            ),
        }
        if render_table["duration_seconds"] <= 0:
            raise TrainingError(f"{song_path}.duration_seconds: must be > 0")
        if raw.get("notes") is not None:
            render_table["notes"] = _string(raw.get("notes"), f"{song_path}.notes")
        try:
            song = config_from_dict({"render": render_table, "sources": sources}, origin=sample_id)
        except Exception as error:  # RenderConfigError / unknown-key errors carry the path
            raise TrainingError(f"{song_path}: invalid render configuration: {error}") from error
        songs.append(song)
    return DatasetPlan(
        name=name,
        seed=seed,
        ratios=ratio_map,
        slots=slots,
        window_seconds=window_seconds,
        songs=tuple(songs),
    )


# --------------------------------------------------------------------------- #
# Dataset fingerprint / load helpers
# --------------------------------------------------------------------------- #


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataset_fingerprint(index: DatasetIndex) -> dict[str, Any]:
    """Content digest of an index for checkpointing and resume verification.

    The digest covers the index plan, every sample record (``sample_sha256``
    already covers the whole record) and every recorded file digest, so a
    rebuilt index over changed audio/labels changes the digest even when all
    shapes stay the same.
    """

    records = []
    for entry in index.samples:
        records.append(
            {
                "sample_id": entry.sample_id,
                "split": entry.split,
                "path": entry.path,
                "sample_sha256": entry.sample_sha256,
                "content_sha256": {
                    key: entry.content_sha256[key] for key in sorted(entry.content_sha256)
                },
                "groups": _plain(entry.groups),
                "source_ids": list(entry.source_ids),
                "labels": _plain(entry.labels),
            }
        )
    records.sort(key=lambda item: item["sample_id"])
    per_split = {
        split: sha256_canonical([item for item in records if item["split"] == split])
        for split in SPLITS
    }
    counts = {split: 0 for split in SPLITS}
    for item in records:
        counts[item["split"]] = counts.get(item["split"], 0) + 1
    return {
        "index_version": index.index_version,
        "plan": _plain(index.plan),
        "digest": sha256_canonical(records),
        "per_split": per_split,
        "split_counts": counts,
        "sample_count": len(records),
    }


def index_file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_verified_index(index_path: str | Path, *, data_root: str | Path | None = None) -> DatasetIndex:
    """Load an index and verify every referenced file before training starts."""

    return DatasetIndex.load(index_path, data_root=data_root, verify_files=True)


# --------------------------------------------------------------------------- #
# Synthetic smoke corpus (no DawDreamer)
# --------------------------------------------------------------------------- #


def _sine_bursts(
    duration_seconds: float,
    sample_rate: int,
    notes: Sequence[tuple[float, float, float]],
    *,
    freq_hz: float,
    decay_seconds: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Deterministic decaying sine bursts ``(start, length, amplitude)``."""

    frames = int(round(duration_seconds * sample_rate))
    t = np.arange(frames, dtype=np.float64) / sample_rate
    buffer = np.zeros(frames, dtype=np.float64)
    for start, length, amplitude in notes:
        mask = (t >= start) & (t < start + length)
        local = t[mask] - start
        envelope = np.exp(-local / max(decay_seconds, 1.0 / sample_rate))
        buffer[mask] += amplitude * envelope * np.sin(2.0 * np.pi * freq_hz * (local + phase))
    return buffer


def _write_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.asarray(samples, dtype=np.float64)
    clipped = np.clip(np.round(payload * 32767.0), -32768.0, 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(clipped.tobytes())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SyntheticSource:
    source_id: str
    preset_ref: str
    sample_ref: str
    notes: tuple[tuple[float, float, float], ...]
    freq_hz: float
    decay_seconds: float


@dataclass(frozen=True)
class _SmokeSong:
    sample_id: str
    composition: str
    duration_seconds: float
    preset_assets: tuple[str, ...]
    sample_assets: tuple[str, ...]
    sources: tuple[SyntheticSource, ...] = field(default_factory=tuple)


def _smoke_songs(seed: int) -> tuple[_SmokeSong, ...]:
    """Four short songs in three disjoint asset families.

    ``smoke-train-01`` and ``smoke-train-02`` share one preset/sample asset so
    the index must keep them in the same split component; ``val``/``test`` use
    assets that never appear in train.  This is a synthetic fixture corpus, not
    a DawDreamer render: it proves the training pipeline and leakage handling,
    not audio quality.
    """

    train_shared_preset = "aat/smoke/train/tone-a"
    train_shared_origin = "generated/smoke/train/tone-a"
    return (
        _SmokeSong(
            sample_id="smoke-train-01",
            composition="comp-smoke-train-a",
            duration_seconds=6.0,
            preset_assets=(train_shared_preset, "aat/smoke/train/low-b"),
            sample_assets=(train_shared_origin, "generated/smoke/train/low-b"),
            sources=(
                SyntheticSource(
                    "s01",
                    train_shared_preset,
                    train_shared_origin,
                    ((0.0, 0.45, 0.55), (1.0, 0.45, 0.5), (2.0, 0.5, 0.55),
                     (3.5, 0.4, 0.5), (4.5, 0.5, 0.55)),
                    freq_hz=440.0,
                    decay_seconds=0.25,
                ),
                SyntheticSource(
                    "s02",
                    "aat/smoke/train/low-b",
                    "generated/smoke/train/low-b",
                    ((0.5, 0.8, 0.6), (2.5, 0.8, 0.6), (4.8, 0.8, 0.6)),
                    freq_hz=110.0,
                    decay_seconds=0.5,
                ),
            ),
        ),
        _SmokeSong(
            sample_id="smoke-train-02",
            composition="comp-smoke-train-b",
            duration_seconds=5.0,
            preset_assets=(train_shared_preset,),
            sample_assets=(train_shared_origin,),
            sources=(
                SyntheticSource(
                    "s01",
                    train_shared_preset,
                    train_shared_origin,
                    ((0.2, 0.4, 0.6), (1.4, 0.4, 0.55), (3.0, 0.45, 0.6), (4.0, 0.4, 0.55)),
                    freq_hz=440.0,
                    decay_seconds=0.25,
                ),
            ),
        ),
        _SmokeSong(
            sample_id="smoke-val-01",
            composition="comp-smoke-val-a",
            duration_seconds=5.0,
            preset_assets=("aat/smoke/val/tone-c", "aat/smoke/val/low-d"),
            sample_assets=("generated/smoke/val/tone-c", "generated/smoke/val/low-d"),
            sources=(
                SyntheticSource(
                    "s01",
                    "aat/smoke/val/tone-c",
                    "generated/smoke/val/tone-c",
                    ((0.0, 0.35, 0.6), (0.9, 0.35, 0.5), (2.4, 0.4, 0.6), (3.6, 0.35, 0.5)),
                    freq_hz=523.25,
                    decay_seconds=0.18,
                ),
                SyntheticSource(
                    "s02",
                    "aat/smoke/val/low-d",
                    "generated/smoke/val/low-d",
                    ((0.4, 0.7, 0.55), (1.8, 0.7, 0.5), (3.2, 0.7, 0.55), (4.4, 0.6, 0.5)),
                    freq_hz=146.83,
                    decay_seconds=0.4,
                ),
            ),
        ),
        _SmokeSong(
            sample_id="smoke-test-01",
            composition="comp-smoke-test-a",
            duration_seconds=5.0,
            preset_assets=("aat/smoke/test/tone-e",),
            sample_assets=("generated/smoke/test/tone-e",),
            sources=(
                SyntheticSource(
                    "s01",
                    "aat/smoke/test/tone-e",
                    "generated/smoke/test/tone-e",
                    ((0.1, 0.3, 0.6), (1.2, 0.3, 0.55), (2.0, 0.35, 0.6), (3.3, 0.3, 0.55), (4.2, 0.3, 0.6)),
                    freq_hz=659.25,
                    decay_seconds=0.2,
                ),
            ),
        ),
    )


def write_synthetic_sample(
    directory: str | Path,
    *,
    sample_id: str,
    composition: str,
    sources: Sequence[SyntheticSource],
    duration_seconds: float,
    sample_rate: int = 16000,
    seed: int = 0,
    preset_assets: Sequence[str] | None = None,
    sample_assets: Sequence[str] | None = None,
    label_config: LabelConfig | None = None,
) -> Path:
    """Write one labeled synthetic protocol sample (real documents + PCM).

    Reused by :func:`make_smoke_dataset` and by tests that need a sample which
    is too short for a 2 s center window (``activity.valid`` all False).
    """

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    label_config = label_config if label_config is not None else LabelConfig()
    stems: dict[str, np.ndarray] = {}
    for source in sources:
        signal = _sine_bursts(
            duration_seconds,
            sample_rate,
            source.notes,
            freq_hz=source.freq_hz,
            decay_seconds=source.decay_seconds,
        )
        stems[source.source_id] = signal
        _write_pcm16(target / "stems" / f"{source.source_id}.wav", signal, sample_rate)
    mix = np.sum(np.stack(list(stems.values())), axis=0)
    _write_pcm16(target / "mix.wav", mix, sample_rate)
    source_ids = tuple(stems)

    SourceRegistry(
        sources=tuple(
            SourceEntry(source_id=source_id, index=index, renderer="smoke-synth")
            for index, source_id in enumerate(source_ids)
        ),
        sample_id=sample_id,
    ).save(target / "sources.json")
    Controls(sample_id=sample_id, events=()).save(target / "controls.json")
    result = label_stems(
        stems,
        source_ids=source_ids,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        track_start_seconds=0.0,
        config=label_config,
        sample_id=sample_id,
    )
    result.activity.save(target)

    content_sha256: dict[str, str] = {
        "mix.wav": _sha256_file(target / "mix.wav"),
        "sources.json": _sha256_file(target / "sources.json"),
        "controls.json": _sha256_file(target / "controls.json"),
    }
    for source_id in source_ids:
        relative = f"stems/{source_id}.wav"
        content_sha256[relative] = _sha256_file(target / relative)
    content_sha256["activity.json"] = _sha256_file(target / "activity.json")
    content_sha256["activity.npz"] = _sha256_file(target / "activity.npz")
    SampleManifest.from_json_dict(
        {
            "schema_version": "0.1.0",
            "kind": "sample_manifest",
            "sample_id": sample_id,
            "stage": "labeled",
            "seed": seed,
            "sample_rate": sample_rate,
            "duration_seconds": float(duration_seconds),
            "track_start_seconds": 0.0,
            "groups": {
                "composition": composition,
                "preset": list(preset_assets if preset_assets is not None else ()),
                "sample_origin": list(sample_assets if sample_assets is not None else ()),
            },
            "mix_path": "mix.wav",
            "stem_paths": {source_id: f"stems/{source_id}.wav" for source_id in source_ids},
            "sources_path": "sources.json",
            "controls_path": "controls.json",
            "activity_metadata_path": "activity.json",
            "activity_arrays_path": "activity.npz",
            "versions": {"renderer": "aat-smoke-synth", "protocol": "0.1.0"},
            "content_sha256": content_sha256,
            "notes": "synthetic sample for issue #8; not a DawDreamer render",
        }
    ).save(target / "manifest.json")
    return target


def make_smoke_dataset(
    dataset_root: str | Path,
    *,
    seed: int = 20260929,
    sample_rate: int = 16000,
    force: bool = False,
) -> dict[str, Any]:
    """Write a tiny labeled corpus with real protocol documents and an index.

    Returns a summary (``data_root``, ``index_path``, split counts, leakage
    evidence).  The corpus is deterministic for a given ``seed``; existing
    files are refused unless ``force`` is true.
    """

    root = Path(dataset_root)
    if root.exists() and any(root.iterdir()) and not force:
        raise TrainingError(
            f"{root}: already exists and is not empty; pass force=True/--overwrite to regenerate"
        )
    samples_root = root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    label_config = LabelConfig()
    written: list[str] = []
    for song in _smoke_songs(seed):
        directory = samples_root / song.sample_id
        write_synthetic_sample(
            directory,
            sample_id=song.sample_id,
            composition=song.composition,
            sources=song.sources,
            duration_seconds=song.duration_seconds,
            sample_rate=sample_rate,
            seed=seed,
            preset_assets=song.preset_assets,
            sample_assets=song.sample_assets,
            label_config=label_config,
        )
        written.append(song.sample_id)

    index = build_dataset_index(
        samples_root,
        seed=seed,
        ratios=(0.5, 0.25, 0.25),
        slots=8,
        window_seconds=2.0,
        index_dir=root,
    )
    index_path = root / SMOKE_INDEX_FILENAME
    index.save(index_path)
    summary = index.summary
    return {
        "data_root": str(samples_root),
        "index_path": str(index_path),
        "sample_ids": written,
        "split_counts": dict(summary["split_counts"]),
        "cross_split_assets": dict(summary["cross_split_assets"]),
        "leak_free": bool(summary["leak_free"]),
        "empty_splits": list(summary["empty_splits"]),
        "warnings": list(summary["warnings"]),
        "fingerprint": dataset_fingerprint(index),
        "synthetic": True,
    }


__all__ = [
    "DatasetPlan",
    "SMOKE_INDEX_FILENAME",
    "dataset_fingerprint",
    "index_file_sha256",
    "load_dataset_plan",
    "load_verified_index",
    "SyntheticSource",
    "make_smoke_dataset",
    "write_synthetic_sample",
]
