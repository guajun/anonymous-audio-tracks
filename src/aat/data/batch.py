"""Multi-center window batches sampled from one song at a time.

A batch is a tuple of :class:`SourceWindowBlock` objects: one block per sampled
song, never a mix of two songs.  Each block keeps a *fixed* source column order
across all of its center rows (the order of ``sources.json``), so a source is
never re-assigned between rows:

``source_ids`` / ``slot_ids``
    The real sources and the full ``K`` slot mapping (``None`` marks padded
    empty slots).
``activity`` ``[N, K]``
    Center-time activity labels copied from the acoustic ``activity.npz``;
    padding slots are exactly 0.
``source_present`` ``[N, K]``
    ``True`` for a real source column even when that source is silent in the
    row, ``False`` for padding slots.  This is what separates "silent source"
    from "empty slot": a silent source has ``source_present=True`` and
    ``activity=0``, a padded slot has both ``False``/``0``.
``center_valid`` ``[N]``
    Protocol ``activity.valid``: the labeler's model window fits fully inside
    the rendered audio, so the row may be used as supervision.  ``False`` rows
    must not be supervised.
``audio`` / ``audio_valid`` ``[N, W]``
    Mono mix windows (arithmetic channel mean) and the shared-windowing mask of
    real samples vs zero padding.  A ``center_valid`` row normally has
    ``audio_valid.all()``; when ``window_seconds`` is overridden away from the
    labeling window the two masks can differ, and both must then be gated.

Same-source non-adjacent windows are supported through ``min_center_gap`` and
reported explicitly in ``same_source_pairs``.  ``source_note_ranges`` is
generation-time evidence copied from ``controls.json``: it records that the
sample contains different pitches, but MIDI never rewrites the acoustic labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ..contracts import ActivityData, ContractError
from ..contracts.errors import WindowError
from ..labels import LabelError, read_wav
from ..windowing import extract_windows_at_times, window_sample_count
from .errors import DatasetError
from .index import DatasetIndex, SampleEntry
from .split import SPLITS


class _UnusableSample(Exception):
    """Internal signal for an unsampleable (but capacity-valid) sample."""


@dataclass(frozen=True)
class SourceWindowBlock:
    """One song's multi-center window batch with fixed source columns."""

    sample_id: str
    path: str
    split: str
    sample_rate: int
    window_seconds: float
    window_samples: int
    hop_seconds: float | None
    track_start_seconds: float
    slots: int
    source_ids: tuple[str, ...]
    slot_ids: tuple[str | None, ...]
    center_times: np.ndarray
    center_indices: np.ndarray
    center_valid: np.ndarray
    activity: np.ndarray
    source_present: np.ndarray
    audio: np.ndarray
    audio_valid: np.ndarray
    same_source_pairs: tuple[tuple[int, int, int], ...]
    source_note_ranges: tuple[tuple[int | None, int | None], ...]
    activity_sha256: str
    mix_sha256: str

    @property
    def centers(self) -> int:
        return int(self.center_times.size)

    @property
    def real_sources(self) -> int:
        return len(self.source_ids)

    def to_json_dict(self, *, include_audio: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "sample_id": self.sample_id,
            "path": self.path,
            "split": self.split,
            "sample_rate": self.sample_rate,
            "window_seconds": self.window_seconds,
            "window_samples": self.window_samples,
            "hop_seconds": self.hop_seconds,
            "track_start_seconds": self.track_start_seconds,
            "slots": self.slots,
            "source_ids": list(self.source_ids),
            "slot_ids": list(self.slot_ids),
            "centers": self.centers,
            "center_times": [float(value) for value in self.center_times],
            "center_indices": [int(value) for value in self.center_indices],
            "center_valid": [bool(value) for value in self.center_valid],
            "activity": [[round(float(value), 6) for value in row] for row in self.activity],
            "source_present": [[bool(value) for value in row] for row in self.source_present],
            "audio_shape": [int(size) for size in self.audio.shape],
            "audio_valid_fraction": (
                float(self.audio_valid.mean()) if self.audio_valid.size else 0.0
            ),
            "audio_valid_all": bool(self.audio_valid.all()) if self.audio_valid.size else False,
            "same_source_pairs": [[int(slot), int(a), int(b)] for slot, a, b in self.same_source_pairs],
            "source_note_ranges": [
                [note_min, note_max] for note_min, note_max in self.source_note_ranges
            ],
            "activity_sha256": self.activity_sha256,
            "mix_sha256": self.mix_sha256,
        }
        if include_audio:
            data["audio"] = [[float(value) for value in row] for row in self.audio]
            data["audio_valid"] = [[bool(value) for value in row] for row in self.audio_valid]
        return data


@dataclass(frozen=True)
class DatasetBatch:
    """A deterministic multi-song batch of per-song window blocks."""

    blocks: tuple[SourceWindowBlock, ...]
    seed: int
    split: str
    requested_items: int
    centers_per_item: int
    min_center_gap: int
    valid_only: bool
    activity_threshold: float
    skipped: tuple[tuple[str, str], ...]

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(block.sample_id for block in self.blocks)

    def to_json_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "split": self.split,
            "requested_items": self.requested_items,
            "returned_items": len(self.blocks),
            "centers_per_item": self.centers_per_item,
            "min_center_gap": self.min_center_gap,
            "valid_only": self.valid_only,
            "activity_threshold": self.activity_threshold,
            "skipped": [
                {"sample_id": sample_id, "reason": reason}
                for sample_id, reason in self.skipped
            ],
            "blocks": [
                block.to_json_dict() if include_arrays else _block_summary(block)
                for block in self.blocks
            ],
        }


def sample_batch(
    index: DatasetIndex,
    data_root: str | Path,
    *,
    seed: int,
    split: str = "train",
    items: int | None = None,
    centers_per_item: int = 4,
    min_center_gap: int = 0,
    valid_only: bool = True,
    activity_threshold: float = 0.5,
    on_unusable: str = "error",
    window_seconds: float | None = None,
    slots: int | None = None,
) -> DatasetBatch:
    """Sample one multi-center block per selected song from ``index``.

    * ``items``: number of songs (``None`` = every song in the split).  Songs
      are selected deterministically from ``seed``; requesting more songs than
      the split holds is an error, never a silent repeat.
    * ``centers_per_item``: center rows per song (>= 1).
    * ``min_center_gap``: minimum distance between selected centers in center
      grid steps.  ``0``/``1`` allow adjacent windows; ``2`` forbids immediate
      neighbors, which is what "same-source non-adjacent windows" needs.
    * ``valid_only``: restrict centers to ``activity.valid`` rows (default).
      Set ``False`` to include boundary rows for mask inspection; the
      ``center_valid`` mask still marks them as unusable supervision.
    * ``on_unusable``: ``"error"`` (default) or ``"skip"`` for songs with no
      valid centers / too few usable centers; skips are reported in
      ``DatasetBatch.skipped``.  Exceeding the slot capacity ``K`` always
      raises, regardless of this option.

    The returned blocks never combine two songs.  Requested window extraction
    uses :func:`aat.windowing.extract_windows_at_times`, so time rounding and
    edge zero padding follow the shared protocol on any sample rate and
    non-zero ``track_start_seconds``.
    """

    seed_value = _require_seed(seed)
    centers_value = _require_positive_int(centers_per_item, "centers_per_item")
    gap_value = _require_non_negative_int(min_center_gap, "min_center_gap")
    if split not in SPLITS:
        raise DatasetError(f"split: expected one of {list(SPLITS)}, got {split!r}")
    if on_unusable not in ("error", "skip"):
        raise DatasetError(
            f"on_unusable: expected 'error' or 'skip', got {on_unusable!r}"
        )
    if not isinstance(valid_only, bool):
        raise DatasetError(
            f"valid_only: expected a boolean, got {type(valid_only).__name__}"
        )
    threshold = _require_threshold(activity_threshold)
    root = Path(data_root)
    effective_slots = (
        _require_positive_int(slots, "slots") if slots is not None else int(index.plan["slots"])
    )
    override_window = (
        None
        if window_seconds is None
        else _require_positive_number(window_seconds, "window_seconds")
    )

    entries = index.samples_for_split(split)
    if items is None:
        requested = len(entries)
    else:
        requested = _require_positive_int(items, "items")
        if requested > len(entries):
            raise DatasetError(
                f"items: requested {requested} song(s) from split {split!r} but only "
                f"{len(entries)} sample(s) are available"
            )
    rng = np.random.default_rng(seed_value)
    if requested == len(entries):
        selected = list(range(len(entries)))
    else:
        selected = sorted(int(value) for value in rng.permutation(len(entries))[:requested])

    blocks: list[SourceWindowBlock] = []
    skipped: list[tuple[str, str]] = []
    for position in selected:
        entry = entries[position]
        if len(entry.source_ids) > effective_slots:
            raise DatasetError(
                f"sample '{entry.sample_id}': {len(entry.source_ids)} source(s) exceed the "
                f"slots capacity K={effective_slots}; silent sources still count"
            )
        reason = _unusable_reason(entry)
        if reason is not None:
            _handle_unusable(entry.sample_id, reason, on_unusable, skipped)
            continue
        try:
            blocks.append(
                _build_block(
                    entry,
                    data_root=root,
                    rng=rng,
                    centers_per_item=centers_value,
                    min_center_gap=gap_value,
                    valid_only=valid_only,
                    activity_threshold=threshold,
                    window_seconds=override_window,
                    slots=effective_slots,
                )
            )
        except _UnusableSample as exc:
            _handle_unusable(entry.sample_id, str(exc), on_unusable, skipped)
    return DatasetBatch(
        blocks=tuple(blocks),
        seed=seed_value,
        split=split,
        requested_items=requested,
        centers_per_item=centers_value,
        min_center_gap=gap_value,
        valid_only=valid_only,
        activity_threshold=threshold,
        skipped=tuple(skipped),
    )


def _build_block(
    entry: SampleEntry,
    *,
    data_root: Path,
    rng: np.random.Generator,
    centers_per_item: int,
    min_center_gap: int,
    valid_only: bool,
    activity_threshold: float,
    window_seconds: float | None,
    slots: int,
) -> SourceWindowBlock:
    directory = data_root / PurePosixPath(entry.path)
    metadata = PurePosixPath(str(entry.labels["metadata_path"]))
    try:
        activity = ActivityData.load(directory / metadata.parent, metadata_filename=metadata.name)
    except (ContractError, OSError, LabelError) as exc:
        raise DatasetError(
            f"sample '{entry.sample_id}': cannot read activity labels: {exc}"
        ) from exc
    if tuple(activity.source_ids) != entry.source_ids:
        raise DatasetError(
            f"sample '{entry.sample_id}': activity source columns changed since the index "
            "was built; rebuild the index"
        )
    if int(activity.center_times.size) != entry.labels["center_count"] or int(
        activity.valid.sum()
    ) != entry.labels["valid_count"]:
        raise DatasetError(
            f"sample '{entry.sample_id}': activity arrays no longer match the index; "
            "rebuild the index"
        )
    if activity.sample_rate != entry.sample_rate:
        raise DatasetError(
            f"sample '{entry.sample_id}': activity sample_rate {activity.sample_rate} does "
            f"not match index sample_rate {entry.sample_rate}"
        )

    window = (
        float(entry.labels["center_window_seconds"])
        if window_seconds is None
        else float(window_seconds)
    )
    try:
        window_samples = window_sample_count(window, entry.sample_rate)
    except WindowError as exc:
        raise DatasetError(
            f"sample '{entry.sample_id}': invalid window_seconds={window}: {exc}"
        ) from exc

    total_rows = int(activity.center_times.size)
    if valid_only:
        candidates = np.nonzero(activity.valid)[0].astype(np.int64)
        if candidates.size == 0:
            raise _UnusableSample("no valid center windows (activity.valid is all False)")
    else:
        candidates = np.arange(total_rows, dtype=np.int64)
        if candidates.size == 0:
            raise _UnusableSample("activity has no center times")
    if candidates.size < centers_per_item:
        raise _UnusableSample(
            f"only {candidates.size} usable center(s) but {centers_per_item} requested"
        )

    mix = PurePosixPath(str(entry.audio["mix_path"]))
    try:
        wav = read_wav(directory / mix)
    except (LabelError, OSError) as exc:
        raise DatasetError(
            f"sample '{entry.sample_id}': cannot read {entry.audio['mix_path']}: {exc}"
        ) from exc
    if wav.sample_rate != entry.sample_rate:
        raise DatasetError(
            f"sample '{entry.sample_id}': mix sample_rate {wav.sample_rate} does not match "
            f"index sample_rate {entry.sample_rate}"
        )
    if wav.frames != int(entry.audio["frames"]):
        raise DatasetError(
            f"sample '{entry.sample_id}': mix frame count {wav.frames} no longer matches the "
            f"index ({entry.audio['frames']}); rebuild the index"
        )
    mono = wav.samples.mean(axis=1) if wav.channels > 1 else wav.samples[:, 0]

    audio_end = entry.track_start_seconds + wav.frames / wav.sample_rate
    times = activity.center_times
    inside = (times >= entry.track_start_seconds - 1e-9) & (times <= audio_end + 1e-9)
    candidates = candidates[inside[candidates]]
    if candidates.size == 0:
        raise _UnusableSample("no center time lies inside the actual audio span")
    if candidates.size < centers_per_item:
        raise _UnusableSample(
            f"only {candidates.size} in-audio center(s) but {centers_per_item} requested"
        )

    picked = _select_centers(candidates, centers_per_item, min_center_gap, rng)
    if picked is None:
        raise _UnusableSample(
            f"cannot select {centers_per_item} center(s) with min_center_gap={min_center_gap}"
        )
    picked_array = np.asarray(picked, dtype=np.int64)
    center_times = np.ascontiguousarray(times[picked_array], dtype=np.float64)
    try:
        windows, audio_valid = extract_windows_at_times(
            mono,
            center_times,
            wav.sample_rate,
            window,
            origin_seconds=entry.track_start_seconds,
        )
    except WindowError as exc:
        raise DatasetError(
            f"sample '{entry.sample_id}': cannot extract center windows: {exc}"
        ) from exc

    count = picked_array.size
    source_count = len(entry.source_ids)
    slot_ids: tuple[str | None, ...] = tuple(entry.source_ids) + (None,) * (slots - source_count)
    activity_block = np.zeros((count, slots), dtype=np.float32)
    activity_block[:, :source_count] = activity.activity[picked_array]
    source_present = np.zeros((count, slots), dtype=bool)
    source_present[:, :source_count] = True
    center_valid = np.ascontiguousarray(activity.valid[picked_array], dtype=bool)
    pairs = _same_source_pairs(
        activity_block,
        center_valid,
        picked_array,
        min_center_gap,
        activity_threshold,
        source_count,
    )
    note_ranges = tuple(
        (
            entry.controls["sources"][source_id]["note_min"],
            entry.controls["sources"][source_id]["note_max"],
        )
        for source_id in entry.source_ids
    )
    return SourceWindowBlock(
        sample_id=entry.sample_id,
        path=entry.path,
        split=entry.split,
        sample_rate=int(entry.sample_rate),
        window_seconds=float(window),
        window_samples=int(window_samples),
        hop_seconds=entry.labels["hop_seconds"],
        track_start_seconds=float(entry.track_start_seconds),
        slots=int(slots),
        source_ids=tuple(entry.source_ids),
        slot_ids=slot_ids,
        center_times=center_times,
        center_indices=picked_array,
        center_valid=center_valid,
        activity=activity_block,
        source_present=source_present,
        audio=np.ascontiguousarray(windows, dtype=np.float32),
        audio_valid=np.ascontiguousarray(audio_valid, dtype=bool),
        same_source_pairs=pairs,
        source_note_ranges=note_ranges,
        activity_sha256=entry.content_sha256.get(str(entry.labels["arrays_path"]), ""),
        mix_sha256=entry.content_sha256.get(str(entry.audio["mix_path"]), ""),
    )


def _select_centers(
    candidates: np.ndarray,
    count: int,
    min_center_gap: int,
    rng: np.random.Generator,
) -> list[int] | None:
    """Pick ``count`` sorted center indices with pairwise grid distance.

    Deterministic for a given ``rng`` state.  The backward greedy ``max_from``
    table decides feasibility; the forward pass picks a random index inside the
    range that still allows completing the selection, so the result is a valid
    (not uniform) sample.  Returns ``None`` when no valid selection exists.
    """

    values = np.asarray(candidates, dtype=np.int64)
    size = int(values.size)
    if count < 1 or count > size:
        return None
    step = max(int(min_center_gap), 1)
    max_from = np.zeros(size, dtype=np.int64)
    for index in range(size - 1, -1, -1):
        following = int(
            np.searchsorted(values, int(values[index]) + step, side="left")
        )
        max_from[index] = 1 + (int(max_from[following]) if following < size else 0)
    if int(max_from[0]) < count:
        return None

    picks: list[int] = []
    low = 0
    for position in range(count):
        remaining = count - position
        high = low
        while high + 1 < size and int(max_from[high + 1]) >= remaining:
            high += 1
        choice = int(rng.integers(low, high + 1))
        picks.append(int(values[choice]))
        low = int(np.searchsorted(values, int(values[choice]) + step, side="left"))
    return picks


def _same_source_pairs(
    activity: np.ndarray,
    center_valid: np.ndarray,
    center_indices: np.ndarray,
    min_center_gap: int,
    threshold: float,
    source_count: int,
) -> tuple[tuple[int, int, int], ...]:
    """Consecutive active rows of one source, far enough apart in the grid."""

    pairs: list[tuple[int, int, int]] = []
    for slot in range(source_count):
        active_rows = [
            row
            for row in range(activity.shape[0])
            if bool(center_valid[row]) and float(activity[row, slot]) >= threshold
        ]
        for left, right in zip(active_rows, active_rows[1:]):
            if int(center_indices[right]) - int(center_indices[left]) >= min_center_gap:
                pairs.append((slot, left, right))
    return tuple(pairs)


def _unusable_reason(entry: SampleEntry) -> str | None:
    if not entry.source_ids:
        return "no sources"
    if int(entry.labels["valid_count"]) == 0:
        return "no valid center windows (activity.valid is all False)"
    if int(entry.labels["center_count"]) == 0:
        return "activity has no center times"
    return None


def _handle_unusable(
    sample_id: str,
    reason: str,
    on_unusable: str,
    skipped: list[tuple[str, str]],
) -> None:
    if on_unusable == "error":
        raise DatasetError(f"sample '{sample_id}': {reason}")
    skipped.append((sample_id, reason))


def _block_summary(block: SourceWindowBlock) -> dict[str, Any]:
    data = block.to_json_dict(include_audio=False)
    data["audio"] = {
        "shape": [int(size) for size in block.audio.shape],
        "valid_fraction": float(block.audio_valid.mean()) if block.audio_valid.size else 0.0,
    }
    return data


def _require_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetError(f"seed: expected an integer, got {type(value).__name__}")
    if value < 0:
        raise DatasetError(f"seed: must be >= 0, got {value}")
    return int(value)


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetError(f"{name}: expected an integer, got {type(value).__name__}")
    if value < 1:
        raise DatasetError(f"{name}: must be >= 1, got {value}")
    return int(value)


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetError(f"{name}: expected an integer, got {type(value).__name__}")
    if value < 0:
        raise DatasetError(f"{name}: must be >= 0, got {value}")
    return int(value)


def _require_positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError(f"{name}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise DatasetError(f"{name}: must be finite and > 0, got {value!r}")
    return number


def _require_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError(
            f"activity_threshold: expected a number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise DatasetError(
            f"activity_threshold: must be finite and in [0, 1], got {value!r}"
        )
    return number


__all__ = [
    "DatasetBatch",
    "SourceWindowBlock",
    "sample_batch",
]
