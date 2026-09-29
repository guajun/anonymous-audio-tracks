"""Synthetic smoke corpus, grouped splits and dataset fingerprints (issue #8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aat.training.dataset import (
    DatasetPlan,
    dataset_fingerprint,
    load_dataset_plan,
    make_smoke_dataset,
)
from aat.training.errors import TrainingError

pytestmark = pytest.mark.ml


def test_smoke_corpus_is_leak_free_and_covers_all_splits(smoke_corpus):
    assert smoke_corpus["leak_free"] is True
    assert smoke_corpus["cross_split_assets"] == {
        "composition": 0,
        "preset": 0,
        "sample_origin": 0,
    }
    assert smoke_corpus["split_counts"]["train"] >= 1
    assert smoke_corpus["split_counts"]["val"] >= 1
    assert smoke_corpus["split_counts"]["test"] >= 1
    assert not smoke_corpus["empty_splits"]


def test_smoke_corpus_families_never_cross_splits(tmp_path: Path):
    dataset = make_smoke_dataset(tmp_path / "family-check", seed=20260929)
    from aat.training.dataset import load_verified_index

    index = load_verified_index(dataset["index_path"], data_root=dataset["data_root"])
    assets: dict[str, set[str]] = {}
    for entry in index.samples:
        for name in ("preset", "sample_origin"):
            for value in entry.groups[name]:
                assets.setdefault(value, set()).add(entry.split)
        assets.setdefault(entry.groups["composition"], set()).add(entry.split)
    assert all(len(splits) == 1 for splits in assets.values())
    train_ids = {entry.sample_id for entry in index.samples if entry.split == "train"}
    val_ids = {entry.sample_id for entry in index.samples if entry.split == "val"}
    test_ids = {entry.sample_id for entry in index.samples if entry.split == "test"}
    assert train_ids and val_ids and test_ids
    assert train_ids.isdisjoint(val_ids | test_ids)


def test_fingerprint_is_stable_and_changes_with_content(tmp_path: Path):
    first = make_smoke_dataset(tmp_path / "fingerprint-a", seed=20260929)
    second = make_smoke_dataset(tmp_path / "fingerprint-b", seed=20260929)
    assert first["fingerprint"]["digest"] == second["fingerprint"]["digest"]

    from aat.training.dataset import load_verified_index
    from tests.training.support import rehash_manifest_mix, rewrite_wav_different_content

    index = load_verified_index(first["index_path"], data_root=first["data_root"])
    entry = next(entry for entry in index.samples if entry.split == "train")
    sample_dir = Path(first["data_root"]) / entry.path
    rewrite_wav_different_content(sample_dir / "mix.wav")
    rehash_manifest_mix(sample_dir)
    # Rebuild the index over the changed sample; the recorded content digest
    # must change, which is exactly what resume compares.
    from aat.data import build_dataset_index

    rebuilt = build_dataset_index(
        first["data_root"],
        seed=20260929,
        ratios=(0.5, 0.25, 0.25),
        slots=8,
        window_seconds=2.0,
        index_dir=Path(first["index_path"]).parent,
    )
    assert dataset_fingerprint(rebuilt)["digest"] != first["fingerprint"]["digest"]


def test_committed_dataset_plan_matches_families():
    plan = load_dataset_plan("configs/train/dataset_local.toml")
    assert isinstance(plan, DatasetPlan)
    assert plan.sample_ids == (
        "train-01",
        "train-02",
        "train-03",
        "train-04",
        "val-01",
        "val-02",
        "test-01",
        "test-02",
    )
    train_presets = set()
    val_presets = set()
    test_presets = set()
    for song in plan.songs:
        target = (
            train_presets
            if song.sample_id.startswith("train")
            else val_presets
            if song.sample_id.startswith("val")
            else test_presets
        )
        target.update(song.presets)
        target.update(song.sample_origins)
    assert train_presets.isdisjoint(val_presets | test_presets)
    assert val_presets.isdisjoint(test_presets)


def test_existing_corpus_needs_explicit_overwrite(tmp_path: Path):
    make_smoke_dataset(tmp_path / "existing", seed=20260929)
    with pytest.raises(TrainingError, match="already exists"):
        make_smoke_dataset(tmp_path / "existing", seed=20260929)
