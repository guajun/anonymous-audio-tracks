"""Padding masks, padded source columns and zero-supervision diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aat.data import sample_batch
from aat.models import SourceQueryHead
from aat.training.batching import training_batch_from_blocks
from aat.training.dataset import SyntheticSource, write_synthetic_sample
from aat.training.encoding import FakeEncoderAdapter, window_start_seconds
from aat.training.trainer import train_from_config
from tests.training.support import read_log, smoke_config

pytestmark = pytest.mark.ml


def _encode_blocks(config, index, data_root, batch):
    encoder = FakeEncoderAdapter(config.encoder)
    encoded = []
    for block in batch.blocks:
        starts = window_start_seconds(
            block.center_times,
            sample_rate=block.sample_rate,
            window_seconds=2.0,
            origin_seconds=block.track_start_seconds,
        )
        encoded.append(
            encoder.encode_windows(
                block.audio,
                block.sample_rate,
                window_start_seconds=starts,
                valid_samples=block.audio_valid,
            )
        )
    return training_batch_from_blocks(batch.blocks, encoded)


def _short_sample_corpus(root: Path):
    """One 1 s labeled sample: no 2 s center window fits (all valid False)."""

    from aat.data import build_dataset_index

    sample_dir = root / "samples" / "short-01"
    source = SyntheticSource(
        "s01",
        "aat/short/tone-v1",
        "generated/short/tone-v1",
        ((0.1, 0.5, 0.6),),
        440.0,
        0.2,
    )
    write_synthetic_sample(
        sample_dir,
        sample_id="short-01",
        composition="comp-short",
        sources=(source,),
        duration_seconds=1.0,
        sample_rate=16000,
        preset_assets=("aat/short/tone-v1",),
        sample_assets=("generated/short/tone-v1",),
    )
    index = build_dataset_index(
        root / "samples",
        seed=20260929,
        ratios=(1.0, 0.0, 0.0),
        slots=8,
        window_seconds=2.0,
        index_dir=root,
    )
    index_path = root / "index.json"
    index.save(index_path)
    return {"index_path": str(index_path), "data_root": str(root / "samples")}


def test_padded_source_columns_never_enter_the_loss(smoke_corpus, tmp_path: Path):
    from aat.training.dataset import load_verified_index

    config = smoke_config(smoke_corpus, tmp_path / "padded", steps=1)
    index = load_verified_index(smoke_corpus["index_path"], data_root=smoke_corpus["data_root"])
    batch = sample_batch(
        index,
        smoke_corpus["data_root"],
        seed=3,
        split="train",
        items=2,
        centers_per_item=3,
        min_center_gap=5,
        window_seconds=2.0,
        slots=8,
    )
    tensor_batch = _encode_blocks(config, index, smoke_corpus["data_root"], batch)
    assert tensor_batch.source_columns == 2
    # smoke-train-02 has one source; its padded second column is False and 0.
    padded_rows = [row for row, group in enumerate(tensor_batch.groups) if group.sources == 1]
    assert padded_rows
    for row in padded_rows:
        assert not bool(tensor_batch.source_valid[row, 1])
        assert float(tensor_batch.activity[row, :, 1].abs().max()) == 0.0
    # All frames come from real audio for valid centers.
    assert bool(tensor_batch.frame_valid.all())


def test_boundary_windows_have_padding_and_masked_frames(tmp_path: Path):
    corpus = _short_sample_corpus(tmp_path / "short")
    from aat.training.dataset import load_verified_index

    index = load_verified_index(corpus["index_path"], data_root=corpus["data_root"])
    batch = sample_batch(
        index,
        corpus["data_root"],
        seed=11,
        split="train",
        items=1,
        centers_per_item=4,
        min_center_gap=0,
        valid_only=False,
        window_seconds=2.0,
        slots=8,
    )
    block = batch.blocks[0]
    assert not bool(block.center_valid.any())
    assert not bool(block.audio_valid.all())
    config = smoke_config(corpus, tmp_path / "boundary", steps=1, valid_only=False)
    tensor_batch = _encode_blocks(config, index, corpus["data_root"], batch)
    assert bool(tensor_batch.center_valid.any()) is False
    # At least one token was computed from zero padding and is masked.
    assert not bool(tensor_batch.frame_valid.all())
    model = SourceQueryHead(feature_dim=16, slots=8)
    with torch.no_grad():
        groups, windows = tensor_batch.features.shape[:2]
        output = model(
            tensor_batch.features.reshape(groups * windows, *tensor_batch.features.shape[2:]),
            frame_times=tensor_batch.frame_times.reshape(groups * windows, -1),
            frame_valid=tensor_batch.frame_valid.reshape(groups * windows, -1),
            center_times=tensor_batch.center_times.reshape(groups * windows),
            center_valid=tensor_batch.center_valid.reshape(groups * windows),
        )
    slot_valid = output.slot_valid.view(groups, windows, 8)
    assert not bool(slot_valid[~tensor_batch.center_valid].any())


def test_zero_effective_supervision_is_not_success(tmp_path: Path):
    corpus = _short_sample_corpus(tmp_path / "short")
    config = smoke_config(
        corpus,
        tmp_path / "zero",
        steps=2,
        interval_steps=1,
        valid_only=False,
        centers_per_item=4,
        min_center_gap=0,
    )
    summary = train_from_config(config)
    assert summary.status == "no_effective_supervision"
    assert summary.totals["activity_terms"] == 0
    assert summary.totals["steps_without_activity_supervision"] == 2
    records = read_log(summary.train_log_path)
    assert all(record["effective_supervision"]["zero_activity_supervision"] for record in records)
    assert all(record["stats"]["skipped_groups"] >= 1 for record in records)
    # Diagnostics are still persisted for review.
    assert Path(summary.checkpoint_path).is_file()
    assert Path(summary.summary_path).is_file()
