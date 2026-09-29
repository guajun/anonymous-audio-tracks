"""Held-out evaluation protocol, baselines and undefined reasons (issue #8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from aat.training.trainer import train_from_config
from tests.training.support import smoke_config

pytestmark = pytest.mark.ml


def test_eval_report_protocol_baselines_and_undefined_reasons(smoke_corpus, tmp_path: Path):
    config = smoke_config(
        smoke_corpus,
        tmp_path / "eval-run",
        steps=2,
        interval_steps=2,
        eval_enabled=True,
        eval_splits=("val",),
        max_songs=1,
    )
    summary = train_from_config(config)
    assert "val" in summary.eval_paths
    report = json.loads(Path(summary.eval_paths["val"]).read_text(encoding="utf-8"))

    protocol = report["protocol"]
    assert protocol["activity_threshold"] == 0.5
    assert "whole-song" in protocol["assignment"]
    assert "per-frame" in protocol["assignment"]
    assert "fake" in protocol["fake_vs_real"]

    assert set(report["baselines_micro"]) == {"all_inactive", "all_active", "no_identity"}
    assert report["effective"]["valid_centers"] > 0
    assert report["effective"]["songs"] == 1
    assert report["per_source"]

    # The all-active baseline must cover every active reference frame; the
    # all-inactive baseline can never produce a true positive.
    assert report["baselines_micro"]["all_active"]["recall"] == 1.0
    assert report["baselines_micro"]["all_inactive"]["true_positives"] == 0
    assert report["baselines_micro"]["all_inactive"]["precision"] is None

    micro = report["micro"]
    if micro["true_positives"] + micro["false_positives"] == 0:
        assert any("precision undefined" in reason for reason in report["undefined_reasons"])
    assert report["songs"][0]["sample_id"]
    assert report["songs"][0]["valid_centers"] == report["effective"]["valid_centers"]


def test_automatic_training_eval_matches_reloaded_checkpoint_with_dropout(
    smoke_corpus, tmp_path: Path
):
    """Automatic post-training eval must use eval mode, not the training model mode."""

    from aat.training.dataset import load_verified_index
    from aat.training.evaluate import evaluate_split
    from aat.training.inference import load_head_from_checkpoint

    config = smoke_config(
        smoke_corpus,
        tmp_path / "dropout-eval",
        steps=2,
        interval_steps=2,
        eval_enabled=True,
        eval_splits=("val",),
        max_songs=1,
        dropout=0.5,
    )
    summary = train_from_config(config)
    automatic = json.loads(Path(summary.eval_paths["val"]).read_text(encoding="utf-8"))

    inference = load_head_from_checkpoint(summary.checkpoint_path, device="cpu")
    index = load_verified_index(config.data.index, data_root=config.data.data_root)
    manual = evaluate_split(
        index=index,
        data_root=index.resolve_data_root(
            index_path=config.data.index, data_root=config.data.data_root
        ),
        split="val",
        inference=inference,
        config=inference.config,
        max_songs=1,
    )
    assert manual["micro"] == automatic["micro"]
    assert manual["baselines_micro"] == automatic["baselines_micro"]
    assert manual["per_source"] == automatic["per_source"]
