"""Held-out evaluation protocol, baselines and undefined reasons (issue #8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
