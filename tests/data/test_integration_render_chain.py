"""Real DawDreamer -> label_sample -> build_dataset_index -> sample_batch chain.

Marked ``integration``: it self-skips without the optional render extra.  One
small real render is used to prove the chain works end to end; it is not used to
claim split ratios, dataset size or model quality.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dawdreamer")

from aat.contracts import ActivityData  # noqa: E402
from aat.data import DatasetIndex, sample_batch  # noqa: E402
from aat.render.config import config_from_dict  # noqa: E402
from aat.render.pipeline import render_sample  # noqa: E402

from . import fixtures  # noqa: E402

pytestmark = pytest.mark.integration

SMOKE_CI = fixtures.REPO_ROOT / "configs" / "render" / "smoke_ci.toml"


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(fixtures.REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_real_render_label_index_batch_chain(tmp_path: Path) -> None:
    with SMOKE_CI.open("rb") as handle:
        config_data = tomllib.load(handle)
    config_data["render"]["sample_id"] = "chain-0001"
    config_data["render"]["composition"] = "comp-chain-01"
    config = config_from_dict(config_data, origin="smoke_ci.toml")

    data_root = tmp_path / "data"
    sample_dir = data_root / "chain-0001"
    result = render_sample(config, sample_dir)
    assert result.manifest.stage == "rendered"
    assert (sample_dir / "mix.wav").is_file()

    labeled = _run(fixtures.LABEL_CLI, str(sample_dir))
    assert labeled.returncode == 0, labeled.stderr
    assert "labeled 2 stem(s)" in labeled.stdout
    activity = ActivityData.load(sample_dir)
    assert activity.center_times.size > 0
    assert activity.source_ids == ("s01", "s02")

    index_path = tmp_path / "index" / "index.json"
    built = _run(
        fixtures.BUILD_CLI,
        "build",
        "--data-root",
        str(data_root),
        "--out",
        str(index_path),
        "--seed",
        "20260929",
        "--ratios",
        "0.8,0.1,0.1",
        "--slots",
        "8",
        "--window-seconds",
        "2.0",
    )
    assert built.returncode == 0, built.stderr
    report = json.loads(built.stdout)
    assert report["sample_count"] == 1
    # A single component cannot fill three sets; the real ratios are reported.
    assert report["split_counts"] == {"train": 1, "val": 0, "test": 0}
    assert report["empty_splits"] == ["val", "test"]
    assert report["leak_free"] is True

    index = DatasetIndex.load(index_path)
    assert index.samples[0].sample_id == "chain-0001"
    assert index.samples[0].controls["sources"]["s02"]["note_min"] == 45
    assert index.samples[0].controls["sources"]["s02"]["note_max"] == 48

    batch = sample_batch(
        index,
        data_root,
        seed=20260929,
        split="train",
        items=1,
        centers_per_item=4,
        min_center_gap=2,
    )
    assert len(batch) == 1
    assert batch.skipped == ()
    block = batch.blocks[0]
    assert block.source_ids == ("s01", "s02")
    assert block.activity.shape == (4, 8)
    assert block.center_valid.all()
    # Real acoustic labels contain activity, padding slots stay zero.
    assert block.activity[:, :2].max() > 0.0
    assert np.all(block.activity[:, 2:] == 0.0)
    assert not block.source_present[:, 2:].any()
    assert block.audio_valid.shape == (4, block.window_samples)
    assert block.audio_valid.all()
    # Different-pitch evidence comes from controls and never from the labels.
    assert block.source_note_ranges[0] == (36, 36)
    assert block.source_note_ranges[1] == (45, 48)

    assert np.array_equal(
        block.activity[:, :2], activity.activity[block.center_indices]
    )
    assert np.array_equal(block.center_valid, activity.valid[block.center_indices])

    cli_batch = _run(
        fixtures.BUILD_CLI,
        "batch",
        "--index",
        str(index_path),
        "--split",
        "train",
        "--items",
        "1",
        "--centers",
        "3",
        "--min-gap",
        "2",
        "--seed",
        "1",
    )
    assert cli_batch.returncode == 0, cli_batch.stderr
    payload = json.loads(cli_batch.stdout)
    assert payload["returned_items"] == 1
    assert len(payload["blocks"][0]["center_times"]) == 3
    assert payload["blocks"][0]["source_ids"] == ["s01", "s02"]
