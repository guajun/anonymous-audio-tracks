"""Training config parsing, pinned modes and resume identity (issue #8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aat.training.config import TrainConfig
from aat.training.errors import TrainingError

pytestmark = pytest.mark.ml


def _config(**overrides):
    payload = {
        "run": {"name": "cfg", "out_dir": "runs/cfg", "seed": 7, "device": "cpu"},
        "encoder": {"mode": "fake", "feature_dim": 16, "fake_seed": 7},
        "data": {"index": "index.json", "split": "train"},
        "head": {"slots": 8},
        "optim": {"steps": 10, "lr": 1e-3},
        "eval": {"enabled": False},
    }
    for table, values in overrides.items():
        payload[table] = {**payload[table], **values}
    return TrainConfig.from_dict(payload, origin="cfg")


def test_defaults_and_pinned_policy():
    config = _config()
    assert config.encoder.window_seconds == 2.0
    assert config.encoder.attention == "block_diagonal"
    assert config.encoder.extraction == "independent_windows"
    assert config.encoder.dtype == "float32"
    assert config.head.slots == 8
    assert config.optim.grad_clip_norm == 1.0


def test_unknown_key_is_rejected():
    with pytest.raises(TrainingError, match="unknown key"):
        TrainConfig.from_dict(
            {
                "run": {"name": "cfg", "out_dir": "runs/cfg", "seed": 1},
                "encoder": {"mode": "fake", "mystery": 1},
                "data": {"index": "index.json"},
            },
            origin="cfg",
        )


def test_non_pinned_window_dtype_attention_rejected():
    with pytest.raises(TrainingError, match="window_seconds"):
        _config(encoder={"window_seconds": 1.0})
    with pytest.raises(TrainingError, match="dtype"):
        _config(encoder={"dtype": "bfloat16"})
    with pytest.raises(TrainingError, match="attention"):
        _config(encoder={"attention": "unmasked_global"})
    with pytest.raises(TrainingError, match="extraction"):
        _config(encoder={"extraction": "single_buffer"})


def test_aut_mode_requires_checkpoint_identity():
    with pytest.raises(TrainingError, match="model_dir"):
        _config(encoder={"mode": "aut", "feature_dim": 2048})
    with pytest.raises(TrainingError, match="revision"):
        _config(
            encoder={
                "mode": "aut",
                "feature_dim": 2048,
                "model_dir": "/checkpoint",
            }
        )
    aut = _config(
        encoder={
            "mode": "aut",
            "feature_dim": 2048,
            "model_dir": "/checkpoint",
            "revision": "26291f793822fb6be9555850f06dfe95f2d7e695",
        }
    )
    assert aut.encoder.mode == "aut"


def test_fake_mode_rejects_checkpoint_fields():
    with pytest.raises(TrainingError, match="model_dir"):
        _config(encoder={"model_dir": "/checkpoint"})
    with pytest.raises(TrainingError, match="revision"):
        _config(encoder={"revision": "abc"})


def test_identity_excludes_paths_steps_and_eval_but_keeps_hyperparameters():
    base = _config()
    same = _config(run={"out_dir": "runs/elsewhere"}, optim={"steps": 99})
    assert base.identity_sha256() == same.identity_sha256()
    changed_lr = _config(optim={"lr": 2e-3})
    assert base.identity_sha256() != changed_lr.identity_sha256()
    changed_slots = _config(head={"slots": 6})
    assert base.identity_sha256() != changed_slots.identity_sha256()
    changed_seed = _config(run={"seed": 8})
    assert base.identity_sha256() != changed_seed.identity_sha256()


def test_layer_final_roundtrip():
    config = _config()
    payload = config.to_dict()
    assert payload["encoder"]["layer"] is None
    restored = TrainConfig.from_dict(payload, origin="roundtrip")
    assert restored.encoder.layer is None
    assert restored.encoder.identity_dict() == config.encoder.identity_dict()


def test_toml_file_roundtrip(tmp_path: Path):
    path = tmp_path / "train.toml"
    path.write_text(
        """
[run]
name = "toml"
out_dir = "runs/toml"
seed = 5

[data]
index = "index.json"
split = "train"
""",
        encoding="utf-8",
    )
    config = TrainConfig.from_toml(path)
    assert config.run.name == "toml"
    assert config.encoder.mode == "fake"
