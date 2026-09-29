"""Fake-encoder smoke training, resume consistency and rejection (issue #8)."""

from __future__ import annotations

from pathlib import Path

import json

import pytest

torch = pytest.importorskip("torch")

from aat.training.checkpoint import capture_rng_state, load_checkpoint, step_seed
from aat.training.errors import ResumeMismatchError, TrainingError
from aat.training.trainer import train_from_config
from tests.training.support import (
    copy_corpus,
    read_log,
    rehash_manifest_mix,
    rewrite_wav_different_content,
    smoke_config,
)

pytestmark = pytest.mark.ml


def test_fake_smoke_training_writes_finite_losses_and_checkpoint(smoke_corpus, tmp_path: Path):
    config = smoke_config(smoke_corpus, tmp_path / "smoke", steps=3, interval_steps=3)
    summary = train_from_config(config)
    assert summary.status == "completed"
    assert summary.encoder_mode == "fake"
    assert "fake-encoder-smoke" in summary.result_kind

    records = read_log(summary.train_log_path)
    assert len(records) == 3
    for record in records:
        for name in ("total", "activity", "empty_slots", "positive", "negative"):
            assert isinstance(record["loss"][name], float)
            assert record["loss"][name] == record["loss"][name]  # not NaN
        assert record["batch_sha256"]
        assert record["stats"]["activity_terms"] >= 1
        assert "supervision_masked_groups" in record["stats"]
        assert record["effective_supervision"]["has_activity_supervision"] is True

    payload = load_checkpoint(summary.checkpoint_path)
    assert payload["step"] == 2
    assert payload["format_version"] == 1
    assert payload["sampler"]["scheme"] == "seed-sequence-v1"
    assert payload["git"] and "sha" in payload["git"]
    assert payload["dataset"]["fingerprint"]["digest"] == summary.dataset_digest
    assert payload["encoder"]["identity"]["mode"] == "fake"
    assert payload["rng"]["torch"] is not None
    for value in payload["model_state"].values():
        assert bool(torch.isfinite(value).all())

    run = json.loads((tmp_path / "smoke" / "run.json").read_text(encoding="utf-8"))
    assert run["encoder_mode"] == "fake"
    assert run["result_kind"].startswith("fake-encoder-smoke")


def test_resume_matches_uninterrupted_run_bitwise(smoke_corpus, tmp_path: Path):
    uninterrupted = smoke_config(smoke_corpus, tmp_path / "uninterrupted", steps=3, interval_steps=3)
    train_from_config(uninterrupted)

    partial = smoke_config(smoke_corpus, tmp_path / "resumed", steps=1, interval_steps=1)
    train_from_config(partial)
    extended = smoke_config(smoke_corpus, tmp_path / "resumed", steps=3, interval_steps=1)
    summary = train_from_config(extended, resume_from=tmp_path / "resumed")
    assert summary.resumed is True

    left = load_checkpoint(tmp_path / "uninterrupted" / "checkpoint.pt")
    right = load_checkpoint(tmp_path / "resumed" / "checkpoint.pt")
    assert left["step"] == right["step"] == 2
    assert set(left["model_state"]) == set(right["model_state"])
    for key in left["model_state"]:
        assert torch.equal(left["model_state"][key], right["model_state"][key]), key
    assert _optimizer_state_equal(left["optimizer_state"], right["optimizer_state"])
    assert torch.equal(
        torch.as_tensor(left["rng"]["torch"]), torch.as_tensor(right["rng"]["torch"])
    )

    left_log = read_log(tmp_path / "uninterrupted" / "train_log.jsonl")
    right_log = read_log(tmp_path / "resumed" / "train_log.jsonl")
    assert [record["batch_sha256"] for record in left_log] == [
        record["batch_sha256"] for record in right_log
    ]
    for first, second in zip(left_log, right_log):
        for name in ("total", "activity", "empty_slots", "positive", "negative"):
            assert first["loss"][name] == second["loss"][name]
        assert first["step_seed"] == second["step_seed"]


def _optimizer_state_equal(left, right) -> bool:
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return set(left) == set(right) and all(_optimizer_state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _optimizer_state_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def test_step_seed_sequence_is_history_independent():
    assert step_seed(20260929, 7) == step_seed(20260929, 7)
    assert step_seed(20260929, 7) != step_seed(20260929, 8)
    assert step_seed(20260929, 7) != step_seed(20260930, 7)


def test_rng_capture_restore_roundtrip():
    import random

    import numpy as np

    state = capture_rng_state()
    random.random()
    np.random.rand()
    torch.rand(3)
    from aat.training.checkpoint import restore_rng_state

    restore_rng_state(state)
    first = (random.random(), float(np.random.rand()), torch.rand(3))
    restore_rng_state(state)
    second = (random.random(), float(np.random.rand()), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


@pytest.mark.parametrize(
    "changed",
    [
        {"optim": {"lr": 2e-3}},
        {"head": {"slots": 6}},
        {"encoder": {"mode": "fake", "feature_dim": 16, "fake_seed": 12345}},
        {"run": {"seed": 4242}},
        {"data": {"centers_per_item": 3, "min_center_gap": 5}},
    ],
)
def test_resume_rejects_config_changes(smoke_corpus, tmp_path: Path, changed):
    out = tmp_path / "reject"
    config = smoke_config(smoke_corpus, out, steps=1, interval_steps=1)
    train_from_config(config)

    payload = {
        "run": {
            "name": "test-run",
            "out_dir": str(out),
            "seed": 20260929,
            "device": "cpu",
        },
        "encoder": {"mode": "fake", "feature_dim": 16, "fake_seed": 20260929},
        "data": {
            "index": str(smoke_corpus["index_path"]),
            "data_root": str(smoke_corpus["data_root"]),
            "split": "train",
            "groups_per_step": 1,
            "centers_per_item": 4,
            "min_center_gap": 5,
        },
        "head": {"slots": 8},
        "optim": {"steps": 2, "lr": 1e-3},
        "checkpoint": {"interval_steps": 1},
        "eval": {"enabled": False},
    }
    for table, values in changed.items():
        payload[table] = {**payload[table], **values}
    from aat.training.config import TrainConfig

    with pytest.raises(ResumeMismatchError, match="refusing to resume"):
        train_from_config(
            TrainConfig.from_dict(payload, origin="changed"),
            resume_from=out,
        )


def test_resume_rejects_changed_data_without_rebuild(smoke_corpus, tmp_path: Path):
    corpus = copy_corpus(smoke_corpus, tmp_path / "corpus")
    config = smoke_config(corpus, tmp_path / "run", steps=1, interval_steps=1)
    train_from_config(config)
    sample_dir = Path(corpus["data_root"]) / "smoke-train-01"
    rewrite_wav_different_content(sample_dir / "mix.wav")
    with pytest.raises(TrainingError, match="cannot load/verify dataset index"):
        train_from_config(config, resume_from=tmp_path / "run")


def test_resume_rejects_rebuilt_data_digest(smoke_corpus, tmp_path: Path):
    corpus = copy_corpus(smoke_corpus, tmp_path / "corpus")
    config = smoke_config(corpus, tmp_path / "run", steps=1, interval_steps=1)
    train_from_config(config)

    sample_dir = Path(corpus["data_root"]) / "smoke-train-01"
    rewrite_wav_different_content(sample_dir / "mix.wav")
    rehash_manifest_mix(sample_dir)
    from aat.data import build_dataset_index

    rebuilt = build_dataset_index(
        corpus["data_root"],
        seed=20260929,
        ratios=(0.5, 0.25, 0.25),
        slots=8,
        window_seconds=2.0,
        index_dir=Path(corpus["index_path"]).parent,
    )
    rebuilt_path = Path(corpus["index_path"]).parent / "index-rebuilt.json"
    rebuilt.save(rebuilt_path)
    changed = smoke_config(
        corpus, tmp_path / "run", steps=2, interval_steps=1, index_path=str(rebuilt_path)
    )
    with pytest.raises(ResumeMismatchError, match="dataset.content"):
        train_from_config(changed, resume_from=tmp_path / "run")


def test_overwrite_is_explicit_and_archives(smoke_corpus, tmp_path: Path):
    out = tmp_path / "overwrite"
    config = smoke_config(smoke_corpus, out, steps=1, interval_steps=1)
    train_from_config(config)
    with pytest.raises(TrainingError, match="already contains"):
        train_from_config(config)
    train_from_config(config, overwrite=True)
    backups = sorted(path.name for path in out.glob("*.bak-*"))
    assert backups, "overwrite must archive, not delete"
    assert (out / "checkpoint.pt").is_file()


def test_resume_after_crash_between_checkpoints_reconciles_log(smoke_corpus, tmp_path: Path):
    """Interrupted *between* checkpoints: tail is evidence, canonical log stays unique."""

    from aat.training.trainer import Trainer

    out = tmp_path / "crash"
    config = smoke_config(smoke_corpus, out, steps=4, interval_steps=2)
    trainer = Trainer(config, log=lambda _message: None)
    original = trainer._run_step

    def crash(step):
        if step == 3:
            raise RuntimeError("simulated interruption")
        return original(step)

    trainer._run_step = crash
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer.run()

    assert [record["step"] for record in read_log(trainer.train_log_path)] == [0, 1, 2]
    checkpoint = load_checkpoint(trainer.checkpoint_path)
    assert checkpoint["step"] == 1
    # Simulate a partial write of the next JSONL line.
    with open(trainer.train_log_path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"step": 2, "partial"')

    resumed_config = smoke_config(smoke_corpus, out, steps=4, interval_steps=2)
    summary = train_from_config(resumed_config, resume_from=out)
    assert summary.steps_completed == 4
    assert [record["step"] for record in read_log(out / "train_log.jsonl")] == [0, 1, 2, 3]

    evidence_files = sorted(out.glob("train_log.discarded-*.jsonl"))
    assert len(evidence_files) == 1
    evidence = evidence_files[0].read_text(encoding="utf-8").splitlines()
    parsed_evidence = [json.loads(line) for line in evidence if line.startswith("{") and line.endswith("}")]
    assert any(record.get("step") == 2 for record in parsed_evidence)
    assert any("partial" in line for line in evidence)

    # A real interruption must resume to exactly the uninterrupted state.
    uninterrupted_out = tmp_path / "uninterrupted"
    train_from_config(smoke_config(smoke_corpus, uninterrupted_out, steps=4, interval_steps=2))
    left = load_checkpoint(uninterrupted_out / "checkpoint.pt")
    right = load_checkpoint(out / "checkpoint.pt")
    assert left["step"] == right["step"] == 3
    for key in left["model_state"]:
        assert torch.equal(left["model_state"][key], right["model_state"][key]), key
    assert _optimizer_state_equal(left["optimizer_state"], right["optimizer_state"])
    assert torch.equal(torch.as_tensor(left["rng"]["torch"]), torch.as_tensor(right["rng"]["torch"]))
    left_log = read_log(uninterrupted_out / "train_log.jsonl")
    right_log = read_log(out / "train_log.jsonl")
    assert [record["batch_sha256"] for record in left_log] == [
        record["batch_sha256"] for record in right_log
    ]
    for first, second in zip(left_log, right_log):
        for name in ("total", "activity", "empty_slots", "positive", "negative"):
            assert first["loss"][name] == second["loss"][name]


def test_resume_refuses_incompatible_log_prefix(smoke_corpus, tmp_path: Path):
    from aat.training.errors import CheckpointError

    out = tmp_path / "bad-prefix"
    config = smoke_config(smoke_corpus, out, steps=2, interval_steps=1)
    train_from_config(config)
    log_path = out / "train_log.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps({**json.loads(lines[0]), "step": 5})
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(CheckpointError, match="incompatible"):
        train_from_config(
            smoke_config(smoke_corpus, out, steps=3, interval_steps=1),
            resume_from=out,
        )


def test_cpu_resources_are_explicit_not_fabricated(smoke_corpus, tmp_path: Path):
    out = tmp_path / "resources"
    config = smoke_config(smoke_corpus, out, steps=1, interval_steps=1)
    summary = train_from_config(config)
    resources = summary.resources
    assert resources["device"] == "cpu"
    assert resources["cuda_available"] is False
    assert resources["peak_allocated_bytes"] is None
    assert resources["peak_reserved_bytes"] is None
    assert "unavailable" in resources["unavailable_reason"]
    assert resources["elapsed_seconds"] > 0
    assert "measurement_scope" in resources and "reset_semantics" in resources

    summary_json = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["resources"]["peak_allocated_bytes"] is None
    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["resources"]["state"] == "finished"
    assert run_json["resources"]["peak_reserved_bytes"] is None
    payload = load_checkpoint(summary.checkpoint_path)
    assert payload["resources"]["peak_allocated_bytes"] is None
    assert payload["resources"]["state"] == "in_progress"
