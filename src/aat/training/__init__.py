"""Minimum reproducible training for the frozen-AuT E/P head (issue #8).

Light, torch-free imports (config, dataset preparation, checkpoint identity,
baselines documentation) are available directly; the modules that require the
``ml`` extra (torch training loop, checkpoints, evaluation) are imported lazily
so a render-only environment can still run ``prepare-data``.

Public entry points:

* :class:`TrainConfig` - validated TOML training configuration.
* :func:`aat.training.dataset.make_smoke_dataset` - tiny synthetic corpus with
  real protocol documents for the CPU fake-encoder smoke and tests.
* :class:`aat.training.Trainer` / :func:`train_from_config` - training loop.
* :func:`aat.training.load_head_from_checkpoint` - minimal inference interface
  for issue #11 (no test modules, no dataset index required).
* :func:`aat.training.evaluate_split` - fixed-protocol held-out evaluation and
  constant baselines.

Fake-encoder runs are engineering smoke evidence only and are labelled as such
in every artifact; real results require the frozen AuT encoder.
"""

from __future__ import annotations

from typing import Any

from .config import (
    CheckpointConfig,
    DataConfig,
    EncoderConfig,
    EvalConfig,
    HeadConfig,
    LossConfig,
    OptimConfig,
    RunConfig,
    TrainConfig,
    default_smoke_config,
)
from .dataset import (
    DatasetPlan,
    dataset_fingerprint,
    index_file_sha256,
    load_dataset_plan,
    load_verified_index,
    make_smoke_dataset,
)
from .errors import CheckpointError, ResumeMismatchError, TrainingError

_LAZY = {
    "AutEncoderAdapter": "aat.training.encoding",
    "BASELINES": "aat.training.evaluate",
    "DataIdentity": "aat.training.checkpoint",
    "EncodedWindows": "aat.training.encoding",
    "EncoderAdapter": "aat.training.encoding",
    "EvaluationProtocol": "aat.training.evaluate",
    "FakeEncoderAdapter": "aat.training.encoding",
    "HeadInference": "aat.training.inference",
    "Trainer": "aat.training.trainer",
    "TrainingBatch": "aat.training.batching",
    "TrainingSummary": "aat.training.trainer",
    "build_encoder": "aat.training.encoding",
    "checkpoint_model_config": "aat.training.checkpoint",
    "evaluate_song": "aat.training.evaluate",
    "evaluate_split": "aat.training.evaluate",
    "load_checkpoint": "aat.training.checkpoint",
    "load_head_from_checkpoint": "aat.training.inference",
    "restore_rng_state": "aat.training.checkpoint",
    "save_checkpoint": "aat.training.checkpoint",
    "step_seed": "aat.training.checkpoint",
    "train_from_config": "aat.training.trainer",
    "training_batch_from_blocks": "aat.training.batching",
    "verify_resume": "aat.training.checkpoint",
    "window_start_seconds": "aat.training.encoding",
}

__all__ = [
    "BASELINES",
    "CheckpointConfig",
    "CheckpointError",
    "DataConfig",
    "DataIdentity",
    "DatasetPlan",
    "EvalConfig",
    "EncoderConfig",
    "EncoderAdapter",
    "EvaluationProtocol",
    "HeadConfig",
    "HeadInference",
    "LossConfig",
    "OptimConfig",
    "ResumeMismatchError",
    "RunConfig",
    "TrainConfig",
    "Trainer",
    "TrainingBatch",
    "TrainingError",
    "TrainingSummary",
    "build_encoder",
    "checkpoint_model_config",
    "dataset_fingerprint",
    "default_smoke_config",
    "evaluate_song",
    "evaluate_split",
    "index_file_sha256",
    "load_checkpoint",
    "load_dataset_plan",
    "load_head_from_checkpoint",
    "load_verified_index",
    "make_smoke_dataset",
    "restore_rng_state",
    "save_checkpoint",
    "step_seed",
    "train_from_config",
    "training_batch_from_blocks",
    "verify_resume",
    "window_start_seconds",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, name)
