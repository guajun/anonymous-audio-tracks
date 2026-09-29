"""Training checkpoints with strict resume verification (issue #8).

A checkpoint records everything needed to stop and continue exactly:

* model and AdamW optimizer state, completed step;
* Python, NumPy, torch and CUDA RNG states;
* the explicit sampler scheme (per-step seed derivation), so the next sampled
  batch is reproducible without an implicit RNG stream;
* the full config snapshot plus its canonical hash, and the identity subset
  used for resume checks;
* Git SHA, ``uv.lock`` digest, Python/NumPy/torch/CUDA versions;
* dataset index/split/content fingerprint and the encoder identity/provenance;
* last loss components and the accumulated effective-supervision counters.

Writes are atomic (temp file + ``os.replace``).  ``load_checkpoint`` refuses
unknown formats/versions; :func:`verify_resume` refuses to continue when the
config identity, dataset fingerprint or encoder identity changed, so data,
shapes and hyper-parameters can never be silently swapped under a run.
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .config import TrainConfig, canonical_json, sha256_canonical
from .errors import CheckpointError, ResumeMismatchError

#: File name of the latest checkpoint inside a run directory.
CHECKPOINT_FILENAME = "checkpoint.pt"
#: Checkpoint format markers; unknown values are rejected on load.
CHECKPOINT_FORMAT = "aat-training-checkpoint"
CHECKPOINT_VERSION = 1
#: Explicit sampler scheme recorded in every checkpoint.
SAMPLER_SCHEME = "seed-sequence-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: str | Path) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    return hashlib.sha256(source.read_bytes()).hexdigest()


def repository_root() -> Path | None:
    """Best-effort repository root for source checkouts (installed wheels -> None)."""

    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return None


def git_provenance(root: Path | None = None) -> dict[str, Any]:
    """Record HEAD and dirty state without failing outside a Git checkout."""

    root = root if root is not None else repository_root()
    if root is None:
        return {"sha": None, "dirty": None, "source": "unavailable"}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"sha": sha or None, "dirty": bool(status.strip()), "source": "git"}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"sha": None, "dirty": None, "source": f"unavailable: {error}"}


def uv_lock_sha256(root: Path | None = None) -> str | None:
    root = root if root is not None else repository_root()
    if root is None:
        return None
    return file_sha256(root / "uv.lock")


def runtime_versions() -> dict[str, Any]:
    cuda_version: str | None
    try:
        cuda_version = torch.version.cuda
    except AttributeError:  # pragma: no cover - very old torch
        cuda_version = None
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": cuda_version,
        "platform": platform.platform(),
    }


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG that could affect the next step."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": {
            "version": python_state[0],
            "keys": list(python_state[1]),
            "gauss_next": python_state[2],
        },
        "numpy": {
            "legacy": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    python_state = state.get("python")
    if not isinstance(python_state, Mapping):
        raise CheckpointError("checkpoint.rng.python is missing")
    random.setstate(
        (
            python_state["version"],
            tuple(int(value) for value in python_state["keys"]),
            python_state["gauss_next"],
        )
    )
    numpy_state = state.get("numpy")
    if not isinstance(numpy_state, Mapping):
        raise CheckpointError("checkpoint.rng.numpy is missing")
    np.random.set_state(
        (
            str(numpy_state["legacy"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch_state = state.get("torch")
    if torch_state is None:
        raise CheckpointError("checkpoint.rng.torch is missing")
    torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def step_seed(base_seed: int, step: int) -> int:
    """Deterministic per-step batch seed (the explicit sampler scheme).

    The sequence depends only on ``(base_seed, step)``, so an interrupted and
    resumed run draws exactly the same batch for every step as an uninterrupted
    run; no hidden RNG stream position is involved.
    """

    sequence = np.random.SeedSequence([int(base_seed), int(step), 0xA17])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


@dataclass(frozen=True)
class DataIdentity:
    """Dataset identity recorded in the checkpoint."""

    index_path: str
    index_sha256: str | None
    data_root: str
    fingerprint: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_path": self.index_path,
            "index_sha256": self.index_sha256,
            "data_root": self.data_root,
            "fingerprint": dict(self.fingerprint),
        }

    @property
    def digest(self) -> str:
        fingerprint = self.fingerprint
        return str(fingerprint.get("digest", ""))

    @property
    def per_split(self) -> Mapping[str, Any]:
        fingerprint = self.fingerprint
        return fingerprint.get("per_split", {})  # type: ignore[return-value]


def _diff_messages(expected: Any, actual: Any, path: str, limit: int = 12) -> list[str]:
    """Recursive difference report for resume identity checks."""

    messages: list[str] = []

    def walk(left: Any, right: Any, where: str) -> None:
        if len(messages) >= limit:
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                child = f"{where}.{key}" if where else str(key)
                if key not in left:
                    messages.append(f"{child}: missing in checkpoint")
                elif key not in right:
                    messages.append(f"{child}: missing in current config")
                else:
                    walk(left[key], right[key], child)
            return
        if isinstance(left, list) and isinstance(right, list):
            if left != right:
                messages.append(f"{where}: checkpoint {left!r} vs current {right!r}")
            return
        if left != right:
            messages.append(f"{where}: checkpoint {left!r} vs current {right!r}")

    walk(expected, actual, path)
    return messages


def verify_resume(
    payload: Mapping[str, Any],
    *,
    config: TrainConfig,
    dataset: DataIdentity,
    encoder_identity: Mapping[str, Any],
) -> None:
    """Refuse to resume unless config, data and encoder identity all match."""

    problems: list[str] = []

    stored_identity = payload.get("config_identity")
    if not isinstance(stored_identity, Mapping):
        problems.append("checkpoint has no config_identity")
    else:
        problems.extend(_diff_messages(stored_identity, config.identity_dict(), "config"))

    stored_dataset = payload.get("dataset")
    if not isinstance(stored_dataset, Mapping):
        problems.append("checkpoint has no dataset identity")
    else:
        stored_fingerprint = stored_dataset.get("fingerprint")
        if not isinstance(stored_fingerprint, Mapping):
            problems.append("checkpoint.dataset.fingerprint is missing")
        elif stored_fingerprint.get("digest") != dataset.digest:
            problems.append(
                "dataset.content: the checkpoint was trained on a different index/content "
                f"(checkpoint digest {stored_fingerprint.get('digest')}, current {dataset.digest})"
            )
        else:
            stored_split = stored_fingerprint.get("per_split", {}).get(config.data.split)
            current_split = dict(dataset.per_split).get(config.data.split)
            if stored_split != current_split:
                problems.append(
                    f"dataset.split '{config.data.split}': checkpoint split digest differs "
                    "from the current index"
                )

    stored_encoder = payload.get("encoder")
    if not isinstance(stored_encoder, Mapping) or not isinstance(
        stored_encoder.get("identity"), Mapping
    ):
        problems.append("checkpoint has no encoder identity")
    else:
        problems.extend(
            _diff_messages(stored_encoder["identity"], dict(encoder_identity), "encoder")
        )

    if problems:
        detail = "\n".join(f"  - {message}" for message in problems[:20])
        raise ResumeMismatchError(
            "refusing to resume: the checkpoint does not match the current "
            f"config/data/encoder identity:\n{detail}"
        )


def save_checkpoint(
    path: str | Path,
    *,
    step: int,
    model_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    config: TrainConfig,
    dataset: DataIdentity,
    encoder_identity: Mapping[str, Any],
    encoder_provenance: Mapping[str, Any],
    last_step_record: Mapping[str, Any] | None,
    totals: Mapping[str, Any],
    resources: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write the latest checkpoint."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "created_at_utc": utc_now(),
        "step": int(step),
        "sampler": {"scheme": SAMPLER_SCHEME, "base_seed": int(config.run.seed)},
        "model_config": checkpoint_model_config(config),
        "model_state": dict(model_state),
        "optimizer_config": config.optim.to_dict(),
        "optimizer_state": optimizer_state,
        "rng": capture_rng_state(),
        "versions": runtime_versions(),
        "config": config.to_dict(),
        "config_sha256": config.config_sha256(),
        "config_identity": config.identity_dict(),
        "config_identity_sha256": config.identity_sha256(),
        "git": git_provenance(),
        "uv_lock_sha256": uv_lock_sha256(),
        "dataset": dataset.to_dict(),
        "encoder": {
            "identity": dict(encoder_identity),
            "provenance": dict(encoder_provenance),
        },
        "last_step": dict(last_step_record) if last_step_record is not None else None,
        "totals": dict(totals),
        "resources": dict(resources) if resources is not None else None,
    }
    temporary = target.with_name(target.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except OSError as error:
        raise CheckpointError(f"cannot write checkpoint {target}: {error}") from error
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass
    return target


def checkpoint_model_config(config: TrainConfig) -> dict[str, Any]:
    return {
        "feature_dim": config.encoder.feature_dim,
        "slots": config.head.slots,
        "embedding_dim": 128,
        "d_model": config.head.d_model,
        "num_heads": config.head.num_heads,
        "dropout": config.head.dropout,
        "time_frequencies": config.head.time_frequencies,
        "base_frequency_hz": config.head.base_frequency_hz,
        "ffn_multiplier": config.head.ffn_multiplier,
    }


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate a checkpoint document."""

    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"checkpoint {source} does not exist")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - surface any torch load failure
        raise CheckpointError(f"cannot load checkpoint {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise CheckpointError(f"checkpoint {source} is not a mapping")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"checkpoint {source}: expected format {CHECKPOINT_FORMAT!r}, "
            f"got {payload.get('format')!r}"
        )
    if payload.get("format_version") != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"checkpoint {source}: unsupported format_version "
            f"{payload.get('format_version')!r} (this code writes {CHECKPOINT_VERSION})"
        )
    sampler = payload.get("sampler")
    if not isinstance(sampler, Mapping) or sampler.get("scheme") != SAMPLER_SCHEME:
        raise CheckpointError(
            f"checkpoint {source}: unknown sampler scheme {sampler!r}; cannot guarantee "
            "a strictly restorable continuation"
        )
    if "model_state" not in payload or "optimizer_state" not in payload:
        raise CheckpointError(f"checkpoint {source}: missing model/optimizer state")
    return dict(payload)


def model_config_sha256(config: TrainConfig) -> str:
    return sha256_canonical(checkpoint_model_config(config))


__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "SAMPLER_SCHEME",
    "DataIdentity",
    "capture_rng_state",
    "checkpoint_model_config",
    "file_sha256",
    "git_provenance",
    "load_checkpoint",
    "model_config_sha256",
    "repository_root",
    "restore_rng_state",
    "runtime_versions",
    "save_checkpoint",
    "step_seed",
    "utc_now",
    "uv_lock_sha256",
    "verify_resume",
]
