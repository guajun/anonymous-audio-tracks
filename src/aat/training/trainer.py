"""Minimum reproducible training loop for the frozen-AuT E/P head (issue #8).

One step = sample one multi-center block per song (same song only, fixed source
columns, non-adjacent centers), encode **independent** 2 s windows, run the
K-query head, apply the permutation-invariant ``head_loss``, and update AdamW.
Batches are drawn from an explicit per-step seed sequence, so an interrupted
and resumed run follows exactly the same batch order as an uninterrupted run.

Every step is appended to ``train_log.jsonl`` (machine-readable loss curve and
effective-supervision counters); checkpoints are written atomically to
``checkpoint.pt`` at the configured interval and at the end.  A run whose total
``activity_terms`` stays zero is reported as ``no_effective_supervision`` and
never as success.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from aat.contracts.jsonio import dumps_json
from aat.data import DatasetIndex, sample_batch
from aat.losses import LossWeights, head_loss

from .batching import TrainingBatch, training_batch_from_blocks
from .checkpoint import (
    CHECKPOINT_FILENAME,
    DataIdentity,
    checkpoint_model_config,
    git_provenance,
    load_checkpoint,
    restore_rng_state,
    runtime_versions,
    save_checkpoint,
    step_seed,
    utc_now,
    uv_lock_sha256,
    verify_resume,
)
from .config import TrainConfig
from .dataset import dataset_fingerprint, index_file_sha256, load_verified_index
from .encoding import EncoderAdapter, build_encoder, window_start_seconds
from .errors import CheckpointError, ResumeMismatchError, TrainingError
from .evaluate import evaluate_split
from .inference import HeadInference, build_head_from_model_config

#: Known artifacts inside a run directory; only these are archived on overwrite.
RUN_ARTIFACTS = (
    "run.json",
    CHECKPOINT_FILENAME,
    "train_log.jsonl",
    "summary.json",
    "eval_val.json",
    "eval_test.json",
)

FRESH_TOTALS: dict[str, int] = {
    "steps": 0,
    "groups": 0,
    "activity_terms": 0,
    "empty_terms": 0,
    "positive_terms": 0,
    "negative_terms": 0,
    "matched_groups": 0,
    "skipped_groups": 0,
    "ambiguous_groups": 0,
    "truncated_groups": 0,
    "identity_masked_groups": 0,
    "supervision_masked_groups": 0,
    "steps_with_activity_supervision": 0,
    "steps_without_activity_supervision": 0,
}


@dataclass(frozen=True)
class TrainingSummary:
    """Final, machine-readable outcome of one training invocation."""

    run_dir: str
    status: str
    result_kind: str
    encoder_mode: str
    resumed: bool
    first_step: int | None
    last_step: int | None
    steps_completed: int
    checkpoint_path: str
    train_log_path: str
    summary_path: str
    eval_paths: Mapping[str, str]
    config_sha256: str
    dataset_digest: str
    totals: Mapping[str, Any]
    git: Mapping[str, Any]
    uv_lock_sha256: str | None
    versions: Mapping[str, Any]
    resources: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "status": self.status,
            "result_kind": self.result_kind,
            "encoder_mode": self.encoder_mode,
            "resumed": self.resumed,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "steps_completed": self.steps_completed,
            "checkpoint_path": self.checkpoint_path,
            "train_log_path": self.train_log_path,
            "summary_path": self.summary_path,
            "eval_paths": dict(self.eval_paths),
            "config_sha256": self.config_sha256,
            "dataset_digest": self.dataset_digest,
            "totals": dict(self.totals),
            "git": dict(self.git),
            "uv_lock_sha256": self.uv_lock_sha256,
            "versions": dict(self.versions),
            "resources": dict(self.resources),
        }


def _archive_stamp() -> str:
    return re.sub(r"[^0-9A-Za-z]", "", utc_now())


class Trainer:
    """Stateful trainer; constructed once per ``train`` invocation."""

    def __init__(
        self,
        config: TrainConfig,
        *,
        device: str | None = None,
        run_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
        overwrite: bool = False,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.log = log if log is not None else (lambda _message: None)
        self.device = torch.device(device or config.run.device)
        # Resource measurement starts before the encoder is loaded so the
        # recorded CUDA peak covers encoder residency plus training/eval.
        self._resources_started = time.perf_counter()
        peak_index = self._cuda_device_index()
        if peak_index is not None:
            # An explicit device index is rejected before the CUDA context
            # exists on torch 2.14, so initialize/select the device first.
            torch.cuda.init()
            if peak_index != int(torch.cuda.current_device()):
                torch.cuda.set_device(peak_index)
            torch.cuda.reset_peak_memory_stats(peak_index)
        self.encoder: EncoderAdapter = build_encoder(config.encoder, device=str(self.device))

        self.index_path = Path(config.data.index)
        self.data_root = (
            Path(config.data.data_root) if config.data.data_root is not None else None
        )
        try:
            self.index: DatasetIndex = load_verified_index(
                self.index_path, data_root=self.data_root
            )
        except Exception as error:  # noqa: BLE001 - dataset errors carry diagnostics
            raise TrainingError(f"cannot load/verify dataset index: {error}") from error
        self.resolved_data_root = self.index.resolve_data_root(
            index_path=self.index_path, data_root=self.data_root
        )
        self._check_dataset_compatibility()

        fingerprint = dataset_fingerprint(self.index)
        self.dataset_identity = DataIdentity(
            index_path=str(self.index_path),
            index_sha256=index_file_sha256(self.index_path),
            data_root=str(self.resolved_data_root),
            fingerprint=fingerprint,
        )
        self.git = git_provenance()
        self.uv_lock = uv_lock_sha256()
        self.versions = runtime_versions()

        split_entries = self.index.samples_for_split(config.data.split)
        if len(split_entries) < config.data.groups_per_step:
            raise TrainingError(
                f"split {config.data.split!r} holds {len(split_entries)} song(s) but "
                f"data.groups_per_step={config.data.groups_per_step}; add data or lower the "
                "batch size"
            )

        self.resumed = resume_from is not None
        if self.resumed:
            self._resume(resume_from, run_dir=run_dir)
        else:
            self._start_fresh(run_dir=run_dir, overwrite=overwrite)

    # -- setup -------------------------------------------------------------- #

    def _check_dataset_compatibility(self) -> None:
        plan = self.index.plan
        if float(plan["window_seconds"]) != self.config.encoder.window_seconds:
            raise TrainingError(
                f"dataset index was built with window_seconds={plan['window_seconds']} but "
                f"the encoder uses {self.config.encoder.window_seconds}; rebuild the index "
                "or fix the config"
            )

    def _start_fresh(self, *, run_dir: str | Path | None, overwrite: bool) -> None:
        target = Path(run_dir) if run_dir is not None else Path(self.config.run.out_dir)
        existing = [name for name in RUN_ARTIFACTS if (target / name).exists()] if target.exists() else []
        if existing and not overwrite:
            raise TrainingError(
                f"run directory {target} already contains {existing}; pass --overwrite to "
                "archive them (nothing is deleted) or --resume to continue"
            )
        if existing and overwrite:
            stamp = _archive_stamp()
            for name in existing:
                os.replace(target / name, target / f"{name}.bak-{stamp}")
            self.log(f"archived {len(existing)} existing artifact(s) as *.bak-{stamp}")
        target.mkdir(parents=True, exist_ok=True)
        self.run_dir = target
        self.checkpoint_path = target / CHECKPOINT_FILENAME
        self.train_log_path = target / "train_log.jsonl"
        self.eval_paths: dict[str, str] = {}

        torch.manual_seed(int(self.config.run.seed))
        self.model = build_head_from_model_config(checkpoint_model_config(self.config))
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.step_offset = 0
        self.last_step_record: dict[str, Any] | None = None
        self.totals: dict[str, int] = dict(FRESH_TOTALS)
        self._write_run_json(status="running")
        self.log(
            f"run {self.config.run.name}: {self._result_kind()} "
            f"({self.config.encoder.mode} encoder, {self.device})"
        )

    def _resume(self, resume_from: str | Path, *, run_dir: str | Path | None) -> None:
        source = Path(resume_from)
        checkpoint_path = source if source.is_file() else source / CHECKPOINT_FILENAME
        target = checkpoint_path.parent
        if run_dir is not None and Path(run_dir) != target:
            raise TrainingError(
                f"--out {run_dir} differs from the checkpoint directory {target}; "
                "resume in place so a run is never forked silently"
            )
        payload = load_checkpoint(checkpoint_path)
        verify_resume(
            payload,
            config=self.config,
            dataset=self.dataset_identity,
            encoder_identity=self.encoder.identity(),
        )
        stored_model_config = payload["model_config"]
        current_model_config = checkpoint_model_config(self.config)
        if stored_model_config != current_model_config:
            raise ResumeMismatchError(
                "refusing to resume: model shape/hyper-parameters differ\n"
                f"  checkpoint: {stored_model_config}\n  current: {current_model_config}"
            )
        self.run_dir = target
        self.checkpoint_path = checkpoint_path
        self.train_log_path = target / "train_log.jsonl"
        self._reconcile_train_log(int(payload["step"]))
        self.model = build_head_from_model_config(stored_model_config)
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        try:
            self.model.load_state_dict(payload["model_state"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer_state"])
        except (RuntimeError, ValueError) as error:
            raise ResumeMismatchError(
                f"refusing to resume: model/optimizer state does not load ({error})"
            ) from error
        self.step_offset = int(payload["step"]) + 1
        self.totals = dict(FRESH_TOTALS)
        self.totals.update(payload.get("totals", {}))
        self.last_step_record = payload.get("last_step")
        self.eval_paths = {}
        restore_rng_state(payload["rng"])
        if self.step_offset > self.config.optim.steps:
            raise TrainingError(
                f"checkpoint is at step {self.step_offset - 1} but optim.steps="
                f"{self.config.optim.steps}; pass --steps to extend the budget"
            )
        self.log(
            f"resumed {target} at step {payload['step']} (next step {self.step_offset}); "
            "config/data/encoder identity verified"
        )

    def _build_optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.optim.lr),
            weight_decay=float(self.config.optim.weight_decay),
        )

    def _result_kind(self) -> str:
        if self.config.encoder.mode == "fake":
            return "fake-encoder-smoke (engineering pipeline only, not a real model result)"
        return "aut-frozen-training (real frozen AuT features)"

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(dumps_json(dict(payload)) + "\n", encoding="utf-8", newline="\n")

    def _cuda_device_index(self) -> int | None:
        """Integer CUDA index for allocator stats, or ``None`` when not on CUDA."""

        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        if self.device.index is not None:
            return int(self.device.index)
        return int(torch.cuda.current_device())

    def _resource_snapshot(self, *, state: str) -> dict[str, Any]:
        """Device/timing/peak-memory record with explicit scope and reset semantics.

        CUDA values are per-process torch allocator statistics and are ``None``
        on CPU (never a fabricated zero).
        """

        payload: dict[str, Any] = {
            "state": state,
            "device": str(self.device),
            "cuda_available": bool(torch.cuda.is_available()),
            "elapsed_seconds": round(time.perf_counter() - self._resources_started, 6),
            "measurement_scope": (
                "per-process torch CUDA allocator statistics from Trainer construction "
                "(before encoder load) through this point; other processes are not observable"
            ),
            "reset_semantics": (
                "torch.cuda.reset_peak_memory_stats(device) once at Trainer construction "
                "before the encoder is loaded; values are per-process peaks"
            ),
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            index = (
                int(self.device.index)
                if self.device.index is not None
                else int(torch.cuda.current_device())
            )
            payload["device_index"] = index
            payload["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(index))
            payload["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(index))
        else:
            payload["unavailable_reason"] = (
                "CUDA allocator peaks are unavailable for this device (not applicable); "
                "null is reported instead of a fabricated zero"
            )
        return payload

    def _reconcile_train_log(self, checkpoint_step: int) -> None:
        """Drop unaudited log tail and keep the canonical log aligned with state.

        Records after ``checkpoint_step`` were written by a step that never
        produced a checkpoint, so after resume they are recomputed from the
        same per-step seed.  The discarded tail (and any partial last line from
        an interrupted write) is preserved as ``train_log.discarded-<stamp>.jsonl``
        evidence; the canonical log is atomically rewritten to the checkpoint
        prefix.  A prefix that does not exactly cover steps ``0..checkpoint_step``
        is refused instead of being silently spliced.
        """

        if not self.train_log_path.exists():
            raise CheckpointError(
                f"train log {self.train_log_path} is missing but the checkpoint is at "
                f"step {checkpoint_step}; cannot resume with a fabricated history"
            )
        try:
            text = self.train_log_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise CheckpointError(
                f"train log {self.train_log_path} is not valid UTF-8: {error}"
            ) from error

        partial = ""
        if text and not text.endswith("\n"):
            cutoff = text.rfind("\n")
            if cutoff == -1:
                partial, text = text, ""
            else:
                partial, text = text[cutoff + 1 :], text[: cutoff + 1]

        kept: list[str] = []
        discarded: list[str] = []
        corrupt = False
        for index, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                corrupt = True
                discarded.append(line)
                continue
            step = record.get("step")
            if not isinstance(step, int) or isinstance(step, bool):
                raise CheckpointError(
                    f"train log {self.train_log_path} record {index}: missing integer 'step'"
                )
            if corrupt:
                discarded.append(line)
                continue
            if step <= checkpoint_step:
                if step != len(kept):
                    raise CheckpointError(
                        f"train log {self.train_log_path} is incompatible with checkpoint step "
                        f"{checkpoint_step}: expected prefix step {len(kept)} at record {index}, "
                        f"found {step}; refusing to splice"
                    )
                kept.append(line)
            else:
                discarded.append(line)
        if len(kept) != checkpoint_step + 1:
            raise CheckpointError(
                f"train log {self.train_log_path} is incompatible with checkpoint step "
                f"{checkpoint_step}: usable prefix has {len(kept)} record(s), expected "
                f"{checkpoint_step + 1}"
            )
        if discarded or partial:
            evidence = self.run_dir / f"train_log.discarded-{_archive_stamp()}.jsonl"
            suffix = 0
            while evidence.exists():
                suffix += 1
                evidence = self.run_dir / (
                    f"train_log.discarded-{_archive_stamp()}-{suffix}.jsonl"
                )
            body = list(discarded)
            if partial:
                body.append(partial)
            evidence.write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")
            self.log(
                f"preserved {len(body)} discarded/partial train-log line(s) as {evidence.name}"
            )
        temporary = self.train_log_path.with_name(self.train_log_path.name + ".tmp")
        temporary.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, self.train_log_path)

    def _write_run_json(
        self, *, status: str, resources: Mapping[str, Any] | None = None
    ) -> None:
        """Write run metadata; the final call replaces the running snapshot."""

        if resources is None:
            resources = {
                "state": "running",
                "device": str(self.device),
                "measurement_scope": (
                    "per-process torch CUDA allocator statistics from Trainer construction "
                    "(before encoder load) through run end; measured on finish"
                ),
                "reset_semantics": (
                    "torch.cuda.reset_peak_memory_stats(device) once at Trainer construction"
                ),
            }
        fingerprint = self.dataset_identity.fingerprint
        payload = {
            "run": self.config.run.name,
            "status": status,
            "created_at_utc": utc_now(),
            "result_kind": self._result_kind(),
            "encoder_mode": self.config.encoder.mode,
            "device": str(self.device),
            "config": self.config.to_dict(),
            "config_sha256": self.config.config_sha256(),
            "git": self.git,
            "uv_lock_sha256": self.uv_lock,
            "versions": self.versions,
            "encoder": {
                "identity": self.encoder.identity(),
                "provenance": self.encoder.provenance(),
            },
            "dataset": {
                "index_path": str(self.index_path),
                "index_sha256": self.dataset_identity.index_sha256,
                "data_root": str(self.resolved_data_root),
                "digest": fingerprint.get("digest"),
                "per_split": fingerprint.get("per_split"),
                "split_counts": fingerprint.get("split_counts"),
                "leak_free": self.index.summary.get("leak_free"),
                "cross_split_assets": self.index.summary.get("cross_split_assets"),
            },
            "sampler": {"scheme": "seed-sequence-v1", "base_seed": self.config.run.seed},
            "resources": dict(resources),
        }
        self._write_json(self.run_dir / "run.json", payload)

    # -- steps -------------------------------------------------------------- #

    def _sample_training_batch(self, step: int) -> tuple[TrainingBatch, int]:
        seed = step_seed(int(self.config.run.seed), step)
        data = self.config.data
        batch = sample_batch(
            self.index,
            self.resolved_data_root,
            seed=seed,
            split=data.split,
            items=data.groups_per_step,
            centers_per_item=data.centers_per_item,
            min_center_gap=data.min_center_gap,
            valid_only=data.valid_only,
            activity_threshold=data.activity_threshold,
            on_unusable="error",
            window_seconds=self.config.encoder.window_seconds,
            slots=self.config.head.slots,
            verify_digests=data.verify_digests,
        )
        if len(batch.blocks) != data.groups_per_step:
            raise TrainingError(
                f"step {step}: requested {data.groups_per_step} group(s) but got "
                f"{len(batch.blocks)}; unusable samples must fail loudly"
            )
        encoded = []
        for block in batch.blocks:
            starts = window_start_seconds(
                block.center_times,
                sample_rate=block.sample_rate,
                window_seconds=self.config.encoder.window_seconds,
                origin_seconds=block.track_start_seconds,
            )
            encoded.append(
                self.encoder.encode_windows(
                    block.audio,
                    block.sample_rate,
                    window_start_seconds=starts,
                    valid_samples=block.audio_valid,
                )
            )
        tensors = training_batch_from_blocks(batch.blocks, encoded).to_device(self.device)
        return tensors, seed

    def _forward_loss(self, batch: TrainingBatch):
        groups, windows = batch.features.shape[:2]
        features = batch.features.reshape(groups * windows, *batch.features.shape[2:])
        frame_times = batch.frame_times.reshape(groups * windows, batch.frame_times.shape[2])
        frame_valid = batch.frame_valid.reshape(groups * windows, batch.frame_valid.shape[2])
        center_times = batch.center_times.reshape(groups * windows)
        center_valid = batch.center_valid.reshape(groups * windows)
        output = self.model(
            features,
            frame_times=frame_times,
            frame_valid=frame_valid,
            center_times=center_times,
            center_valid=center_valid,
        )
        slots = self.config.head.slots
        embeddings = output.embeddings.view(groups, windows, slots, output.embeddings.shape[-1])
        logits = output.activity_logits.view(groups, windows, slots)
        slot_valid = output.slot_valid.view(groups, windows, slots)
        loss_config = self.config.loss
        return head_loss(
            embeddings,
            logits,
            activity_target=batch.activity,
            source_valid=batch.source_valid,
            composition_ids=batch.composition_ids,
            source_ids=batch.source_ids,
            center_valid=batch.center_valid,
            slot_valid=slot_valid,
            weights=LossWeights(
                activity=loss_config.w_activity,
                empty_slots=loss_config.w_empty_slots,
                positive=loss_config.w_positive,
                negative=loss_config.w_negative,
            ),
            identity_activity_threshold=loss_config.identity_activity_threshold,
            negative_margin=loss_config.negative_margin,
            max_optimal_assignments=loss_config.max_optimal_assignments,
        )

    def _run_step(self, step: int) -> dict[str, Any]:
        started = time.perf_counter()
        batch, seed = self._sample_training_batch(step)
        loss = self._forward_loss(batch)
        total_value = float(loss.total.detach().cpu())
        if not np.isfinite(total_value):
            raise TrainingError(
                f"step {step}: non-finite total loss ({total_value}); refusing to continue"
            )
        self.optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        parameters = [parameter for parameter in self.model.parameters() if parameter.grad is not None]
        for parameter in parameters:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise TrainingError(
                    f"step {step}: non-finite gradient in parameter {tuple(parameter.shape)}"
                )
        if self.config.optim.grad_clip_norm is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.config.optim.grad_clip_norm)
                )
            )
        else:
            total_squares = sum(
                float(parameter.grad.detach().pow(2).sum().cpu())
                for parameter in parameters
            )
            grad_norm = float(total_squares ** 0.5)
        self.optimizer.step()

        stats = loss.stats
        components = loss.summary()
        activity_terms = int(stats.activity_terms)
        has_activity_supervision = activity_terms > 0
        record: dict[str, Any] = {
            "step": step,
            "step_seed": seed,
            "split": self.config.data.split,
            "sample_ids": list(batch.sample_ids),
            "groups": len(batch.sample_ids),
            "windows": int(batch.features.shape[0] * batch.features.shape[1]),
            "source_counts": [group.sources for group in batch.groups],
            "center_valid": sum(group.center_valid for group in batch.groups),
            "active_labels": sum(group.active_labels for group in batch.groups),
            "batch_sha256": batch.content_sha256,
            "loss": {
                "total": components["total"],
                "activity": components["activity"],
                "empty_slots": components["empty_slots"],
                "positive": components["positive"],
                "negative": components["negative"],
            },
            "loss_weights": dict(loss.weights),
            "stats": {
                "groups": stats.groups,
                "matched_groups": stats.matched_groups,
                "skipped_groups": stats.skipped_groups,
                "ambiguous_groups": stats.ambiguous_groups,
                "truncated_groups": stats.truncated_groups,
                "identity_masked_groups": stats.identity_masked_groups,
                "supervision_masked_groups": stats.supervision_masked_groups,
                "activity_terms": stats.activity_terms,
                "empty_terms": stats.empty_terms,
                "positive_terms": stats.positive_terms,
                "negative_terms": stats.negative_terms,
            },
            "effective_supervision": {
                "activity_terms": activity_terms,
                "has_activity_supervision": has_activity_supervision,
                "zero_activity_supervision": not has_activity_supervision,
            },
            "grad_norm": grad_norm,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - started,
            "encoder_mode": self.config.encoder.mode,
            "device": str(self.device),
        }
        self._accumulate_totals(record)
        return record

    def _accumulate_totals(self, record: Mapping[str, Any]) -> None:
        stats = record["stats"]
        self.totals["steps"] += 1
        self.totals["groups"] += int(record["groups"])
        for key in (
            "activity_terms",
            "empty_terms",
            "positive_terms",
            "negative_terms",
            "matched_groups",
            "skipped_groups",
            "ambiguous_groups",
            "truncated_groups",
            "identity_masked_groups",
            "supervision_masked_groups",
        ):
            self.totals[key] = self.totals.get(key, 0) + int(stats[key])
        if record["effective_supervision"]["has_activity_supervision"]:
            self.totals["steps_with_activity_supervision"] += 1
        else:
            self.totals["steps_without_activity_supervision"] += 1

    def _append_log(self, record: Mapping[str, Any]) -> None:
        # Documents are written pretty via dumps_json; the JSONL log needs one
        # compact line per step for machine consumption.
        line = json.dumps(dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with open(self.train_log_path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    def _write_checkpoint(self, step: int) -> None:
        save_checkpoint(
            self.checkpoint_path,
            step=step,
            model_state={key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            optimizer_state=self.optimizer.state_dict(),
            config=self.config,
            dataset=self.dataset_identity,
            encoder_identity=self.encoder.identity(),
            encoder_provenance=self.encoder.provenance(),
            last_step_record=self.last_step_record,
            totals=self.totals,
            resources=self._resource_snapshot(state="in_progress"),
        )

    # -- orchestration ------------------------------------------------------ #

    def run(self) -> TrainingSummary:
        total_steps = int(self.config.optim.steps)
        first_step = self.step_offset
        last_step: int | None = None
        for step in range(self.step_offset, total_steps):
            record = self._run_step(step)
            self.last_step_record = record
            self._append_log(record)
            last_step = step
            self.log(
                f"step {step + 1}/{total_steps}: total={record['loss']['total']:.4f} "
                f"activity={record['loss']['activity']:.4f} "
                f"empty={record['loss']['empty_slots']:.4f} "
                f"pos={record['loss']['positive']:.4f} "
                f"neg={record['loss']['negative']:.4f} "
                f"terms={record['stats']['activity_terms']} "
                f"grad={record['grad_norm']:.3f}"
            )
            if (step + 1) % self.config.checkpoint.interval_steps == 0 or step == total_steps - 1:
                self._write_checkpoint(step)
        if last_step is None:
            raise TrainingError(
                "nothing to do: the checkpoint already reached optim.steps; pass --steps to extend"
            )

        status = (
            "completed" if self.totals.get("activity_terms", 0) > 0 else "no_effective_supervision"
        )
        if status != "completed":
            self.log(
                "WARNING: no effective activity supervision over the whole run "
                f"(activity_terms={self.totals.get('activity_terms', 0)}); "
                "this run is not a successful training result"
            )

        eval_paths: dict[str, str] = {}
        if self.config.eval.enabled:
            inference = HeadInference(
                config=self.config,
                model=self.model,
                encoder=self.encoder,
                device=self.device,
                checkpoint_path=str(self.checkpoint_path),
                step=last_step,
            )
            for split in self.config.eval.splits:
                report = evaluate_split(
                    index=self.index,
                    data_root=self.resolved_data_root,
                    split=split,
                    inference=inference,
                    config=self.config,
                    git_commit=self.git.get("sha"),
                )
                report_path = self.run_dir / f"eval_{split}.json"
                self._write_json(report_path, report)
                eval_paths[split] = str(report_path)
                micro = report["micro"]
                self.log(
                    f"eval {split}: f1={micro['f1']} precision={micro['precision']} "
                    f"recall={micro['recall']} (valid_centers={report['effective']['valid_centers']})"
                )

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self._cuda_device_index())
        resources = self._resource_snapshot(state="finished")
        summary = TrainingSummary(
            run_dir=str(self.run_dir),
            status=status,
            result_kind=self._result_kind(),
            encoder_mode=self.config.encoder.mode,
            resumed=self.resumed,
            first_step=first_step if first_step <= last_step else None,
            last_step=last_step,
            steps_completed=int(self.totals.get("steps", 0)),
            checkpoint_path=str(self.checkpoint_path),
            train_log_path=str(self.train_log_path),
            summary_path=str(self.run_dir / "summary.json"),
            eval_paths=eval_paths,
            config_sha256=self.config.config_sha256(),
            dataset_digest=self.dataset_identity.digest,
            totals=dict(self.totals),
            git=dict(self.git),
            uv_lock_sha256=self.uv_lock,
            versions=dict(self.versions),
            resources=resources,
        )
        self._write_json(self.run_dir / "summary.json", summary.to_dict())
        self._write_run_json(status=status, resources=resources)
        return summary


def train_from_config(
    config: TrainConfig,
    *,
    device: str | None = None,
    out_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    overwrite: bool = False,
    log: Callable[[str], None] | None = None,
) -> TrainingSummary:
    """Convenience entry point used by the CLI and tests."""

    trainer = Trainer(
        config,
        device=device,
        run_dir=out_dir,
        resume_from=resume_from,
        overwrite=overwrite,
        log=log,
    )
    return trainer.run()


__all__ = [
    "FRESH_TOTALS",
    "RUN_ARTIFACTS",
    "Trainer",
    "TrainingSummary",
    "train_from_config",
]
