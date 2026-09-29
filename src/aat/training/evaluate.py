"""Held-out evaluation and constant baselines (issue #8).

The evaluation protocol is fixed before a run (default activity threshold 0.5)
and never tuned on the evaluated split: every song is predicted on its valid
label centers, associated into whole-song trajectories with the issue #9
tracker (one-to-one, identity-aware) and scored with
:func:`aat.evaluation.evaluate_trajectory`, which uses a **global** per-song
source-to-track mapping instead of per-frame re-matching.  Three constant
baselines are scored with the exact same protocol:

``all_inactive``
    no candidate at all (TP=0, FN=all active labels; precision undefined);
``all_active``
    every slot active in every window with a constant identity;
``no_identity``
    the model's activity with all embeddings replaced by one constant vector,
    so association has no identity information.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aat.contracts import ActivityData, PredictionData
from aat.contracts.documents import AudioSpan, RunProvenance
from aat.data import DatasetIndex, SampleEntry
from aat.evaluation import EvaluationConfig, evaluate_trajectory
from aat.labels import read_wav
from aat.tracking import PredictionSequence, TrackingConfig, WindowPrediction, associate_sequence

from .checkpoint import utc_now
from .config import TrainConfig
from .errors import TrainingError
from .inference import HeadInference

BASELINES = ("all_inactive", "all_active", "no_identity")


@dataclass(frozen=True)
class EvaluationProtocol:
    """Frozen scoring protocol recorded in every report."""

    threshold: float
    tracking: TrackingConfig
    time_tolerance_seconds: float = 1e-6


def tracking_config_from(config: TrainConfig) -> TrackingConfig:
    evaluation = config.eval
    return TrackingConfig(
        activity_threshold=evaluation.threshold,
        match_threshold=evaluation.match_threshold,
        retention_seconds=evaluation.retention_seconds,
        prototype_alpha=evaluation.prototype_alpha,
        birth_threshold=evaluation.birth_threshold,
        max_exact_slots=evaluation.max_exact_slots,
    )


def _validate_git_commit(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if 7 <= len(text) <= 40 and all(character in "0123456789abcdef" for character in text):
        return text
    return None


def _provenance(
    config: TrainConfig,
    *,
    data_kind: str,
    step: int | None,
    git_commit: str | None,
    notes: str | None = None,
) -> RunProvenance:
    return RunProvenance(
        run_id=f"aat-train-eval-step{step}" if step is not None else "aat-train-eval",
        data_kind=data_kind,
        model_id=f"source-query-head:step={step}:encoder={config.encoder.mode}"
        if step is not None
        else f"source-query-head:encoder={config.encoder.mode}",
        git_commit=_validate_git_commit(git_commit),
        config_hash=config.config_sha256()[:16],
        created_at_utc=utc_now(),
        notes=notes,
    )


def _trajectory(
    predictions: PredictionData,
    *,
    reference_entry: SampleEntry,
    audio_seconds: float,
    provenance: RunProvenance,
    protocol: EvaluationProtocol,
):
    return associate_sequence(
        PredictionSequence.from_prediction(predictions),
        AudioSpan(
            duration_seconds=float(audio_seconds),
            track_start_seconds=float(reference_entry.track_start_seconds),
        ),
        provenance,
        config=protocol.tracking,
        sample_id=reference_entry.sample_id,
    )


def _constant_embeddings(rows: int, slots: int) -> np.ndarray:
    embeddings = np.zeros((rows, slots, 128), dtype=np.float32)
    embeddings[..., 0] = 1.0
    return embeddings


def _baseline_predictions(
    model_predictions: PredictionData,
    name: str,
) -> PredictionData:
    rows = int(model_predictions.center_times.shape[0])
    slots = int(model_predictions.slots)
    if name == "all_inactive":
        return PredictionData(
            center_times=model_predictions.center_times,
            embeddings=np.zeros((rows, slots, 128), dtype=np.float32),
            activity=np.zeros((rows, slots), dtype=np.float32),
            slot_valid=np.zeros((rows, slots), dtype=bool),
            center_valid=model_predictions.center_valid,
            slots=slots,
            embedding_dim=128,
            hop_seconds=model_predictions.hop_seconds,
            sample_id=model_predictions.sample_id,
            provenance=model_predictions.provenance,
        )
    if name == "all_active":
        embeddings = _constant_embeddings(rows, slots)
        return PredictionData(
            center_times=model_predictions.center_times,
            embeddings=embeddings,
            activity=np.ones((rows, slots), dtype=np.float32),
            slot_valid=np.ones((rows, slots), dtype=bool),
            center_valid=model_predictions.center_valid,
            slots=slots,
            embedding_dim=128,
            hop_seconds=model_predictions.hop_seconds,
            sample_id=model_predictions.sample_id,
            provenance=model_predictions.provenance,
        )
    if name == "no_identity":
        return PredictionData(
            center_times=model_predictions.center_times,
            embeddings=_constant_embeddings(rows, slots),
            activity=np.array(model_predictions.activity, dtype=np.float32, copy=True),
            slot_valid=np.array(model_predictions.slot_valid, dtype=bool, copy=True),
            center_valid=model_predictions.center_valid,
            slots=slots,
            embedding_dim=128,
            hop_seconds=model_predictions.hop_seconds,
            sample_id=model_predictions.sample_id,
            provenance=model_predictions.provenance,
        )
    raise TrainingError(f"unknown baseline {name!r}; expected one of {list(BASELINES)}")


def _metrics_dict(evaluation) -> dict[str, Any]:
    return evaluation.to_dict()


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positives"]) for row in rows)
    fp = sum(int(row["false_positives"]) for row in rows)
    fn = sum(int(row["false_negatives"]) for row in rows)
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    denominator = 2 * tp + fp + fn
    f1 = (2 * tp / denominator) if denominator > 0 else None
    onset_rows = [
        (float(row["boundary_onset_mae"]), int(row["boundary_segments_compared"]))
        for row in rows
        if row["boundary_onset_mae"] is not None
    ]
    offset_rows = [
        (float(row["boundary_offset_mae"]), int(row["boundary_segments_compared"]))
        for row in rows
        if row["boundary_offset_mae"] is not None
    ]

    def weighted(entries: list[tuple[float, int]]) -> float | None:
        total = sum(weight for _, weight in entries)
        if total <= 0:
            return None
        return sum(value * weight for value, weight in entries) / total

    return {
        "songs": len(rows),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "id_switch_count": sum(int(row["id_switch_count"]) for row in rows),
        "ambiguous_owner_frames": sum(int(row["ambiguous_owner_frames"]) for row in rows),
        "reference_source_count": sum(int(row["reference_source_count"]) for row in rows),
        "predicted_track_count": sum(int(row["predicted_track_count"]) for row in rows),
        "source_count_abs_error": sum(int(row["source_count_abs_error"]) for row in rows),
        "boundary_onset_mae": weighted(onset_rows),
        "boundary_offset_mae": weighted(offset_rows),
        "boundary_segments_compared": sum(int(row["boundary_segments_compared"]) for row in rows),
        "boundary_segments_missed": sum(int(row["boundary_segments_missed"]) for row in rows),
    }


def _undefined_reasons(aggregate: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if aggregate["precision"] is None:
        reasons.append("precision undefined: no positive predictions (TP+FP=0)")
    if aggregate["recall"] is None:
        reasons.append("recall undefined: no positive reference labels (TP+FN=0)")
    if aggregate["f1"] is None:
        reasons.append("f1 undefined: no positives in either side (2TP+FP+FN=0)")
    if aggregate["boundary_onset_mae"] is None:
        reasons.append("boundary onset MAE undefined: no comparable active segments")
    if aggregate["boundary_offset_mae"] is None:
        reasons.append("boundary offset MAE undefined: no comparable active segments")
    return reasons


def evaluate_song(
    entry: SampleEntry,
    *,
    data_root: str | Path,
    inference: HeadInference,
    config: TrainConfig,
    protocol: EvaluationProtocol,
    git_commit: str | None,
) -> dict[str, Any]:
    """Evaluated model predictions plus constant baselines for one song."""

    directory = Path(data_root) / entry.path
    reference = ActivityData.load(directory)
    wav = read_wav(directory / str(entry.audio["mix_path"]))
    if wav.sample_rate != entry.sample_rate:
        raise TrainingError(
            f"sample {entry.sample_id!r}: mix sample rate {wav.sample_rate} != "
            f"index {entry.sample_rate}"
        )
    mono = wav.samples.mean(axis=1) if wav.channels > 1 else wav.samples[:, 0]
    valid_centers = np.asarray(reference.center_times, dtype=np.float64)[
        np.asarray(reference.valid, dtype=bool)
    ]

    provenance = _provenance(
        config,
        data_kind="model",
        step=inference.step,
        git_commit=git_commit,
    )
    predictions = inference.predict_at_times(
        mono,
        valid_centers,
        sample_rate=wav.sample_rate,
        origin_seconds=float(entry.track_start_seconds),
        sample_id=entry.sample_id,
        hop_seconds=float(reference.hop_seconds) if reference.hop_seconds else None,
        provenance=provenance,
    )
    audio_seconds = wav.frames / wav.sample_rate
    trajectory = _trajectory(
        predictions,
        reference_entry=entry,
        audio_seconds=audio_seconds,
        provenance=provenance,
        protocol=protocol,
    )
    evaluation = evaluate_trajectory(
        reference,
        trajectory,
        EvaluationConfig(
            activity_threshold=protocol.threshold,
            time_tolerance_seconds=protocol.time_tolerance_seconds,
        ),
    )
    valid = np.asarray(reference.valid, dtype=bool)
    active = (np.asarray(reference.activity, dtype=np.float64) >= protocol.threshold) & valid[:, None]
    result: dict[str, Any] = {
        "sample_id": entry.sample_id,
        "split": entry.split,
        "path": entry.path,
        "duration_seconds": audio_seconds,
        "valid_centers": int(valid.sum()),
        "active_labels": int(active.sum()),
        "metrics": _metrics_dict(evaluation),
        "baselines": {},
    }
    if config.eval.baselines:
        for name in BASELINES:
            baseline_predictions = _baseline_predictions(predictions, name)
            baseline_provenance = _provenance(
                config,
                data_kind="mock",
                step=inference.step,
                git_commit=git_commit,
                notes=f"analytic {name} baseline; not a trained model",
            )
            baseline_trajectory = _trajectory(
                baseline_predictions,
                reference_entry=entry,
                audio_seconds=audio_seconds,
                provenance=baseline_provenance,
                protocol=protocol,
            )
            baseline_evaluation = evaluate_trajectory(
                reference,
                baseline_trajectory,
                EvaluationConfig(
                    activity_threshold=protocol.threshold,
                    time_tolerance_seconds=protocol.time_tolerance_seconds,
                ),
            )
            result["baselines"][name] = _metrics_dict(baseline_evaluation)
    return result


def evaluate_split(
    *,
    index: DatasetIndex,
    data_root: str | Path,
    split: str,
    inference: HeadInference,
    config: TrainConfig,
    max_songs: int | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Evaluate one held-out split and return a machine-readable report."""

    if split not in ("train", "val", "test"):
        raise TrainingError(f"eval split: unknown split {split!r}")
    protocol = EvaluationProtocol(
        threshold=config.eval.threshold,
        tracking=tracking_config_from(config),
    )
    entries = sorted(index.samples_for_split(split), key=lambda entry: entry.sample_id)
    limit = config.eval.max_songs if max_songs is None else max_songs
    entries = entries[: int(limit)]
    songs: list[dict[str, Any]] = []
    for entry in entries:
        songs.append(
            evaluate_song(
                entry,
                data_root=data_root,
                inference=inference,
                config=config,
                protocol=protocol,
                git_commit=git_commit,
            )
        )
    micro = _aggregate([song["metrics"] for song in songs])
    baselines = {
        name: _aggregate([song["baselines"][name] for song in songs])
        for name in BASELINES
        if songs and name in songs[0]["baselines"]
    }
    per_source: list[dict[str, Any]] = []
    for song in songs:
        for source_id, row in song["metrics"]["per_source"].items():
            per_source.append({"sample_id": song["sample_id"], **row})
    return {
        "split": split,
        "protocol": {
            "activity_threshold": protocol.threshold,
            "time_tolerance_seconds": protocol.time_tolerance_seconds,
            "assignment": "whole-song one-to-one global mapping (no per-frame rematch)",
            "tracking": protocol.tracking.as_params(),
            "baselines": list(BASELINES),
            "fake_vs_real": (
                "fake-encoder smoke: pipeline evidence only, not a real model result"
                if config.encoder.mode == "fake"
                else "frozen AuT features (real encoder)"
            ),
        },
        "checkpoint": inference.describe(),
        "effective": {
            "songs": len(songs),
            "valid_centers": sum(song["valid_centers"] for song in songs),
            "active_labels": sum(song["active_labels"] for song in songs),
        },
        "micro": micro,
        "undefined_reasons": _undefined_reasons(micro),
        "baselines_micro": baselines,
        "baselines_undefined_reasons": {
            name: _undefined_reasons(values) for name, values in baselines.items()
        },
        "per_source": per_source,
        "songs": songs,
    }


__all__ = [
    "BASELINES",
    "EvaluationProtocol",
    "evaluate_song",
    "evaluate_split",
    "tracking_config_from",
]
