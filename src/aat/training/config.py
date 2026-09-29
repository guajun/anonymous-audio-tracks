"""Strict training configuration for the frozen-AuT minimum run (issue #8).

Every knob that changes the trained state lives in one frozen dataclass tree so
that it can be validated, hashed, snapshotted into a checkpoint and compared on
resume.  Unknown keys, wrong types and out-of-range values are rejected with the
offending key path instead of being silently ignored.

Two encoder modes are explicitly separated:

``fake``
    :class:`aat.encoders.fake.FakeAutEncoder` (NumPy only).  Results are an
    engineering smoke, never a real model result.
``aut``
    the frozen Qwen3-Omni AuT encoder from issue #5, forced to the audited
    block-diagonal SDPA path, float32, independent 2 s windows.

The config file uses TOML (``tomllib`` from the standard library).  Relative
paths are resolved by the caller (the training CLI runs from the repository
root).  No path in the committed examples is machine-specific.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .errors import TrainingError

#: Semantic marker of the pinned extraction policy for this issue.
PINNED_WINDOW_SECONDS = 2.0
#: Audited semantic attention mode (see docs/AUT_PROBE.md section 4.4).
PINNED_ATTENTION = "block_diagonal"
#: Extraction mode: independent fixed windows everywhere (train and inference).
PINNED_EXTRACTION = "independent_windows"
#: This issue pins float32 AuT features to reduce batch-shape differences.
PINNED_DTYPE = "float32"

ENCODER_MODES = ("fake", "aut")
SPLITS = ("train", "val", "test")


def canonical_json(payload: Any) -> str:
    """Deterministic compact JSON for hashing (rejects NaN/Inf)."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_canonical(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _check_keys(table: Mapping[str, Any], allowed: tuple[str, ...], path: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise TrainingError(
            f"{path}: unknown key(s) {unknown}; allowed keys: {list(allowed)}"
        )


def _require(table: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in table:
        raise TrainingError(f"{path}.{key}: missing required value")
    return table[key]


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TrainingError(f"{path}: expected a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise TrainingError(f"{path}: must not be empty")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingError(f"{path}: expected an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise TrainingError(f"{path}: must be >= {minimum}, got {value}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError(f"{path}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise TrainingError(f"{path}: must be finite, got {value!r}")
    if minimum is not None:
        too_small = number <= minimum if exclusive_minimum else number < minimum
        if too_small:
            comparator = ">" if exclusive_minimum else ">="
            raise TrainingError(f"{path}: must be {comparator} {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise TrainingError(f"{path}: must be <= {maximum}, got {number}")
    return number


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingError(f"{path}: expected a boolean, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class RunConfig:
    """Run identity and output location."""

    name: str
    out_dir: str
    seed: int
    device: str = "cpu"
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "out_dir": self.out_dir,
            "seed": self.seed,
            "device": self.device,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class EncoderConfig:
    """Encoder mode, provenance expectations and the fixed batch policy."""

    mode: str = "fake"
    feature_dim: int = 16
    fake_seed: int = 20260929
    model_dir: str | None = None
    model_id: str | None = None
    revision: str | None = None
    layer: int | None = None
    dtype: str = PINNED_DTYPE
    attention: str = PINNED_ATTENTION
    extraction: str = PINNED_EXTRACTION
    window_seconds: float = PINNED_WINDOW_SECONDS
    batch_windows: int = 8

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "feature_dim": self.feature_dim,
            "fake_seed": self.fake_seed,
            "model_dir": self.model_dir,
            "model_id": self.model_id,
            "revision": self.revision,
            "layer": self.layer,
            "dtype": self.dtype,
            "attention": self.attention,
            "extraction": self.extraction,
            "window_seconds": self.window_seconds,
            "batch_windows": self.batch_windows,
        }

    def identity_dict(self) -> dict[str, Any]:
        """Semantic encoder identity used for resume checks.

        ``model_dir`` is a local path and is deliberately excluded: the loaded
        provenance (model id/revision/layer) plus dtype/attention/extraction are
        the identity.  The caller additionally compares the resolved load
        report when the checkpoint was written by a real run.
        """

        return {
            "mode": self.mode,
            "feature_dim": self.feature_dim,
            "fake_seed": self.fake_seed,
            "model_id": self.model_id,
            "revision": self.revision,
            "layer": self.layer,
            "dtype": self.dtype,
            "attention": self.attention,
            "extraction": self.extraction,
            "window_seconds": self.window_seconds,
            "batch_windows": self.batch_windows,
        }


@dataclass(frozen=True)
class DataConfig:
    """Dataset index/split and same-song multi-center batch sampling."""

    index: str
    data_root: str | None = None
    split: str = "train"
    groups_per_step: int = 1
    centers_per_item: int = 4
    min_center_gap: int = 5
    valid_only: bool = True
    activity_threshold: float = 0.5
    verify_digests: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "data_root": self.data_root,
            "split": self.split,
            "groups_per_step": self.groups_per_step,
            "centers_per_item": self.centers_per_item,
            "min_center_gap": self.min_center_gap,
            "valid_only": self.valid_only,
            "activity_threshold": self.activity_threshold,
            "verify_digests": self.verify_digests,
        }

    def identity_dict(self) -> dict[str, Any]:
        # ``index``/``data_root`` are paths; the dataset digest is the real
        # identity and is stored separately in the checkpoint.
        return {
            key: value
            for key, value in self.to_dict().items()
            if key not in ("index", "data_root")
        }


@dataclass(frozen=True)
class HeadConfig:
    """`aat.models.SourceQueryHead` hyper-parameters."""

    slots: int = 8
    d_model: int = 128
    num_heads: int = 4
    dropout: float = 0.0
    time_frequencies: int = 6
    base_frequency_hz: float = 1.0
    ffn_multiplier: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": self.slots,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "time_frequencies": self.time_frequencies,
            "base_frequency_hz": self.base_frequency_hz,
            "ffn_multiplier": self.ffn_multiplier,
        }


@dataclass(frozen=True)
class LossConfig:
    """Component weights and matching/identity parameters of ``head_loss``."""

    w_activity: float = 1.0
    w_empty_slots: float = 0.5
    w_positive: float = 1.0
    w_negative: float = 1.0
    identity_activity_threshold: float = 0.5
    negative_margin: float = 0.25
    max_optimal_assignments: int = 64

    def to_dict(self) -> dict[str, Any]:
        return {
            "w_activity": self.w_activity,
            "w_empty_slots": self.w_empty_slots,
            "w_positive": self.w_positive,
            "w_negative": self.w_negative,
            "identity_activity_threshold": self.identity_activity_threshold,
            "negative_margin": self.negative_margin,
            "max_optimal_assignments": self.max_optimal_assignments,
        }


@dataclass(frozen=True)
class OptimConfig:
    """AdamW optimizer settings and step budget."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    steps: int = 60
    grad_clip_norm: float | None = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "steps": self.steps,
            "grad_clip_norm": self.grad_clip_norm,
        }

    def identity_dict(self) -> dict[str, Any]:
        # The step budget may be extended explicitly on the CLI; it is not a
        # hyper-parameter of the achieved state.
        return {key: value for key, value in self.to_dict().items() if key != "steps"}


@dataclass(frozen=True)
class CheckpointConfig:
    interval_steps: int = 20

    def to_dict(self) -> dict[str, Any]:
        return {"interval_steps": self.interval_steps}


@dataclass(frozen=True)
class EvalConfig:
    """Held-out evaluation protocol; the threshold is fixed for the run."""

    enabled: bool = True
    splits: tuple[str, ...] = ("val",)
    threshold: float = 0.5
    max_songs: int = 4
    max_windows_per_forward: int = 16
    match_threshold: float = 0.7
    retention_seconds: float = 1.0
    prototype_alpha: float = 0.9
    birth_threshold: float | None = None
    max_exact_slots: int = 16
    baselines: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "splits": list(self.splits),
            "threshold": self.threshold,
            "max_songs": self.max_songs,
            "max_windows_per_forward": self.max_windows_per_forward,
            "match_threshold": self.match_threshold,
            "retention_seconds": self.retention_seconds,
            "prototype_alpha": self.prototype_alpha,
            "birth_threshold": self.birth_threshold,
            "max_exact_slots": self.max_exact_slots,
            "baselines": self.baselines,
        }


@dataclass(frozen=True)
class TrainConfig:
    """Complete validated configuration for one training run."""

    run: RunConfig
    encoder: EncoderConfig
    data: DataConfig
    head: HeadConfig
    loss: LossConfig
    optim: OptimConfig
    checkpoint: CheckpointConfig
    eval: EvalConfig

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "encoder": self.encoder.to_dict(),
            "data": self.data.to_dict(),
            "head": self.head.to_dict(),
            "loss": self.loss.to_dict(),
            "optim": self.optim.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "eval": self.eval.to_dict(),
        }

    def config_sha256(self) -> str:
        return sha256_canonical(self.to_dict())

    def identity_dict(self) -> dict[str, Any]:
        """Fields that must match when resuming a checkpoint.

        Paths and operationally irrelevant knobs (output directory, run name,
        eval protocol, checkpoint interval, step budget, device) are excluded;
        data and encoder identity are compared separately through the stored
        digest/provenance.
        """

        return {
            "seed": self.run.seed,
            "encoder": self.encoder.identity_dict(),
            "data": self.data.identity_dict(),
            "head": self.head.to_dict(),
            "loss": self.loss.to_dict(),
            "optim": self.optim.identity_dict(),
        }

    def identity_sha256(self) -> str:
        return sha256_canonical(self.identity_dict())

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, origin: str = "<config>") -> "TrainConfig":
        if not isinstance(payload, Mapping):
            raise TrainingError(f"{origin}: expected a TOML table")
        _check_keys(
            payload,
            ("run", "encoder", "data", "head", "loss", "optim", "checkpoint", "eval"),
            origin,
        )
        table = payload

        run_table = _check_subtable(table, "run", origin)
        _check_keys(run_table, ("name", "out_dir", "seed", "device", "notes"), f"{origin}.run")
        run = RunConfig(
            name=_string(_require(run_table, "name", f"{origin}.run"), f"{origin}.run.name"),
            out_dir=_string(
                _require(run_table, "out_dir", f"{origin}.run"), f"{origin}.run.out_dir"
            ),
            seed=_integer(_require(run_table, "seed", f"{origin}.run"), f"{origin}.run.seed", minimum=0),
            device=_string(run_table.get("device", "cpu"), f"{origin}.run.device"),
            notes=(
                None
                if run_table.get("notes") is None
                else _string(run_table.get("notes"), f"{origin}.run.notes", allow_empty=True)
            ),
        )

        encoder_table = _optional_subtable(table, "encoder", origin)
        _check_keys(
            encoder_table,
            (
                "mode",
                "feature_dim",
                "fake_seed",
                "model_dir",
                "model_id",
                "revision",
                "layer",
                "dtype",
                "attention",
                "extraction",
                "window_seconds",
                "batch_windows",
            ),
            f"{origin}.encoder",
        )
        encoder = _parse_encoder(encoder_table, origin)

        data_table = _check_subtable(table, "data", origin)
        _check_keys(
            data_table,
            (
                "index",
                "data_root",
                "split",
                "groups_per_step",
                "centers_per_item",
                "min_center_gap",
                "valid_only",
                "activity_threshold",
                "verify_digests",
            ),
            f"{origin}.data",
        )
        data = _parse_data(data_table, origin)

        head_table = _optional_subtable(table, "head", origin)
        _check_keys(
            head_table,
            (
                "slots",
                "d_model",
                "num_heads",
                "dropout",
                "time_frequencies",
                "base_frequency_hz",
                "ffn_multiplier",
            ),
            f"{origin}.head",
        )
        head = _parse_head(head_table, origin)

        loss_table = _optional_subtable(table, "loss", origin)
        _check_keys(
            loss_table,
            (
                "w_activity",
                "w_empty_slots",
                "w_positive",
                "w_negative",
                "identity_activity_threshold",
                "negative_margin",
                "max_optimal_assignments",
            ),
            f"{origin}.loss",
        )
        loss = _parse_loss(loss_table, origin)

        optim_table = _optional_subtable(table, "optim", origin)
        _check_keys(optim_table, ("lr", "weight_decay", "steps", "grad_clip_norm"), f"{origin}.optim")
        optim = _parse_optim(optim_table, origin)

        checkpoint_table = _optional_subtable(table, "checkpoint", origin)
        _check_keys(checkpoint_table, ("interval_steps",), f"{origin}.checkpoint")
        checkpoint = CheckpointConfig(
            interval_steps=_integer(
                checkpoint_table.get("interval_steps", 20),
                f"{origin}.checkpoint.interval_steps",
                minimum=1,
            )
        )

        eval_table = _optional_subtable(table, "eval", origin)
        _check_keys(
            eval_table,
            (
                "enabled",
                "splits",
                "threshold",
                "max_songs",
                "max_windows_per_forward",
                "match_threshold",
                "retention_seconds",
                "prototype_alpha",
                "birth_threshold",
                "max_exact_slots",
                "baselines",
            ),
            f"{origin}.eval",
        )
        evaluation = _parse_eval(eval_table, origin)

        config = cls(
            run=run,
            encoder=encoder,
            data=data,
            head=head,
            loss=loss,
            optim=optim,
            checkpoint=checkpoint,
            eval=evaluation,
        )
        config._validate()
        return config

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrainConfig":
        source = Path(path)
        if not source.is_file():
            raise TrainingError(f"{source}: training config file does not exist")
        try:
            with open(source, "rb") as handle:
                payload = tomllib.load(handle)
        except tomllib.TOMLDecodeError as error:
            raise TrainingError(f"{source}: invalid TOML: {error}") from error
        return cls.from_dict(payload, origin=source.name)

    # -- validation --------------------------------------------------------- #

    def _validate(self) -> None:
        encoder = self.encoder
        if encoder.mode not in ENCODER_MODES:
            raise TrainingError(
                f"encoder.mode: expected one of {list(ENCODER_MODES)}, got {encoder.mode!r}"
            )
        if encoder.window_seconds != PINNED_WINDOW_SECONDS:
            raise TrainingError(
                "encoder.window_seconds: issue #8 pins independent "
                f"{PINNED_WINDOW_SECONDS:g}s windows; got {encoder.window_seconds:g}. "
                "A different window needs protocol/experiment review first."
            )
        if encoder.attention != PINNED_ATTENTION:
            raise TrainingError(
                f"encoder.attention: issue #8 uses the audited {PINNED_ATTENTION!r} path; "
                f"got {encoder.attention!r}"
            )
        if encoder.extraction != PINNED_EXTRACTION:
            raise TrainingError(
                f"encoder.extraction: train and inference must both use "
                f"{PINNED_EXTRACTION!r}; got {encoder.extraction!r}"
            )
        if encoder.dtype != PINNED_DTYPE:
            raise TrainingError(
                f"encoder.dtype: issue #8 pins {PINNED_DTYPE!r} to reduce batch-shape "
                f"differences; got {encoder.dtype!r}"
            )
        if encoder.mode == "fake":
            if encoder.model_dir is not None:
                raise TrainingError("encoder.model_dir must not be set in fake mode")
            if encoder.revision is not None or encoder.model_id is not None:
                raise TrainingError(
                    "encoder.revision/model_id only apply to the real AuT mode"
                )
        else:
            if not encoder.model_dir:
                raise TrainingError(
                    "encoder.model_dir: required in aut mode (pinned checkpoint directory)"
                )
            if not encoder.revision:
                raise TrainingError(
                    "encoder.revision: required in aut mode so a locally supplied "
                    "checkpoint cannot be silently treated as the audited revision"
                )
        if encoder.mode == "aut" and encoder.feature_dim != 2048:
            raise TrainingError(
                "encoder.feature_dim: the audited AuT encoder emits 2048 features; "
                f"got {encoder.feature_dim}"
            )
        if encoder.batch_windows < 1:
            raise TrainingError("encoder.batch_windows: must be >= 1")

        data = self.data
        if data.split not in SPLITS:
            raise TrainingError(f"data.split: expected one of {list(SPLITS)}, got {data.split!r}")

        head = self.head
        if head.d_model % head.num_heads != 0:
            raise TrainingError(
                f"head.d_model ({head.d_model}) must be divisible by head.num_heads "
                f"({head.num_heads})"
            )
        if head.slots > 16:
            raise TrainingError(
                "head.slots: the exact group matching used by #7 is limited to K <= 16"
            )
        if self.loss.identity_activity_threshold > 1.0:
            raise TrainingError("loss.identity_activity_threshold must be inside [0, 1]")

        evaluation = self.eval
        unknown_splits = [split for split in evaluation.splits if split not in SPLITS]
        if unknown_splits:
            raise TrainingError(f"eval.splits: unknown split(s) {unknown_splits}")
        if not evaluation.splits:
            raise TrainingError("eval.splits: at least one split is required")

    def with_overrides(
        self,
        *,
        out_dir: str | None = None,
        model_dir: str | None = None,
        steps: int | None = None,
        device: str | None = None,
    ) -> "TrainConfig":
        """Return a validated copy with explicit CLI overrides applied."""

        from dataclasses import replace

        encoder = self.encoder
        if model_dir is not None:
            encoder = replace(encoder, model_dir=model_dir)
        run = self.run
        if out_dir is not None:
            run = replace(run, out_dir=out_dir)
        if device is not None:
            run = replace(run, device=device)
        optim = self.optim
        if steps is not None:
            if steps < 1:
                raise TrainingError(f"--steps: must be >= 1, got {steps}")
            optim = replace(optim, steps=steps)
        result = replace(self, run=run, encoder=encoder, optim=optim)
        result._validate()
        return result


def _check_subtable(payload: Mapping[str, Any], key: str, origin: str) -> Mapping[str, Any]:
    if key not in payload:
        raise TrainingError(f"{origin}: missing required table [{key}]")
    value = payload[key]
    if not isinstance(value, Mapping):
        raise TrainingError(f"{origin}.{key}: expected a table")
    return value


def _optional_subtable(payload: Mapping[str, Any], key: str, origin: str) -> Mapping[str, Any]:
    """Return an optional subtable; a missing table means "all defaults"."""

    if key not in payload:
        return {}
    value = payload[key]
    if not isinstance(value, Mapping):
        raise TrainingError(f"{origin}.{key}: expected a table")
    return value


def _parse_encoder(table: Mapping[str, Any], origin: str) -> EncoderConfig:
    path = f"{origin}.encoder"
    layer_raw = table.get("layer", "final")
    if layer_raw is None:
        layer = None
    elif isinstance(layer_raw, bool):
        raise TrainingError(f"{path}.layer: expected 'final' or an integer, got a boolean")
    elif isinstance(layer_raw, int):
        layer = _integer(layer_raw, f"{path}.layer", minimum=0)
    elif isinstance(layer_raw, str) and layer_raw == "final":
        layer = None
    else:
        raise TrainingError(f"{path}.layer: expected 'final' or an integer, got {layer_raw!r}")

    def optional_string(key: str) -> str | None:
        value = table.get(key)
        if value is None:
            return None
        return _string(value, f"{path}.{key}")

    return EncoderConfig(
        mode=_string(table.get("mode", "fake"), f"{path}.mode"),
        feature_dim=_integer(
            table.get("feature_dim", 16), f"{path}.feature_dim", minimum=1
        ),
        fake_seed=_integer(table.get("fake_seed", 20260929), f"{path}.fake_seed", minimum=0),
        model_dir=optional_string("model_dir"),
        model_id=optional_string("model_id"),
        revision=optional_string("revision"),
        layer=layer,
        dtype=_string(table.get("dtype", PINNED_DTYPE), f"{path}.dtype"),
        attention=_string(table.get("attention", PINNED_ATTENTION), f"{path}.attention"),
        extraction=_string(table.get("extraction", PINNED_EXTRACTION), f"{path}.extraction"),
        window_seconds=_number(
            table.get("window_seconds", PINNED_WINDOW_SECONDS),
            f"{path}.window_seconds",
            minimum=0.0,
            exclusive_minimum=True,
        ),
        batch_windows=_integer(table.get("batch_windows", 8), f"{path}.batch_windows", minimum=1),
    )


def _parse_data(table: Mapping[str, Any], origin: str) -> DataConfig:
    path = f"{origin}.data"
    data_root = table.get("data_root")
    return DataConfig(
        index=_string(_require(table, "index", path), f"{path}.index"),
        data_root=None if data_root is None else _string(data_root, f"{path}.data_root"),
        split=_string(table.get("split", "train"), f"{path}.split"),
        groups_per_step=_integer(
            table.get("groups_per_step", 1), f"{path}.groups_per_step", minimum=1
        ),
        centers_per_item=_integer(
            table.get("centers_per_item", 4), f"{path}.centers_per_item", minimum=1
        ),
        min_center_gap=_integer(
            table.get("min_center_gap", 5), f"{path}.min_center_gap", minimum=0
        ),
        valid_only=_boolean(table.get("valid_only", True), f"{path}.valid_only"),
        activity_threshold=_number(
            table.get("activity_threshold", 0.5),
            f"{path}.activity_threshold",
            minimum=0.0,
            maximum=1.0,
        ),
        verify_digests=_boolean(table.get("verify_digests", True), f"{path}.verify_digests"),
    )


def _parse_head(table: Mapping[str, Any], origin: str) -> HeadConfig:
    path = f"{origin}.head"
    return HeadConfig(
        slots=_integer(table.get("slots", 8), f"{path}.slots", minimum=1),
        d_model=_integer(table.get("d_model", 128), f"{path}.d_model", minimum=1),
        num_heads=_integer(table.get("num_heads", 4), f"{path}.num_heads", minimum=1),
        dropout=_number(table.get("dropout", 0.0), f"{path}.dropout", minimum=0.0, maximum=1.0),
        time_frequencies=_integer(
            table.get("time_frequencies", 6), f"{path}.time_frequencies", minimum=1
        ),
        base_frequency_hz=_number(
            table.get("base_frequency_hz", 1.0),
            f"{path}.base_frequency_hz",
            minimum=0.0,
            exclusive_minimum=True,
        ),
        ffn_multiplier=_integer(
            table.get("ffn_multiplier", 2), f"{path}.ffn_multiplier", minimum=1
        ),
    )


def _parse_loss(table: Mapping[str, Any], origin: str) -> LossConfig:
    path = f"{origin}.loss"
    return LossConfig(
        w_activity=_number(table.get("w_activity", 1.0), f"{path}.w_activity", minimum=0.0),
        w_empty_slots=_number(
            table.get("w_empty_slots", 0.5), f"{path}.w_empty_slots", minimum=0.0
        ),
        w_positive=_number(table.get("w_positive", 1.0), f"{path}.w_positive", minimum=0.0),
        w_negative=_number(table.get("w_negative", 1.0), f"{path}.w_negative", minimum=0.0),
        identity_activity_threshold=_number(
            table.get("identity_activity_threshold", 0.5),
            f"{path}.identity_activity_threshold",
            minimum=0.0,
            maximum=1.0,
        ),
        negative_margin=_number(
            table.get("negative_margin", 0.25), f"{path}.negative_margin"
        ),
        max_optimal_assignments=_integer(
            table.get("max_optimal_assignments", 64),
            f"{path}.max_optimal_assignments",
            minimum=1,
        ),
    )


def _parse_optim(table: Mapping[str, Any], origin: str) -> OptimConfig:
    path = f"{origin}.optim"
    if "grad_clip_norm" not in table:
        grad_clip_value: Any = 1.0
    else:
        raw = table["grad_clip_norm"]
        grad_clip_value = None if raw is None or raw == "none" else raw
    return OptimConfig(
        lr=_number(table.get("lr", 1e-3), f"{path}.lr", minimum=0.0, exclusive_minimum=True),
        weight_decay=_number(
            table.get("weight_decay", 0.0), f"{path}.weight_decay", minimum=0.0
        ),
        steps=_integer(table.get("steps", 60), f"{path}.steps", minimum=1),
        grad_clip_norm=(
            None
            if grad_clip_value is None
            else _number(
                grad_clip_value, f"{path}.grad_clip_norm", minimum=0.0, exclusive_minimum=True
            )
        ),
    )


def _parse_eval(table: Mapping[str, Any], origin: str) -> EvalConfig:
    path = f"{origin}.eval"
    splits_raw = table.get("splits", ["val"])
    if not isinstance(splits_raw, list) or any(not isinstance(item, str) for item in splits_raw):
        raise TrainingError(f"{path}.splits: expected an array of strings")
    birth = table.get("birth_threshold")
    return EvalConfig(
        enabled=_boolean(table.get("enabled", True), f"{path}.enabled"),
        splits=tuple(splits_raw),
        threshold=_number(table.get("threshold", 0.5), f"{path}.threshold", minimum=0.0, maximum=1.0),
        max_songs=_integer(table.get("max_songs", 4), f"{path}.max_songs", minimum=1),
        max_windows_per_forward=_integer(
            table.get("max_windows_per_forward", 16),
            f"{path}.max_windows_per_forward",
            minimum=1,
        ),
        match_threshold=_number(
            table.get("match_threshold", 0.7), f"{path}.match_threshold", minimum=-1.0, maximum=1.0
        ),
        retention_seconds=_number(
            table.get("retention_seconds", 1.0), f"{path}.retention_seconds", minimum=0.0
        ),
        prototype_alpha=_number(
            table.get("prototype_alpha", 0.9), f"{path}.prototype_alpha", minimum=0.0, maximum=1.0
        ),
        birth_threshold=(
            None
            if birth is None
            else _number(birth, f"{path}.birth_threshold", minimum=0.0, maximum=1.0)
        ),
        max_exact_slots=_integer(
            table.get("max_exact_slots", 16), f"{path}.max_exact_slots", minimum=1
        ),
        baselines=_boolean(table.get("baselines", True), f"{path}.baselines"),
    )


def default_smoke_config(
    *,
    data_root: str | Path,
    index_path: str | Path,
    out_dir: str | Path,
    steps: int,
    seed: int = 20260929,
    eval_splits: tuple[str, ...] = ("val",),
) -> TrainConfig:
    """Deterministic CPU fake-encoder smoke config used by ``train.py smoke``."""

    payload = {
        "run": {"name": "fake-smoke", "out_dir": str(out_dir), "seed": seed, "device": "cpu"},
        "encoder": {"mode": "fake", "feature_dim": 16, "fake_seed": seed, "batch_windows": 8},
        "data": {
            "index": str(index_path),
            "data_root": str(data_root),
            "split": "train",
            "groups_per_step": 1,
            "centers_per_item": 4,
            "min_center_gap": 5,
        },
        "head": {"slots": 8},
        "loss": {},
        "optim": {"steps": steps, "lr": 3e-3},
        "checkpoint": {"interval_steps": max(1, steps)},
        "eval": {"splits": list(eval_splits), "max_songs": 4},
    }
    return TrainConfig.from_dict(payload, origin="smoke-defaults")


def config_field_names() -> tuple[str, ...]:
    """Field names of the top-level config, for diagnostics."""

    return tuple(field.name for field in fields(TrainConfig))


__all__ = [
    "CheckpointConfig",
    "DataConfig",
    "ENCODER_MODES",
    "EvalConfig",
    "EncoderConfig",
    "HeadConfig",
    "LossConfig",
    "OptimConfig",
    "PINNED_ATTENTION",
    "PINNED_DTYPE",
    "PINNED_EXTRACTION",
    "PINNED_WINDOW_SECONDS",
    "RunConfig",
    "SPLITS",
    "TrainConfig",
    "canonical_json",
    "config_field_names",
    "default_smoke_config",
    "sha256_canonical",
]
