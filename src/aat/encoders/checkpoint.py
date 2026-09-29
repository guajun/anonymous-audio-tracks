"""Pinned Qwen3-Omni checkpoint facts and strict encoder-only weight selection.

The M2 probe must never instantiate the full Omni model.  This module keeps
the audited facts (model id, revision, tensor prefix, checkpoint layout) and
the pure-python plumbing that decides *which shard files are needed* and
verifies the filtered state dict *before* any tensor touches the encoder:

* ``model.safetensors.index.json`` is parsed and every ``thinker.audio_tower.*``
  tensor is mapped to the shard that stores it.
* The shard plan is reported with tensor counts and byte totals so the probe
  can check disk space before downloading, and can report down/loaded bytes
  separately from the encoder-only bytes it feeds into the model.
* ``check_state_dict_coverage`` refuses missing/unexpected keys explicitly;
  the loader never relies on ``strict=False``.

Nothing here imports torch, transformers or safetensors.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import EncoderCheckpointError

#: Official checkpoint id.  The audio tower is the only component this issue loads.
AUT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
#: Checkpoint revision audited for M2 (also pinned in docs/AUT_PROBE.md).
AUT_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
#: Nesting of the audio encoder config inside the Omni ``config.json``.
AUT_CONFIG_AUDIO_PATH = ("thinker_config", "audio_config")
#: Prefix of every audio-tower tensor in the checkpoint shards.
AUT_TENSOR_PREFIX = "thinker.audio_tower."
#: Index file and preprocessor config of the official checkpoint.
AUT_INDEX_FILENAME = "model.safetensors.index.json"
AUT_PREPROCESSOR_FILENAME = "preprocessor_config.json"

# --------------------------------------------------------------------------- #
# Audited layout facts (measured by reading the remote safetensors headers by
# HTTP range before any shard was downloaded; see docs/AUT_PROBE.md).
# --------------------------------------------------------------------------- #
#: Total bytes of all 15 checkpoint shards according to the index metadata.
AUT_CHECKPOINT_TOTAL_BYTES = 70_519_637_090
#: Number of shards in the official checkpoint.
AUT_CHECKPOINT_SHARD_COUNT = 15
#: Audio-tower tensors in the checkpoint (all live in shard 1 of 15).
AUT_TENSOR_COUNT = 525
#: Parameter count of the standalone ``Qwen3OmniMoeAudioEncoder`` built from
#: the official ``thinker_config.audio_config`` (all 525 tensors are BF16).
AUT_PARAM_COUNT = 647_927_168
#: BF16 payload bytes of those 525 tensors (525 * tensor bytes, audited).
AUT_ENCODER_TENSOR_BYTES = 1_295_854_336
#: Tensor payload of shard ``model-00001-of-00015.safetensors`` (4.654 GiB),
#: of which ``AUT_ENCODER_TENSOR_BYTES`` (1.207 GiB) belong to the audio tower.
AUT_FIRST_SHARD_TENSOR_BYTES = 4_997_712_096

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EncoderCheckpointError(message)


@dataclass(frozen=True)
class CheckpointProvenance:
    """Identity of the locally available checkpoint files.

    ``source`` is one of ``explicit`` (caller supplied id/revision),
    ``hf-metadata`` (read from ``hf_hub_download`` local-dir metadata) or
    ``unverified`` (local files without revision metadata).  ``unverified``
    must never be silently replaced by the audited default revision: a feature
    document may only claim the pinned revision when the bytes can be tied to
    it.
    """

    model_id: str = ""
    revision: str = ""
    source: str = "unverified"


_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def discover_checkpoint_provenance(model_dir: Path | str) -> CheckpointProvenance:
    """Read ``hf_hub_download`` local-dir metadata to tie files to a revision.

    ``huggingface_hub`` writes ``.cache/huggingface/download/<name>.metadata``
    next to files downloaded with ``local_dir=...``.  Older hub versions write
    a three-line text file (commit hash, etag, mtime), newer ones a JSON
    object; both are accepted.  Multiple distinct commit hashes are rejected
    because the directory would mix revisions.
    """

    directory = Path(model_dir) / ".cache" / "huggingface" / "download"
    revisions: set[str] = set()
    if directory.is_dir():
        for metadata_path in sorted(directory.glob("*.metadata")):
            try:
                text = metadata_path.read_text(encoding="utf-8").strip()
            except OSError as error:  # pragma: no cover - unusual filesystem error
                raise EncoderCheckpointError(
                    f"cannot read checkpoint metadata {metadata_path}: {error}"
                ) from error
            candidate: Any = None
            try:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    candidate = payload.get("commit_hash")
            except json.JSONDecodeError:
                lines = text.splitlines()
                candidate = lines[0].strip() if lines else None
            if isinstance(candidate, str) and _SHA40.match(candidate):
                revisions.add(candidate)
    if len(revisions) > 1:
        raise EncoderCheckpointError(
            f"local checkpoint {directory.parent.parent.parent} mixes revisions: "
            f"{sorted(revisions)}"
        )
    if revisions:
        return CheckpointProvenance(revision=revisions.pop(), source="hf-metadata")
    return CheckpointProvenance(source="unverified")


def merge_checkpoint_provenance(
    discovered: CheckpointProvenance,
    *,
    model_id: str | None = None,
    revision: str | None = None,
) -> CheckpointProvenance:
    """Combine explicit caller identity with local download metadata."""

    if model_id is None and revision is None:
        return discovered
    if (
        revision is not None
        and discovered.revision
        and revision != discovered.revision
    ):
        raise EncoderCheckpointError(
            f"requested revision {revision!r} but the local download metadata "
            f"records {discovered.revision!r}; refusing to mislabel the weights"
        )
    return CheckpointProvenance(
        model_id=model_id if model_id else discovered.model_id,
        revision=revision if revision else discovered.revision,
        source="explicit",
    )


@dataclass(frozen=True)
class CheckpointIndex:
    """Parsed ``model.safetensors.index.json``."""

    weight_map: Mapping[str, str]
    metadata: Mapping[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path | str) -> "CheckpointIndex":
        index_path = Path(path)
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except OSError as error:  # pragma: no cover - exercised by callers
            raise EncoderCheckpointError(f"cannot read index {index_path}: {error}") from error
        except json.JSONDecodeError as error:
            raise EncoderCheckpointError(f"index {index_path} is not valid JSON: {error}") from error
        _require(isinstance(payload, dict), f"index {index_path}: expected a JSON object")
        weight_map = payload.get("weight_map")
        _require(
            isinstance(weight_map, dict) and weight_map,
            f"index {index_path}: missing non-empty 'weight_map'",
        )
        for key, shard in weight_map.items():
            _require(isinstance(key, str) and key, f"index: bad tensor key {key!r}")
            _require(
                isinstance(shard, str) and shard.endswith(".safetensors"),
                f"index: tensor {key!r} maps to non-safetensors shard {shard!r}",
            )
        metadata = payload.get("metadata", {})
        _require(isinstance(metadata, dict), f"index {index_path}: 'metadata' must be an object")
        return cls(weight_map=dict(weight_map), metadata=dict(metadata), path=index_path)


@dataclass(frozen=True)
class ShardPlan:
    """Which shards contain the audio tower and how much they carry."""

    prefix: str
    tensor_names: tuple[str, ...]
    shards: tuple[str, ...]
    tensor_counts: Mapping[str, int]

    @property
    def total_tensors(self) -> int:
        return len(self.tensor_names)

    @property
    def shard_count(self) -> int:
        return len(self.shards)


def plan_encoder_shards(
    index: CheckpointIndex,
    prefix: str = AUT_TENSOR_PREFIX,
) -> ShardPlan:
    """Map every encoder tensor to its shard and return the minimal shard set."""

    selected = {name: shard for name, shard in index.weight_map.items() if name.startswith(prefix)}
    _require(
        selected,
        f"index {index.path}: no tensor starts with prefix {prefix!r}; refusing to load an "
        "empty or mis-specified encoder state dict",
    )
    counts: dict[str, int] = {}
    for shard in selected.values():
        counts[shard] = counts.get(shard, 0) + 1
    return ShardPlan(
        prefix=prefix,
        tensor_names=tuple(sorted(selected)),
        shards=tuple(sorted(counts)),
        tensor_counts=dict(counts),
    )


def normalize_encoder_keys(
    state: Mapping[str, Any],
    prefix: str = AUT_TENSOR_PREFIX,
) -> dict[str, Any]:
    """Strip the checkpoint prefix and normalize keys for the standalone class."""

    normalized: dict[str, Any] = {}
    for key, value in state.items():
        _require(
            key.startswith(prefix),
            f"state dict key {key!r} does not start with the audited prefix {prefix!r}",
        )
        stripped = key[len(prefix) :]
        _require(stripped, f"state dict key {key!r} has an empty suffix")
        _require(
            stripped not in normalized,
            f"duplicate encoder tensor after prefix normalization: {stripped!r}",
        )
        normalized[stripped] = value
    return normalized


def check_state_dict_coverage(
    expected_keys: Sequence[str],
    provided_keys: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(missing, unexpected)`` and raise if either is non-empty.

    This is the explicit replacement for ``strict=False``: the probe fails
    loudly if a tensor name changed, disappeared or leaked in from another
    component of the Omni checkpoint.
    """

    expected = set(expected_keys)
    provided = set(provided_keys)
    missing = tuple(sorted(expected - provided))
    unexpected = tuple(sorted(provided - expected))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={list(missing)}")
        if unexpected:
            details.append(f"unexpected={list(unexpected)}")
        raise EncoderCheckpointError(
            "encoder state dict does not match the standalone audio encoder: " + "; ".join(details)
        )
    return missing, unexpected


def tensor_nbytes(shape: Sequence[int], dtype: str) -> int:
    """Byte size of one tensor from its shape and a safetensors dtype string."""

    _require(dtype in _DTYPE_BYTES, f"unsupported safetensors dtype {dtype!r}")
    count = 1
    for dimension in shape:
        _require(
            isinstance(dimension, int) and dimension >= 0,
            f"tensor shape {shape!r} is not a sequence of non-negative ints",
        )
        count *= int(dimension)
    return count * _DTYPE_BYTES[dtype]


def header_encoder_bytes(
    header: Mapping[str, Any],
    prefix: str = AUT_TENSOR_PREFIX,
) -> dict[str, Any]:
    """Byte/dtype accounting for encoder tensors in a safetensors header."""

    selected = {key: value for key, value in header.items() if key.startswith(prefix)}
    total = 0
    dtypes: dict[str, int] = {}
    for key, entry in selected.items():
        _require(isinstance(entry, dict), f"safetensors header entry {key!r} is not an object")
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        _require(
            isinstance(shape, list) and isinstance(dtype, str),
            f"safetensors header entry {key!r} lacks shape/dtype",
        )
        total += tensor_nbytes(shape, dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
    return {"tensor_count": len(selected), "bytes": total, "dtypes": dtypes}


def check_disk_space(path: Path | str, required_bytes: int, *, safety_factor: float = 1.2) -> dict[str, int]:
    """Fail before downloading when the target filesystem cannot hold the shards."""

    if required_bytes < 0:
        raise EncoderCheckpointError(f"required_bytes: must be >= 0, got {required_bytes}")
    usage = shutil.disk_usage(str(path))
    margin = int(required_bytes * float(safety_factor))
    if usage.free < margin:
        raise EncoderCheckpointError(
            f"not enough disk space at {path}: need >= {margin} bytes "
            f"({required_bytes} x {safety_factor}), free {usage.free} bytes"
        )
    return {"required_bytes": required_bytes, "required_with_margin": margin, "free_bytes": usage.free}


def audio_config_from_checkpoint(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract ``thinker_config.audio_config`` and reject a wrong config shape."""

    node: Any = config
    for step in AUT_CONFIG_AUDIO_PATH:
        _require(
            isinstance(node, Mapping) and step in node,
            f"checkpoint config: missing {'.'.join(AUT_CONFIG_AUDIO_PATH)}",
        )
        node = node[step]
    _require(
        isinstance(node, Mapping),
        f"checkpoint config: {'.'.join(AUT_CONFIG_AUDIO_PATH)} must be an object",
    )
    return node
