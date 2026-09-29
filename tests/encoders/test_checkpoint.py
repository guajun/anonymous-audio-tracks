"""Checkpoint index/selection plumbing (pure Python, no torch/no network)."""

from __future__ import annotations

import json

import pytest

from aat.encoders import (
    CheckpointIndex,
    CheckpointProvenance,
    EncoderCheckpointError,
    audio_config_from_checkpoint,
    check_disk_space,
    check_state_dict_coverage,
    discover_checkpoint_provenance,
    header_encoder_bytes,
    merge_checkpoint_provenance,
    normalize_encoder_keys,
    plan_encoder_shards,
)
from aat.encoders.checkpoint import AUT_TENSOR_PREFIX, tensor_nbytes

# Keep the audited facts referenced so documentation drift fails tests.
from aat.encoders.checkpoint import (
    AUT_CHECKPOINT_SHARD_COUNT,
    AUT_ENCODER_TENSOR_BYTES,
    AUT_PARAM_COUNT,
    AUT_TENSOR_COUNT,
)


def _index_payload() -> dict:
    weight_map = {}
    for layer in range(4):
        weight_map[f"thinker.audio_tower.layers.{layer}.self_attn.q_proj.weight"] = (
            "model-00001-of-00015.safetensors"
        )
    weight_map["thinker.audio_tower.conv2d1.weight"] = "model-00002-of-00015.safetensors"
    weight_map["thinker.text.layers.0.mlp.weight"] = "model-00003-of-00015.safetensors"
    weight_map["talker.code_predictor.weight"] = "model-00003-of-00015.safetensors"
    return {"metadata": {"total_size": 123}, "weight_map": weight_map}


def _write_index(tmp_path, payload) -> CheckpointIndex:
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return CheckpointIndex.load(path)


def test_index_roundtrip_and_validation(tmp_path):
    index = _write_index(tmp_path, _index_payload())
    assert len(index.weight_map) == 7
    assert index.metadata["total_size"] == 123

    (tmp_path / "bad.json").write_text("[]", encoding="utf-8")
    with pytest.raises(EncoderCheckpointError):
        CheckpointIndex.load(tmp_path / "bad.json")

    (tmp_path / "no-map.json").write_text('{"metadata": {}}', encoding="utf-8")
    with pytest.raises(EncoderCheckpointError, match="weight_map"):
        CheckpointIndex.load(tmp_path / "no-map.json")


def test_plan_selects_only_encoder_shards(tmp_path):
    plan = plan_encoder_shards(_write_index(tmp_path, _index_payload()))
    assert plan.total_tensors == 5
    assert plan.shards == (
        "model-00001-of-00015.safetensors",
        "model-00002-of-00015.safetensors",
    )
    assert plan.tensor_counts["model-00001-of-00015.safetensors"] == 4
    assert all(name.startswith(AUT_TENSOR_PREFIX) for name in plan.tensor_names)


def test_plan_rejects_empty_selection(tmp_path):
    payload = _index_payload()
    payload["weight_map"] = {"thinker.text.foo": "model-00003-of-00015.safetensors"}
    with pytest.raises(EncoderCheckpointError, match="no tensor starts"):
        plan_encoder_shards(_write_index(tmp_path, payload))


def test_normalize_strips_prefix_and_rejects_foreign_or_duplicate_keys():
    state = {
        f"{AUT_TENSOR_PREFIX}conv2d1.weight": 1,
        f"{AUT_TENSOR_PREFIX}conv2d1.bias": 2,
    }
    normalized = normalize_encoder_keys(state)
    assert set(normalized) == {"conv2d1.weight", "conv2d1.bias"}
    with pytest.raises(EncoderCheckpointError, match="prefix"):
        normalize_encoder_keys({"thinker.text.weight": 1})


def test_coverage_reports_missing_and_unexpected_explicitly():
    missing, unexpected = check_state_dict_coverage(["a", "b"], ["a", "b"])
    assert missing == () and unexpected == ()
    with pytest.raises(EncoderCheckpointError, match="missing"):
        check_state_dict_coverage(["a", "b"], ["a"])
    with pytest.raises(EncoderCheckpointError, match="unexpected"):
        check_state_dict_coverage(["a"], ["a", "talker.weight"])


def test_tensor_nbytes_and_header_accounting():
    assert tensor_nbytes([2, 3], "BF16") == 12
    assert tensor_nbytes([], "F32") == 4
    with pytest.raises(EncoderCheckpointError):
        tensor_nbytes([1], "COMPLEX64")
    header = {
        "__metadata__": {"format": "pt"},
        f"{AUT_TENSOR_PREFIX}conv2d1.weight": {
            "dtype": "BF16",
            "shape": [480, 1, 3, 3],
            "data_offsets": [0, 8640],
        },
        f"{AUT_TENSOR_PREFIX}conv2d1.bias": {
            "dtype": "BF16",
            "shape": [480],
            "data_offsets": [8640, 9600],
        },
        "thinker.text.foo": {"dtype": "F32", "shape": [4], "data_offsets": [9600, 9616]},
    }
    account = header_encoder_bytes(header)
    assert account["tensor_count"] == 2
    assert account["bytes"] == (480 * 9 + 480) * 2
    assert account["dtypes"] == {"BF16": 2}


def test_disk_check_fails_before_download_on_impossible_size(tmp_path):
    check_disk_space(tmp_path, 1024)
    with pytest.raises(EncoderCheckpointError, match="disk space"):
        check_disk_space(tmp_path, 10**18)


def test_audio_config_extraction_requires_nested_path():
    payload = {"thinker_config": {"audio_config": {"d_model": 1280}}}
    assert audio_config_from_checkpoint(payload)["d_model"] == 1280
    with pytest.raises(EncoderCheckpointError, match="thinker_config.audio_config"):
        audio_config_from_checkpoint({"thinker_config": {}})


def test_audited_layout_constants_are_internally_consistent():
    # 525 tensors x 2 bytes = 1,295,854,336 bytes of BF16 payload.  The value
    # below is the audited remote shard header total (docs/AUT_PROBE.md).
    assert AUT_TENSOR_COUNT == 525
    assert AUT_PARAM_COUNT == 647_927_168
    assert AUT_ENCODER_TENSOR_BYTES == 1_295_854_336
    assert AUT_CHECKPOINT_SHARD_COUNT == 15


def test_discover_provenance_reads_hf_local_dir_metadata(tmp_path):
    metadata_dir = tmp_path / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    revision = "a" * 40
    # Older hub versions: commit hash / etag / mtime lines.
    (metadata_dir / "config.json.metadata").write_text(
        revision + "\n" + "b" * 64 + "\n1.0\n", encoding="utf-8"
    )
    provenance = discover_checkpoint_provenance(tmp_path)
    assert provenance.revision == revision
    assert provenance.source == "hf-metadata"
    assert provenance.model_id == ""

    # Newer hub versions write JSON.
    (metadata_dir / "config.json.metadata").write_text(
        json.dumps({"commit_hash": "c" * 40}), encoding="utf-8"
    )
    assert discover_checkpoint_provenance(tmp_path).revision == "c" * 40

    empty = tmp_path / "empty"
    empty.mkdir()
    assert discover_checkpoint_provenance(empty).source == "unverified"


def test_discover_provenance_rejects_mixed_revisions(tmp_path):
    metadata_dir = tmp_path / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "a.metadata").write_text("a" * 40 + "\n", encoding="utf-8")
    (metadata_dir / "b.metadata").write_text("b" * 40 + "\n", encoding="utf-8")
    with pytest.raises(EncoderCheckpointError, match="mixes revisions"):
        discover_checkpoint_provenance(tmp_path)


def test_merge_provenance_prefers_explicit_and_rejects_conflicts():
    discovered = CheckpointProvenance(revision="a" * 40, source="hf-metadata")
    assert merge_checkpoint_provenance(discovered) == discovered
    merged = merge_checkpoint_provenance(discovered, model_id="local/test")
    assert merged.model_id == "local/test"
    assert merged.revision == "a" * 40
    assert merged.source == "explicit"
    explicit = merge_checkpoint_provenance(discovered, revision="a" * 40)
    assert explicit.source == "explicit"
    with pytest.raises(EncoderCheckpointError, match="local download metadata"):
        merge_checkpoint_provenance(discovered, revision="c" * 40)
