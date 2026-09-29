"""Real-class integration test on a tiny synthetic checkpoint (no network).

Runs only when torch + transformers + safetensors are installed (the pinned ML
extra); the base CPU CI environment skips it and exercises the fake instead.
The tiny checkpoint uses the *real* upstream classes and the same strict
filtering/loading path as the probe, so the loading contract is covered
without downloading the 30B model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

pytestmark = pytest.mark.ml

from aat.encoders import AUT_SAMPLE_RATE, EncoderCheckpointError, EncoderInputError
from aat.encoders.aut import AUT_TENSOR_PREFIX, AutEncoder
from aat.encoders.grid import aut_output_length

TINY_AUDIO_CONFIG = {
    "model_type": "qwen3_omni_moe_audio_encoder",
    "d_model": 32,
    "encoder_attention_heads": 4,
    "encoder_ffn_dim": 64,
    "encoder_layers": 2,
    "output_dim": 16,
    "num_mel_bins": 128,
    "n_window": 50,
    "n_window_infer": 800,
    "conv_chunksize": 500,
    "downsample_hidden_size": 8,
    "max_source_positions": 1500,
    "activation_function": "gelu",
    "scale_embedding": False,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
}

PREPROCESSOR = {
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "sampling_rate": 16000,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "dither": 0.0,
    "padding_value": 0.0,
}


def _build_tiny_checkpoint(tmp_path, *, drop_key=None, extra_key=None, audio_overrides=None):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
    )

    audio_config = dict(TINY_AUDIO_CONFIG)
    if audio_overrides:
        audio_config.update(audio_overrides)
    torch.manual_seed(0)
    config = Qwen3OmniMoeAudioEncoderConfig(**audio_config)
    model = Qwen3OmniMoeAudioEncoder(config)
    state = {
        f"{AUT_TENSOR_PREFIX}{key}": value.detach().contiguous()
        for key, value in model.state_dict().items()
    }
    index = {key: "model-00001-of-00001.safetensors" for key in state}
    if drop_key is not None:
        index.pop(drop_key)
        state.pop(drop_key)
    if extra_key is not None:
        state[extra_key] = torch.zeros(2)
        index[extra_key] = "model-00001-of-00001.safetensors"

    shard = tmp_path / "model-00001-of-00001.safetensors"
    safetensors_torch.save_file(state, str(shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": index}), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3_omni_moe", "thinker_config": {"audio_config": audio_config}}
        ),
        encoding="utf-8",
    )
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(PREPROCESSOR), encoding="utf-8")
    return model


def _signal(seconds: float, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(seconds * AUT_SAMPLE_RATE))) / AUT_SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 330.0 * t) + 0.05 * rng.standard_normal(t.size)).astype(
        np.float32
    )


@pytest.fixture(scope="module")
def tiny_encoder(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny-aut")
    model = _build_tiny_checkpoint(directory)
    encoder = AutEncoder.from_checkpoint(directory, device="cpu", dtype="float32")
    return encoder, model


def test_load_report_proves_encoder_only_loading(tiny_encoder):
    encoder, model = tiny_encoder
    report = encoder.load_report
    assert report.class_name == "Qwen3OmniMoeAudioEncoder"
    assert report.tensor_count == len(model.state_dict())
    assert report.param_count == sum(p.numel() for p in model.parameters())
    assert report.device == "cpu"
    assert report.dtype == "float32"
    assert report.mel_config["feature_size"] == 128
    # A locally built random checkpoint has no download metadata; it must not
    # attest the audited default revision.
    assert report.revision == ""
    assert report.model_id == ""
    assert report.revision_source == "unverified"
    names = encoder.module_class_names()
    assert "Qwen3OmniMoeAudioEncoder" in names
    for forbidden in ("Qwen3OmniMoeForConditionalGeneration", "Qwen3OmniMoeTalkerForConditionalGeneration"):
        assert forbidden not in names


def test_provenance_is_explicit_or_read_from_hf_metadata(tmp_path):
    directory = tmp_path / "prov"
    directory.mkdir()
    _build_tiny_checkpoint(directory)

    encoder = AutEncoder.from_checkpoint(
        directory, device="cpu", dtype="float32", model_id="local/test", revision="rev-1"
    )
    assert encoder.load_report.model_id == "local/test"
    assert encoder.load_report.revision == "rev-1"
    assert encoder.load_report.revision_source == "explicit"
    item = encoder.extract(_signal(1.0))
    assert item.preprocessing["model"]["revision"] == "rev-1"
    assert item.preprocessing["model"]["revision_source"] == "explicit"

    metadata_dir = directory / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "config.json.metadata").write_text(
        "a" * 40 + "\n" + "b" * 64 + "\n1.0\n", encoding="utf-8"
    )
    discovered = AutEncoder.from_checkpoint(directory, device="cpu", dtype="float32")
    assert discovered.load_report.revision == "a" * 40
    assert discovered.load_report.revision_source == "hf-metadata"

    with pytest.raises(EncoderCheckpointError, match="local download metadata"):
        AutEncoder.from_checkpoint(
            directory, device="cpu", dtype="float32", revision="c" * 40
        )


def test_preprocessing_snapshot_tracks_attention_mode_and_extraction_mode(tiny_encoder):
    encoder, _ = tiny_encoder
    audio = _signal(1.0)
    masked = encoder.extract(audio)
    unmasked = encoder.with_masked_attention(False).extract(audio)
    assert masked.preprocessing["attention"]["mode"] == "block_diagonal"
    assert unmasked.preprocessing["attention"]["mode"] == "unmasked_global"
    assert masked.preprocessing["extraction_mode"] == "single_buffer"
    assert unmasked.preprocessing["attention"] != masked.preprocessing["attention"]

    starts = np.array([0.0, 2.0])
    windows = np.stack([audio, audio])
    batch = encoder.extract_windows(windows, AUT_SAMPLE_RATE, window_start_seconds=starts)
    assert batch.preprocessing["extraction_mode"] == "independent_windows"
    window = batch.window(0)
    assert window.preprocessing == batch.preprocessing
    assert window.preprocessing["attention"]["mode"] == "block_diagonal"


def test_grid_assumptions_are_validated_before_extraction(tmp_path):
    directory = tmp_path / "bad-window"
    directory.mkdir()
    _build_tiny_checkpoint(directory, audio_overrides={"n_window": 40})
    with pytest.raises(EncoderCheckpointError, match="n_window"):
        AutEncoder.from_checkpoint(directory, device="cpu", dtype="float32")


def test_unaudited_transformers_version_is_rejected(tmp_path, monkeypatch):
    import transformers

    directory = tmp_path / "unaudited-version"
    directory.mkdir()
    _build_tiny_checkpoint(directory)
    monkeypatch.setattr(transformers, "__version__", "9.9.9")
    with pytest.raises(EncoderCheckpointError, match="audited"):
        AutEncoder.from_checkpoint(directory, device="cpu", dtype="float32")


def test_attention_block_boundary_is_eight_full_chunks(tiny_encoder):
    encoder, _ = tiny_encoder
    # 8.0 s = eight 100-frame chunks = 104 tokens -> one attention block.
    encoder.extract(np.zeros(int(8.0 * AUT_SAMPLE_RATE), dtype=np.float32))
    assert encoder.last_attention_info["cu_seqlens"] == [0, 104]
    # 8.05 s has one tail-chunk token more and therefore starts a second block.
    encoder.extract(np.zeros(int(8.05 * AUT_SAMPLE_RATE), dtype=np.float32))
    assert encoder.last_attention_info["cu_seqlens"] == [0, 104, 105]
    # 16.0 s = two full blocks.
    encoder.extract(np.zeros(int(16.0 * AUT_SAMPLE_RATE), dtype=np.float32))
    assert encoder.last_attention_info["cu_seqlens"] == [0, 104, 208]


def test_real_class_extract_shapes_times_and_finite_values(tiny_encoder):
    encoder, _ = tiny_encoder
    audio = _signal(2.5)
    item = encoder.extract(audio, AUT_SAMPLE_RATE)
    assert item.token_count == aut_output_length(audio.size // 160) == 33
    assert item.feature_dim == 16
    assert np.all(np.isfinite(item.features))
    assert item.valid.all()
    assert item.frame_times[0] == 0.0
    assert item.frame_times[-1] == pytest.approx(2.48)
    document = item.to_feature_data()
    assert document.feature_dim == 16


def test_real_class_batch_equals_single_on_cpu(tiny_encoder):
    encoder, _ = tiny_encoder
    full = _signal(8.0)
    starts = np.array([1.0, 2.0, 4.0])
    windows = np.stack(
        [full[int(s * AUT_SAMPLE_RATE) : int((s + 2.0) * AUT_SAMPLE_RATE)] for s in starts]
    )
    batch = encoder.extract_windows(windows, AUT_SAMPLE_RATE, window_start_seconds=starts)
    for index, start in enumerate(starts):
        single = encoder.extract(windows[index], origin_seconds=float(start))
        np.testing.assert_allclose(batch.features[index], single.features, atol=1e-6)
        np.testing.assert_array_equal(batch.valid[index], single.valid)


def test_masked_attention_info_exposes_block_structure(tiny_encoder):
    encoder, _ = tiny_encoder
    audio = _signal(2.0)
    encoder.extract(audio)
    info = encoder.last_attention_info
    assert info["masked_attention"] is True
    assert info["mask_installed"] is True
    assert info["mask_shape"] == [1, 1, 26, 26]
    assert info["cu_seqlens"] == [0, 26]

    windows = np.stack([audio, audio, audio])
    starts = np.array([0.0, 2.0, 4.0])
    encoder.extract_windows(windows, AUT_SAMPLE_RATE, window_start_seconds=starts)
    info = encoder.last_attention_info
    assert info["cu_seqlens"] == [0, 26, 52, 78]
    assert info["mask_shape"] == [1, 1, 78, 78]

    unmasked = encoder.with_masked_attention(False)
    unmasked.extract_windows(windows, AUT_SAMPLE_RATE, window_start_seconds=starts)
    assert unmasked.last_attention_info["mask_installed"] is False


def test_sdpa_ignores_cu_seqlens_unless_the_block_mask_is_injected(tiny_encoder):
    """Documents the upstream behaviour the wrapper corrects.

    ``sdpa_attention_forward`` ignores ``cu_seq_lens_*``; without the mask that
    ``Qwen3OmniMoeAudioEncoder.forward`` never applies, sample 2's output
    depends on sample 1 (and on a long buffer, all tokens attend globally).
    """

    encoder, model = tiny_encoder
    attention = model.layers[0].self_attn
    torch.manual_seed(0)
    h1 = torch.randn(26, 32)
    h2 = torch.randn(26, 32)
    h1_alt = torch.randn(26, 32) * 5.0
    cu = torch.tensor([0, 26, 52], dtype=torch.int32)
    with torch.no_grad():
        unmasked = attention(torch.cat([h1, h2]), cu_seqlens=cu)
        unmasked_alt = attention(torch.cat([h1_alt, h2]), cu_seqlens=cu)
        mask = model._prepare_attention_mask(torch.cat([h1, h2]), cu)
        assert mask is not None
        masked = attention(torch.cat([h1, h2]), cu_seqlens=cu, attention_mask=mask)
        masked_alt = attention(torch.cat([h1_alt, h2]), cu_seqlens=cu, attention_mask=mask)
    leaked = float((unmasked[26:] - unmasked_alt[26:]).abs().max())
    contained = float((masked[26:] - masked_alt[26:]).abs().max())
    assert leaked > 1e-4
    assert contained < 1e-6


def test_real_class_window_padding_masks_edges(tiny_encoder):
    from aat.windowing import extract_windows_at_times

    encoder, _ = tiny_encoder
    track = _signal(1.0)
    windows, valid = extract_windows_at_times(
        track, np.array([0.0]), AUT_SAMPLE_RATE, 2.0, origin_seconds=0.0
    )
    batch = encoder.extract_windows(
        windows, AUT_SAMPLE_RATE, window_start_seconds=np.array([-1.0]), valid_samples=valid
    )
    assert not batch.valid[0][:4].any()
    assert batch.valid[0][-2:].all()


def test_hook_on_last_layer_matches_plain_forward(tiny_encoder):
    encoder, _ = tiny_encoder
    audio = _signal(1.0)
    plain = encoder.extract(audio)
    hooked = encoder.extract(audio, layer=1)  # two layers: index 1 is the last
    np.testing.assert_allclose(hooked.features, plain.features, atol=0.0, rtol=0.0)
    assert hooked.layer == "layers.1"


def test_hook_layer_range_is_enforced(tiny_encoder):
    encoder, _ = tiny_encoder
    audio = _signal(1.0)
    with pytest.raises(EncoderInputError, match=r"\[0, 1\]"):
        encoder.extract(audio, layer=2)
    with pytest.raises(EncoderInputError, match="integer or None"):
        encoder.extract(audio, layer=1.5)


def test_strict_loading_rejects_missing_tensor(tmp_path):
    directory = tmp_path / "missing"
    directory.mkdir()
    drop = f"{AUT_TENSOR_PREFIX}layers.0.fc1.weight"
    _build_tiny_checkpoint(directory, drop_key=drop)
    with pytest.raises(EncoderCheckpointError, match="missing"):
        AutEncoder.from_checkpoint(directory, device="cpu", dtype="float32")


def test_strict_loading_rejects_unexpected_tensor(tmp_path):
    directory = tmp_path / "unexpected"
    directory.mkdir()
    _build_tiny_checkpoint(directory, extra_key=f"{AUT_TENSOR_PREFIX}bogus.weight")
    with pytest.raises(EncoderCheckpointError, match="unexpected"):
        AutEncoder.from_checkpoint(directory, device="cpu", dtype="float32")


def test_preprocessor_validation_rejects_wrong_sample_rate(tmp_path):
    directory = tmp_path / "bad-preproc"
    directory.mkdir()
    _build_tiny_checkpoint(directory)
    payload = dict(PREPROCESSOR)
    payload["sampling_rate"] = 44100
    (directory / "preprocessor_config.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EncoderCheckpointError, match="preprocessor config"):
        AutEncoder.from_checkpoint(directory, device="cpu", dtype="float32")


def test_official_formula_port_matches_installed_transformers():
    import transformers

    # 4.57.0 and 4.57.1 contain byte-identical qwen3_omni_moe sources.
    if transformers.__version__ not in ("4.57.0", "4.57.1"):
        pytest.skip(f"pinned transformers is 4.57.1 (audited 4.57.0 too), found {transformers.__version__}")
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        _get_feat_extract_output_lengths,
    )

    lengths = torch.arange(0, 401, dtype=torch.long)
    upstream = _get_feat_extract_output_lengths(lengths).tolist()
    for mel_len, expected in zip(lengths.tolist(), upstream):
        assert aut_output_length(mel_len) == expected


def _noise(seconds: float, rate: int, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(seconds * rate))) / rate
    return (0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.05 * rng.standard_normal(t.size)).astype(
        np.float32
    )


def test_real_and_fake_share_padded_window_validity(tiny_encoder):
    from aat.encoders import FakeAutEncoder
    from aat.windowing import extract_windows_at_times

    encoder, _ = tiny_encoder
    fake = FakeAutEncoder()
    track = _signal(1.0)
    centers = np.array([0.25, 0.5, 0.75, 1.0])
    windows, valid = extract_windows_at_times(
        track, centers, AUT_SAMPLE_RATE, 2.0, origin_seconds=0.0
    )
    starts = centers - 1.0
    real_batch = encoder.extract_windows(
        windows, AUT_SAMPLE_RATE, window_start_seconds=starts, valid_samples=valid
    )
    fake_batch = fake.extract_windows(
        windows, AUT_SAMPLE_RATE, window_start_seconds=starts, valid_samples=valid
    )
    np.testing.assert_array_equal(real_batch.valid, fake_batch.valid)


def test_equal_rate_endpoint_validity_is_half_open(tiny_encoder):
    """A real span of [0, 1319) must not validate the token ending at 1320."""

    from aat.encoders import FakeAutEncoder

    encoder, _ = tiny_encoder
    fake = FakeAutEncoder()
    audio = _signal(2.0)
    valid = np.zeros(audio.size, dtype=bool)
    valid[:1319] = True
    starts = np.array([0.0])
    real_batch = encoder.extract_windows(
        audio[None, :], AUT_SAMPLE_RATE, window_start_seconds=starts, valid_samples=valid[None, :]
    )
    fake_batch = fake.extract_windows(
        audio[None, :], AUT_SAMPLE_RATE, window_start_seconds=starts, valid_samples=valid[None, :]
    )
    assert real_batch.valid[0, 0] is np.bool_(False)
    assert fake_batch.valid[0, 0] is np.bool_(False)
    np.testing.assert_array_equal(real_batch.valid, fake_batch.valid)

    # Same span but one sample longer: token 0's support [0, 1320) is now real.
    valid[1319] = True
    longer = fake.extract_windows(
        audio[None, :], AUT_SAMPLE_RATE, window_start_seconds=starts, valid_samples=valid[None, :]
    )
    assert longer.valid[0, 0]


def test_real_and_fake_validity_match_for_44k_windows(tiny_encoder):
    from aat.encoders import FakeAutEncoder
    from aat.windowing import extract_windows_at_times

    encoder, _ = tiny_encoder
    fake = FakeAutEncoder()
    track = _noise(1.0, 44100)
    centers = np.array([0.5, 1.0])
    windows, valid = extract_windows_at_times(track, centers, 44100, 2.0, origin_seconds=0.0)
    starts = centers - 1.0
    real_batch = encoder.extract_windows(
        windows, 44100, window_start_seconds=starts, valid_samples=valid
    )
    fake_batch = fake.extract_windows(
        windows, 44100, window_start_seconds=starts, valid_samples=valid
    )
    assert np.all(np.isfinite(real_batch.features))
    np.testing.assert_array_equal(real_batch.valid, fake_batch.valid)
