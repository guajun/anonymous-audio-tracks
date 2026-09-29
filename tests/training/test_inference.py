"""Checkpoint -> inference interface for downstream consumers (issue #8)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aat.contracts import ActivityData, PredictionData
from aat.labels import read_wav
from aat.tracking import build_prediction_data
from aat.training.checkpoint import load_checkpoint
from aat.training.errors import CheckpointError
from aat.training.inference import build_head_from_model_config, load_head_from_checkpoint
from aat.training.trainer import train_from_config
from tests.training.support import smoke_config

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory, smoke_corpus):
    out = tmp_path_factory.mktemp("inference-run")
    config = smoke_config(smoke_corpus, out, steps=2, interval_steps=2)
    summary = train_from_config(config)
    return smoke_corpus, Path(summary.checkpoint_path)


def _sample_audio(corpus, sample_id="smoke-val-01"):
    entry_root = Path(corpus["data_root"]) / sample_id
    reference = ActivityData.load(entry_root)
    wav = read_wav(entry_root / "mix.wav")
    audio = wav.samples.mean(axis=1) if wav.channels > 1 else wav.samples[:, 0]
    centers = np.asarray(reference.center_times)[np.asarray(reference.valid)][:8]
    return audio, wav.sample_rate, centers


def test_predict_at_times_satisfies_prediction_contract(trained_checkpoint):
    corpus, checkpoint = trained_checkpoint
    inference = load_head_from_checkpoint(checkpoint, device="cpu")
    audio, sample_rate, centers = _sample_audio(corpus)
    prediction = inference.predict_at_times(audio, centers, sample_rate=sample_rate)
    assert isinstance(prediction, PredictionData)
    assert prediction.slots == 8
    assert prediction.embeddings.shape == (len(centers), 8, 128)
    norms = np.linalg.norm(prediction.embeddings, axis=2)
    valid = prediction.slot_valid
    assert np.allclose(norms[valid], 1.0, atol=1e-3)
    assert np.all(prediction.activity[~valid] == 0.0)
    assert inference.describe()["encoder_mode"] == "fake"


def test_predictor_for_centers_matches_predict_at_times(trained_checkpoint):
    corpus, checkpoint = trained_checkpoint
    inference = load_head_from_checkpoint(checkpoint, device="cpu")
    audio, sample_rate, centers = _sample_audio(corpus)
    direct = inference.predict_at_times(audio, centers, sample_rate=sample_rate)
    callback = inference.predictor_for_centers(
        centers, sample_rate=sample_rate, audio_duration_seconds=len(audio) / sample_rate
    )
    via_callback = build_prediction_data(
        audio,
        centers,
        sample_rate=sample_rate,
        window_seconds=2.0,
        predictor=callback,
        slots=8,
    )
    assert np.allclose(via_callback.embeddings, direct.embeddings, atol=1e-6)
    assert np.allclose(via_callback.activity, direct.activity, atol=1e-6)
    assert np.array_equal(via_callback.slot_valid, direct.slot_valid)


def test_boundary_padded_centers_produce_canonical_invalid_slots(trained_checkpoint):
    _, checkpoint = trained_checkpoint
    inference = load_head_from_checkpoint(checkpoint, device="cpu")
    audio = np.ones(32000, dtype=np.float32)

    start_edge = inference.predict_at_times(audio, [0.0, 1.0], sample_rate=16000)
    assert start_edge.center_valid.tolist() == [False, True]
    assert not start_edge.slot_valid[0].any()
    assert np.all(start_edge.embeddings[0] == 0.0)
    assert np.all(start_edge.activity[0] == 0.0)
    assert start_edge.slot_valid[1].any()

    end_edge = inference.predict_at_times(audio, [1.0, 2.0], sample_rate=16000)
    assert end_edge.center_valid.tolist() == [True, False]
    assert not end_edge.slot_valid[1].any()
    assert np.all(end_edge.activity[1] == 0.0)


def test_boundary_masks_with_nonzero_origin_and_callback_consistency(trained_checkpoint):
    _, checkpoint = trained_checkpoint
    inference = load_head_from_checkpoint(checkpoint, device="cpu")
    audio = np.ones(48000, dtype=np.float32)  # 3 s at 16 kHz, absolute origin 10 s
    centers = np.array([10.5, 11.0, 12.0, 13.0])
    direct = inference.predict_at_times(
        audio, centers, sample_rate=16000, origin_seconds=10.0
    )
    assert direct.center_valid.tolist() == [False, True, True, False]
    assert not direct.slot_valid[0].any()
    assert not direct.slot_valid[3].any()

    callback = inference.predictor_for_centers(
        centers,
        sample_rate=16000,
        origin_seconds=10.0,
        audio_duration_seconds=3.0,
    )
    built = build_prediction_data(
        audio,
        centers,
        sample_rate=16000,
        window_seconds=2.0,
        predictor=callback,
        slots=8,
        origin_seconds=10.0,
    )
    assert built.center_valid.tolist() == [False, True, True, False]
    assert np.array_equal(built.slot_valid, direct.slot_valid)
    assert np.allclose(built.activity, direct.activity, atol=1e-6)
    # The tracking callback re-normalizes in float32, so allow float-rounding
    # differences while requiring the boundary rows to stay canonical zeros.
    assert np.allclose(built.embeddings, direct.embeddings, atol=1e-6)
    assert np.array_equal(built.embeddings[[0, 3]], np.zeros_like(built.embeddings[[0, 3]]))


def test_predict_windows_validates_pinned_length_rate_and_alignment(trained_checkpoint):
    _, checkpoint = trained_checkpoint
    inference = load_head_from_checkpoint(checkpoint, device="cpu")
    from aat.training.errors import TrainingError

    with pytest.raises(TrainingError, match="exactly 32000 samples"):
        inference.predict_windows(np.ones((1, 16000), dtype=np.float32), sample_rate=16000)
    with pytest.raises(TrainingError, match="sample_rate"):
        inference.predict_windows(np.ones((1, 32000), dtype=np.float32), sample_rate=0)
    with pytest.raises(TrainingError, match="sample_rate"):
        inference.predict_windows(np.ones((1, 32000), dtype=np.float32), sample_rate=16000.0)
    with pytest.raises(TrainingError, match="inconsistent"):
        inference.predict_windows(
            np.ones((1, 32000), dtype=np.float32),
            sample_rate=16000,
            window_start_seconds=np.array([0.0]),
            center_times=np.array([100.0]),
        )
    with pytest.raises(TrainingError, match="one entry per window"):
        inference.predict_windows(
            np.ones((1, 32000), dtype=np.float32),
            sample_rate=16000,
            center_times=np.array([0.0, 1.0]),
        )
    # The extraction-convention starts stay accepted (half-up sample rounding).
    from aat.training.encoding import window_start_seconds

    centers = np.array([4.0, 5.0])
    starts = window_start_seconds(centers, sample_rate=16000, window_seconds=2.0, origin_seconds=0.0)
    inference.predict_windows(
        np.ones((2, 32000), dtype=np.float32),
        sample_rate=16000,
        window_start_seconds=starts,
        center_times=centers,
    )


def test_inference_restores_mode_and_does_not_consume_rng(trained_checkpoint):
    _, checkpoint = trained_checkpoint
    from aat.training.config import TrainConfig
    from aat.training.encoding import build_encoder
    from aat.training.inference import HeadInference, build_head_from_model_config
    from aat.training.checkpoint import load_checkpoint as read_checkpoint

    payload = read_checkpoint(checkpoint)
    config = TrainConfig.from_dict(payload["config"], origin="dropout-test")

    model = build_head_from_model_config(payload["model_config"])
    model.load_state_dict(payload["model_state"])
    # Force a non-trivial dropout probability so train mode would consume RNG.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.5
    model.train()
    inference = HeadInference(
        config=config,
        model=model,
        encoder=build_encoder(config.encoder),
        device=torch.device("cpu"),
        checkpoint_path=str(checkpoint),
    )
    window = np.ones((1, 32000), dtype=np.float32)
    state = torch.get_rng_state()
    first = inference.predict_windows(window, sample_rate=16000)
    second = inference.predict_windows(window, sample_rate=16000)
    assert torch.equal(state, torch.get_rng_state()), "inference must not consume torch RNG"
    assert np.array_equal(first.activity, second.activity)
    assert np.array_equal(first.embeddings, second.embeddings)
    assert model.training is True, "predict_windows must restore the original mode"


def test_encoder_identity_mismatch_is_rejected(trained_checkpoint, tmp_path: Path):
    _, checkpoint = trained_checkpoint
    payload = load_checkpoint(checkpoint)
    payload["encoder"]["identity"]["fake_seed"] = 424242
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(CheckpointError, match="encoder identity mismatch"):
        load_head_from_checkpoint(tampered, device="cpu")


def test_model_config_mismatch_is_rejected(trained_checkpoint, tmp_path: Path):
    _, checkpoint = trained_checkpoint
    payload = load_checkpoint(checkpoint)
    payload["model_config"]["slots"] = 4
    tampered = tmp_path / "tampered-model.pt"
    torch.save(payload, tampered)
    with pytest.raises(CheckpointError):
        load_head_from_checkpoint(tampered, device="cpu")


def test_build_head_from_model_config_validates_missing_keys():
    with pytest.raises(CheckpointError, match="missing keys"):
        build_head_from_model_config({"feature_dim": 16, "slots": 8})
