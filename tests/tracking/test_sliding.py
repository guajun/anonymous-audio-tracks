"""Sliding-window callback entry tests: fake model forward, full chain."""

from __future__ import annotations

import numpy as np
import pytest

from aat.tracking import (
    TrackingError,
    WindowPrediction,
    build_prediction_data,
    predict_windows,
    track_audio,
)
from aat.windowing import center_times

from . import helpers

SAMPLE_RATE = 16000


def _two_slot_energy_predictor(windows: np.ndarray):
    """Deterministic fake head: slot 0 = any energy, slot 1 = loud windows."""

    energy = np.mean(np.abs(windows), axis=1)
    batches = windows.shape[0]
    embeddings = np.zeros((batches, 2, helpers.EMBEDDING_DIM), dtype=np.float32)
    embeddings[:, 0, 0] = 1.0
    embeddings[:, 1, 1] = 1.0
    activity = np.zeros((batches, 2), dtype=np.float32)
    activity[:, 0] = np.where(energy > 0.01, 0.9, 0.0)
    activity[:, 1] = np.where(energy > 0.05, 0.9, 0.0)
    slot_valid = np.repeat((energy > 0.01)[:, None], 2, axis=1)
    return embeddings, activity, slot_valid


def test_fake_callback_full_chain_produces_valid_protocol_output():
    samples = np.concatenate(
        [
            np.full(SAMPLE_RATE // 2, 0.2, dtype=np.float32),
            np.zeros(SAMPLE_RATE // 2, dtype=np.float32),
        ]
    )

    predictions, trajectory = track_audio(
        samples,
        sample_rate=SAMPLE_RATE,
        predictor=_two_slot_energy_predictor,
        window_seconds=0.02,
        hop_seconds=0.02,
        slots=2,
        provenance=helpers.provenance("sliding-test"),
    )

    assert predictions.center_times.shape == (51,)
    assert predictions.embeddings.shape == (51, 2, 128)
    assert predictions.slot_valid[1].all()
    assert not predictions.slot_valid[-1].any()
    # Protocol validation is part of construction and of this round trip.
    predictions.validate()

    assert len(trajectory.tracks) == 2
    assert trajectory.provenance.run_id == "sliding-test"
    first, second = trajectory.tracks
    first_active = sum(value >= 0.5 for value in first.activity)
    second_active = sum(value >= 0.5 for value in second.activity)
    assert first_active == second_active == 25
    assert first.activity[-1] == pytest.approx(0.0)
    assert trajectory.audio.duration_seconds == pytest.approx(1.0)
    assert trajectory.to_json_dict()["kind"] == "trajectory"


def test_predict_windows_normalizes_callback_embeddings():
    def predictor(windows):
        batches = windows.shape[0]
        embeddings = np.zeros((batches, 1, helpers.EMBEDDING_DIM), dtype=np.float32)
        embeddings[:, 0, 0] = 3.0
        activity = np.full((batches, 1), 0.9, dtype=np.float32)
        return WindowPrediction(embeddings=embeddings, activity=activity)

    batch = predict_windows(np.zeros((2, 16), dtype=np.float32), predictor, slots=1)

    assert batch.slot_valid.all()
    np.testing.assert_allclose(np.linalg.norm(batch.embeddings, axis=2), 1.0, rtol=1e-5)


def test_zero_embedding_becomes_an_invalid_slot():
    def predictor(windows):
        batches = windows.shape[0]
        embeddings = np.zeros((batches, 2, helpers.EMBEDDING_DIM), dtype=np.float32)
        activity = np.full((batches, 2), 0.9, dtype=np.float32)
        return embeddings, activity

    batch = predict_windows(np.zeros((2, 16), dtype=np.float32), predictor, slots=2)

    assert not batch.slot_valid.any()
    assert np.all(batch.activity == 0.0)
    assert np.all(batch.embeddings == 0.0)


def test_explicit_valid_slot_with_zero_embedding_is_rejected():
    def predictor(windows):
        batches = windows.shape[0]
        embeddings = np.zeros((batches, 1, helpers.EMBEDDING_DIM), dtype=np.float32)
        activity = np.zeros((batches, 1), dtype=np.float32)
        slot_valid = np.ones((batches, 1), dtype=bool)
        return embeddings, activity, slot_valid

    with pytest.raises(TrackingError):
        predict_windows(np.zeros((1, 16), dtype=np.float32), predictor, slots=1)


def test_clip_origin_keeps_original_track_absolute_times():
    samples = np.full(SAMPLE_RATE // 5, 0.2, dtype=np.float32)

    def always_active(windows):
        batches = windows.shape[0]
        embeddings = np.zeros((batches, 1, helpers.EMBEDDING_DIM), dtype=np.float32)
        embeddings[:, 0, 0] = 1.0
        activity = np.full((batches, 1), 0.9, dtype=np.float32)
        return embeddings, activity

    predictions, trajectory = track_audio(
        samples,
        sample_rate=SAMPLE_RATE,
        predictor=always_active,
        window_seconds=0.02,
        hop_seconds=0.02,
        slots=1,
        provenance=helpers.provenance(),
        track_start_seconds=12.0,
    )

    assert predictions.center_times[0] == pytest.approx(12.0)
    assert trajectory.audio.track_start_seconds == pytest.approx(12.0)
    assert trajectory.tracks[0].center_times[0] == pytest.approx(12.02)
    assert trajectory.tracks[0].center_times[-1] == pytest.approx(12.18)


def test_empty_audio_never_calls_the_predictor():
    def forbidden(windows):  # pragma: no cover - must not be called
        raise AssertionError("predictor must not run on empty audio")

    predictions, trajectory = track_audio(
        np.zeros(0, dtype=np.float32),
        sample_rate=SAMPLE_RATE,
        predictor=forbidden,
        window_seconds=0.02,
        hop_seconds=0.02,
        slots=1,
        provenance=helpers.provenance(),
    )

    assert predictions.center_times.size == 0
    assert trajectory.audio.duration_seconds == 0.0
    assert trajectory.tracks == ()


def test_build_prediction_data_records_center_validity():
    samples = np.full(SAMPLE_RATE, 0.2, dtype=np.float32)
    times = center_times(1.0, 0.02, origin_seconds=0.0)

    prediction = build_prediction_data(
        samples,
        times,
        sample_rate=SAMPLE_RATE,
        window_seconds=0.02,
        predictor=_two_slot_energy_predictor,
        slots=2,
        provenance=helpers.provenance(),
    )

    assert not prediction.center_valid[0]
    assert prediction.center_valid[1]
    assert not prediction.center_valid[-1]
