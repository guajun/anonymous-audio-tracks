"""Feature containers and the window/slice comparison helper (no torch)."""

from __future__ import annotations

import numpy as np
import pytest

from aat.encoders import (
    AutFeatures,
    AutWindowBatch,
    EncoderError,
    EncoderInputError,
    aut_token_grid,
    compare_aligned_tokens,
)


def _features(grid, dim=3, value=1.0):
    return np.full((grid.token_count, dim), value, dtype=np.float32)


def test_build_derives_times_mask_and_preserves_shapes():
    grid = aut_token_grid(150)
    features = _features(grid, dim=5)
    valid = np.ones(grid.token_count, dtype=bool)
    item = AutFeatures.build(
        features=features,
        valid=valid,
        grid=grid,
        buffer_origin_seconds=2.0,
        model_id="unit",
        revision="rev",
    )
    assert item.feature_dim == 5
    assert item.token_count == grid.token_count
    assert item.frame_times.shape == (grid.token_count,)
    assert item.frame_times[0] == pytest.approx(2.0)
    assert item.hop_seconds == 0.08


def test_build_rejects_shape_and_finiteness_violations():
    grid = aut_token_grid(100)
    with pytest.raises(EncoderError, match="2-D"):
        AutFeatures.build(
            features=np.zeros(grid.token_count),
            valid=np.ones(grid.token_count, dtype=bool),
            grid=grid,
            buffer_origin_seconds=0.0,
        )
    bad = _features(grid)
    bad[0, 0] = np.nan
    with pytest.raises(EncoderError, match="NaN"):
        AutFeatures.build(
            features=bad,
            valid=np.ones(grid.token_count, dtype=bool),
            grid=grid,
            buffer_origin_seconds=0.0,
        )
    with pytest.raises(EncoderError, match="boolean"):
        AutFeatures.build(
            features=_features(grid),
            valid=np.ones(grid.token_count, dtype=np.int8),
            grid=grid,
            buffer_origin_seconds=0.0,
        )


def test_feature_data_projection_roundtrip(tmp_path):
    grid = aut_token_grid(100)
    preprocessing = {
        "model": {"revision": "abc123", "revision_source": "explicit"},
        "attention": {"mode": "block_diagonal", "backend": "sdpa"},
        "extraction_mode": "single_buffer",
    }
    item = AutFeatures.build(
        features=_features(grid, dim=4),
        valid=np.ones(grid.token_count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=0.0,
        layer="final",
        model_id="unit",
        revision="abc123",
        preprocessing=preprocessing,
    )
    document = item.to_feature_data(sample_id="sample-1")
    assert document.feature_dim == 4
    assert document.sample_rate == 16000
    assert document.frame_times[0] == 0.0
    assert document.preprocessing == preprocessing
    metadata_path, arrays_path = document.save(tmp_path)
    assert metadata_path.name == "feature.json"
    loaded = type(document).load(tmp_path)
    np.testing.assert_array_equal(loaded.features, item.features)
    np.testing.assert_array_equal(loaded.valid, item.valid)
    assert loaded.preprocessing == preprocessing
    assert loaded.feature_name == "aut.qwen3-omni-moe.audio_tower.final"


def test_negative_times_are_rejected_by_the_protocol_projection():
    grid = aut_token_grid(100)
    item = AutFeatures.build(
        features=_features(grid),
        valid=np.ones(grid.token_count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=-0.5,
    )
    assert item.frame_times[0] == pytest.approx(-0.5)
    with pytest.raises(EncoderInputError, match="negative"):
        item.to_feature_data()
    clipped = item.drop_before_track_origin()
    assert np.all(clipped.frame_times >= 0.0)
    assert clipped.token_count == int(np.sum(item.frame_times >= 0.0))


def test_drop_before_track_origin_matches_numpy_selection():
    grid = aut_token_grid(100)
    item = AutFeatures.build(
        features=_features(grid),
        valid=np.ones(grid.token_count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=-0.33,
    )
    clipped = item.drop_before_track_origin()
    expected_rows = int(np.sum(item.frame_times >= 0.0))
    assert clipped.token_count == expected_rows
    assert clipped.grid.mel_len == item.grid.mel_len
    assert np.all(clipped.frame_times >= 0.0)


def test_compare_aligned_tokens_reports_identity_and_difference():
    grid = aut_token_grid(200)
    count = grid.token_count
    reference = AutFeatures.build(
        features=np.ones((count, 4), dtype=np.float32),
        valid=np.ones(count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=0.0,
    )
    identical = compare_aligned_tokens(reference, reference)
    assert identical["matched_tokens"] == count
    assert identical["elementwise_mae"] == 0.0
    assert identical["mean_token_mae"] == 0.0
    assert identical["mean_token_max_abs"] == 0.0
    assert identical["min_cosine"] == pytest.approx(1.0)

    modified = np.ones((count, 4), dtype=np.float32)
    modified[5, 0] = 2.0  # one channel off by 1.0 on one token
    candidate = AutFeatures.build(
        features=modified,
        valid=np.ones(count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=0.0,
    )
    comparison = compare_aligned_tokens(reference, candidate)
    assert comparison["matched_tokens"] == count
    assert comparison["unmatched_candidate_tokens"] == 0
    assert comparison["elementwise_mae"] == pytest.approx(1.0 / (4 * count))
    assert comparison["mean_token_mae"] == pytest.approx(0.25 / count)
    assert comparison["max_token_mae"] == pytest.approx(0.25)
    assert comparison["mean_token_max_abs"] == pytest.approx(1.0 / count)
    assert comparison["max_abs_diff"] == pytest.approx(1.0)
    assert comparison["min_cosine"] < 0.99
    assert comparison["per_token_mae"][5] == pytest.approx(0.25)
    assert comparison["per_token_max_abs_diff"][5] == pytest.approx(1.0)

    # A half-hop shift moves every candidate time off the reference grid: the
    # comparison reports them as unmatched instead of forcing a bad pairing.
    shifted = AutFeatures.build(
        features=modified,
        valid=np.ones(count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=-0.01,
    )
    shifted_comparison = compare_aligned_tokens(reference, shifted)
    assert shifted_comparison["matched_tokens"] == 0
    assert shifted_comparison["unmatched_candidate_tokens"] == count
    assert shifted_comparison["elementwise_mae"] is None


def test_compare_elementwise_mae_is_not_mean_token_max():
    """Hand-computable regression for the metric definition.

    Reference ``[0, 0, 0, 0]`` vs candidate ``[1, 0, 0, 0]``: elementwise MAE
    is 0.25, while the mean of per-token max errors is 1.0.  The two must not be
    conflated.
    """

    grid = aut_token_grid(100)
    count = grid.token_count
    reference = AutFeatures.build(
        features=np.zeros((count, 4), dtype=np.float32),
        valid=np.ones(count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=0.0,
    )
    candidate_features = np.zeros((count, 4), dtype=np.float32)
    candidate_features[0, 0] = 1.0
    candidate = AutFeatures.build(
        features=candidate_features,
        valid=np.ones(count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=0.0,
    )
    comparison = compare_aligned_tokens(reference, candidate)
    assert comparison["elementwise_mae"] == pytest.approx(1.0 / (4 * count))
    assert comparison["mean_token_mae"] == pytest.approx(0.25 / count)
    assert comparison["max_token_mae"] == pytest.approx(0.25)
    assert comparison["mean_token_max_abs"] == pytest.approx(1.0 / count)
    assert comparison["max_abs_diff"] == pytest.approx(1.0)


def test_compare_handles_empty_reference_and_invalid_tokens():
    grid = aut_token_grid(100)
    empty_grid = aut_token_grid(0)
    empty = AutFeatures.build(
        features=np.zeros((0, 4), dtype=np.float32),
        valid=np.zeros(0, dtype=bool),
        grid=empty_grid,
        buffer_origin_seconds=0.0,
    )
    candidate = AutFeatures.build(
        features=np.ones((grid.token_count, 4), dtype=np.float32),
        valid=np.ones(grid.token_count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=0.0,
    )
    empty_comparison = compare_aligned_tokens(empty, candidate)
    assert empty_comparison["matched_tokens"] == 0
    assert empty_comparison["unmatched_candidate_tokens"] == grid.token_count
    assert empty_comparison["elementwise_mae"] is None

    reference = AutFeatures.build(
        features=np.zeros((grid.token_count, 4), dtype=np.float32),
        valid=np.ones(grid.token_count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=0.0,
    )
    invalid = np.ones(grid.token_count, dtype=bool)
    invalid[3] = False
    invalid_candidate = AutFeatures.build(
        features=np.ones((grid.token_count, 4), dtype=np.float32),
        valid=invalid,
        grid=grid,
        buffer_origin_seconds=0.0,
    )
    all_tokens = compare_aligned_tokens(reference, invalid_candidate)
    assert all_tokens["matched_tokens"] == grid.token_count
    assert all_tokens["invalid_matched_tokens"] == 1
    assert all_tokens["masked_out_tokens"] == 0
    valid_only = compare_aligned_tokens(reference, invalid_candidate, valid_only=True)
    assert valid_only["matched_tokens"] == grid.token_count - 1
    assert valid_only["masked_out_tokens"] == 1
    assert valid_only["invalid_matched_tokens"] == 0


def test_window_batch_validates_and_preserves_provenance():
    grid = aut_token_grid(100)
    preprocessing = {"attention": {"mode": "block_diagonal"}, "extraction_mode": "independent_windows"}
    batch = AutWindowBatch(
        features=np.ones((2, grid.token_count, 6), dtype=np.float32),
        valid=np.ones((2, grid.token_count), dtype=bool),
        frame_times=np.stack([grid.frame_times(1.0), grid.frame_times(3.0)]),
        window_start_seconds=np.array([1.0, 3.0]),
        grid=grid,
        model_id="unit",
        revision="rev",
        preprocessing=preprocessing,
    )
    assert batch.window_count == 2
    assert batch.token_count == grid.token_count
    assert batch.feature_dim == 6
    window = batch.window(1)
    assert window.frame_times[0] == pytest.approx(3.0)
    assert window.preprocessing == preprocessing
    assert window.model_id == "unit"
    assert window.revision == "rev"
    with pytest.raises(EncoderError, match="token axis"):
        AutWindowBatch(
            features=np.ones((1, grid.token_count + 1, 6), dtype=np.float32),
            valid=np.ones((1, grid.token_count + 1), dtype=bool),
            frame_times=np.stack([grid.frame_times(0.0) for _ in range(1)]),
            window_start_seconds=np.array([0.0]),
            grid=grid,
        )


def test_compare_rejects_dimension_mismatch():
    grid = aut_token_grid(50)
    a = AutFeatures.build(
        features=np.zeros((grid.token_count, 2), dtype=np.float32),
        valid=np.ones(grid.token_count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=0.0,
    )
    b = AutFeatures.build(
        features=np.zeros((grid.token_count, 3), dtype=np.float32),
        valid=np.ones(grid.token_count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=0.0,
    )
    with pytest.raises(EncoderError, match="dims differ"):
        compare_aligned_tokens(a, b)
