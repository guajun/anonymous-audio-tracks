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
    item = AutFeatures.build(
        features=_features(grid, dim=4),
        valid=np.ones(grid.token_count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=0.0,
        layer="final",
        model_id="unit",
        revision="rev",
    )
    document = item.to_feature_data(sample_id="sample-1")
    assert document.feature_dim == 4
    assert document.sample_rate == 16000
    assert document.frame_times[0] == 0.0
    metadata_path, arrays_path = document.save(tmp_path)
    assert metadata_path.name == "feature.json"
    loaded = type(document).load(tmp_path)
    np.testing.assert_array_equal(loaded.features, item.features)
    np.testing.assert_array_equal(loaded.valid, item.valid)


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
    reference = AutFeatures.build(
        features=np.ones((grid.token_count, 4), dtype=np.float32),
        valid=np.ones(grid.token_count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=0.0,
    )
    identical = compare_aligned_tokens(reference, reference)
    assert identical["matched_tokens"] == grid.token_count
    assert identical["max_abs_diff"] == 0.0
    assert identical["min_cosine"] == pytest.approx(1.0)

    shifted_features = np.ones((grid.token_count, 4), dtype=np.float32)
    shifted_features[5, 0] = 2.0  # changes direction, not just magnitude
    candidate = AutFeatures.build(
        features=shifted_features,
        valid=np.ones(grid.token_count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=0.0,
    )
    comparison = compare_aligned_tokens(reference, candidate)
    assert comparison["matched_tokens"] == grid.token_count
    assert comparison["unmatched_candidate_tokens"] == 0
    assert comparison["max_abs_diff"] == pytest.approx(1.0)
    assert comparison["min_cosine"] < 0.99

    # A half-hop shift moves every candidate time off the reference grid: the
    # comparison reports them as unmatched instead of forcing a bad pairing.
    shifted = AutFeatures.build(
        features=shifted_features,
        valid=np.ones(grid.token_count, dtype=bool),
        grid=grid,
        buffer_origin_seconds=-0.01,
    )
    shifted_comparison = compare_aligned_tokens(reference, shifted)
    assert shifted_comparison["matched_tokens"] == 0
    assert shifted_comparison["unmatched_candidate_tokens"] == grid.token_count


def test_window_batch_validates_and_exposes_windows():
    grid = aut_token_grid(100)
    batch = AutWindowBatch(
        features=np.ones((2, grid.token_count, 6), dtype=np.float32),
        valid=np.ones((2, grid.token_count), dtype=bool),
        frame_times=np.stack([grid.frame_times(1.0), grid.frame_times(3.0)]),
        window_start_seconds=np.array([1.0, 3.0]),
        grid=grid,
    )
    assert batch.window_count == 2
    assert batch.token_count == grid.token_count
    assert batch.feature_dim == 6
    window = batch.window(1)
    assert window.frame_times[0] == pytest.approx(3.0)
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
