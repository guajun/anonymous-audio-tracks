"""Fake AuT encoder: grid/mask/time behaviour, windows vs slices (no torch).

These tests are the CPU-CI stand-ins for the real probe: they use only NumPy
and the deterministic local fake, so no network, checkpoint or torch is
required.  The real model numbers live in ``docs/AUT_PROBE.md``.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aat.encoders import (
    AUT_SAMPLE_RATE,
    FAKE_FEATURE_DIM,
    FakeAutEncoder,
    aut_output_length,
    compare_aligned_tokens,
)
from aat.windowing import extract_windows_at_times


def _signal(seconds: float, rate: int = AUT_SAMPLE_RATE, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(seconds * rate))) / rate
    tone = 0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 440.0 * t + 0.4)
    noise = 0.05 * rng.standard_normal(t.size)
    return (tone + noise).astype(np.float32)


@pytest.fixture()
def encoder() -> FakeAutEncoder:
    return FakeAutEncoder()


def test_extract_returns_finite_features_mask_and_absolute_times(encoder):
    audio = _signal(2.5)
    item = encoder.extract(audio, AUT_SAMPLE_RATE, origin_seconds=0.0)
    mel_len = audio.size // 160
    assert item.token_count == aut_output_length(mel_len) == 33
    assert item.feature_dim == FAKE_FEATURE_DIM
    assert item.features.shape == (33, FAKE_FEATURE_DIM)
    assert item.valid.shape == (33,)
    assert item.frame_times.shape == (33,)
    assert item.valid.all()
    assert np.all(np.isfinite(item.features))
    assert np.all(np.isfinite(item.frame_times))
    assert item.frame_times[0] == 0.0
    assert item.frame_times[-1] == pytest.approx(2.48)


@pytest.mark.parametrize("seconds", [0.5, 1.0, 2.0, 2.5, 3.13])
def test_different_lengths_and_non_integral_chunks(encoder, seconds):
    audio = _signal(seconds)
    item = encoder.extract(audio)
    mel_len = audio.size // 160
    assert item.token_count == aut_output_length(mel_len)
    assert np.all(np.diff(item.frame_times) > 0.0)
    assert np.all(np.isfinite(item.features))


def test_non_zero_origin_shifts_times_without_touching_features(encoder):
    audio = _signal(1.0)
    base = encoder.extract(audio, origin_seconds=0.0)
    shifted = encoder.extract(audio, origin_seconds=120.25)
    assert np.allclose(shifted.frame_times, base.frame_times + 120.25)
    np.testing.assert_array_equal(shifted.features, base.features)


def test_left_and_right_padding_are_masked(encoder):
    track = _signal(1.0)
    window_seconds = 2.0
    # Window centered at 0.0 on the track: left half is zero padding.
    windows, valid = extract_windows_at_times(
        track, np.array([0.0, 0.5, 1.0]), AUT_SAMPLE_RATE, window_seconds, origin_seconds=0.0
    )
    starts = np.array([0.0, 0.5, 1.0]) - window_seconds / 2.0
    batch = encoder.extract_windows(
        windows, AUT_SAMPLE_RATE, window_start_seconds=starts, valid_samples=valid
    )
    assert batch.features.shape[1] == 26  # 2 s = 200 mel frames -> 26 tokens
    # t=0.0 window: first tokens touch left padding -> invalid; last tok valid.
    assert not batch.valid[0][:4].any()
    assert batch.valid[0][-2:].all()
    # t=1.0 window: right half padded -> trailing tokens invalid.
    assert batch.valid[2][-4:].sum() < 4
    assert batch.valid[2][:8].all()
    # t=0.5 window covers [-0.5, 1.5]: both ends are padded, the middle is real.
    assert not batch.valid[1][:8].any()
    assert batch.valid[1][8:19].all()
    assert not batch.valid[1][19:].any()
    assert np.all(np.isfinite(batch.features))


def test_aligned_independent_windows_reproduce_full_slice_for_local_encoder(encoder):
    full_audio = _signal(10.0)
    reference = encoder.extract(full_audio)
    assert reference.token_count == 130
    for start in (1.0, 3.0, 5.0):
        window = full_audio[int(start * AUT_SAMPLE_RATE) : int((start + 2.0) * AUT_SAMPLE_RATE)]
        candidate = encoder.extract(window, origin_seconds=start)
        comparison = compare_aligned_tokens(reference, candidate)
        assert comparison["matched_tokens"] == 26
        assert comparison["unmatched_candidate_tokens"] == 0
        # Even aligned windows differ at the two boundary frames: the window's
        # own STFT frames 0/1 and 199 are built with zero-padding at the window
        # edge, unlike the interior frames of the full track.  This is exactly
        # the boundary effect the real probe quantifies.
        assert comparison["max_abs_diff"] > 0.0
        interior = np.zeros(26, dtype=bool)
        interior[2:24] = True
        interior_comparison = compare_aligned_tokens(reference, candidate.subset(interior))
        assert interior_comparison["matched_tokens"] == 22
        # The fake is strictly local and the 1 s chunk grid is aligned, so
        # interior tokens are bit-identical to the full-slice tokens here; the
        # real encoder's attention context makes this only approximate and is
        # measured by the probe instead.
        assert interior_comparison["max_abs_diff"] == 0.0


def test_misaligned_window_changes_grid_and_features(encoder):
    full_audio = _signal(10.0)
    reference = encoder.extract(full_audio)
    start = 1.5
    window = full_audio[int(start * AUT_SAMPLE_RATE) : int((start + 2.0) * AUT_SAMPLE_RATE)]
    candidate = encoder.extract(window, origin_seconds=start)
    exact = compare_aligned_tokens(reference, candidate)
    assert exact["matched_tokens"] == 0  # no token time coincides with the full grid
    loose = compare_aligned_tokens(reference, candidate, atol_seconds=0.02)
    assert loose["matched_tokens"] > 0
    assert loose["max_abs_diff"] > 0.0


def test_batch_of_windows_equals_individual_extractions(encoder):
    full_audio = _signal(8.0)
    starts = np.array([1.0, 2.0, 4.0])
    windows = np.stack(
        [
            full_audio[int(start * AUT_SAMPLE_RATE) : int((start + 2.0) * AUT_SAMPLE_RATE)]
            for start in starts
        ]
    )
    batch = encoder.extract_windows(windows, AUT_SAMPLE_RATE, window_start_seconds=starts)
    for index, start in enumerate(starts):
        single = encoder.extract(windows[index], origin_seconds=float(start))
        np.testing.assert_array_equal(batch.features[index], single.features)
        np.testing.assert_array_equal(batch.valid[index], single.valid)
        np.testing.assert_array_equal(batch.frame_times[index], single.frame_times)


def test_profile_feature_document_is_protocol_valid(encoder, tmp_path):
    audio = _signal(1.0)
    item = encoder.extract(audio)
    document = item.to_feature_data(sample_id="fake-0001")
    path, arrays_path = document.save(tmp_path)
    assert path.name == "feature.json"
    assert arrays_path.name == "feature.npz"
    loaded = type(document).load(tmp_path)
    assert loaded.feature_dim == FAKE_FEATURE_DIM
    np.testing.assert_array_equal(loaded.features, item.features)
    np.testing.assert_array_equal(loaded.frame_times, item.frame_times)


def test_short_audio_below_stft_minimum_is_rejected(encoder):
    with pytest.raises(Exception, match="minimum"):
        encoder.extract(np.zeros(120, dtype=np.float32))


def test_importing_encoders_does_not_import_torch_or_transformers():
    """The fake path must stay usable on the base CPU environment."""

    repo_root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; import aat.encoders; import aat.encoders.fake; "
        "assert 'torch' not in sys.modules, sorted(k for k in sys.modules if 'torch' in k); "
        "assert 'transformers' not in sys.modules; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env={**__import__("os").environ, "PYTHONPATH": str(repo_root / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_fake_window_token_count_uses_the_shared_grid(encoder):
    window_seconds = 2.0
    audio = _signal(window_seconds)
    windows = np.stack([audio, audio])
    starts = np.array([0.0, 1.0])
    batch = encoder.extract_windows(windows, window_start_seconds=starts)
    expected = aut_output_length(int(window_seconds * AUT_SAMPLE_RATE) // 160)
    assert math.isfinite(float(batch.features.mean()))
    assert batch.token_count == expected
