"""Token-grid maths of the AuT encoder (pure NumPy, no torch)."""

from __future__ import annotations

import numpy as np
import pytest

from aat.encoders import (
    AUT_CHUNK_MEL_FRAMES,
    AUT_MIN_AUDIO_SAMPLES,
    AUT_TOKENS_PER_FULL_CHUNK,
    EncoderInputError,
    aut_output_length,
    aut_token_grid,
    token_valid_mask,
)


def _upstream_formula(mel_len: int) -> int:
    """Independent transcription of ``_get_feat_extract_output_lengths``."""

    leave = mel_len % 100
    feat = (leave - 1) // 2 + 1
    return (mel_len // 100) * 13 + ((feat - 1) // 2 + 1 - 1) // 2 + 1


def test_output_length_matches_upstream_formula_on_all_short_lengths():
    for mel_len in range(0, 501):
        assert aut_output_length(mel_len) == _upstream_formula(mel_len)


@pytest.mark.parametrize(
    ("mel_len", "tokens"),
    [
        (0, 0),
        (1, 1),
        (12, 2),
        (13, 2),
        (25, 4),
        (50, 7),
        (80, 10),
        (99, 13),
        (100, 13),
        (101, 14),
        (150, 20),
        (200, 26),
        (250, 33),
        (300, 39),
    ],
)
def test_output_length_known_values(mel_len, tokens):
    assert aut_output_length(mel_len) == tokens


def test_full_chunk_yields_thirteen_tokens_not_twelve_point_five():
    # 100 mel frames = 1 s, but padding-1 stride-2 convolutions give 13 tokens,
    # so a "12.5 Hz / 80 ms" claim would be wrong at chunk granularity.
    assert aut_output_length(AUT_CHUNK_MEL_FRAMES) == AUT_TOKENS_PER_FULL_CHUNK == 13


def test_grid_is_consistent_with_formula_and_time_axis():
    grid = aut_token_grid(250)
    assert grid.token_count == 33
    assert np.all(np.diff(grid.mel_centers) > 0)
    assert grid.mel_centers[0] == 0
    assert grid.mel_centers[12] == 96
    assert grid.mel_centers[13] == 100
    assert np.all(grid.field_stop > grid.field_start)
    # tokens stay inside their own chunk: no context leaks across chunks
    assert np.all(grid.field_start >= grid.chunk_index * 100)
    assert np.all(grid.field_stop <= np.minimum((grid.chunk_index + 1) * 100, 250))
    assert np.all(grid.field_start <= grid.mel_centers)
    assert np.all(grid.mel_centers < grid.field_stop)


def test_frame_times_have_40ms_chunk_boundary_gap():
    grid = aut_token_grid(200)
    times = grid.frame_times(0.0)
    interior = np.diff(times)
    assert np.allclose(interior[:12], 0.08)
    # 0.96 s -> 1.00 s: the chunk boundary gap is 40 ms, not 80 ms.
    assert times[12] == pytest.approx(0.96)
    assert times[13] == pytest.approx(1.00)
    assert times[13] - times[12] == pytest.approx(0.04)


def test_frame_times_shift_with_origin():
    grid = aut_token_grid(100)
    times = grid.frame_times(123.5)
    assert times[0] == pytest.approx(123.5)
    assert times[-1] == pytest.approx(123.5 + 0.96)


def test_full_buffer_has_no_invalid_tokens():
    grid = aut_token_grid(250)
    mask = token_valid_mask(grid, 250 * 160)
    assert mask.all()
    assert mask.dtype == np.bool_


def test_padding_invalidates_only_tokens_touching_the_padded_span():
    n_samples = 32000  # 2 s
    grid = aut_token_grid(200)
    # Real audio only in the middle second: very conservative mask.
    mask = token_valid_mask(grid, n_samples, real_start=12000, real_stop=20000)
    assert not mask[0]  # token at 0.0 s starts in the first padded region
    assert not mask[-1]  # token at 1.96 s reaches into the second padded region
    assert np.flatnonzero(mask).tolist() == [11, 12, 13, 14, 15]
    # A token whose field is fully inside a wider real span stays valid.
    shifted = token_valid_mask(grid, n_samples, real_start=3000, real_stop=25000)
    assert shifted[16]


def test_sample_support_uses_mel_field_not_chunk_padding():
    grid = aut_token_grid(100)
    start, stop = grid.sample_support(16000 * 10)
    # token 0: field frames [0, 7] -> samples [-200, 1320) clipped to [0, 1320)
    assert start[0] == 0
    assert stop[0] == 7 * 160 + 200
    # last full token: centre frame 96, field [89, 103] clipped to chunk end 99
    assert grid.field_stop[-1] == 100
    assert stop[-1] == 99 * 160 + 200


def test_grid_rejects_bad_mel_lengths():
    with pytest.raises(EncoderInputError):
        aut_output_length(-1)
    with pytest.raises(EncoderInputError):
        aut_output_length(3.5)  # type: ignore[arg-type]
    with pytest.raises(EncoderInputError):
        token_valid_mask(aut_token_grid(10), 1600, real_start=1700, real_stop=1600)


def test_minimum_audio_samples_follows_stft_padding_requirement():
    # torch.stft(center=True) pads n_fft // 2 and requires padding < length.
    assert AUT_MIN_AUDIO_SAMPLES == 201
