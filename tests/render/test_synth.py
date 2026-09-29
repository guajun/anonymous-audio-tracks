"""Deterministic self-generated sample buffers (no DawDreamer, no assets)."""

from __future__ import annotations

import numpy as np
import pytest

from aat.render.errors import RenderConfigError
from aat.render.synth import SAMPLE_TYPES, render_sample, sample_duration_seconds


@pytest.mark.parametrize("sample_type", SAMPLE_TYPES)
def test_every_sample_type_is_finite_and_bounded(sample_type: str) -> None:
    data = render_sample(sample_type, None, 44100, seed=123)
    assert data.dtype == np.float32
    assert data.ndim == 2
    assert data.shape[0] in (1, 2)
    assert data.shape[1] > 0
    assert np.all(np.isfinite(data))
    assert np.max(np.abs(data)) > 0.0
    assert np.max(np.abs(data)) <= 1.0


def test_same_seed_is_byte_identical() -> None:
    first = render_sample("pluck", {"freq_hz": 110.0}, 44100, seed=7)
    second = render_sample("pluck", {"freq_hz": 110.0}, 44100, seed=7)
    assert first.tobytes() == second.tobytes()


def test_noise_based_types_change_with_seed() -> None:
    for sample_type in ("snare", "hat", "pluck"):
        first = render_sample(sample_type, None, 44100, seed=1)
        second = render_sample(sample_type, None, 44100, seed=2)
        assert not np.array_equal(first, second), sample_type


def test_duration_matches_response() -> None:
    for sample_type in SAMPLE_TYPES:
        duration = sample_duration_seconds(sample_type, None)
        data = render_sample(sample_type, None, 8000, seed=3)
        assert data.shape[1] == round(duration * 8000)


def test_unknown_type_and_parameter_are_rejected() -> None:
    with pytest.raises(RenderConfigError):
        render_sample("piano", None, 44100, seed=0)
    with pytest.raises(RenderConfigError):
        render_sample("kick", {"bogus": 1.0}, 44100, seed=0)


def test_pad_envelope_fits_duration() -> None:
    with pytest.raises(RenderConfigError):
        render_sample(
            "pad",
            {"duration_seconds": 1.0, "attack_seconds": 0.7, "release_seconds": 0.6},
            44100,
            seed=0,
        )


def test_stereo_pad_has_two_channels() -> None:
    data = render_sample("pad", {"stereo": True}, 44100, seed=0)
    assert data.shape[0] == 2
    assert not np.array_equal(data[0], data[1])
