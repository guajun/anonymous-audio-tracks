"""Pure-NumPy helpers of ``scripts/probe_aut.py`` (no torch, no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from aat.encoders.aut import _real_region_after_resample

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "probe_aut.py"
_spec = importlib.util.spec_from_file_location("probe_aut_under_test", _SCRIPT)
probe_aut = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(probe_aut)


def test_synthetic_signal_is_deterministic_and_finite():
    first = probe_aut.synthetic_signal(1.5, seed=11)
    second = probe_aut.synthetic_signal(1.5, seed=11)
    other = probe_aut.synthetic_signal(1.5, seed=12)
    assert first.dtype == np.float32
    assert first.shape == (24000,)
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_synthetic_signal_respects_requested_rate():
    signal = probe_aut.synthetic_signal(0.5, rate=44100)
    assert signal.shape == (22050,)


def test_edge_interior_split_uses_first_and_last_tokens():
    count = 26
    diffs = [1.0] * 8 + [0.5] * 10 + [2.0] * 8
    cosines = [0.9] * 8 + [0.99] * 10 + [0.7] * 8
    summary = probe_aut._edge_interior(
        {"per_token_max_abs_diff": diffs, "per_token_cosine": cosines}
    )
    assert summary["edge_tokens"] == 16
    assert summary["interior_tokens"] == 10
    assert summary["edge_mean_abs_diff"] == pytest.approx(1.5)
    assert summary["interior_mean_abs_diff"] == pytest.approx(0.5)
    assert summary["edge_min_cosine"] == pytest.approx(0.7)
    assert summary["interior_min_cosine"] == pytest.approx(0.99)


def test_edge_interior_split_handles_short_windows():
    summary = probe_aut._edge_interior(
        {"per_token_max_abs_diff": [1.0, 2.0, 3.0, 4.0], "per_token_cosine": [1.0] * 4}
    )
    assert summary["edge_tokens"] + summary["interior_tokens"] == 4
    assert summary["edge_mean_abs_diff"] is not None


def test_median_helper_matches_numpy():
    assert probe_aut._median([3.0, 1.0, 2.0]) == pytest.approx(2.0)


def test_real_region_after_resample_maps_contiguous_span():
    valid = np.zeros(44100, dtype=bool)
    valid[11025:33075] = True  # 0.25 s .. 0.75 s at 44.1 kHz
    region = _real_region_after_resample(valid, 44100, 16000, 16000)
    assert region is not None
    start, stop = region
    assert start == pytest.approx(4000, abs=1)
    assert stop == pytest.approx(12000, abs=1)
    assert _real_region_after_resample(np.zeros(100, dtype=bool), 16000, 16000, 100) is None
