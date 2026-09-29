"""Hysteresis, release-tail and segmentation tests."""

from __future__ import annotations

import numpy as np
import pytest

from aat.labels import (
    LabelError,
    active_segments,
    hysteresis_states,
    label_envelope,
    release_extension,
)


def test_schmitt_trigger_holds_inside_band():
    levels = np.array([-100.0, -20.0, -31.0, -31.0, -34.0, -31.0, -100.0])
    states = hysteresis_states(levels, -30.0, -33.0)
    assert states.tolist() == [False, True, True, True, False, False, False]


def test_release_extends_only_falling_edges():
    levels = np.array([-100.0, -20.0, -31.0, -31.0, -34.0, -31.0, -100.0])
    labels = label_envelope(levels, -30.0, -33.0, release_hold_samples=2)
    assert labels.state.tolist() == [False, True, True, True, False, False, False]
    assert labels.released.tolist() == [False, False, False, False, True, True, False]
    assert labels.active.tolist() == [False, True, True, True, True, True, False]


def test_no_chatter_below_on_threshold():
    levels = np.array([-31.0, -32.0, -31.5, -32.5])
    assert not label_envelope(levels, -30.0, -33.0, 0).active.any()


def test_segments_are_half_open_runs():
    mask = np.array([False, False, True, True, True, False, True, True, False])
    assert active_segments(mask) == [(2, 5), (6, 8)]
    assert active_segments(np.empty(0, dtype=bool)) == []


def test_empty_and_invalid_inputs():
    assert hysteresis_states(np.empty(0), -30.0, -33.0).size == 0
    with pytest.raises(LabelError):
        hysteresis_states(np.array([0.0]), -30.0, -20.0)
    with pytest.raises(LabelError):
        release_extension(np.array([True]), -1)


def test_release_hold_zero_adds_no_tail():
    states = np.array([True, False, False])
    released, active = release_extension(states, 0)
    assert not released.any()
    np.testing.assert_array_equal(active, states)
