"""Center-local activation tests on programmatic waveforms.

Covered cases: pulse, slow attack, sustain, digital silence, repeated notes,
hysteresis bridging, release/tail after note-off and threshold configurability.
None of these are delegated to control events.
"""

from __future__ import annotations

import numpy as np

from aat.labels import LabelConfig

from . import signals
from .support import activity_at, label_one, state_at

BINARY = LabelConfig(probability_mode="binary")


def test_pulse_is_center_local_not_a_two_second_window_or():
    signal = signals.pulse_buffer(3.0, 1.0)
    result = label_one(signal, 3.0)

    assert activity_at(result, 1.0) > 0.9
    assert activity_at(result, 1.02) > 0.9
    # A 2-second model window centered at 1.5 contains the pulse at 1.0, but a
    # center-local label must stay silent there.
    assert activity_at(result, 1.5) == 0.0
    assert activity_at(result, 0.5) == 0.0
    active = result.activity.activity[:, 0] >= 0.5
    assert np.count_nonzero(active) <= 6


def test_slow_attack_does_not_force_an_onset_label():
    signal = signals.db_ramp_tone(
        2.0, 0.5, attack_seconds=0.4, hold_seconds=0.6, db_start=-70.0, db_end=-9.0309
    )
    result = label_one(signal, 2.0)

    assert activity_at(result, 0.5) < 0.1  # onset is still ~70 dB down
    assert activity_at(result, 0.55) < 0.1
    assert activity_at(result, 0.75) > 0.9  # after the audible part of the ramp
    assert activity_at(result, 1.0) > 0.9


def test_sustained_region_is_active_and_silence_is_not():
    signal = signals.pulse_buffer(3.0, 0.5, pulse_seconds=1.0, freq=440.0)
    result = label_one(signal, 3.0)

    assert activity_at(result, 0.7) > 0.9
    assert activity_at(result, 1.2) > 0.9
    assert activity_at(result, 0.2) == 0.0
    assert activity_at(result, 2.0) == 0.0
    source = result.summary["sources"][0]
    assert source["silent"] is False
    assert source["n_segments"] == 1


def test_digital_silence_is_all_zero_and_flagged():
    result = label_one(signals.silence(2.0), 2.0)

    assert np.all(result.activity.activity == 0.0)
    source = result.summary["sources"][0]
    assert source["silent"] is True
    assert source["low_snr"] is False
    assert source["active_centers"] == 0
    assert any("inactive" in warning for warning in result.summary["warnings"])
    assert result.activity.valid.dtype == np.bool_


def test_repeated_notes_stay_separate_segments():
    duration = 3.0
    signal = signals.silence(duration)
    for onset in (0.5, 0.75, 1.0, 1.25):
        signals.add_at(signal, signals.tone(0.05, freq=1000.0), onset)
    config = LabelConfig(model_window_seconds=0.1)
    result = label_one(signal, duration, config=config)

    for onset in (0.5, 0.75, 1.0, 1.25):
        assert activity_at(result, onset) > 0.9
    for gap in (0.65, 0.9, 1.15, 1.4):
        assert activity_at(result, gap) == 0.0
    assert result.summary["sources"][0]["n_segments"] == 4


def test_tail_remains_active_after_note_off():
    signal = signals.sustain_then_tail(3.0, 0.5, 0.7, tau_seconds=0.15)
    soft = label_one(signal, 3.0)
    binary = label_one(signal, 3.0, config=BINARY)

    # The acoustic tail is still sounding well after the nominal note-off.
    assert activity_at(soft, 0.9) > 0.9
    assert activity_at(soft, 1.0) > 0.9
    assert 0.1 < activity_at(soft, 1.35) < 0.8  # decay through the threshold band
    assert activity_at(soft, 1.5) < 0.01
    assert state_at(binary, 1.3) is True
    assert state_at(binary, 1.5) is False
    # Centers inside the hysteresis band are reported as ambiguous, not hidden.
    assert soft.summary["sources"][0]["ambiguous_fraction"] > 0.0


def test_release_seconds_extend_the_active_tail():
    signal = signals.sustain_then_tail(2.0, 0.5, 0.7, tau_seconds=0.15)
    short = label_one(signal, 2.0, config=LabelConfig(probability_mode="binary", release_seconds=0.05))
    long = label_one(signal, 2.0, config=LabelConfig(probability_mode="binary", release_seconds=0.3))

    assert state_at(short, 1.45) is False
    assert state_at(long, 1.45) is True
    assert state_at(long, 1.7) is False


def test_hysteresis_bridges_a_short_dip_below_the_off_threshold():
    signal = signals.constant_tone_with_dip(
        2.0, first_on_seconds=0.5, first_off_seconds=0.7, second_on_seconds=0.74
    )
    bridged = label_one(signal, 2.0, config=LabelConfig(probability_mode="binary", release_seconds=0.1))
    unbridged = label_one(signal, 2.0, config=LabelConfig(probability_mode="binary", release_seconds=0.0))

    assert state_at(bridged, 0.6) is True
    assert state_at(bridged, 0.72) is True
    assert state_at(bridged, 0.8) is True
    assert state_at(unbridged, 0.72) is False


def test_thresholds_are_configurable_and_recorded():
    signal = signals.pulse_buffer(2.0, 1.0, pulse_seconds=0.5, amplitude=0.005)  # ~ -49 dBFS
    strict = label_one(signal, 2.0, config=LabelConfig(abs_on_db=-45.0))
    sensitive = label_one(signal, 2.0, config=LabelConfig(abs_on_db=-60.0))

    assert activity_at(strict, 1.0) == 0.0
    assert activity_at(sensitive, 1.0) > 0.9
    assert strict.summary["config"]["abs_on_db"] == -45.0
    assert sensitive.summary["config"]["abs_on_db"] == -60.0
    assert strict.summary["config_sha256"] != sensitive.summary["config_sha256"]


def test_low_snr_stems_are_flagged_but_still_labelled():
    # ~ -53 dBFS pulse with an explicit -58 dBFS floor: active, but only ~5 dB
    # above the floor, so the summary must call the decision ambiguous.
    signal = signals.pulse_buffer(3.0, 1.0, pulse_seconds=0.5, amplitude=0.003166)
    config = LabelConfig(
        noise_floor_db=-58.0,
        abs_on_db=-60.0,
        rel_on_db=2.0,
        min_snr_db=10.0,
    )
    result = label_one(signal, 3.0, config=config)

    assert activity_at(result, 1.1) > 0.9
    source = result.summary["sources"][0]
    assert source["silent"] is False
    assert source["low_snr"] is True
    assert any("ambiguous" in warning for warning in result.summary["warnings"])


def test_binary_and_soft_modes_agree_on_clear_regions():
    signal = signals.pulse_buffer(2.0, 1.0)
    soft = label_one(signal, 2.0)
    binary = label_one(signal, 2.0, config=BINARY)

    assert activity_at(soft, 1.0) == activity_at(binary, 1.0) == 1.0
    assert activity_at(soft, 0.2) == activity_at(binary, 0.2) == 0.0


def test_anti_phase_stereo_keeps_its_activity():
    mono = signals.pulse_buffer(2.0, 1.0)
    stereo = np.stack([mono, -mono], axis=1)
    from .support import label_one as _label

    result = _label(stereo, 2.0)
    assert activity_at(result, 1.0) > 0.9
