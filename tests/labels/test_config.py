"""LabelConfig validation, serialisation and fingerprint stability."""

from __future__ import annotations

import pytest

from aat.labels import LABELER_VERSION, LabelConfig, LabelConfigError


def test_defaults_are_valid_and_roundtrip():
    config = LabelConfig()
    data = config.to_dict()
    assert data["labeler_version"] == LABELER_VERSION
    assert set(data) == set(LabelConfig.field_names())
    assert LabelConfig.from_dict(data) == config


def test_fingerprint_is_stable_and_threshold_sensitive():
    config = LabelConfig()
    assert config.fingerprint() == LabelConfig().fingerprint()
    assert len(config.fingerprint()) == 64
    assert LabelConfig(abs_on_db=-50.0).fingerprint() != config.fingerprint()
    assert LabelConfig(noise_floor_db=-70.0).fingerprint() != config.fingerprint()


def test_from_dict_rejects_unknown_fields_and_missing_fields():
    with pytest.raises(LabelConfigError, match="unknown"):
        LabelConfig.from_dict({**LabelConfig().to_dict(), "threshold_db": -40.0})
    with pytest.raises(LabelConfigError, match="missing"):
        LabelConfig.from_dict({"hop_seconds": 0.02})


def test_from_dict_allows_partial_overrides():
    config = LabelConfig.from_dict({"abs_on_db": -50.0}, allow_partial=True)
    assert config.abs_on_db == -50.0
    assert config.hop_seconds == LabelConfig().hop_seconds


@pytest.mark.parametrize(
    "overrides",
    [
        {"hop_seconds": 0.0},
        {"frame_seconds": 0.01, "frame_hop_seconds": 0.02},
        {"full_scale": 0.0},
        {"probability_mode": "hard"},
        {"hysteresis_db": -1.0},
        {"release_seconds": -0.1},
        {"noise_floor_percentile": 101.0},
        {"noise_floor_db": -200.0},
        {"abs_on_db": -200.0},
        {"summary_active_probability": 1.5},
        {"labeler_version": ""},
    ],
)
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(LabelConfigError):
        LabelConfig(**overrides)


def test_disable_hysteresis_and_binary_mode_are_expressible():
    config = LabelConfig(hysteresis_db=0.0, release_seconds=0.0, probability_mode="binary")
    assert config.hysteresis_db == 0.0
    assert config.probability_mode == "binary"
