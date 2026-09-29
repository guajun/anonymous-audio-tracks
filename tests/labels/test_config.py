"""Configuration validation, hashing and reconstruction tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from aat.labels import CONFIG_VERSION, LabelConfig, LabelError


def test_default_config_roundtrip_and_sha():
    config = LabelConfig()
    payload = config.to_label_params()
    assert payload["version"] == CONFIG_VERSION
    assert payload["sha256"] == config.config_sha256()
    rebuilt = LabelConfig.from_label_params(json.loads(json.dumps(payload)))
    assert rebuilt == config
    assert rebuilt.config_sha256() == config.config_sha256()


def test_config_sha_is_stable_and_sensitive():
    assert LabelConfig().config_sha256() == LabelConfig().config_sha256()
    changed = replace(LabelConfig(), absolute_threshold_dbfs=-61.0)
    assert changed.config_sha256() != LabelConfig().config_sha256()
    assert len(changed.config_sha256()) == 64


def test_partial_dict_uses_defaults_and_rejects_unknown_keys():
    config = LabelConfig.from_dict({"hop_seconds": 0.04})
    assert config.hop_seconds == 0.04
    assert config.center_window_seconds == LabelConfig().center_window_seconds
    with pytest.raises(LabelError, match="unknown"):
        LabelConfig.from_dict({"hop_seconds": 0.04, "mystery": 1})


def test_from_label_params_rejects_tampering_and_unknown_version():
    payload = LabelConfig().to_label_params()
    payload["config"]["absolute_threshold_dbfs"] = -20.0
    with pytest.raises(LabelError, match="sha256"):
        LabelConfig.from_label_params(payload)

    payload = LabelConfig().to_label_params()
    payload["version"] = "older"
    with pytest.raises(LabelError, match="version"):
        LabelConfig.from_label_params(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("hop_seconds", 0.0),
        ("center_window_seconds", -1.0),
        ("energy_window_seconds", 0.0),
        ("peak_relative_threshold_db", 0.0),
        ("noise_floor_percentile", 101.0),
        ("noise_floor_margin_db", -1.0),
        ("hysteresis_db", -0.1),
        ("release_hold_seconds", -1.0),
        ("low_snr_threshold_db", -1.0),
        ("ambiguity_margin_db", -1.0),
        ("ambiguity_fraction_threshold", 1.5),
        ("hysteresis_db", True),
        ("absolute_threshold_dbfs", float("nan")),
    ],
)
def test_invalid_values_rejected(key, value):
    with pytest.raises(LabelError):
        replace(LabelConfig(), **{key: value})
