"""Surge XT probe failure semantics (no plugin required, no fake pass)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from aat.render.errors import SurgeProbeError
from aat.render.probe_surge import SurgeProbeResult, probe_surge


def test_missing_plugin_path_is_explicit_failure() -> None:
    result = probe_surge(None)
    assert isinstance(result, SurgeProbeResult)
    assert result.status == "failed"
    assert result.reason == "not_configured"
    assert result.passed is False
    assert result.latency_samples is None


def test_empty_plugin_path_is_explicit_failure() -> None:
    assert probe_surge("   ").reason == "not_configured"


def test_nonexistent_plugin_path_is_explicit_failure(tmp_path) -> None:
    result = probe_surge(tmp_path / "Surge XT.vst3")
    assert result.status == "failed"
    assert result.reason == "path_not_found"
    assert "does not exist" in (result.detail or "")


def test_result_is_json_serialisable() -> None:
    payload = probe_surge(None).to_json_dict()
    assert payload["kind"] == "surge_probe"
    assert payload["status"] == "failed"
    assert json.loads(json.dumps(payload))["reason"] == "not_configured"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"block_size": 0},
        {"duration_seconds": 0.01},
        {"note": 128},
        {"velocity": 0},
    ],
)
def test_invalid_arguments_raise(kwargs) -> None:
    with pytest.raises(SurgeProbeError):
        probe_surge(None, **kwargs)


def test_probe_surge_config_has_no_plugin_path() -> None:
    config = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "configs" / "render" / "surge_probe.toml").read_text(
            encoding="utf-8"
        )
    )
    assert "plugin_path" not in config.get("surge", {})
