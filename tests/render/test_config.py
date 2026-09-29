"""Strict render-config loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aat.render.config import config_from_dict, load_config
from aat.render.errors import RenderConfigError

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "render"


def _note(step: int = 0, note: int = 60) -> dict:
    return {"step": step, "note": note, "velocity": 100, "length_steps": 1}


def _source(source_id: str, *, sample_type: str = "kick") -> dict:
    payload = {
        "id": source_id,
        "sample": {"type": sample_type, "params": {"duration_seconds": 0.3}},
        "pattern": {"step_seconds": 0.25, "loop_steps": 4, "notes": [_note()]},
    }
    if sample_type == "pad":
        payload["sample"]["params"] = {"duration_seconds": 0.4, "attack_seconds": 0.05, "release_seconds": 0.1}
    return payload


def _base_dict() -> dict:
    return {
        "render": {
            "sample_id": "unit-0001",
            "composition": "comp-unit-01",
            "seed": 7,
            "sample_rate": 8000,
            "block_size": 64,
            "bpm": 100.0,
            "duration_seconds": 2.0,
            "tail_seconds": 0.5,
        },
        "sources": [_source("s01"), _source("s02", sample_type="pluck")],
    }


def test_committed_configs_load() -> None:
    smoke = load_config(CONFIG_DIR / "smoke.toml")
    assert smoke.source_ids == ("s01", "s02", "s03", "s04")
    assert 15.0 <= smoke.duration_seconds <= 30.0
    assert smoke.total_seconds == pytest.approx(23.0)
    assert smoke.surge.plugin_path is None
    assert smoke.presets and smoke.sample_origins
    assert len(smoke.presets) == len(set(smoke.presets))

    ci = load_config(CONFIG_DIR / "smoke_ci.toml")
    assert ci.source_ids == ("s01", "s02")
    assert ci.total_seconds == pytest.approx(2.6)

    probe = load_config(CONFIG_DIR / "surge_probe.toml")
    assert probe.surge.probe_duration_seconds == pytest.approx(1.5)


def test_default_sample_seed_is_derived_from_render_seed() -> None:
    config = config_from_dict(_base_dict())
    assert config.sources[0].sample.seed == 7
    assert config.sources[1].sample.seed == 8
    payload = _base_dict()
    payload["sources"][0]["sample"]["seed"] = 999
    assert config_from_dict(payload).sources[0].sample.seed == 999


def test_config_to_dict_is_json_serialisable_and_path_free() -> None:
    config = config_from_dict(_base_dict())
    text = json.dumps(config.to_dict())
    assert "unit-0001" in text
    assert ":\\" not in text and ":/" not in text


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["render"].update({"unknown": 1}), "unknown key"),
        (lambda d: d["render"].update({"sample_rate": 100}), "sample_rate"),
        (lambda d: d["render"].update({"duration_seconds": 0.5}), "duration_seconds"),
        (lambda d: d["sources"].pop(), "expected 2..8"),
        (lambda d: d["sources"][0].update({"id": "s02"}), "duplicate id"),
        (lambda d: d["sources"][0]["sample"].update({"type": "piano"}), "unsupported sample type"),
        (lambda d: d["sources"][0]["sample"]["params"].update({"bogus": 1}), "unknown key"),
        (lambda d: d["sources"][0]["pattern"].update({"notes": []}), "non-empty array"),
        (lambda d: d["sources"][0]["pattern"]["notes"][0].update({"step": 9}), "loop_steps"),
        (lambda d: d["sources"][0].update({"gain": 9.0}), "gain"),
        (lambda d: d["sources"][0].update({"effects": [{"type": "reverb"}]}), "unsupported effect"),
        (lambda d: d["sources"][0].update({"effects": [{"type": "gain", "gain": 0.0}]}), "gain"),
        (lambda d: d["sources"][0].update({"amp": {"release_ms": 0.0}}), "release_ms"),
        (lambda d: d["surge"].update({"plugin_path": ""}) if "surge" in d else d.update({"surge": {"plugin_path": ""}}), "plugin_path"),
    ],
)
def test_invalid_configs_are_rejected(mutate, message: str) -> None:
    payload = _base_dict()
    mutate(payload)
    with pytest.raises(RenderConfigError) as excinfo:
        config_from_dict(payload)
    assert message in str(excinfo.value)


def test_missing_config_file_reports_clean_error(tmp_path) -> None:
    with pytest.raises(RenderConfigError, match="file not found"):
        load_config(tmp_path / "missing.toml")
