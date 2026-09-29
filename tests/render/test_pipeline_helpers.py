"""Pipeline helpers that keep absolute and render-local time separate.

These tests need no DawDreamer: they build a score and verify the tail boundary
is computed in the same coordinate system as the rendered buffer.
"""

from __future__ import annotations

import pytest

from aat.render.config import config_from_dict
from aat.render.pipeline import required_tail_seconds
from aat.render.timeline import expand_score


def _clip_config(track_start_seconds: float) -> dict:
    return {
        "render": {
            "sample_id": "clip-unit-0001",
            "composition": "comp-clip-unit-01",
            "seed": 5,
            "sample_rate": 8000,
            "block_size": 64,
            "bpm": 120.0,
            "duration_seconds": 2.0,
            "tail_seconds": 0.5,
            "track_start_seconds": track_start_seconds,
        },
        "sources": [
            {
                "id": "s01",
                "sample": {"type": "kick", "params": {"duration_seconds": 0.2}},
                "amp": {"attack_ms": 1.0, "release_ms": 100.0},
                "pattern": {
                    "step_seconds": 0.5,
                    "loop_steps": 4,
                    "notes": [{"step": 0, "note": 36, "length_steps": 1}],
                },
            },
            {
                "id": "s02",
                "sample": {"type": "pluck", "params": {"duration_seconds": 0.2, "freq_hz": 110.0, "decay_seconds": 0.15}},
                "amp": {"attack_ms": 1.0, "release_ms": 100.0},
                "pattern": {
                    "step_seconds": 0.5,
                    "loop_steps": 4,
                    "notes": [{"step": 2, "note": 45, "length_steps": 1}],
                },
            },
        ],
    }


def test_required_tail_is_render_local_not_absolute() -> None:
    config = config_from_dict(_clip_config(track_start_seconds=12.0))
    score = expand_score(config)
    # s02: local start 1.0s, 0.5s note, 0.1s release -> 1.6s render-local.
    assert required_tail_seconds(config, score) == pytest.approx(1.6)


def test_required_tail_does_not_depend_on_track_start() -> None:
    zero = config_from_dict(_clip_config(track_start_seconds=0.0))
    shifted = config_from_dict(_clip_config(track_start_seconds=12.0))
    assert required_tail_seconds(zero, expand_score(zero)) == pytest.approx(
        required_tail_seconds(shifted, expand_score(shifted))
    )


def test_required_tail_uses_max_release_boundary() -> None:
    payload = _clip_config(track_start_seconds=0.0)
    payload["sources"][1]["amp"]["release_ms"] = 700.0
    config = config_from_dict(payload)
    # s02 boundary: 1.0 + 0.5 + 0.7 = 2.2s, later than s01's 0.6s.
    assert required_tail_seconds(config, expand_score(config)) == pytest.approx(2.2)
