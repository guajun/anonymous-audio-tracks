"""Pattern expansion and protocol control events."""

from __future__ import annotations

from pathlib import Path

import pytest

from aat.render.config import config_from_dict, load_config
from aat.render.timeline import build_controls, expand_score, source_events

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "render"


def test_smoke_ci_score_expansion() -> None:
    config = load_config(CONFIG_DIR / "smoke_ci.toml")
    score = expand_score(config)
    assert [event.source_id for event in score].count("s01") == 3
    assert [event.source_id for event in score].count("s02") == 2
    times = [event.start_seconds for event in score]
    assert times == sorted(times)
    assert all(event.start_seconds >= config.track_start_seconds for event in score)
    assert all(event.duration_seconds > 0.0 for event in score)


def test_pattern_repeats_across_the_musical_duration() -> None:
    config = load_config(CONFIG_DIR / "smoke_ci.toml")
    events = source_events(config, "s02")
    # s02 loops every 2 s over a 2 s composition: one repetition.
    assert {event.repetition for event in events} == {0}
    long_config = config_from_dict(
        {
            "render": {
                "sample_id": "loop-0001",
                "composition": "comp-loop-01",
                "seed": 1,
                "sample_rate": 8000,
                "block_size": 64,
                "bpm": 120.0,
                "duration_seconds": 5.0,
                "tail_seconds": 0.5,
            },
            "sources": [
                {
                    "id": "s01",
                    "sample": {"type": "kick", "params": {"duration_seconds": 0.2}},
                    "pattern": {
                        "step_seconds": 0.5,
                        "loop_steps": 2,
                        "notes": [{"step": 0, "note": 36}],
                    },
                },
                {
                    "id": "s02",
                    "sample": {"type": "hat", "params": {"duration_seconds": 0.2}},
                    "pattern": {
                        "step_seconds": 0.5,
                        "loop_steps": 2,
                        "notes": [{"step": 1, "note": 42}],
                    },
                },
            ],
        }
    )
    events = source_events(long_config, "s01")
    assert len(events) == 5
    assert [event.start_seconds for event in events] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_controls_validate_against_the_protocol() -> None:
    config = load_config(CONFIG_DIR / "smoke_ci.toml")
    score = expand_score(config)
    controls = build_controls(config, score)
    assert controls.sample_id == config.sample_id
    controls.validate_source_references(config.source_ids)
    times = [event.time_seconds for event in controls.events]
    assert times == sorted(times)
    ons = [event for event in controls.events if event.event_type == "note_on"]
    offs = [event for event in controls.events if event.event_type == "note_off"]
    assert len(ons) == len(offs) == len(score)
    for note_on in ons:
        matching = [
            event
            for event in offs
            if event.source_id == note_on.source_id and event.data["note"] == note_on.data["note"]
        ]
        expected = note_on.time_seconds + note_on.data["duration_seconds"]
        assert any(event.time_seconds == expected for event in matching)


def test_absolute_times_shift_with_track_start() -> None:
    payload = {
        "render": {
            "sample_id": "clip-0001",
            "composition": "comp-clip-01",
            "seed": 2,
            "sample_rate": 8000,
            "block_size": 64,
            "bpm": 120.0,
            "duration_seconds": 2.0,
            "tail_seconds": 0.5,
            "track_start_seconds": 12.0,
        },
        "sources": [
            {
                "id": "s01",
                "sample": {"type": "kick", "params": {"duration_seconds": 0.2}},
                "pattern": {"step_seconds": 0.5, "loop_steps": 4, "notes": [{"step": 0, "note": 36}]},
            },
            {
                "id": "s02",
                "sample": {"type": "hat", "params": {"duration_seconds": 0.2}},
                "pattern": {"step_seconds": 0.5, "loop_steps": 4, "notes": [{"step": 1, "note": 42}]},
            },
        ],
    }
    config = config_from_dict(payload)
    assert config.track_start_seconds == pytest.approx(12.0)
    assert min(event.start_seconds for event in expand_score(config)) == pytest.approx(12.0)
