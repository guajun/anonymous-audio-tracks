"""Synthetic, pure-data fixtures for the protocol tests.

Nothing here touches real audio, models, weights or datasets.  The ``.npz``
payloads are created by the tests at runtime from these small arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aat.contracts.version import SCHEMA_VERSION


def _digest(seed: int) -> str:
    return format(seed, "064x")


def _base_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sample_manifest",
        "sample_id": "synth-0001",
        "stage": "rendered",
        "seed": 20260929,
        "sample_rate": 16000,
        "duration_seconds": 4.0,
        "track_start_seconds": 0.0,
        "groups": {
            "composition": "comp-synth-01",
            "preset": ["preset-a", "preset-b"],
            "sample_origin": ["surge-factory"],
        },
        "mix_path": "mix.wav",
        "stem_paths": {"s01": "stems/s01.wav", "s02": "stems/s02.wav"},
        "sources_path": "sources.json",
        "controls_path": "controls.json",
        "versions": {"renderer": "dawdreamer-test", "protocol": SCHEMA_VERSION},
        "content_sha256": {
            "mix.wav": _digest(0),
            "stems/s01.wav": _digest(1),
            "stems/s02.wav": _digest(2),
            "sources.json": _digest(3),
            "controls.json": _digest(4),
        },
        "render_latency_seconds": 0.001,
        "tail_seconds": 0.5,
    }


def manifest_dict() -> dict[str, Any]:
    """A valid ``rendered`` (pre-label) manifest."""

    return _base_manifest()


def labeled_manifest_dict() -> dict[str, Any]:
    """A valid ``labeled`` manifest with the activity sidecar pair."""

    data = _base_manifest()
    data["stage"] = "labeled"
    data["activity_metadata_path"] = "activity.json"
    data["activity_arrays_path"] = "activity.npz"
    data["content_sha256"]["activity.json"] = _digest(5)
    data["content_sha256"]["activity.npz"] = _digest(6)
    return data


def manifest_for_assets(
    composition: str,
    presets: list[str],
    origins: list[str],
) -> dict[str, Any]:
    """A rendered manifest with explicit shared-asset group lists."""

    data = _base_manifest()
    data["groups"] = {
        "composition": composition,
        "preset": list(presets),
        "sample_origin": list(origins),
    }
    return data


def sources_dict() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sources",
        "sample_id": "synth-0001",
        "sources": [
            {
                "source_id": "s01",
                "index": 0,
                "renderer": "test-synth",
                "preset_ref": "presets/pad-a.fxp",
                "seed": 11,
            },
            {
                "source_id": "s02",
                "index": 1,
                "renderer": "test-sampler",
                "sample_ref": "samples/kick-a.wav",
            },
        ],
    }


def controls_dict() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "controls",
        "sample_id": "synth-0001",
        "events": [
            {
                "time_seconds": 0.0,
                "source_id": "s01",
                "event_type": "note_on",
                "data": {"note": 60, "velocity": 100},
            },
            {
                "time_seconds": 0.25,
                "source_id": "s01",
                "event_type": "note_off",
                "data": {"note": 60},
            },
            {
                "time_seconds": 0.25,
                "source_id": "s02",
                "event_type": "note_on",
                "data": {"note": 36, "velocity": 120},
            },
            {
                "time_seconds": 0.5,
                "source_id": "s02",
                "event_type": "param",
                "data": {"name": "cutoff_hz", "value": 800.0},
            },
        ],
    }


def provenance_dict(data_kind: str = "model") -> dict[str, Any]:
    return {
        "run_id": "run-20260929-01",
        "data_kind": data_kind,
        "model_id": "test-output-head",
        "git_commit": "e281aff",
        "config_hash": "cfg-abc123",
        "created_at_utc": "2026-09-29T12:00:00Z",
    }


def trajectory_dict() -> dict[str, Any]:
    """A whole-track trajectory with one silent-identity point (``null`` slot)."""

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "trajectory",
        "sample_id": "synth-0001",
        "slots": 8,
        "audio": {"duration_seconds": 4.0, "track_start_seconds": 0.0},
        "provenance": provenance_dict(),
        "tracks": [
            {
                "track_id": "trk-0001",
                "center_times": [0.0, 0.02, 0.04],
                "activity": [0.0, 0.8, 0.9],
                "confidence": [0.5, 0.9, 0.95],
                "slot_indices": [1, None, 1],
            },
            {
                "track_id": "trk-0002",
                "center_times": [0.02, 0.04, 0.06],
                "activity": [0.0, 0.0, 0.0],
            },
        ],
        "params": {"activity_threshold": 0.5, "match_threshold": 0.7},
    }


def empty_trajectory_dict() -> dict[str, Any]:
    """Empty audio: zero duration and no tracks."""

    data = trajectory_dict()
    data["audio"] = {"duration_seconds": 0.0, "track_start_seconds": 0.0}
    data["tracks"] = []
    return data


def clip_trajectory_dict() -> dict[str, Any]:
    """A clip that starts later on the original-track axis."""

    data = trajectory_dict()
    data["audio"] = {"duration_seconds": 0.5, "track_start_seconds": 12.0}
    data["tracks"] = [
        {
            "track_id": "trk-clip-0001",
            "center_times": [12.0, 12.24, 12.48],
            "activity": [0.0, 0.7, 0.9],
            "slot_indices": [2, 2, None],
        }
    ]
    return data


def activity_arrays() -> dict[str, Any]:
    return {
        "center_times": np.array([0.0, 0.02, 0.04, 0.06], dtype=np.float64),
        "activity": np.array(
            [[0.0, 0.0], [0.5, 0.9], [0.0, 0.0], [0.8, 0.0]], dtype=np.float32
        ),
        "valid": np.array([False, True, True, True], dtype=bool),
        "source_ids": ("s01", "s02"),
        "sample_rate": 16000,
    }


def feature_arrays() -> dict[str, Any]:
    frames = 5
    hop_seconds = 0.04
    return {
        "frame_times": np.arange(frames, dtype=np.float64) * hop_seconds,
        "features": (np.arange(frames * 4, dtype=np.float32).reshape(frames, 4) / 10.0),
        "valid": np.array([True, True, True, True, True], dtype=bool),
        "feature_name": "synth-test-feature",
        "feature_dim": 4,
        "sample_rate": 16000,
        "frame_origin_seconds": 0.0,
        "hop_seconds": hop_seconds,
    }


def prediction_arrays(
    n_windows: int = 3,
    slots: int = 8,
    embedding_dim: int = 128,
) -> dict[str, Any]:
    rng = np.random.default_rng(20260929)
    embeddings = rng.normal(size=(n_windows, slots, embedding_dim)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    activity = rng.random(size=(n_windows, slots)).astype(np.float32)
    slot_valid = np.ones((n_windows, slots), dtype=bool)
    center_valid = np.ones(n_windows, dtype=bool)
    if n_windows >= 2:
        center_valid[0] = False
        slot_valid[1, 0] = False
        embeddings[1, 0] = 0.0
        activity[1, 0] = 0.0
    return {
        "center_times": np.arange(n_windows, dtype=np.float64) * 0.02,
        "embeddings": embeddings,
        "activity": activity,
        "slot_valid": slot_valid,
        "center_valid": center_valid,
        "hop_seconds": 0.02,
        "sample_id": "synth-0001",
    }
