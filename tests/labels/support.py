"""Shared helpers for labeler tests."""

from __future__ import annotations

import numpy as np

from aat.labels import LabelConfig, LabelResult, label_stems

from . import signals


def label_one(
    signal: np.ndarray,
    duration_seconds: float | None = None,
    *,
    config: LabelConfig | None = None,
    origin_seconds: float = 0.0,
    sample_rate: int = signals.SAMPLE_RATE,
) -> LabelResult:
    if duration_seconds is None:
        duration_seconds = signal.shape[0] / sample_rate
    return label_stems(
        {"s01": signal},
        sample_rate,
        duration_seconds=duration_seconds,
        origin_seconds=origin_seconds,
        config=config,
    )


def activity_at(result: LabelResult, seconds: float, source: int = 0) -> float:
    times = result.activity.center_times
    right = int(np.searchsorted(times, seconds))
    if right <= 0:
        index = 0
    elif right >= times.size:
        index = times.size - 1
    else:
        index = right if (times[right] - seconds) <= (seconds - times[right - 1]) else right - 1
    return float(result.activity.activity[index, source])


def state_at(result: LabelResult, seconds: float, source_id: str = "s01") -> bool:
    times = result.activity.center_times
    right = int(np.searchsorted(times, seconds))
    if right <= 0:
        index = 0
    elif right >= times.size:
        index = times.size - 1
    else:
        index = right if (times[right] - seconds) <= (seconds - times[right - 1]) else right - 1
    return bool(result.diagnostics[source_id].active_state[index])
