"""Masked multi-source E/P output head (issue #7).

The head consumes per-window audio feature frames and emits the protocol
candidates ``E[N, K, 128]`` (unit identities) and ``P[N, K]`` (center-time
activity).  Training-time matching and losses live in :mod:`aat.losses`.
"""

from __future__ import annotations

from .head import (
    HeadConfig,
    HeadOutput,
    MultiSourceHead,
    SinusoidalTimeEncoding,
    head_output_to_prediction_arrays,
)

__all__ = [
    "HeadConfig",
    "HeadOutput",
    "MultiSourceHead",
    "SinusoidalTimeEncoding",
    "head_output_to_prediction_arrays",
]
