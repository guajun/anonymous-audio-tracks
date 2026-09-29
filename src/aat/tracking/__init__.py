"""Cross-window source tracking.

The public pipeline is:

``PredictionData`` -> :class:`PredictionSequence` -> :func:`associate_sequence`
-> protocol ``trajectory.json`` (:class:`aat.contracts.documents.Trajectory`).

Association is cosine-gated, one-to-one and identity-aware: it maximises the
number of gated matches, breaks ties by total cosine, updates prototypes only
from candidates with ``P >= activity_threshold``, keeps silent identities for
``retention_seconds`` and starts new tracks for unsupported active candidates.
Slots are never identities.

``predict_windows`` / ``track_audio`` provide the sliding-window callback entry
used with fake callbacks in tests and with a future trained head.
"""

from __future__ import annotations

from .config import TrackingConfig
from .errors import TrackingError
from .matching import cosine_similarity, maximum_assignment
from .sequence import PredictionSequence
from .sliding import (
    PredictorCallback,
    WindowPrediction,
    build_prediction_data,
    predict_windows,
    track_audio,
)
from .tracker import as_prediction_sequence, associate_sequence, track_prediction

__all__ = [
    "PredictorCallback",
    "PredictionSequence",
    "TrackingConfig",
    "TrackingError",
    "WindowPrediction",
    "as_prediction_sequence",
    "associate_sequence",
    "build_prediction_data",
    "cosine_similarity",
    "maximum_assignment",
    "predict_windows",
    "track_audio",
    "track_prediction",
]
