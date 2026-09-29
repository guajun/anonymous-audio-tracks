"""Feature-to-embedding model head (issue #7).

Public entry points:

* :class:`aat.models.SourceQueryHead` - K learnable queries over masked,
  time-stamped frame features, emitting unit ``E[..., K, 128]`` and center
  activity logits ``P[..., K]``.
* :class:`aat.models.HeadOutput` - typed output plus
  :meth:`HeadOutput.to_prediction_data` for the shared ``PredictionData``
  contract.

Only fake features are needed; no AuT weights or audio are loaded here.
"""

from __future__ import annotations

from .head import HeadOutput, SourceQueryHead, head_output_to_prediction_data

__all__ = [
    "HeadOutput",
    "SourceQueryHead",
    "head_output_to_prediction_data",
]
