"""Whole-song trajectory evaluation.

``evaluate_trajectory`` compares center-activity labels with a tracked
``trajectory.json`` using one whole-song source-to-track mapping.  See
``docs/TRACKING.md`` for metric definitions and limitations.
"""

from __future__ import annotations

from .errors import EvaluationError
from .metrics import (
    EvaluationConfig,
    SourceEvaluation,
    TrajectoryEvaluation,
    evaluate_trajectory,
)

__all__ = [
    "EvaluationConfig",
    "EvaluationError",
    "SourceEvaluation",
    "TrajectoryEvaluation",
    "evaluate_trajectory",
]
