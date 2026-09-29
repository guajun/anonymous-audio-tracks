"""Training, checkpoint and resume errors (issue #8).

The exceptions are intentionally few and explicit:

* :class:`TrainingError` - invalid configuration, insufficient supervision or
  an impossible training request.
* :class:`CheckpointError` - a checkpoint file is unreadable, has the wrong
  format version or cannot be written safely.
* :class:`ResumeMismatchError` - the checkpoint is valid but does not describe
  the current config/data/encoder identity; resuming would silently replace
  data, shapes or hyper-parameters, so it is refused.
"""

from __future__ import annotations


class TrainingError(RuntimeError):
    """A training run cannot proceed as requested."""


class CheckpointError(TrainingError):
    """A training checkpoint is missing, malformed or not safely writable."""


class ResumeMismatchError(TrainingError):
    """Resume was requested with a different config, data or encoder identity."""
