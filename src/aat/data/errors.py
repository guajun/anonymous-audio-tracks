"""Dataset index, split and batch errors."""

from __future__ import annotations


class DatasetError(Exception):
    """Raised when a dataset index, split or batch cannot be produced.

    Messages always name the sample (``sample_id`` and/or data-root relative
    path) or the offending parameter, so a failure can be located without
    guessing which of many samples is broken.
    """


__all__ = ["DatasetError"]
