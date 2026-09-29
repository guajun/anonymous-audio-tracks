"""Errors raised by the audio-encoder layer (issue #5)."""

from __future__ import annotations


class EncoderError(RuntimeError):
    """Base class for AuT encoder/feature extraction failures."""


class EncoderCheckpointError(EncoderError):
    """The checkpoint index, shards or filtered state dict do not match."""


class EncoderInputError(EncoderError):
    """Audio/feature input violates the encoder's expected ranges or shapes."""
