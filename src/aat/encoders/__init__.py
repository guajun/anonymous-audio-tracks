"""Independent Qwen3-Omni AuT audio-encoder feature extraction (issue #5).

The package keeps the real checkpoint path and the CPU fake path explicitly
separate:

* :class:`~aat.encoders.aut.AutEncoder` loads only the
  ``thinker.audio_tower.*`` tensors of the pinned checkpoint into the
  standalone ``Qwen3OmniMoeAudioEncoder`` class (torch + transformers).
* :class:`~aat.encoders.fake.FakeAutEncoder` is a NumPy-only deterministic
  test double that shares the token grid, mask and time axis and requires no
  torch, transformers, network or weights.

Time conventions are defined in :mod:`aat.encoders.grid`; ``frame_times`` are
absolute original-track seconds and the protocol document is
``aat.contracts.FeatureData``.
"""

from __future__ import annotations

from .checkpoint import (
    AUT_ENCODER_TENSOR_BYTES,
    AUT_MODEL_ID,
    AUT_PARAM_COUNT,
    AUT_REVISION,
    AUT_TENSOR_COUNT,
    AUT_TENSOR_PREFIX,
    CheckpointIndex,
    ShardPlan,
    audio_config_from_checkpoint,
    check_disk_space,
    check_state_dict_coverage,
    header_encoder_bytes,
    normalize_encoder_keys,
    plan_encoder_shards,
)
from .errors import EncoderCheckpointError, EncoderError, EncoderInputError
from .fake import FAKE_FEATURE_DIM, FakeAutEncoder
from .features import (
    AUT_FEATURE_NAME_PREFIX,
    AutFeatures,
    AutWindowBatch,
    compare_aligned_tokens,
    feature_name_for_layer,
)
from .grid import (
    AUT_CHUNK_MEL_FRAMES,
    AUT_CONV_STRIDE,
    AUT_FIELD_RADIUS_MEL,
    AUT_MEL_BINS,
    AUT_MEL_HOP_SAMPLES,
    AUT_MIN_AUDIO_SAMPLES,
    AUT_N_FFT,
    AUT_NOMINAL_TOKEN_STEP_SECONDS,
    AUT_SAMPLE_RATE,
    AUT_TOKENS_PER_FULL_CHUNK,
    TokenGrid,
    aut_output_length,
    aut_token_grid,
    token_valid_mask,
)
from .resample import prepare_audio, resample_audio

__all__ = [
    "AUT_CHUNK_MEL_FRAMES",
    "AUT_CONV_STRIDE",
    "AUT_ENCODER_TENSOR_BYTES",
    "AUT_FEATURE_NAME_PREFIX",
    "AUT_FIELD_RADIUS_MEL",
    "AUT_MEL_BINS",
    "AUT_MEL_HOP_SAMPLES",
    "AUT_MIN_AUDIO_SAMPLES",
    "AUT_MODEL_ID",
    "AUT_NOMINAL_TOKEN_STEP_SECONDS",
    "AUT_N_FFT",
    "AUT_PARAM_COUNT",
    "AUT_REVISION",
    "AUT_SAMPLE_RATE",
    "AUT_TENSOR_COUNT",
    "AUT_TENSOR_PREFIX",
    "AUT_TOKENS_PER_FULL_CHUNK",
    "AutFeatures",
    "AutWindowBatch",
    "CheckpointIndex",
    "EncoderCheckpointError",
    "EncoderError",
    "EncoderInputError",
    "FAKE_FEATURE_DIM",
    "FakeAutEncoder",
    "ShardPlan",
    "TokenGrid",
    "aut_output_length",
    "aut_token_grid",
    "audio_config_from_checkpoint",
    "check_disk_space",
    "check_state_dict_coverage",
    "compare_aligned_tokens",
    "feature_name_for_layer",
    "header_encoder_bytes",
    "normalize_encoder_keys",
    "plan_encoder_shards",
    "prepare_audio",
    "resample_audio",
    "token_valid_mask",
]


def __getattr__(name: str):
    """Expose the real encoder lazily so importing the package needs no torch."""

    if name in {"AutEncoder", "MelConfig", "load_filtered_audio_encoder"}:
        from . import aut

        return getattr(aut, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
