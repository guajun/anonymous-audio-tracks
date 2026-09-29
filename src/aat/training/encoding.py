"""Encoder adapters for training: fake (NumPy) and real frozen AuT.

Both adapters expose the same call shape and always encode **independent fixed
windows** in chunks of ``batch_windows`` (the last chunk may be smaller; that
tail behaviour is recorded in the run metadata).  The real adapter forces the
audited block-diagonal attention path and float32 from the config; the fake
adapter is the NumPy test double from issue #5 and is never presented as a real
model result.

This module does not import torch at module scope: the real AuT path imports it
lazily through :mod:`aat.encoders.aut`, so a render-only environment can still
import the training package's light modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from aat.encoders.errors import EncoderError
from aat.encoders.fake import FakeAutEncoder
from aat.windowing import centered_window_bounds, window_sample_count

from .config import EncoderConfig, TrainingError

#: Audited AuT output dimension (``output_dim=2048`` in the pinned config, see
#: docs/AUT_PROBE.md).  The config pins ``feature_dim`` to this value and the
#: first encoded chunk is checked against it.
AUT_FEATURE_DIM = 2048


@dataclass(frozen=True)
class EncodedWindows:
    """Encoder output for ``N`` windows of one song block."""

    features: np.ndarray  # (N, T, F) float32
    frame_valid: np.ndarray  # (N, T) bool
    frame_times: np.ndarray  # (N, T) float64 absolute seconds
    preprocessing: Mapping[str, Any]

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[2])

    @property
    def windows(self) -> int:
        return int(self.features.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.features.shape[1])


def window_start_seconds(
    center_times: np.ndarray,
    *,
    sample_rate: int,
    window_seconds: float,
    origin_seconds: float,
) -> np.ndarray:
    """Absolute start time of each centered window, mirroring extraction.

    Uses the same half-up sample rounding and ``centered_window_bounds`` as
    :func:`aat.windowing.extract_windows_at_times`, so the encoder frame times
    refer to the actual samples that were encoded.
    """

    times = np.asarray(center_times, dtype=np.float64)
    width = window_sample_count(window_seconds, sample_rate)
    local = times - float(origin_seconds)
    center_samples = np.floor(local * sample_rate + 0.5).astype(np.int64)
    starts = np.array(
        [centered_window_bounds(int(center), width)[0] for center in center_samples],
        dtype=np.int64,
    )
    return float(origin_seconds) + starts.astype(np.float64) / float(sample_rate)


class EncoderAdapter:
    """Common interface of the fake and real encoders used by the trainer.

    Subclasses implement :meth:`_encode_chunk`; this base class enforces the
    fixed chunking policy and consistent arrays.
    """

    mode = "abstract"

    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        self.batch_windows = int(config.batch_windows)
        self.window_seconds = float(config.window_seconds)

    # -- provenance --------------------------------------------------------- #

    def provenance(self) -> dict[str, Any]:
        data = {
            "mode": self.mode,
            "window_seconds": self.window_seconds,
            "extraction": self.config.extraction,
            "batch_policy": {
                "batch_windows": self.batch_windows,
                "tail": "smaller final chunk, order preserved",
                "scope": "per song block, windows in center order",
            },
        }
        data.update(self._provenance_extra())
        return data

    def identity(self) -> dict[str, Any]:
        """Semantic identity used to refuse resume under a different encoder."""

        data = {
            "mode": self.mode,
            "feature_dim": self.feature_dim,
            "window_seconds": self.window_seconds,
            "extraction": self.config.extraction,
            "batch_windows": self.batch_windows,
        }
        data.update(self._identity_extra())
        return data

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    def _provenance_extra(self) -> dict[str, Any]:
        return {}

    def _identity_extra(self) -> dict[str, Any]:
        return {}

    def _encode_chunk(
        self,
        windows: np.ndarray,
        sample_rate: int,
        starts: np.ndarray,
        valid_samples: np.ndarray,
    ) -> EncodedWindows:
        raise NotImplementedError

    # -- encoding ----------------------------------------------------------- #

    def encode_windows(
        self,
        windows: np.ndarray,
        sample_rate: int,
        *,
        window_start_seconds: np.ndarray,
        valid_samples: np.ndarray | None = None,
    ) -> EncodedWindows:
        """Encode all ``N`` windows with fixed-size chunks (ordered)."""

        audio = np.asarray(windows)
        if audio.ndim != 2:
            raise TrainingError(
                f"windows: expected a 2-D (N, W) array, got shape {audio.shape}"
            )
        starts = np.asarray(window_start_seconds, dtype=np.float64)
        if starts.shape != (audio.shape[0],):
            raise TrainingError(
                f"window_start_seconds: expected shape {(audio.shape[0],)}, got {starts.shape}"
            )
        if valid_samples is None:
            valid = np.ones_like(audio, dtype=bool)
        else:
            valid = np.asarray(valid_samples)
            if valid.shape != audio.shape or valid.dtype != np.bool_:
                raise TrainingError(
                    f"valid_samples: expected bool {audio.shape}, got {valid.dtype} {valid.shape}"
                )
        if audio.shape[0] == 0:
            raise TrainingError("cannot encode an empty batch of windows")

        chunks: list[EncodedWindows] = []
        for start in range(0, audio.shape[0], self.batch_windows):
            stop = min(start + self.batch_windows, audio.shape[0])
            chunk = self._encode_chunk(
                audio[start:stop], int(sample_rate), starts[start:stop], valid[start:stop]
            )
            if chunk.windows != stop - start:
                raise TrainingError(
                    f"encoder returned {chunk.windows} windows for a {stop - start}-window chunk"
                )
            if chunk.feature_dim != self.feature_dim:
                raise TrainingError(
                    f"encoder returned {chunk.feature_dim} features but the config expects "
                    f"{self.feature_dim}"
                )
            chunks.append(chunk)
        if len(chunks) == 1:
            return chunks[0]
        features = np.concatenate([chunk.features for chunk in chunks], axis=0)
        frame_valid = np.concatenate([chunk.frame_valid for chunk in chunks], axis=0)
        frame_times = np.concatenate([chunk.frame_times for chunk in chunks], axis=0)
        first = chunks[0]
        return EncodedWindows(
            features=np.ascontiguousarray(features, dtype=np.float32),
            frame_valid=np.ascontiguousarray(frame_valid, dtype=bool),
            frame_times=np.ascontiguousarray(frame_times, dtype=np.float64),
            preprocessing=first.preprocessing,
        )


class FakeEncoderAdapter(EncoderAdapter):
    """Deterministic NumPy test double; results are an engineering smoke only."""

    mode = "fake"

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__(config)
        self._encoder = FakeAutEncoder(feature_dim=config.feature_dim, seed=config.fake_seed)

    @property
    def feature_dim(self) -> int:
        return int(self.config.feature_dim)

    def _provenance_extra(self) -> dict[str, Any]:
        return {
            "backend": "fake-frame-energy",
            "fake_seed": self.config.fake_seed,
            "feature_dim": self.feature_dim,
            "model_id": self._encoder.model_id,
            "revision": self._encoder.revision,
            "result_kind": "fake-encoder-smoke (not a real model result)",
        }

    def _identity_extra(self) -> dict[str, Any]:
        return {"fake_seed": self.config.fake_seed}

    def _encode_chunk(
        self,
        windows: np.ndarray,
        sample_rate: int,
        starts: np.ndarray,
        valid_samples: np.ndarray,
    ) -> EncodedWindows:
        batch = self._encoder.extract_windows(
            windows,
            sample_rate,
            window_start_seconds=starts,
            valid_samples=valid_samples,
        )
        return EncodedWindows(
            features=np.ascontiguousarray(batch.features, dtype=np.float32),
            frame_valid=np.ascontiguousarray(batch.valid, dtype=bool),
            frame_times=np.ascontiguousarray(batch.frame_times, dtype=np.float64),
            preprocessing=batch.preprocessing,
        )


class AutEncoderAdapter(EncoderAdapter):
    """Frozen real AuT encoder (issue #5) in the pinned inference mode."""

    mode = "aut"

    def __init__(self, config: EncoderConfig, encoder: Any) -> None:
        super().__init__(config)
        self._encoder = encoder
        self._feature_dim = AUT_FEATURE_DIM

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def load_report_dict(self) -> dict[str, Any]:
        return self._encoder.load_report.to_dict()

    def _provenance_extra(self) -> dict[str, Any]:
        report = self._encoder.load_report
        return {
            "backend": "qwen3-omni-aut",
            "feature_dim": self.feature_dim,
            "model_id": report.model_id,
            "revision": report.revision,
            "revision_source": report.revision_source,
            "transformers_version": report.transformers_version,
            "torch_version": report.torch_version,
            "layer": "final" if self.config.layer is None else int(self.config.layer),
            "dtype": report.dtype,
            "attention": {
                "mode": "block_diagonal" if self._encoder.masked_attention else "unmasked_global",
                "backend": report.attn_implementation,
            },
            "device": report.device,
            "param_count": report.param_count,
            "encoder_bytes": report.encoder_bytes,
            "result_kind": "frozen-aut-features",
        }

    def _identity_extra(self) -> dict[str, Any]:
        report = self._encoder.load_report
        return {
            "model_id": report.model_id,
            "revision": report.revision,
            "layer": self.config.layer,
            "dtype": report.dtype,
            "attention_backend": report.attn_implementation,
            "masked_attention": bool(self._encoder.masked_attention),
        }

    def _encode_chunk(
        self,
        windows: np.ndarray,
        sample_rate: int,
        starts: np.ndarray,
        valid_samples: np.ndarray,
    ) -> EncodedWindows:
        batch = self._encoder.extract_windows(
            windows,
            sample_rate,
            window_start_seconds=starts,
            valid_samples=valid_samples,
            layer=self.config.layer,
        )
        return EncodedWindows(
            features=np.ascontiguousarray(batch.features, dtype=np.float32),
            frame_valid=np.ascontiguousarray(batch.valid, dtype=bool),
            frame_times=np.ascontiguousarray(batch.frame_times, dtype=np.float64),
            preprocessing=batch.preprocessing,
        )


def build_encoder(config: EncoderConfig, *, device: str | None = None) -> EncoderAdapter:
    """Build the encoder adapter selected by ``config.mode``.

    ``device`` overrides ``run.device`` (used by CLI to keep the config
    semantics unchanged while choosing hardware explicitly).
    """

    mode = config.mode
    if mode == "fake":
        return FakeEncoderAdapter(config)
    if mode == "aut":
        from aat.encoders.aut import AutEncoder  # noqa: PLC0415 - lazy torch path

        if not config.model_dir:
            raise TrainingError("encoder.model_dir is required in aut mode")
        target_device = device or "cpu"
        try:
            encoder = AutEncoder.from_checkpoint(
                config.model_dir,
                device=target_device,
                dtype=config.dtype,
                attn_implementation="sdpa",
                model_id=config.model_id,
                revision=config.revision,
            )
        except EncoderError as error:
            raise TrainingError(f"cannot load the AuT checkpoint: {error}") from error
        if not encoder.masked_attention:
            raise TrainingError(
                "the loaded AuT encoder is not on the audited block-diagonal attention path"
            )
        adapter = AutEncoderAdapter(config, encoder)
        if adapter.feature_dim != config.feature_dim:
            raise TrainingError(
                f"loaded AuT emits {adapter.feature_dim} features but config says "
                f"{config.feature_dim}"
            )
        return adapter
    raise TrainingError(f"encoder.mode: unknown mode {mode!r}")


__all__ = [
    "AUT_FEATURE_DIM",
    "AutEncoderAdapter",
    "EncodedWindows",
    "EncoderAdapter",
    "FakeEncoderAdapter",
    "build_encoder",
    "window_start_seconds",
]
