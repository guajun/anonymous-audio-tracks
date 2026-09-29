"""Tiny NumPy-only AuT test double for CPU CI (no torch, no weights).

The fake encoder shares the AuT token grid, time axis and validity mask with
the real path (``aat.encoders.grid``) and produces finite features of a
configurable dimension from a deterministic frame-energy proxy.  It exists so
that the CPU test environment can exercise windows, padding, non-integral
chunks and the independent-window comparison without torch, transformers,
network or checkpoint files.

This fake is *not* a model: it is strictly local (a token depends only on the
mel frames in its own chunk-clipped receptive field and its position inside
the chunk), so an independent window that is aligned to the full-track chunk
grid and fully interior reproduces the corresponding full-slice tokens
exactly, while padded or misaligned windows differ.  The real encoder's
*measured* window-vs-slice differences come from the actual attention context
and are reported by ``scripts/probe_aut.py``; they are never inferred from
this double.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import EncoderInputError
from .features import AutFeatures, AutWindowBatch
from .grid import (
    AUT_MEL_HOP_SAMPLES,
    AUT_MIN_AUDIO_SAMPLES,
    AUT_N_FFT,
    AUT_SAMPLE_RATE,
    aut_token_grid,
    token_valid_mask,
)
from .resample import prepare_audio

#: Feature dimension of the fake; deliberately different from the real 2048.
FAKE_FEATURE_DIM = 16


class FakeAutEncoder:
    """Deterministic local test double of the AuT encoder."""

    def __init__(
        self,
        *,
        feature_dim: int = FAKE_FEATURE_DIM,
        seed: int = 20260929,
        model_id: str = "fake/aut-test-double",
        revision: str = "test",
    ) -> None:
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, (int, np.integer)):
            raise EncoderInputError(f"feature_dim: expected an integer, got {type(feature_dim).__name__}")
        if feature_dim < 1:
            raise EncoderInputError(f"feature_dim: must be >= 1, got {feature_dim}")
        self.feature_dim = int(feature_dim)
        self.model_id = model_id
        self.revision = revision
        rng = np.random.default_rng(int(seed))
        self._projection = rng.standard_normal((5, self.feature_dim)).astype(np.float32)
        self._frequencies = (1.0 + np.arange(self.feature_dim)) * 0.17

    # -- internals ---------------------------------------------------------- #

    def _frame_energy(self, audio: np.ndarray) -> np.ndarray:
        """Frame-wise mean magnitude using the mel-frame sample support."""

        half = AUT_N_FFT // 2
        n_frames = audio.shape[0] // AUT_MEL_HOP_SAMPLES
        if n_frames == 0:
            return np.empty(0, dtype=np.float64)
        frames = np.empty(n_frames, dtype=np.float64)
        for t in range(n_frames):
            center = t * AUT_MEL_HOP_SAMPLES
            start = max(0, center - half)
            stop = min(audio.shape[0], center + half)
            frames[t] = float(np.mean(np.abs(audio[start:stop])))
        return frames

    def _token_features(self, frames: np.ndarray, grid: Any) -> np.ndarray:
        out = np.empty((grid.token_count, self.feature_dim), dtype=np.float32)
        for i in range(grid.token_count):
            block = frames[grid.field_start[i] : grid.field_stop[i]]
            if block.size == 0:  # pragma: no cover - fields always cover one frame
                stats = np.zeros(5, dtype=np.float64)
            else:
                stats = np.array(
                    [
                        float(np.mean(block)),
                        float(np.std(block)),
                        float(np.min(block)),
                        float(np.max(block)),
                        float(np.mean(np.abs(np.diff(block)))) if block.size > 1 else 0.0,
                    ],
                    dtype=np.float64,
                )
            positional = np.sin((grid.token_in_chunk[i] + 1) * self._frequencies)
            out[i] = stats @ self._projection + positional.astype(np.float32)
        return out

    def _extract_one(
        self,
        audio: np.ndarray,
        buffer_origin_seconds: float,
        real_start: int,
        real_stop: int,
    ) -> AutFeatures:
        if audio.shape[0] < AUT_MIN_AUDIO_SAMPLES:
            raise EncoderInputError(
                f"audio: {audio.shape[0]} samples is below the {AUT_MIN_AUDIO_SAMPLES}-sample "
                "mel minimum"
            )
        frames = self._frame_energy(audio)
        grid = aut_token_grid(frames.shape[0])
        features = self._token_features(frames, grid)
        valid = token_valid_mask(grid, audio.shape[0], real_start=real_start, real_stop=real_stop)
        return AutFeatures.build(
            features=features,
            valid=valid,
            grid=grid,
            buffer_origin_seconds=buffer_origin_seconds,
            layer="final",
            model_id=self.model_id,
            revision=self.revision,
            preprocessing={"backend": "fake-frame-energy", "feature_dim": self.feature_dim},
            sample_rate=AUT_SAMPLE_RATE,
        )

    # -- public API mirrored by the real encoder ---------------------------- #

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int = AUT_SAMPLE_RATE,
        *,
        origin_seconds: float = 0.0,
    ) -> AutFeatures:
        samples = prepare_audio(audio, sample_rate)
        return self._extract_one(samples, float(origin_seconds), 0, samples.shape[0])

    def extract_windows(
        self,
        windows: np.ndarray,
        sample_rate: int = AUT_SAMPLE_RATE,
        *,
        window_start_seconds: np.ndarray,
        valid_samples: np.ndarray | None = None,
    ) -> AutWindowBatch:
        window_array = np.asarray(windows)
        if window_array.ndim != 2:
            raise EncoderInputError(
                f"windows: expected a 2-D (N, W) array, got shape {window_array.shape}"
            )
        starts = np.asarray(window_start_seconds, dtype=np.float64)
        if starts.shape != (window_array.shape[0],):
            raise EncoderInputError(
                f"window_start_seconds: expected shape {(window_array.shape[0],)}, got {starts.shape}"
            )
        if valid_samples is None:
            valid_samples = np.ones_like(window_array, dtype=bool)
        valid_array = np.asarray(valid_samples)
        if valid_array.shape != window_array.shape or valid_array.dtype != np.bool_:
            raise EncoderInputError(
                f"valid_samples: expected bool {window_array.shape}, got {valid_array.dtype} "
                f"{valid_array.shape}"
            )

        rows: list[AutFeatures] = []
        for index in range(window_array.shape[0]):
            samples = prepare_audio(window_array[index], sample_rate)
            real = valid_array[index]
            if bool(np.all(real)):
                real_start, real_stop = 0, samples.shape[0]
            else:
                indices = np.flatnonzero(real)
                if indices.size == 0:
                    raise EncoderInputError(
                        f"windows[{index}]: valid_samples is all-False; no real audio to encode"
                    )
                real_start, real_stop = int(indices[0]), int(indices[-1]) + 1
            rows.append(self._extract_one(samples, float(starts[index]), real_start, real_stop))

        features = np.stack([row.features for row in rows], axis=0)
        valid = np.stack([row.valid for row in rows], axis=0)
        frame_times = np.stack([row.frame_times for row in rows], axis=0)
        return AutWindowBatch(
            features=features,
            valid=valid,
            frame_times=frame_times,
            window_start_seconds=starts,
            grid=rows[0].grid,
            layer="final",
            model_id=self.model_id,
            revision=self.revision,
        )
