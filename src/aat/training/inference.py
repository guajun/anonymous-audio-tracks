"""Minimal checkpoint -> inference interface for downstream consumers (#11).

:func:`load_head_from_checkpoint` rebuilds the trained output head and its
encoder adapter from a normal training checkpoint **without importing any test
module and without the dataset index**.  :class:`HeadInference` then offers:

* :meth:`HeadInference.predict_windows` - encoded ``E[N, K, 128]`` / ``P[N, K]``
  for explicit centered windows (with their absolute start/center times);
* :meth:`HeadInference.predict_at_times` - whole-track prediction on the
  original time axis, returning ``aat.contracts.PredictionData`` that feeds
  ``aat.tracking.associate_sequence`` (issue #9) directly;
* :meth:`HeadInference.predictor_for_centers` - a callback for
  ``aat.tracking.track_audio``/``build_prediction_data``.

Inference always uses the same pinned policy as training: independent fixed
2 s windows (``encoder.window_seconds``), the same encoder attention/dtype and
the same ``encoder.batch_windows`` chunking, so train and inference features
come from one path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from aat.contracts import PredictionData
from aat.models import SourceQueryHead
from aat.tracking import WindowPrediction
from aat.windowing import centered_window_bounds, extract_windows_at_times, window_sample_count

from .checkpoint import load_checkpoint
from .config import TrainConfig
from .encoding import EncoderAdapter, build_encoder, window_start_seconds
from .errors import CheckpointError, TrainingError


def _checkpoint_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise CheckpointError("checkpoint.model_config is missing or malformed")
    return model_config


def build_head_from_model_config(model_config: dict[str, Any]) -> SourceQueryHead:
    """Instantiate ``SourceQueryHead`` exactly as the checkpoint recorded it."""

    required = {
        "feature_dim",
        "slots",
        "embedding_dim",
        "d_model",
        "num_heads",
        "dropout",
        "time_frequencies",
        "base_frequency_hz",
        "ffn_multiplier",
    }
    missing = sorted(required - set(model_config))
    if missing:
        raise CheckpointError(f"checkpoint.model_config is missing keys {missing}")
    return SourceQueryHead(
        feature_dim=int(model_config["feature_dim"]),
        slots=int(model_config["slots"]),
        embedding_dim=int(model_config["embedding_dim"]),
        d_model=int(model_config["d_model"]),
        num_heads=int(model_config["num_heads"]),
        dropout=float(model_config["dropout"]),
        time_frequencies=int(model_config["time_frequencies"]),
        base_frequency_hz=float(model_config["base_frequency_hz"]),
        ffn_multiplier=int(model_config["ffn_multiplier"]),
    )


@dataclass
class HeadInference:
    """Trainable head + frozen encoder loaded from one training checkpoint."""

    config: TrainConfig
    model: SourceQueryHead
    encoder: EncoderAdapter
    device: torch.device
    checkpoint_path: str | None = None
    step: int | None = None

    # -- forward ------------------------------------------------------------ #

    def _invalid_center_mask(
        self,
        centers: np.ndarray,
        *,
        sample_rate: int,
        origin_seconds: float,
        audio_duration_seconds: float | None,
    ) -> np.ndarray:
        """Which bound centers have a window that does not fit the audio span.

        With ``audio_duration_seconds`` the exact sample-grid convention of
        :func:`aat.windowing.centered_window_bounds` is used for both edges
        (identical to the extraction path).  Without a duration only the start
        edge is known, so centers whose window must extend before
        ``origin_seconds`` are marked invalid; the end edge cannot be decided
        without the actual audio span and stays valid (documented limitation).
        """

        width = window_sample_count(self.config.encoder.window_seconds, int(sample_rate))
        local = np.asarray(centers, dtype=np.float64) - float(origin_seconds)
        center_samples = np.floor(local * int(sample_rate) + 0.5).astype(np.int64)
        valid = np.zeros(center_samples.shape[0], dtype=bool)
        frames: int | None = None
        if audio_duration_seconds is not None:
            if (
                isinstance(audio_duration_seconds, bool)
                or not isinstance(audio_duration_seconds, (int, float))
                or not np.isfinite(float(audio_duration_seconds))
                or float(audio_duration_seconds) < 0.0
            ):
                raise TrainingError(
                    "audio_duration_seconds: expected a finite value >= 0"
                )
            frames = int(round(float(audio_duration_seconds) * int(sample_rate)))
        for index, center in enumerate(center_samples):
            start, stop = centered_window_bounds(int(center), width)
            inside = start >= 0
            if frames is not None:
                inside = inside and stop <= frames
            valid[index] = inside
        return valid

    def predict_windows(
        self,
        windows: np.ndarray,
        *,
        sample_rate: int,
        window_start_seconds: np.ndarray | None = None,
        center_times: np.ndarray | None = None,
        valid_samples: np.ndarray | None = None,
        center_valid: np.ndarray | None = None,
    ) -> WindowPrediction:
        """Predict E/P for explicit windows under the pinned 2 s policy.

        ``windows`` must contain exactly the pinned window length for
        ``sample_rate`` (no silent resizing).  ``window_start_seconds``
        (absolute time of ``windows[:, 0]``) and ``center_times`` (absolute
        window centers) default to the pinned 2 s centered convention when
        omitted; when both are given they must agree within the actual
        sample-grid rounding (half a sample).

        Padding semantics: when ``center_valid`` is not given, a row is only
        marked valid when ``valid_samples`` is fully True for that row; rows
        with padded samples therefore produce canonical all-zero invalid slots
        (``E = 0``, ``P = 0``) exactly like training-time invalid centers.
        """

        audio = np.asarray(windows)
        if audio.ndim != 2:
            raise TrainingError(f"windows: expected (N, W), got {audio.shape}")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
            raise TrainingError(
                f"sample_rate: expected an integer >= 1, got {type(sample_rate).__name__}"
            )
        rate = int(sample_rate)
        if rate < 1:
            raise TrainingError(f"sample_rate: must be >= 1, got {rate}")
        expected_samples = window_sample_count(self.config.encoder.window_seconds, rate)
        if audio.shape[1] != expected_samples:
            raise TrainingError(
                f"windows: the pinned policy requires exactly {expected_samples} samples "
                f"per {self.config.encoder.window_seconds:g}s window at {rate} Hz; "
                f"got {audio.shape[1]}"
            )
        rows = int(audio.shape[0])
        if rows == 0:
            slots = self.config.head.slots
            return WindowPrediction(
                embeddings=np.zeros((0, slots, 128), dtype=np.float32),
                activity=np.zeros((0, slots), dtype=np.float32),
                slot_valid=np.zeros((0, slots), dtype=bool),
            )

        valid_mask: np.ndarray | None = None
        if valid_samples is not None:
            valid_mask = np.asarray(valid_samples)
            if valid_mask.dtype != np.bool_ or valid_mask.shape != audio.shape:
                raise TrainingError(
                    f"valid_samples: expected bool {audio.shape}, got "
                    f"{valid_mask.dtype} {valid_mask.shape}"
                )

        if center_times is None:
            if window_start_seconds is None:
                starts = np.zeros(rows, dtype=np.float64)
            else:
                starts = np.asarray(window_start_seconds, dtype=np.float64)
            centers = starts + self.config.encoder.window_seconds / 2.0
        else:
            centers = np.asarray(center_times, dtype=np.float64)
            if window_start_seconds is None:
                starts = centers - self.config.encoder.window_seconds / 2.0
            else:
                starts = np.asarray(window_start_seconds, dtype=np.float64)
                tolerance = 1.0 / rate + 1e-9
                offsets = np.abs(starts - (centers - self.config.encoder.window_seconds / 2.0))
                if np.any(offsets > tolerance):
                    raise TrainingError(
                        "window_start_seconds and center_times are inconsistent with the "
                        f"pinned {self.config.encoder.window_seconds:g}s centered extraction "
                        f"(max offset {float(offsets.max()):.6f}s > {tolerance:.6f}s)"
                    )
        if starts.shape != (rows,) or centers.shape != (rows,):
            raise TrainingError("window_start_seconds/center_times must have one entry per window")
        if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(centers)):
            raise TrainingError("window_start_seconds/center_times must be finite")

        if center_valid is None:
            if valid_mask is not None:
                center_valid_t = valid_mask.all(axis=1)
            else:
                center_valid_t = np.ones(rows, dtype=bool)
        else:
            center_valid_t = np.asarray(center_valid)
            if center_valid_t.dtype != np.bool_ or center_valid_t.shape != (rows,):
                raise TrainingError(
                    f"center_valid: expected bool {(rows,)}, got "
                    f"{center_valid_t.dtype} {center_valid_t.shape}"
                )

        encoded = self.encoder.encode_windows(
            audio,
            rate,
            window_start_seconds=starts,
            valid_samples=valid_mask,
        )
        chunk = self.config.encoder.batch_windows
        embeddings: list[np.ndarray] = []
        activity: list[np.ndarray] = []
        slot_valid: list[np.ndarray] = []
        device = self.device
        dtype = torch.float32
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, encoded.windows, chunk):
                    stop = min(start + chunk, encoded.windows)
                    features = torch.from_numpy(
                        np.ascontiguousarray(encoded.features[start:stop], dtype=np.float32)
                    ).to(device=device, dtype=dtype)
                    frame_times = torch.from_numpy(
                        np.ascontiguousarray(encoded.frame_times[start:stop], dtype=np.float64)
                    ).to(device=device)
                    frame_valid = torch.from_numpy(
                        np.ascontiguousarray(encoded.frame_valid[start:stop], dtype=bool)
                    ).to(device=device)
                    centers_t = torch.from_numpy(
                        np.ascontiguousarray(centers[start:stop], dtype=np.float64)
                    ).to(device=device)
                    center_valid_chunk = torch.from_numpy(
                        np.ascontiguousarray(center_valid_t[start:stop], dtype=bool)
                    ).to(device=device)
                    output = self.model.forward(
                        features,
                        frame_times=frame_times,
                        frame_valid=frame_valid,
                        center_times=centers_t,
                        center_valid=center_valid_chunk,
                    )
                    embeddings.append(
                        output.embeddings.detach().to("cpu", dtype=torch.float32).numpy()
                    )
                    activity.append(
                        output.activity_probabilities()
                        .detach()
                        .to("cpu", dtype=torch.float32)
                        .numpy()
                    )
                    slot_valid.append(output.slot_valid.detach().to("cpu").numpy())
        finally:
            self.model.train(was_training)
        return WindowPrediction(
            embeddings=np.concatenate(embeddings, axis=0).astype(np.float32, copy=False),
            activity=np.concatenate(activity, axis=0).astype(np.float32, copy=False),
            slot_valid=np.concatenate(slot_valid, axis=0).astype(bool, copy=False),
        )

    def predict_at_times(
        self,
        audio: np.ndarray,
        center_seconds: Sequence[float] | np.ndarray,
        *,
        sample_rate: int,
        origin_seconds: float = 0.0,
        sample_id: str | None = None,
        hop_seconds: float | None = None,
        provenance: Any | None = None,
    ) -> PredictionData:
        """Extract centered windows at absolute times and return the E/P contract."""

        samples = np.asarray(audio)
        if samples.ndim != 1:
            raise TrainingError(f"audio: expected a 1-D array, got {samples.shape}")
        centers = np.asarray(center_seconds, dtype=np.float64).reshape(-1)
        if centers.shape[0] == 0:
            slots = self.config.head.slots
            return PredictionData(
                center_times=centers,
                embeddings=np.zeros((0, slots, 128), dtype=np.float32),
                activity=np.zeros((0, slots), dtype=np.float32),
                slot_valid=np.zeros((0, slots), dtype=bool),
                center_valid=np.zeros((0,), dtype=bool),
                slots=slots,
                embedding_dim=128,
                hop_seconds=hop_seconds,
                sample_id=sample_id,
                provenance=provenance,
            )
        windows, valid = extract_windows_at_times(
            samples,
            centers,
            int(sample_rate),
            self.config.encoder.window_seconds,
            origin_seconds=float(origin_seconds),
        )
        starts = window_start_seconds(
            centers,
            sample_rate=int(sample_rate),
            window_seconds=self.config.encoder.window_seconds,
            origin_seconds=float(origin_seconds),
        )
        center_valid = valid.all(axis=1)
        prediction = self.predict_windows(
            windows,
            sample_rate=int(sample_rate),
            window_start_seconds=starts,
            center_times=centers,
            valid_samples=valid,
            center_valid=center_valid,
        )
        return PredictionData(
            center_times=centers,
            embeddings=prediction.embeddings,
            activity=prediction.activity,
            slot_valid=prediction.slot_valid,
            center_valid=center_valid,
            slots=self.config.head.slots,
            embedding_dim=128,
            hop_seconds=hop_seconds,
            sample_id=sample_id,
            provenance=provenance,
        )

    def predictor_for_centers(
        self,
        center_seconds: Sequence[float] | np.ndarray,
        *,
        sample_rate: int,
        origin_seconds: float = 0.0,
        audio_duration_seconds: float | None = None,
    ) -> Callable[[np.ndarray], WindowPrediction]:
        """Callback for ``aat.tracking.build_prediction_data``/``track_audio``.

        The callback extracts nothing itself: the caller must generate windows
        for exactly these centers in the same order.  ``predict_at_times`` is
        the safer whole-track entry point.

        ``audio_duration_seconds`` (the analyzed span starting at
        ``origin_seconds``) enables the exact boundary mask so padded edge
        windows are returned as invalid slots; without it only the start edge
        can be known.  Pass it for any track that starts or ends near a center.
        """

        centers = np.asarray(center_seconds, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(centers)):
            raise TrainingError("center_seconds must be finite")
        starts = window_start_seconds(
            centers,
            sample_rate=int(sample_rate),
            window_seconds=self.config.encoder.window_seconds,
            origin_seconds=float(origin_seconds),
        )
        bound_valid = self._invalid_center_mask(
            centers,
            sample_rate=int(sample_rate),
            origin_seconds=float(origin_seconds),
            audio_duration_seconds=audio_duration_seconds,
        )
        expected = int(centers.shape[0])
        state = {"offset": 0}

        def callback(windows: np.ndarray) -> WindowPrediction:
            audio = np.asarray(windows)
            stop = state["offset"] + audio.shape[0]
            if stop > expected:
                raise TrainingError(
                    "predictor_for_centers: received more windows than the bound center grid"
                )
            result = self.predict_windows(
                audio,
                sample_rate=int(sample_rate),
                window_start_seconds=starts[state["offset"] : stop],
                center_times=centers[state["offset"] : stop],
                center_valid=bound_valid[state["offset"] : stop],
            )
            state["offset"] = stop
            return result

        return callback

    # -- metadata ----------------------------------------------------------- #

    def describe(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "step": self.step,
            "config_sha256": self.config.config_sha256(),
            "encoder_mode": self.encoder.mode,
            "encoder_identity": self.encoder.identity(),
            "slots": self.config.head.slots,
            "window_seconds": self.config.encoder.window_seconds,
        }


def load_head_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_dir: str | None = None,
    device: str | None = None,
    encoder: EncoderAdapter | None = None,
) -> HeadInference:
    """Rebuild a trained head (and its encoder) from a normal training checkpoint."""

    payload = load_checkpoint(checkpoint_path)
    config = TrainConfig.from_dict(payload["config"], origin=str(checkpoint_path))
    if model_dir is not None:
        config = config.with_overrides(model_dir=model_dir)
    if device is not None:
        config = config.with_overrides(device=device)
    resolved_device = torch.device(config.run.device)
    adapter = (
        encoder
        if encoder is not None
        else build_encoder(config.encoder, device=str(resolved_device))
    )
    stored_encoder = payload.get("encoder")
    if not isinstance(stored_encoder, dict) or not isinstance(
        stored_encoder.get("identity"), dict
    ):
        raise CheckpointError("checkpoint has no encoder identity; cannot verify inference setup")
    current = adapter.identity()
    mismatches = {
        key: (stored_encoder["identity"].get(key), current.get(key))
        for key in sorted(set(stored_encoder["identity"]) | set(current))
        if stored_encoder["identity"].get(key) != current.get(key)
    }
    if mismatches:
        details = ", ".join(f"{key}: checkpoint {old!r} vs current {new!r}" for key, (old, new) in mismatches.items())
        raise CheckpointError(f"encoder identity mismatch for inference: {details}")

    model_config = _checkpoint_model_config(payload)
    model = build_head_from_model_config(model_config)
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as error:
        raise CheckpointError(f"model state does not match checkpoint model_config: {error}") from error
    model.to(device=resolved_device)
    model.eval()
    return HeadInference(
        config=config,
        model=model,
        encoder=adapter,
        device=resolved_device,
        checkpoint_path=str(checkpoint_path),
        step=int(payload.get("step", -1)),
    )


__all__ = [
    "HeadInference",
    "build_head_from_model_config",
    "load_head_from_checkpoint",
]
