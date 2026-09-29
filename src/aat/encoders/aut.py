"""Real Qwen3-Omni AuT audio encoder, loaded standalone from filtered shards.

This module never instantiates the Omni Thinker/Talker/vision modules.  It

1. reads ``thinker_config.audio_config`` from the checkpoint ``config.json``
   and builds only ``Qwen3OmniMoeAudioEncoderConfig``;
2. builds only the standalone ``Qwen3OmniMoeAudioEncoder`` class;
3. opens the planned safetensors shard(s) and materialises *only* the
   ``thinker.audio_tower.*`` tensors, verifies the key set exactly against the
   class state dict (missing/unexpected keys raise) and loads with
   ``strict=True``;
4. runs the official ``WhisperFeatureExtractor`` log-mel front end with the
   checkpoint's own preprocessor config.

torch/transformers/safetensors are imported lazily so importing
``aat.encoders.aut`` stays cheap and CPU-only for the fake test path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .checkpoint import (
    AUT_INDEX_FILENAME,
    AUT_MODEL_ID,
    AUT_PREPROCESSOR_FILENAME,
    AUT_REVISION,
    AUT_TENSOR_PREFIX,
    CheckpointIndex,
    audio_config_from_checkpoint,
    check_state_dict_coverage,
    normalize_encoder_keys,
    plan_encoder_shards,
)
from .errors import EncoderCheckpointError, EncoderInputError
from .features import AutFeatures, AutWindowBatch
from .grid import (
    AUT_MEL_BINS,
    AUT_MEL_HOP_SAMPLES,
    AUT_MIN_AUDIO_SAMPLES,
    AUT_N_FFT,
    AUT_SAMPLE_RATE,
    aut_output_length,
    aut_token_grid,
    token_valid_mask,
)
from .resample import prepare_audio

#: The single upstream class this issue instantiates.
AUT_ENCODER_CLASS_NAME = "Qwen3OmniMoeAudioEncoder"
#: Recommended transformers pin.  4.57.0 is yanked on PyPI; the
#: ``qwen3_omni_moe`` and Whisper feature-extraction sources are byte-identical
#: in 4.57.0 and 4.57.1 (``modeling_qwen3_omni_moe.py`` sha256
#: ``809eaeb4d40cb0e59965a85b5f85ddc07e9ab7b6cdae84972c711f8cdadec296``).
AUT_TRANSFORMERS_VERSION = "4.57.1"
#: Official mel preprocessor class (declared by the checkpoint preprocessor config).
AUT_MEL_CLASS_NAME = "WhisperFeatureExtractor"

_DTYPES = {"float32": "float32", "bfloat16": "bfloat16", "float16": "float16"}


@dataclass(frozen=True)
class MelConfig:
    """Validated subset of the official checkpoint preprocessor config."""

    feature_size: int
    sampling_rate: int
    hop_length: int
    n_fft: int
    dither: float
    padding_value: float
    chunk_length: float
    n_samples: int
    nb_max_frames: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MelConfig":
        def number(key: str) -> float:
            if key not in payload:
                raise EncoderCheckpointError(f"preprocessor config: missing {key!r}")
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EncoderCheckpointError(f"preprocessor config: {key!r} must be numeric")
            return value

        feature_size = int(number("feature_size"))
        sampling_rate = int(number("sampling_rate"))
        hop_length = int(number("hop_length"))
        n_fft = int(number("n_fft"))
        n_samples = (
            int(number("n_samples"))
            if "n_samples" in payload
            else int(number("chunk_length") * sampling_rate)
        )
        nb_max_frames = (
            int(number("nb_max_frames")) if "nb_max_frames" in payload else n_samples // hop_length
        )
        config = cls(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            dither=float(number("dither")) if "dither" in payload else 0.0,
            padding_value=float(number("padding_value")) if "padding_value" in payload else 0.0,
            chunk_length=float(n_samples) / sampling_rate,
            n_samples=n_samples,
            nb_max_frames=nb_max_frames,
        )
        if (
            config.feature_size != AUT_MEL_BINS
            or config.sampling_rate != AUT_SAMPLE_RATE
            or config.hop_length != AUT_MEL_HOP_SAMPLES
            or config.n_fft != AUT_N_FFT
        ):
            raise EncoderCheckpointError(
                "preprocessor config is not the audited AuT mel front end: "
                f"feature_size={config.feature_size}, sampling_rate={config.sampling_rate}, "
                f"hop_length={config.hop_length}, n_fft={config.n_fft}"
            )
        if config.dither != 0.0:
            raise EncoderCheckpointError(
                f"preprocessor config: dither={config.dither} would make mel extraction non-deterministic"
            )
        if config.nb_max_frames * config.hop_length != config.n_samples:
            raise EncoderCheckpointError(
                "preprocessor config: nb_max_frames * hop_length != n_samples "
                f"({config.nb_max_frames} * {config.hop_length} != {config.n_samples})"
            )
        return config


@dataclass
class EncoderLoadReport:
    """What was actually loaded, for the probe and for review."""

    model_id: str = AUT_MODEL_ID
    revision: str = AUT_REVISION
    class_name: str = AUT_ENCODER_CLASS_NAME
    transformers_version: str = ""
    torch_version: str = ""
    param_count: int = 0
    tensor_count: int = 0
    encoder_bytes: int = 0
    shard_files: tuple[str, ...] = ()
    shard_tensor_counts: Mapping[str, int] = field(default_factory=dict)
    device: str = ""
    dtype: str = ""
    attn_implementation: str = ""
    mel_config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["shard_files"] = list(self.shard_files)
        payload["shard_tensor_counts"] = dict(self.shard_tensor_counts)
        payload["mel_config"] = dict(self.mel_config)
        return payload


def _import_torch() -> Any:
    try:
        import torch  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - environment dependent
        raise EncoderCheckpointError(
            "the real AuT encoder requires torch; install the pinned ML extra "
            "(see docs/AUT_PROBE.md). CPU CI only uses the fake encoder."
        ) from error
    return torch


def load_mel_feature_extractor(model_dir: Path | str) -> tuple[Any, MelConfig]:
    """Build the official ``WhisperFeatureExtractor`` from the checkpoint config."""

    directory = Path(model_dir)
    path = directory / AUT_PREPROCESSOR_FILENAME
    if not path.is_file():
        raise EncoderCheckpointError(
            f"missing {path}; the official mel front end is part of the pinned checkpoint"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EncoderCheckpointError(f"{path} is not valid JSON: {error}") from error
    if payload.get("feature_extractor_type") != AUT_MEL_CLASS_NAME:
        raise EncoderCheckpointError(
            f"{path}: feature_extractor_type must be {AUT_MEL_CLASS_NAME!r}, "
            f"got {payload.get('feature_extractor_type')!r}"
        )
    mel_config = MelConfig.from_payload(payload)
    from transformers import WhisperFeatureExtractor  # noqa: PLC0415

    extractor = WhisperFeatureExtractor(
        feature_size=mel_config.feature_size,
        sampling_rate=mel_config.sampling_rate,
        hop_length=mel_config.hop_length,
        n_fft=mel_config.n_fft,
        chunk_length=mel_config.chunk_length,
        dither=mel_config.dither,
        padding_value=mel_config.padding_value,
        return_attention_mask=False,
    )
    return extractor, mel_config


def load_audio_encoder_config(model_dir: Path | str) -> Any:
    """Build ``Qwen3OmniMoeAudioEncoderConfig`` from ``thinker_config.audio_config``."""

    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise EncoderCheckpointError(f"missing {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EncoderCheckpointError(f"{path} is not valid JSON: {error}") from error
    if payload.get("model_type") != "qwen3_omni_moe":
        raise EncoderCheckpointError(
            f"{path}: model_type {payload.get('model_type')!r} is not the audited checkpoint"
        )
    audio_config = audio_config_from_checkpoint(payload)
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (  # noqa: PLC0415
        Qwen3OmniMoeAudioEncoderConfig,
    )

    return Qwen3OmniMoeAudioEncoderConfig(**dict(audio_config))


def load_filtered_audio_encoder(
    model_dir: Path | str,
    shard_paths: Sequence[Path | str],
    *,
    device: str = "cpu",
    dtype: str | None = None,
    attn_implementation: str = "sdpa",
) -> tuple[Any, EncoderLoadReport]:
    """Instantiate the encoder class and load only its audited tensors."""

    torch = _import_torch()
    directory = Path(model_dir)
    index = CheckpointIndex.load(directory / AUT_INDEX_FILENAME)
    plan = plan_encoder_shards(index)

    provided = {Path(path).name: Path(path) for path in shard_paths}
    missing_shards = [name for name in plan.shards if name not in provided]
    if missing_shards:
        raise EncoderCheckpointError(
            f"missing shard(s) {missing_shards}; audited plan needs {list(plan.shards)}"
        )

    config = load_audio_encoder_config(directory)
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (  # noqa: PLC0415
        Qwen3OmniMoeAudioEncoder,
    )

    config._attn_implementation = attn_implementation
    model = Qwen3OmniMoeAudioEncoder(config)

    from safetensors import safe_open  # noqa: PLC0415

    selected: dict[str, Any] = {}
    encoder_bytes = 0
    for name in plan.shards:
        path = provided[name]
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not a mapping
                if not key.startswith(AUT_TENSOR_PREFIX):
                    continue
                tensor = handle.get_tensor(key)
                selected[key] = tensor
                encoder_bytes += int(tensor.numel()) * int(tensor.element_size())

    normalized = normalize_encoder_keys(selected)
    expected = list(model.state_dict().keys())
    check_state_dict_coverage(expected, list(normalized))
    model.load_state_dict(normalized, strict=True, assign=True)
    del selected, normalized

    resolved_dtype = dtype
    if resolved_dtype is None:
        resolved_dtype = "bfloat16" if str(device).startswith("cuda") else "float32"
    if resolved_dtype not in _DTYPES:
        raise EncoderCheckpointError(
            f"dtype {resolved_dtype!r} not supported; choose from {sorted(_DTYPES)}"
        )
    torch_dtype = getattr(torch, resolved_dtype)
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    import transformers  # noqa: PLC0415

    report = EncoderLoadReport(
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        param_count=int(sum(p.numel() for p in model.parameters())),
        tensor_count=len(expected),
        encoder_bytes=encoder_bytes,
        shard_files=tuple(plan.shards),
        shard_tensor_counts=dict(plan.tensor_counts),
        device=str(device),
        dtype=resolved_dtype,
        attn_implementation=attn_implementation,
    )
    return model, report


def _real_region_after_resample(
    valid_samples: np.ndarray,
    source_rate: int,
    target_rate: int,
    n_out: int,
) -> tuple[int, int] | None:
    """Map the contiguous real-sample span of a window onto the resampled axis."""

    indices = np.flatnonzero(valid_samples)
    if indices.size == 0:
        return None
    start, stop = int(indices[0]), int(indices[-1]) + 1
    ratio = target_rate / source_rate
    start_out = int(np.ceil(start * ratio))
    stop_out = int(np.floor(stop * ratio)) + 1
    start_out = max(0, min(start_out, n_out))
    stop_out = max(start_out, min(stop_out, n_out))
    return start_out, stop_out


class AutEncoder:
    """Independent AuT feature extractor: 16 kHz audio in, tokens + mask + times out.

    ``transformers``' sdpa/eager attention paths ignore the ``cu_seqlens`` that
    the standalone encoder computes and whose only purpose is the ~8.32 s
    block-diagonal attention.  In the upstream ``Qwen3OmniMoeAudioEncoder``
    ``forward`` the documented ``_prepare_attention_mask`` helper is never
    called, so the sdpa path silently runs *unmasked global* attention (and a
    batched forward leaks context between samples).  ``masked_attention=True``
    (the default) injects that block-diagonal mask into every encoder layer
    through a documented ``forward_pre_hook``, reproducing the flash-attention
    varlen semantics.  ``masked_attention=False`` keeps the upstream unmasked
    sdpa behaviour for comparison in the probe.
    """

    def __init__(
        self,
        model: Any,
        feature_extractor: Any,
        mel_config: MelConfig,
        report: EncoderLoadReport,
        *,
        masked_attention: bool = True,
    ) -> None:
        self._model = model
        self._feature_extractor = feature_extractor
        self.mel_config = mel_config
        self.load_report = report
        self.masked_attention = bool(masked_attention)
        self.last_attention_info: dict[str, Any] = {}

    def with_masked_attention(self, masked: bool) -> "AutEncoder":
        """Share the loaded model but run with/without the block attention mask."""

        return AutEncoder(
            self._model,
            self._feature_extractor,
            self.mel_config,
            self.load_report,
            masked_attention=masked,
        )

    # -- loading ------------------------------------------------------------ #

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: Path | str,
        *,
        shard_paths: Sequence[Path | str] | None = None,
        device: str = "cpu",
        dtype: str | None = None,
        attn_implementation: str = "sdpa",
    ) -> "AutEncoder":
        directory = Path(model_dir)
        index = CheckpointIndex.load(directory / AUT_INDEX_FILENAME)
        plan = plan_encoder_shards(index)
        if shard_paths is None:
            shard_paths = [directory / name for name in plan.shards]
        model, report = load_filtered_audio_encoder(
            directory,
            shard_paths,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        feature_extractor, mel_config = load_mel_feature_extractor(directory)
        report.mel_config = {
            "feature_size": mel_config.feature_size,
            "sampling_rate": mel_config.sampling_rate,
            "hop_length": mel_config.hop_length,
            "n_fft": mel_config.n_fft,
            "chunk_length": mel_config.chunk_length,
            "n_samples": mel_config.n_samples,
            "nb_max_frames": mel_config.nb_max_frames,
            "dither": mel_config.dither,
            "padding_value": mel_config.padding_value,
        }
        return cls(model, feature_extractor, mel_config, report)

    # -- mel + forward ------------------------------------------------------ #

    def log_mel(self, audio_16k: np.ndarray) -> np.ndarray:
        """Official log-mel ``(128, T)`` for a 16 kHz float array (no truncation)."""

        samples = np.asarray(audio_16k, dtype=np.float32)
        if samples.ndim != 1:
            raise EncoderInputError(f"audio: expected 1-D, got shape {samples.shape}")
        if samples.shape[0] < AUT_MIN_AUDIO_SAMPLES:
            raise EncoderInputError(
                f"audio: {samples.shape[0]} samples is below the {AUT_MIN_AUDIO_SAMPLES}-sample "
                "minimum required by torch.stft(center=True, n_fft=400)"
            )
        mel = self._feature_extractor(
            samples,
            sampling_rate=AUT_SAMPLE_RATE,
            return_attention_mask=False,
            padding=True,  # official processor behaviour: pad to longest, never to n_samples
            truncation=False,
            return_tensors="np",
        )["input_features"]
        mel = np.asarray(mel, dtype=np.float32)[0]
        if mel.shape[1] > self.mel_config.nb_max_frames:
            raise EncoderInputError(
                f"mel length {mel.shape[1]} exceeds the checkpoint maximum "
                f"{self.mel_config.nb_max_frames} frames ({self.mel_config.chunk_length}s); "
                "split the audio into segments instead of truncating silently"
            )
        return mel

    def _install_block_attention_mask(self) -> tuple[list[Any], dict[str, Any]]:
        """Inject the encoder's own block-diagonal mask into every layer.

        The upstream ``forward`` computes ``cu_seqlens`` but never calls
        ``_prepare_attention_mask`` for the sdpa/eager backends; the layer
        signature does accept ``attention_mask``.  A ``forward_pre_hook`` that
        adds the keyword is the minimal way to reproduce the intended FA2
        varlen behaviour without reimplementing the forward pass.
        """

        cache: dict[str, Any] = {}
        model = self._model

        def hook(_module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
            if kwargs.get("attention_mask") is not None:
                return args, kwargs
            if "mask" not in cache:
                hidden_states, cu_seqlens = args[0], args[1]
                cache["mask"] = model._prepare_attention_mask(hidden_states, cu_seqlens)
                cache["cu_seqlens"] = cu_seqlens
            kwargs["attention_mask"] = cache["mask"]
            return args, kwargs

        handles = [layer.register_forward_pre_hook(hook, with_kwargs=True) for layer in model.layers]
        return handles, cache

    def _forward_mel_batch(
        self,
        mel_batch: Sequence[np.ndarray],
        layer: int | None,
    ) -> list[np.ndarray]:
        torch = _import_torch()
        mel_lengths = [int(mel.shape[1]) for mel in mel_batch]
        input_features = np.concatenate(mel_batch, axis=1)
        tensor = torch.from_numpy(np.ascontiguousarray(input_features))
        tensor = tensor.to(device=self.load_report.device, dtype=getattr(torch, self.load_report.dtype))
        lengths = torch.tensor(mel_lengths, dtype=torch.long, device=tensor.device)

        captured: dict[str, Any] = {}
        handles: list[Any] = []
        if layer is not None:
            if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
                raise EncoderInputError(f"layer: expected an integer or None, got {layer!r}")
            total_layers = len(self._model.layers)
            if not 0 <= int(layer) < total_layers:
                raise EncoderInputError(
                    f"layer: must be in [0, {total_layers - 1}] for this checkpoint, got {layer}"
                )
            layer = int(layer)

            def _hook(_module: Any, _inputs: Any, output: Any) -> None:
                captured["hidden"] = output[0] if isinstance(output, tuple) else output

            handles.append(self._model.layers[layer].register_forward_hook(_hook))

        needs_mask = self.masked_attention and self.load_report.attn_implementation != "flash_attention_2"
        mask_cache: dict[str, Any] = {}
        if needs_mask:
            mask_handles, mask_cache = self._install_block_attention_mask()
            handles.extend(mask_handles)
        try:
            with torch.inference_mode():
                outputs = self._model(input_features=tensor, feature_lens=lengths)
                hidden = outputs.last_hidden_state
                if layer is not None:
                    with torch.inference_mode():
                        head = captured["hidden"]
                        hidden = self._model.proj2(
                            self._model.act(self._model.proj1(self._model.ln_post(head)))
                        )
        finally:
            for handle in handles:
                handle.remove()
        mask = mask_cache.get("mask")
        cu_seqlens = mask_cache.get("cu_seqlens")
        self.last_attention_info = {
            "masked_attention": self.masked_attention,
            "mask_installed": bool(mask_cache),
            "mask_shape": list(mask.shape) if mask is not None else None,
            "cu_seqlens": [int(value) for value in cu_seqlens.tolist()] if cu_seqlens is not None else None,
        }

        counts = [aut_output_length(length) for length in mel_lengths]
        pieces = torch.split(hidden, counts, dim=0)
        return [piece.detach().to("cpu", dtype=torch.float32).numpy() for piece in pieces]

    # -- public API --------------------------------------------------------- #

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int = AUT_SAMPLE_RATE,
        *,
        origin_seconds: float = 0.0,
        layer: int | None = None,
    ) -> AutFeatures:
        samples = prepare_audio(audio, sample_rate)
        mel = self.log_mel(samples)
        grid = aut_token_grid(mel.shape[1])
        features = self._forward_mel_batch([mel], layer)[0]
        if features.shape[0] != grid.token_count:
            raise EncoderCheckpointError(
                f"encoder returned {features.shape[0]} tokens, grid says {grid.token_count}"
            )
        valid = token_valid_mask(grid, samples.shape[0])
        return AutFeatures.build(
            features=features,
            valid=valid,
            grid=grid,
            buffer_origin_seconds=float(origin_seconds),
            layer="final" if layer is None else f"layers.{layer}",
            model_id=self.load_report.model_id,
            revision=self.load_report.revision,
            preprocessing=self._preprocessing_snapshot(),
            sample_rate=AUT_SAMPLE_RATE,
        )

    def extract_windows(
        self,
        windows: np.ndarray,
        sample_rate: int = AUT_SAMPLE_RATE,
        *,
        window_start_seconds: np.ndarray,
        valid_samples: np.ndarray | None = None,
        layer: int | None = None,
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

        mels: list[np.ndarray] = []
        resampled: list[np.ndarray] = []
        regions: list[tuple[int, int]] = []
        for index in range(window_array.shape[0]):
            samples = prepare_audio(window_array[index], sample_rate)
            region = _real_region_after_resample(
                valid_array[index], int(sample_rate), AUT_SAMPLE_RATE, samples.shape[0]
            )
            if region is None:
                raise EncoderInputError(
                    f"windows[{index}]: valid_samples is all-False; no real audio to encode"
                )
            mels.append(self.log_mel(samples))
            resampled.append(samples)
            regions.append(region)

        grids = [aut_token_grid(mel.shape[1]) for mel in mels]
        token_counts = {grid.token_count for grid in grids}
        if len(token_counts) != 1:
            raise EncoderInputError(
                "extract_windows requires equal-length windows; got mel lengths "
                f"{[mel.shape[1] for mel in mels]}"
            )
        grid = grids[0]
        feature_rows = self._forward_mel_batch(mels, layer)

        features = np.stack(feature_rows, axis=0)
        valid = np.stack(
            [
                token_valid_mask(grids[i], resampled[i].shape[0], real_start=regions[i][0], real_stop=regions[i][1])
                for i in range(len(mels))
            ],
            axis=0,
        )
        frame_times = starts[:, None] + grid.mel_centers.astype(np.float64)[None, :] * 0.01
        return AutWindowBatch(
            features=features,
            valid=valid,
            frame_times=frame_times,
            window_start_seconds=starts,
            grid=grid,
            layer="final" if layer is None else f"layers.{layer}",
            model_id=self.load_report.model_id,
            revision=self.load_report.revision,
        )

    def _preprocessing_snapshot(self) -> dict[str, Any]:
        return {
            "model_id": self.load_report.model_id,
            "revision": self.load_report.revision,
            "mel_feature_extractor": AUT_MEL_CLASS_NAME,
            "mel": {
                "feature_size": self.mel_config.feature_size,
                "sampling_rate": self.mel_config.sampling_rate,
                "hop_length": self.mel_config.hop_length,
                "n_fft": self.mel_config.n_fft,
                "dither": self.mel_config.dither,
                "padding_value": self.mel_config.padding_value,
            },
            "chunk_mel_frames": 100,
            "token_center_step_seconds": 0.08,
        }

    # -- introspection ------------------------------------------------------ #

    def module_class_names(self) -> list[str]:
        """Classes present in the loaded module tree (used as loading evidence)."""

        return sorted({type(module).__name__ for module in self._model.modules()})
