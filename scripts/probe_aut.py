#!/usr/bin/env python3
"""Real AuT probe: filtered checkpoint download, short forward passes, resources.

Stages
------

``plan``
    Download only the small index/config files, map every ``thinker.audio_tower.*``
    tensor to its shard, report shard bytes vs free disk and the encoder-only
    byte share.  Nothing large is fetched.
``download``
    Plan, then fetch the planned shard(s) plus the small checkpoint configs
    into ``--model-dir`` and report the bytes that were actually transferred.
``probe``
    Load the encoder standalone (never Thinker/Talker/vision) and run the real
    short forward passes, masks/times checks, aligned-window comparison,
    optional layer hook and CUDA latency/memory measurements.
``all``
    ``download`` then ``probe``.

Example (paths are local to the machine; nothing here is committed)::

    python scripts/probe_aut.py all \
        --model-dir "$WORKDIR/qwen3-omni-aut" --cache-dir "$WORKDIR/hf-cache" \
        --output-dir "$WORKDIR/aut-probe" --device cuda:0 --dtype bfloat16

The report is written to ``--output-dir/aut_probe_report.json`` (ignored by
git).  CPU CI never runs this script; it exercises ``aat.encoders.fake`` only.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

try:  # installed package (uv sync)
    from aat.encoders import (
        AUT_ENCODER_TENSOR_BYTES,
        AUT_MODEL_ID,
        AUT_PARAM_COUNT,
        AUT_REVISION,
        AUT_SAMPLE_RATE,
        AUT_TENSOR_COUNT,
        AUT_TENSOR_PREFIX,
        AutEncoder,
        CheckpointIndex,
        check_disk_space,
        compare_aligned_tokens,
        discover_checkpoint_provenance,
        plan_encoder_shards,
    )
    from aat.windowing import extract_windows_at_times
except ModuleNotFoundError:  # standalone checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from aat.encoders import (
        AUT_ENCODER_TENSOR_BYTES,
        AUT_MODEL_ID,
        AUT_PARAM_COUNT,
        AUT_REVISION,
        AUT_SAMPLE_RATE,
        AUT_TENSOR_COUNT,
        AUT_TENSOR_PREFIX,
        AutEncoder,
        CheckpointIndex,
        check_disk_space,
        compare_aligned_tokens,
        discover_checkpoint_provenance,
        plan_encoder_shards,
    )
    from aat.windowing import extract_windows_at_times

INDEX_FILENAME = "model.safetensors.index.json"
SMALL_FILES = (INDEX_FILENAME, "config.json", "preprocessor_config.json")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def synthetic_signal(seconds: float, *, rate: int = AUT_SAMPLE_RATE, seed: int = 20260929) -> "object":
    """Deterministic, non-sensitive multi-tone test signal (no real audio)."""

    import numpy as np

    rng = np.random.default_rng(seed)
    t = np.arange(int(round(seconds * rate)), dtype=np.float64) / rate
    tones = 0.0
    for index, (frequency, amplitude) in enumerate(
        [(220.0, 0.30), (330.0, 0.22), (523.25, 0.18), (1046.5, 0.10)]
    ):
        envelope = 0.55 + 0.45 * np.sin(2 * np.pi * (0.7 + 0.3 * index) * t + index)
        tones = tones + amplitude * envelope * np.sin(2 * np.pi * frequency * t + 0.2 * index)
    noise = 0.03 * rng.standard_normal(t.size)
    return (tones + noise).astype("float32")


def _median(values: list[float]) -> float:
    import numpy as np

    return float(np.median(np.asarray(values, dtype=np.float64)))


def _edge_interior(comparison: dict, *, n_edge: int = 8) -> dict:
    """Split per-token MAE/max/cosine into window-edge and interior tokens."""

    mae_values = comparison["per_token_mae"]
    max_values = comparison["per_token_max_abs_diff"]
    cosines = comparison["per_token_cosine"]
    count = len(max_values)
    n_edge = min(n_edge, count // 2)
    edge_indices = list(range(0, n_edge)) + list(range(count - n_edge, count))
    interior_indices = [i for i in range(count) if i not in set(edge_indices)]

    def _mean(values: list) -> float | None:
        present = [value for value in values if value is not None]
        if not present:
            return None
        import numpy as np

        return float(np.mean(present))

    def _min(values: list) -> float | None:
        present = [value for value in values if value is not None]
        return min(present) if present else None

    return {
        "edge_tokens": len(edge_indices),
        "interior_tokens": len(interior_indices),
        "edge_mean_mae": _mean([mae_values[i] for i in edge_indices]),
        "interior_mean_mae": _mean([mae_values[i] for i in interior_indices]),
        "edge_mean_token_max_abs": _mean([max_values[i] for i in edge_indices]),
        "interior_mean_token_max_abs": _mean([max_values[i] for i in interior_indices]),
        "edge_min_cosine": _min([cosines[i] for i in edge_indices]),
        "interior_min_cosine": _min([cosines[i] for i in interior_indices]),
    }


def _fetch(
    repo_id: str,
    filename: str,
    revision: str,
    cache_dir: Path | None,
    model_dir: Path,
) -> dict:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    target = model_dir / filename
    existed = target.exists()
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_dir=str(model_dir),
    )
    size = Path(path).stat().st_size
    return {"filename": filename, "bytes": size, "reused": bool(existed)}


def _encoder_bytes_in_shard(shard: Path) -> int:
    from safetensors import safe_open  # noqa: PLC0415

    total = 0
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118
            if key.startswith(AUT_TENSOR_PREFIX):
                tensor = handle.get_tensor(key)
                total += int(tensor.numel()) * int(tensor.element_size())
    return total


def _remote_file_sizes(repo_id: str, revision: str) -> dict[str, int]:
    """File sizes from the Hub model metadata (no weight download)."""

    from huggingface_hub import HfApi  # noqa: PLC0415

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    return {
        sibling.rfilename: int(sibling.size)
        for sibling in info.siblings
        if sibling.size is not None
    }


def _plan(report: dict, model_dir: Path, revision: str, repo_id: str) -> None:
    index = CheckpointIndex.load(model_dir / INDEX_FILENAME)
    plan = plan_encoder_shards(index)
    shard_sizes: dict[str, int] = {}
    missing = []
    for name in plan.shards:
        path = model_dir / name
        if path.exists():
            shard_sizes[name] = path.stat().st_size
        else:
            missing.append(name)
    remote_sizes = _remote_file_sizes(repo_id, revision) if missing else {}
    for name in missing:
        if name not in remote_sizes:
            raise SystemExit(f"cannot determine size of shard {name} before downloading")
        shard_sizes[name] = remote_sizes[name]
    checkpoint = report["checkpoint"]
    checkpoint["revision"] = revision
    checkpoint["total_checkpoint_bytes"] = index.metadata.get("total_size")
    checkpoint["shards_needed"] = list(plan.shards)
    checkpoint["shard_tensor_counts"] = dict(plan.tensor_counts)
    checkpoint["shard_file_bytes"] = shard_sizes
    checkpoint["encoder_tensor_bytes_audited"] = AUT_ENCODER_TENSOR_BYTES
    checkpoint["encoder_tensor_count_audited"] = AUT_TENSOR_COUNT
    checkpoint["encoder_share_of_needed_shards"] = (
        AUT_ENCODER_TENSOR_BYTES / sum(shard_sizes.values()) if shard_sizes else None
    )
    check_disk_space(model_dir, sum(shard_sizes.values()))


def run_plan(args: argparse.Namespace, report: dict) -> None:
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    transfers = []
    for filename in SMALL_FILES:
        transfers.append(
            _fetch(args.model_id, filename, args.revision, Path(args.cache_dir) if args.cache_dir else None, model_dir)
        )
    report["small_file_transfers"] = transfers
    _plan(report, model_dir, args.revision, args.model_id)


def run_download(args: argparse.Namespace, report: dict) -> None:
    run_plan(args, report)
    model_dir = Path(args.model_dir)
    plan = plan_encoder_shards(CheckpointIndex.load(model_dir / INDEX_FILENAME))
    transfers = list(report["small_file_transfers"])
    for name in plan.shards:
        transfers.append(
            _fetch(args.model_id, name, args.revision, Path(args.cache_dir) if args.cache_dir else None, model_dir)
        )
    report["shard_transfers"] = transfers
    report["transfer_summary"] = {
        "files": len(transfers),
        "bytes_total": sum(item["bytes"] for item in transfers),
        "bytes_newly_transferred": sum(item["bytes"] for item in transfers if not item["reused"]),
        "bytes_reused": sum(item["bytes"] for item in transfers if item["reused"]),
    }
    encoder_bytes = sum(_encoder_bytes_in_shard(model_dir / name) for name in plan.shards)
    report["checkpoint"]["encoder_tensor_bytes_loaded"] = encoder_bytes
    if encoder_bytes != AUT_ENCODER_TENSOR_BYTES:
        raise SystemExit(
            f"encoder byte audit mismatch: shard headers say {encoder_bytes}, "
            f"audited value is {AUT_ENCODER_TENSOR_BYTES}"
        )


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #


def _cuda_info(torch, device: str) -> dict:
    if not device.startswith("cuda"):
        return {}
    properties = torch.cuda.get_device_properties(int(device.split(":")[1]) if ":" in device else 0)
    return {
        "device_name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "capability": f"{properties.major}.{properties.minor}",
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def _timed(encoder: AutEncoder, audio, *, repeats: int, warmup: int, device: str, torch, **kwargs) -> dict:
    for _ in range(warmup):
        encoder.extract(audio, **kwargs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before_alloc = torch.cuda.memory_allocated()
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        item = encoder.extract(audio, **kwargs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    result = {
        "repeats": repeats,
        "latencies_seconds": latencies,
        "median_seconds": _median(latencies),
        "mean_seconds": float(sum(latencies) / len(latencies)),
        "audio_seconds": float(audio.shape[0] / AUT_SAMPLE_RATE),
    }
    result["realtime_factor"] = result["audio_seconds"] / result["median_seconds"]
    if device.startswith("cuda"):
        result["peak_allocated_delta_bytes"] = int(torch.cuda.max_memory_allocated() - before_alloc)
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    result["tokens"] = item.token_count
    result["feature_dim"] = item.feature_dim
    return result


def run_probe(args: argparse.Namespace, report: dict) -> None:
    import numpy as np
    import torch

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[1]) if ":" in device else 0)
        torch.cuda.reset_peak_memory_stats()

    report["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "safetensors": __import__("safetensors").__version__,
        "numpy": np.__version__,
        **_cuda_info(torch, device),
    }

    start = time.perf_counter()
    encoder = AutEncoder.from_checkpoint(
        args.model_dir,
        device=device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        model_id=args.model_id,
        revision=args.revision,
    )
    load_seconds = time.perf_counter() - start
    load_report = encoder.load_report.to_dict()
    if load_report["revision_source"] != "explicit" or load_report["revision"] != args.revision:
        raise SystemExit(
            "loaded weights are not tied to the requested revision: "
            f"{load_report['revision']!r} / {load_report['revision_source']!r}"
        )
    if load_report["tensor_count"] != AUT_TENSOR_COUNT:
        raise SystemExit(
            f"loaded {load_report['tensor_count']} tensors, audited checkpoint has {AUT_TENSOR_COUNT}"
        )
    if load_report["param_count"] != AUT_PARAM_COUNT:
        raise SystemExit(
            f"loaded {load_report['param_count']} parameters, audited encoder has {AUT_PARAM_COUNT}"
        )
    classes = encoder.module_class_names()
    forbidden = [
        name
        for name in classes
        if "Thinker" in name or "Talker" in name or "Vision" in name or "Code2Wav" in name
    ]
    if forbidden:
        raise SystemExit(f"forbidden modules instantiated: {forbidden}")
    load = {
        "report": load_report,
        "module_class_names": classes,
        "forbidden_module_classes_present": forbidden,
        "load_seconds": load_seconds,
    }
    if device.startswith("cuda"):
        load["peak_allocated_after_load_bytes"] = int(torch.cuda.max_memory_allocated())
        load["peak_reserved_after_load_bytes"] = int(torch.cuda.max_memory_reserved())
    report["load"] = load

    # 1) smoke forward at several lengths (real weights), including the 8 s
    # attention-block boundary: 104 tokens = eight full 1 s chunks; 8.1 s
    # exceeds one block.
    smoke = []
    for seconds in (0.5, 1.0, 2.0, 2.5, 8.0, 8.1, 16.0):
        audio = synthetic_signal(seconds)
        item = encoder.extract(audio, AUT_SAMPLE_RATE)
        timing = _timed(encoder, audio, repeats=args.repeats, warmup=args.warmup, device=device, torch=torch)
        smoke.append(
            {
                "seconds": seconds,
                "samples": int(audio.shape[0]),
                "mel_frames": int(item.grid.mel_len),
                "tokens": item.token_count,
                "feature_dim": item.feature_dim,
                "features_shape": list(item.features.shape),
                "finite": bool(np.isfinite(item.features).all()),
                "valid_tokens": int(item.valid.sum()),
                "frame_time_first": float(item.frame_times[0]) if item.token_count else None,
                "frame_time_last": float(item.frame_times[-1]) if item.token_count else None,
                "feature_mean_abs": float(np.mean(np.abs(item.features))),
                "feature_std": float(np.std(item.features)),
                "feature_max_abs": float(np.max(np.abs(item.features))),
                "attention": dict(encoder.last_attention_info),
                "timing": timing,
            }
        )
    report["forward"] = {"smoke": smoke}
    report["forward"]["preprocessing_snapshot"] = encoder.extract(
        synthetic_signal(0.5)
    ).preprocessing

    discovered = discover_checkpoint_provenance(args.model_dir)
    if discovered.revision and discovered.revision != args.revision:
        raise SystemExit(
            "local download metadata records revision "
            f"{discovered.revision!r} but the probe requested {args.revision!r}"
        )
    report["checkpoint"]["local_download_metadata"] = {
        "revision": discovered.revision or None,
        "source": discovered.source,
        "matches_requested_revision": (
            discovered.revision == args.revision if discovered.revision else None
        ),
    }

    # 2) 44.1 kHz input end to end (renderer rate), non-zero origin, padding
    from aat.encoders import prepare_audio

    audio_44k = synthetic_signal(2.0, rate=44100)
    resampled = prepare_audio(audio_44k, 44100)
    item_44k = encoder.extract(audio_44k, 44100, origin_seconds=30.0)
    report["forward"]["resample_44k1"] = {
        "input_samples": int(audio_44k.shape[0]),
        "resampled_samples": int(resampled.shape[0]),
        "tokens": item_44k.token_count,
        "finite": bool(np.isfinite(item_44k.features).all()),
        "frame_time_first": float(item_44k.frame_times[0]),
        "frame_time_last": float(item_44k.frame_times[-1]),
    }

    track = synthetic_signal(3.0)
    centers = np.array([0.5, 1.5, 2.5])
    windows, valid = extract_windows_at_times(track, centers, AUT_SAMPLE_RATE, 2.0, origin_seconds=0.0)
    starts = centers - 1.0
    batch = encoder.extract_windows(
        windows, AUT_SAMPLE_RATE, window_start_seconds=starts, valid_samples=valid
    )
    padding_rows = []
    for index, center in enumerate(centers):
        inds = np.flatnonzero(batch.valid[index])
        padding_rows.append(
            {
                "center_seconds": float(center),
                "valid_tokens": int(batch.valid[index].sum()),
                "valid_first_index": int(inds[0]) if inds.size else None,
                "valid_last_index": int(inds[-1]) if inds.size else None,
                "frame_time_min": float(batch.frame_times[index].min()),
                "frame_time_max": float(batch.frame_times[index].max()),
                "finite": bool(np.isfinite(batch.features[index]).all()),
            }
        )
    report["forward"]["padded_edge_windows"] = padding_rows

    # 3) independent windows vs full-slice tokens (real attention context)
    full_audio = synthetic_signal(13.0)
    reference = encoder.extract(full_audio)
    reference_attention = dict(encoder.last_attention_info)
    comparisons = []
    for start in (1.0, 3.0, 5.0, 7.0):
        window = full_audio[int(start * AUT_SAMPLE_RATE) : int((start + 2.0) * AUT_SAMPLE_RATE)]
        candidate = encoder.extract(window, origin_seconds=start)
        comparison = compare_aligned_tokens(reference, candidate)
        comparison["start_seconds"] = start
        comparison["aligned_to_chunk_grid"] = True
        comparison.update(_edge_interior(comparison))
        comparisons.append(comparison)
    misaligned = []
    for start in (1.5, 3.5, 5.5):
        window = full_audio[int(start * AUT_SAMPLE_RATE) : int((start + 2.0) * AUT_SAMPLE_RATE)]
        candidate = encoder.extract(window, origin_seconds=start)
        entry = compare_aligned_tokens(reference, candidate)
        entry["start_seconds"] = start
        entry.update(_edge_interior(entry))
        entry["loose_match"] = compare_aligned_tokens(reference, candidate, atol_seconds=0.04)
        misaligned.append(entry)
    report["forward"]["window_vs_slice"] = {
        "full_audio_seconds": 13.0,
        "reference_tokens": reference.token_count,
        "reference_feature_mean_abs": float(np.mean(np.abs(reference.features))),
        "reference_attention": reference_attention,
        "aligned_windows": comparisons,
        "misaligned_windows": misaligned,
    }

    # window batch vs one-by-one (batch must not change values)
    starts = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    batch_windows = np.stack(
        [full_audio[int(s * AUT_SAMPLE_RATE) : int((s + 2.0) * AUT_SAMPLE_RATE)] for s in starts]
    )
    batched = encoder.extract_windows(batch_windows, AUT_SAMPLE_RATE, window_start_seconds=starts)
    masked_attention_info_batch = dict(encoder.last_attention_info)
    per_window = [encoder.extract(batch_windows[i], origin_seconds=float(starts[i])) for i in range(len(starts))]
    batch_diffs = [
        float(np.max(np.abs(batched.features[i] - per_window[i].features))) for i in range(len(starts))
    ]
    report["forward"]["batch_vs_single_max_abs_diff"] = max(batch_diffs)

    # The upstream sdpa path never applies the block-diagonal attention mask;
    # this control shows what that costs for batching and for a long buffer.
    global_encoder = encoder.with_masked_attention(False)
    global_batch = global_encoder.extract_windows(
        batch_windows, AUT_SAMPLE_RATE, window_start_seconds=starts
    )
    global_singles = [
        global_encoder.extract(batch_windows[i], origin_seconds=float(starts[i])) for i in range(len(starts))
    ]
    global_diffs = [
        float(np.max(np.abs(global_batch.features[i] - global_singles[i].features)))
        for i in range(len(starts))
    ]
    global_full = global_encoder.extract(full_audio)
    report["forward"]["attention_mode_control"] = {
        "masked_batch_vs_single_max_abs_diff": max(batch_diffs),
        "global_batch_vs_single_max_abs_diff": max(global_diffs),
        "masked_vs_global_13s_full_max_abs_diff": float(
            np.max(np.abs(reference.features - global_full.features))
        ),
        "masked_attention_info_batch": masked_attention_info_batch,
        "global_attention_info_batch": dict(global_encoder.last_attention_info),
    }

    # Float32 control: separates bf16 rounding from actual batching effects.
    if device.startswith("cuda") and args.dtype != "float32":
        fp32 = AutEncoder.from_checkpoint(
            args.model_dir, device=device, dtype="float32", attn_implementation=args.attn_implementation
        )
        try:
            fp32_batch = fp32.extract_windows(batch_windows, AUT_SAMPLE_RATE, window_start_seconds=starts)
            fp32_singles = [
                fp32.extract(batch_windows[i], origin_seconds=float(starts[i])) for i in range(len(starts))
            ]
            report["forward"]["dtype_control"] = {
                "float32_batch_vs_single_max_abs_diff": max(
                    float(np.max(np.abs(fp32_batch.features[i] - fp32_singles[i].features)))
                    for i in range(len(starts))
                ),
                "bf16_vs_float32_batch_max_abs_diff": float(
                    np.max(np.abs(batched.features - fp32_batch.features))
                ),
                "bf16_batch_vs_single_max_abs_diff": max(batch_diffs),
            }
        finally:
            del fp32
            torch.cuda.empty_cache()

    # 4) optional intermediate layer through the documented hook
    layer = args.layer
    if layer is not None:
        plain = encoder.extract(synthetic_signal(2.0))
        hooked = encoder.extract(synthetic_signal(2.0), layer=layer)
        last = len(encoder._model.layers) - 1  # noqa: SLF001 - probe introspection
        last_hook = encoder.extract(synthetic_signal(2.0), layer=last)
        report["forward"]["layer"] = {
            "layer": layer,
            "feature_dim": hooked.feature_dim,
            "tokens": hooked.token_count,
            "finite": bool(np.isfinite(hooked.features).all()),
            "last_layer_index": last,
            "last_layer_hook_vs_plain_max_abs_diff": float(
                np.max(np.abs(last_hook.features - plain.features))
            ),
        }

    # 5) batched throughput + peak memory
    batch_timing = _timed_windows(
        encoder, batch_windows, starts, repeats=args.repeats, warmup=args.warmup, device=device, torch=torch
    )
    report["forward"]["batch_timing_8x2s"] = batch_timing

    # keep one real feature bundle as evidence next to the report
    sample = encoder.extract(synthetic_signal(2.5))
    np.savez(
        Path(args.output_dir) / "aut_probe_sample.npz",
        features=sample.features,
        valid=sample.valid,
        frame_times=sample.frame_times,
    )
    report["sample_artifact"] = "aut_probe_sample.npz"
    report["notes"] = [
        "Token times are the mel receptive-field centres at 10 ms mel hop; the "
        "nominal centre step inside a chunk is 80 ms but chunk boundaries add a "
        "40 ms gap. These are not 10 ms-precision timings.",
        "All audio here is synthetic and generated in-process; no music or personal data is involved.",
    ]


def _timed_windows(
    encoder: AutEncoder,
    windows,
    starts,
    *,
    repeats: int,
    warmup: int,
    device: str,
    torch,
) -> dict:
    for _ in range(warmup):
        encoder.extract_windows(windows, AUT_SAMPLE_RATE, window_start_seconds=starts)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before_alloc = torch.cuda.memory_allocated()
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        batch = encoder.extract_windows(windows, AUT_SAMPLE_RATE, window_start_seconds=starts)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    result = {
        "windows": int(windows.shape[0]),
        "window_seconds": 2.0,
        "latencies_seconds": latencies,
        "median_seconds": _median(latencies),
        "audio_seconds": float(windows.shape[0] * windows.shape[1] / AUT_SAMPLE_RATE),
    }
    result["realtime_factor"] = result["audio_seconds"] / result["median_seconds"]
    if device.startswith("cuda"):
        result["peak_allocated_delta_bytes"] = int(torch.cuda.max_memory_allocated() - before_alloc)
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-Omni AuT standalone probe (issue #5)")
    parser.add_argument("--model-id", default=AUT_MODEL_ID)
    parser.add_argument("--revision", default=AUT_REVISION)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa", choices=("sdpa", "eager"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--layer", type=int, default=None, help="optional layer hook index (0-based)")
    parser.add_argument("--stage", default="all", choices=("plan", "download", "probe", "all"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report: dict = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": args.stage,
        "checkpoint": {"model_id": args.model_id, "revision": args.revision},
    }

    if args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.stage in ("plan", "download"):
        (run_plan if args.stage == "plan" else run_download)(args, report)
    elif args.stage in ("probe", "all"):
        if args.output_dir is None:
            raise SystemExit("--output-dir is required for probe/all")
        if args.stage == "all":
            run_download(args, report)
        run_probe(args, report)

    if args.output_dir:
        output = Path(args.output_dir) / "aut_probe_report.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
