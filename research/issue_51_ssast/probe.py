"""P0 only: validate coordinates and costs; never starts training."""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import timm

from features import FrozenSSAST, log_fbank
from aat.data.curriculum import prepare_c0
from aat.labels.wav import read_wav
from aat.labels.energy import mean_square_at_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", choices=["frame", "patch"], required=True)
    parser.add_argument("--checkpoint-origin", choices=["unverified", "user-provided", "official-download"], default="unverified")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(51)
    encoder = FrozenSSAST(args.source, args.checkpoint, model_type=args.model_type)
    data = args.out / "c0"
    if not data.exists():
        prepare_c0(data, counts=(1, 0, 0))
    wave = read_wav(data / "c0-train-00" / "mix.wav")
    mono = torch.tensor(wave.samples.mean(axis=1), dtype=torch.float32)
    mono = torchaudio.functional.resample(mono, wave.sample_rate, 16000)
    # Real source note at 1.2s, with an explicit absolute origin.
    crop = mono[16000:48000].unsqueeze(0)
    zeros = torch.zeros_like(crop)
    impulse = zeros.clone()
    impulse[0, 16000] = 1
    step = zeros.clone()
    step[0, 16000:] = .1
    shifted = torch.roll(crop, 160, -1)
    boundary = torch.cat((zeros[:, :16000], mono[:16000].unsqueeze(0)), dim=1)
    waves = [crop, crop * .5, crop * 2, zeros, impulse, step, shifted, boundary]
    names = ["real", "gain_half", "gain_double", "silence", "impulse",
             "step", "shift_10ms", "left_boundary_pad"]
    features = torch.stack([log_fbank(w) for w in waves])
    assert features.shape == (8, 198, 128), features.shape
    torch.cuda.reset_peak_memory_stats()
    output = encoder(features)
    assert torch.isfinite(output).all()
    repeated = encoder(features)
    repeat_error = float((repeated - output).abs().max())
    costs = {}
    for batch in (1, 4):
        x = features[:batch]
        for _ in range(5):
            encoder(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            encoder(x)
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        costs[str(batch)] = dict(batch_seconds=duration / 20,
                                per_window_seconds=duration / 20 / batch)
    f, t, centers, starts, ends = encoder.grid
    # Check the real discrete convolution support, independently from grid().
    discrete_start = np.arange(t) * encoder.manifest["stride"][1] * 160
    discrete_end = discrete_start + 400 + 160 * (encoder.manifest["kernel"][1] - 1)
    mapped = (discrete_start + discrete_end) / 2 / 16000
    coord_error = float(np.abs(mapped - centers).max())
    assert coord_error <= 1 / 16000
    expected_f = torch.arange(f).repeat_interleave(t)
    expected_t = torch.arange(t).repeat(f)
    order = torch.arange(f * t).reshape(f, t)
    assert torch.equal(order[expected_f, expected_t], torch.arange(f * t))
    # Absolute RMS bypass is measured on original multichannel full-scale input.
    original_indices = np.round((1. + centers) * wave.sample_rate).astype(np.int64)
    rms = np.sqrt(mean_square_at_samples(wave.samples, wave.sample_rate, .02, original_indices))
    valid_waveforms = np.ones((8, 32000), dtype=bool)
    valid_waveforms[-1, :16000] = False
    support_valid_fraction = np.stack([
        np.array([mask[a:b].mean() for a, b in zip(discrete_start, discrete_end)])
        for mask in valid_waveforms])
    response = []
    for i, name in enumerate(names):
        response.append(dict(name=name, finite=bool(torch.isfinite(output[i]).all()),
            feature_norm_mean=float(output[i].norm(dim=-1).mean()),
            difference_from_real=float((output[i] - output[0]).norm(dim=-1).mean()),
            valid_waveform_fraction=.5 if name == "left_boundary_pad" else 1.,
            origin_seconds=-1. if name == "left_boundary_pad" else 1.))
    np.savez_compressed(args.out / "tokens.npz", tokens=output.cpu().numpy(),
        token_centers_local=centers, token_support_starts=starts,
        token_support_ends=ends, absolute_rms_real=rms)
    np.savez_compressed(args.out / "validity.npz", waveform_valid=valid_waveforms,
        token_projection_valid_fraction=support_valid_fraction,
        token_projection_complete=support_valid_fraction == 1.,
        full_context_valid=valid_waveforms.all(axis=1))
    report = dict(stage="P0", model_type=args.model_type,
        scope="unpooled feature diagnostics; not training or source separation",
        manifest=encoder.manifest,
        runtime=dict(python=platform.python_version(), torch=torch.__version__,
                     torchaudio=torchaudio.__version__, timm=timm.__version__,
                     device=torch.cuda.get_device_name(), threads=4),
        fbank_shape=list(features.shape), token_shape=list(output.shape),
        first_center=float(centers[0]), last_center=float(centers[-1]),
        coordinate_error_seconds=coord_error, repeat_max_error=repeat_error,
        gain_probe=response, costs=costs,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        validity_note="projection support masks do not restrict attention; all tokens see the full 2s input",
        original_sample_rate=wave.sample_rate,
        bypass_rms_unit="linear_rms_full_scale; original channel power average",
        minute_track_estimate_seconds_at_hop_20ms=3000 * costs["4"]["per_window_seconds"],
        checkpoint_origin=args.checkpoint_origin,
        provenance_note="origin is declared and recorded; a supplied file is not independently authenticated as the author's original download",
        frame_stage_gate_passed=args.model_type == "frame" and args.checkpoint_origin != "unverified")
    (args.out / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("model_type", "token_shape",
          "coordinate_error_seconds", "repeat_max_error", "costs", "frame_stage_gate_passed")}), flush=True)


if __name__ == "__main__":
    main()
