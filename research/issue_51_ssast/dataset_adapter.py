"""Reuse curriculum/envelope APIs and cache actual 2s frozen features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from features import FrozenSSAST, log_fbank, sha256
from aat.data.curriculum import prepare_c0
from aat.labels.envelope import EnvelopeData
from aat.labels.wav import read_wav
from aat.labels.energy import mean_square_at_samples


def crop_two_seconds(audio, center):
    index = int(np.floor(float(center) * 16000 + .5))
    left, right = index - 16000, index + 16000
    result = torch.zeros(1, 32000)
    lo, hi = max(0, left), min(len(audio), right)
    if hi > lo:
        result[0, lo - left:hi - left] = audio[lo:hi]
    return result, (hi - lo) / 32000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    data = args.out / "c0-data"
    if not data.exists():
        prepare_c0(data)  # Default 4/2/2, renderer seed4600.
    index = json.loads((data / "index.json").read_text())
    enc = FrozenSSAST(args.source, args.checkpoint, model_type="frame")
    rows = []
    for entry in index["entries"]:
        directory = data / entry["directory"]
        labels = EnvelopeData.load(directory)
        wav = read_wav(directory / "mix.wav")
        centers = labels.center_times[labels.valid]
        out = args.out / (entry["sample_id"] + ".npz")
        if out.exists():
            raise ValueError("refusing to reuse unvalidated cache: " + str(out))
        mono = torch.tensor(wav.samples.mean(axis=1), dtype=torch.float32)
        mono = torchaudio.functional.resample(mono, wav.sample_rate, 16000)
        tokens, rms_sequence, valid_fraction = [], [], []
        for offset in range(0, len(centers), args.batch):
            cc = centers[offset:offset + args.batch]
            crops = [crop_two_seconds(mono, c) for c in cc]
            fb = torch.stack([log_fbank(x[0]) for x in crops])
            tokens.append(enc(fb).mean(dim=1).cpu().numpy().astype(np.float16))
            for center, (_, frac) in zip(cc, crops):
                absolute = center - 1. + enc.grid[2]
                positions = np.floor(absolute * wav.sample_rate + .5).astype(np.int64)
                rms_sequence.append(np.sqrt(mean_square_at_samples(
                    wav.samples, wav.sample_rate, .02, positions)))
                valid_fraction.append(frac)
        indices = np.floor(centers * wav.sample_rate + .5).astype(np.int64)
        oracle = np.sqrt(mean_square_at_samples(wav.samples, wav.sample_rate, .02, indices))
        np.savez(out, tokens=np.concatenate(tokens),
            token_rms=np.asarray(rms_sequence, dtype=np.float32),
            center_times=centers, targets=labels.rms[labels.valid],
            context_valid_fraction=np.asarray(valid_fraction),
            mix_rms_oracle=oracle.astype(np.float32), local_token_centers=enc.grid[2])
        row = dict(sample_id=entry["sample_id"], split=entry["split"],
            cache=out.name, cache_sha256=sha256(out), centers=len(centers),
            mix_sha256=entry["mix_sha256"], envelope_sha256=entry["envelope_sha256"],
            boundary_windows=int((np.array(valid_fraction) < 1).sum()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    manifest = dict(stage="P1 C0 feature cache", encoder=enc.manifest,
        renderer_index_sha256=sha256(data / "index.json"),
        representation="frequency mean only; all 197 time tokens retained as float16",
        head_center_seconds=1., context_seconds=2., hop_seconds=.02,
        rms_bypass="original-channel power average, 20ms centered measurement",
        entries=rows)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
