"""Cache two-source course inputs at audited original-time centers."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torchaudio

from dataset_adapter import crop_two_seconds
from features import FrozenSSAST, log_fbank, sha256
from aat.labels.envelope import EnvelopeData
from aat.labels.wav import read_wav
from aat.labels.energy import mean_square_at_samples
from aat.data.local_context import audit_local_context


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--full-window', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    enc = FrozenSSAST(args.source, args.checkpoint, model_type='frame')
    for stage in ('C1', 'C2-low', 'C2-high', 'C3'):
        root = args.data / stage
        if stage == 'C3':
            root = args.data.parent / 'data-v2' / 'C3'
        index = json.loads((root / 'course-index.json').read_text())
        out = args.out / stage
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        start = time.monotonic()
        for entry in index['entries']:
            dest = out / (entry['sample_id']+'.npz')
            d = root / entry['directory']
            labels = EnvelopeData.load(d)
            wav = read_wav(d/'mix.wav')
            # Training uses the complete nonpadding 1..11 scoring range,
            # default20ms grid, labels unaltered, every real2s input audited.
            valid = (labels.center_times >= 1-1e-9)&(labels.center_times <= 11+1e-9)
            centers = labels.center_times[valid]
            audit, _ = audit_local_context(labels, centers, audio_frames=wav.frames)
            if not audit['passed']:
                raise ValueError('actual inference input evidence failed')
            mono = torch.tensor(wav.samples.mean(1), dtype=torch.float32)
            mono = torchaudio.functional.resample(mono, wav.sample_rate, 16000)
            features, bypass = [], []
            for offset in range(0, len(centers), 16):
                cc = centers[offset:offset+16]
                crops = [crop_two_seconds(mono, c) for c in cc]
                if any(frac != 1 for _, frac in crops):
                    raise ValueError('scored course window padding')
                fb = torch.stack([log_fbank(w) for w, _ in crops])
                # Head-receptive-field subset, with full-window encoder attention.
                indices = np.arange(197) if args.full_window else np.arange(96,102)
                features.append(enc(fb).mean(1)[:, indices].cpu().numpy().astype(np.float16))
                for c in cc:
                    pos = np.floor((c-1+enc.grid[2][indices])*wav.sample_rate+.5).astype(np.int64)
                    bypass.append(np.sqrt(mean_square_at_samples(wav.samples, wav.sample_rate, .02, pos)))
            mix = np.sqrt(mean_square_at_samples(wav.samples, wav.sample_rate, .02,
                np.floor(centers*wav.sample_rate+.5).astype(np.int64)))
            np.savez(dest, tokens=np.concatenate(features), token_rms=np.array(bypass,dtype=np.float32),
                center_times=centers, targets=labels.rms[valid], mix_rms_oracle=mix.astype(np.float32))
            row = dict(**entry, cache=dest.name, cache_sha256=sha256(dest),
                actual_center_audit=audit, head_token_indices=indices.tolist())
            rows.append(row)
            print(stage, entry['sample_id'], flush=True)
        manifest = dict(stage=stage, encoder=enc.manifest, data_index=index,
            elapsed_seconds=time.monotonic()-start, entries=rows,
            head_receptive_field_only=not args.full_window, global_encoder_attention_seconds=2.)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
