"""Analyze saved P1 outputs without training or changing acceptance thresholds."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fit', type=Path, required=True)
    args = p.parse_args()
    output = {}
    for seed in (46, 47, 48):
        directory = args.fit / f'seed{seed}'
        result = json.loads((directory / 'result.json').read_text())
        output[str(seed)] = {}
        for split in ('val', 'test'):
            rows = []
            for sample in result['splits'][split]['samples']:
                z = np.load(directory / (sample['sample_id'] + '.npz'), allow_pickle=False)
                a, truth = z['raw_a'], z['target'][:, 0]
                unused = np.arange(8) != sample['slot']
                thresholds = {}
                for threshold in (.0005, .001, .002):
                    gated = np.where(a > threshold, a, 0)
                    thresholds[str(threshold)] = dict(
                        unused_false_positive_fraction=float((a[:, unused] > threshold).mean()),
                        source_count_mae=float(np.abs((gated > 0).sum(1) - (truth > threshold)).mean()),
                        extra_activity_area_fraction=float(gated[:, unused].sum() / truth.sum()))
                norms = np.linalg.norm(a, axis=0)
                cosine = a.T @ a / (norms[:, None] * norms[None, :]).clip(1e-12)
                areas = a.sum(0)
                copy_pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)
                    if cosine[i, j] > .99 and min(areas[i], areas[j]) > .05 * truth.sum()]
                rows.append(dict(sample_id=sample['sample_id'], thresholds=thresholds,
                    raw_extra_area_fraction=float(a[:, unused].sum() / truth.sum()),
                    global_shape_copy_pairs=copy_pairs))
            output[str(seed)][split] = rows
    (args.fit / 'sensitivity.json').write_text(json.dumps(output, indent=2) + '\n')
    directory = args.fit / 'seed46'
    result = json.loads((directory / 'result.json').read_text())
    sample = result['splits']['val']['samples'][0]
    z = np.load(directory / (sample['sample_id'] + '.npz'), allow_pickle=False)
    times, a, truth = z['times'], z['raw_a'], z['target'][:, 0]
    slot = sample['slot']
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(times, truth, color='black', linewidth=2, label='Reference stem RMS')
    axes[0].plot(times, a[:, slot], color='#2674b7', label='Matched candidate raw A')
    axes[0].set_ylabel('Linear RMS (full scale)')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Frozen SSAST frame + RMS bypass, C0 validation, seed46')
    for k in range(8):
        if k != slot:
            axes[1].plot(times, a[:, k], alpha=.75, label=f'Unused slot {k}')
    axes[1].axhline(.001, color='black', linestyle='--', label='Fixed gate 0.001')
    axes[1].set_ylim(0, .005)
    axes[1].set_xlabel('Original audio time (s)')
    axes[1].set_ylabel('Unused candidate raw A')
    axes[1].legend(ncol=4, fontsize=8)
    fig.savefig(args.fit / 'seed46-val-curves.png', dpi=160)


if __name__ == '__main__':
    main()
