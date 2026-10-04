#!/usr/bin/env python3
"""Render model-independent curricula or label an existing rendered sample."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    defaults = {'prepare-c0': 4600, 'prepare-c1': 4610, 'prepare-c1-local': 4640}
    for name, seed in defaults.items():
        command = commands.add_parser(name)
        command.add_argument('--out', required=True)
        command.add_argument('--seed', type=int, default=seed)
        command.add_argument('--counts', nargs=3, type=int, default=(4, 2, 2),
                             metavar=('TRAIN', 'VAL', 'TEST'))
    label = commands.add_parser('label')
    label.add_argument('--sample', required=True)
    label.add_argument('--hop-seconds', type=float, default=.02)
    label.add_argument('--energy-window-seconds', type=float, default=.02)
    args = parser.parse_args(argv)
    if args.command == 'label':
        from aat.labels.envelope import label_sample
        result = label_sample(args.sample, hop_seconds=args.hop_seconds,
                              energy_window_seconds=args.energy_window_seconds)
        print(f'continuous envelopes: {result.rms.shape}; linear RMS, original time axis')
    else:
        from aat.data.curriculum import prepare_c0, prepare_c1, prepare_c1_local
        prepare = dict(zip(defaults, (prepare_c0, prepare_c1, prepare_c1_local)))[args.command]
        result = prepare(args.out, seed=args.seed, counts=tuple(args.counts))
        print(f"completed {result['stage']}: {len(result['entries'])} samples")


if __name__ == '__main__':
    main()
