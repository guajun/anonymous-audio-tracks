"""Dedicated two-source difficulty axes using the shared renderer/labels."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from aat.data.curriculum import prepare_c1_local, c1_local_config
from aat.data.local_context import audit_local_context
from aat.labels.envelope import label_sample, overlap_ratio
from aat.labels.wav import read_wav
from aat.render.config import config_from_dict
from aat.render.pipeline import render_sample
from features import sha256


def config(stage, seed, sample_id):
    base, onsets, owners = c1_local_config(seed, sample_id)
    sources = []
    rng = np.random.default_rng(seed + 5151)
    for i, old in enumerate(base.sources):
        s = copy.deepcopy(old.to_dict())
        s["id"] = s.pop("source_id")
        s.pop("index")
        if stage in ("C2-low", "C2-high"):
            s["amp"]["release_ms"] = 180. if stage == "C2-low" else 260.
            for note in s["pattern"]["notes"]:
                note["length_steps"] = 7
        elif stage == "C3":
            s["amp"].update(attack_ms=60. if i else 10.,
                decay_ms=60. if i else 100.,
                sustain=.85 if i else .35, release_ms=120. if i else 100.)
            for note in s["pattern"]["notes"]:
                note["length_steps"] = int(rng.choice([19, 20, 21]))
                note["velocity"] = int(rng.integers(70, 116))
            if i == 1:
                # The fixed pluck sample decays in a few milliseconds;
                # increasing MIDI length alone did not create sustained sound.
                # Repeated unchanged plucks form ONE anonymous source burst,
                # with independently increasing velocity. No new timbre.
                s['amp'].update(attack_ms=5., decay_ms=20., sustain=.85, release_ms=40.)
                bursts = []
                for note in s['pattern']['notes']:
                    for j in range(10):
                        bursts.append(dict(step=note['step'] + 2*j, note=60,
                            velocity=min(115, 70 + 5*j + int(rng.integers(-3, 4))), length_steps=1))
                s['pattern']['notes'] = bursts
        else:
            raise ValueError(stage)
        sources.append(s)
    cfg = config_from_dict(dict(render=dict(sample_id=sample_id, composition=sample_id,
        seed=seed, sample_rate=44100, block_size=512, bpm=100,
        duration_seconds=12., tail_seconds=1.), sources=sources))
    return cfg, onsets, owners


def prepare(stage, out, seed, counts=(4, 2, 2)):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite curriculum")
    if stage == "C1":
        original = prepare_c1_local(out, seed=seed, counts=counts)
        entries = original["entries"]
    else:
        out.mkdir(parents=True, exist_ok=True)
        entries = []
        for split, count in zip(("train", "val", "test"), counts):
            for i in range(count):
                sid = f'{stage.lower()}-{split}-{i:02d}'
                song_seed = seed + len(entries) * 101
                cfg, onset, owner = config(stage, song_seed, sid)
                d = out / sid
                rendered = render_sample(cfg, d)
                labels = label_sample(d)
                w = read_wav(d / 'mix.wav')
                centers = labels.center_times[(labels.center_times >= 1-1e-9) & (labels.center_times <= 11+1e-9)]
                audit, arrays = audit_local_context(labels, centers, audio_frames=w.frames)
                # Require actual complete local events; if sustained events
                # merge or the context contract fails, do not train on them.
                if not audit['passed']:
                    raise ValueError(f'{stage} local context failed: {audit}')
                np.savez_compressed(d / 'local-context.npz', **arrays)
                (d / 'local-context.json').write_text(json.dumps(audit, indent=2)+'\n')
                entries.append(dict(sample_id=sid, split=split, directory=sid, seed=song_seed,
                    onsets_seconds=onset.tolist(), owners=[cfg.sources[j].source_id for j in owner],
                    amp=[s.amp.to_dict() for s in cfg.sources], gain=[s.gain for s in cfg.sources],
                    local_context=audit, stem_sum=rendered.report['stem_sum'],
                    mix_sha256=sha256(d / 'mix.wav'), envelope_sha256=sha256(d / 'envelope.npz')))
    for e in entries:
        labels = label_sample(out / e['directory'])
        score = (labels.center_times >= 1-1e-9) & (labels.center_times <= 11+1e-9)
        a = labels.rms[score] > .001
        count = a.sum(1)
        e['source_count'] = len(labels.source_ids)
        e['actual_overlap_ratio_all'] = overlap_ratio(labels.rms[labels.valid])
        e['actual_overlap_ratio_scored'] = float((count >= 2).sum() / max(1, (count >= 1).sum()))
        e['scored_overlap_centers'] = int((count >= 2).sum())
        e['scoring_start_seconds'], e['scoring_end_seconds'] = 1., 11.
        e['source_events'] = e['local_context']['sources']
    manifest = dict(version='issue51-courses-v1', stage=stage, seed=seed, counts=list(counts),
        anonymous_source_semantics='one pad stem and one pluck stem; notes within a stem are one source',
        fixed_timbres='pad/pluck; no chord, similar-timbre or effects axis',
        ambiguity_policy='completely synchronous indistinguishable shapes excluded from this initial set',
        entries=entries)
    (out / 'course-index.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(stage=stage, overlaps=[e['actual_overlap_ratio_scored'] for e in entries])), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['C1', 'C2-low', 'C2-high', 'C3'])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for stage, seed in (('C1', 4640), ('C2-low', 51520), ('C2-high', 51540), ('C3', 51560)):
        if args.stage and args.stage != stage:
            continue
        prepare(stage, args.out / stage, seed)


if __name__ == '__main__':
    main()
