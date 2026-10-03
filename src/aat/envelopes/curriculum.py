"""Small deterministic C0 corpus using the existing DawDreamer renderer."""
from __future__ import annotations

from pathlib import Path
import copy
import hashlib
import numpy as np
from aat.contracts.jsonio import dump_json
from aat.render.config import config_from_dict
from aat.render.pipeline import render_sample
from aat.labels.wav import read_wav
from .labels import label_sample, overlap_ratio
from .local_context import audit_local_context,CONSTRAINT_VERSION


def c0_config(seed: int, sample_id: str):
    rng = np.random.default_rng(seed)
    # One voice shared intentionally for C0 engineering validation. Parameters
    # and performances are held out; this is not an unseen-timbre benchmark.
    amp = {"attack_ms": float(rng.choice([10, 40, 100, 200])),
           "decay_ms": float(rng.choice([40, 100, 200])),
           "sustain": float(rng.choice([0.35, 0.6, 0.9])),
           "release_ms": float(rng.choice([100, 200, 400]))}
    notes = [{"step": step, "note": 60, "velocity": int(rng.integers(65, 116)),
              "length_steps": int(rng.integers(3, 6))} for step in (12, 40, 68)]
    return config_from_dict({
        "render": {"sample_id": sample_id, "composition": sample_id, "seed": seed,
                   "sample_rate": 44100, "block_size": 512, "bpm": 100,
                   "duration_seconds": 8.0, "tail_seconds": 2.0},
        "sources": [{"id": "s01", "gain": float(rng.uniform(0.3, 0.65)), "center_note": 60,
                     "preset_ref": "aat/c0/fixed-pad-v1", "sample_ref": "generated/c0/fixed-pad-v1",
                     "sample": {"type": "pad", "seed": 46,
                                "params": {"duration_seconds": 1.5, "attack_seconds": 0.05,
                                           "release_seconds": 0.15, "freq_hz": 261.626,
                                           "detune_cents": 0.0, "stereo": False}},
                     "amp": amp, "pattern": {"step_seconds": 0.1, "notes": notes}}],
    })


def prepare_c0(out: str | Path, *, seed=4600, counts=(4,2,2)):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("C0 output must be empty; refusing to overwrite prior data")
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for split, count in zip(("train", "val", "test"), counts):
        for index in range(count):
            sample_id = f"c0-{split}-{index:02d}"
            song_seed = seed + len(entries)*101
            config = c0_config(song_seed, sample_id)
            directory = out / sample_id
            rendered = render_sample(config, directory)
            labels = label_sample(directory)
            # Notes are spaced 2.8s, exceeding sample duration + maximum release.
            ratio = overlap_ratio(labels.rms[labels.valid])
            if ratio != 0.0:
                raise ValueError("C0 contains cross-source overlap")
            before_peaks=[]
            for onset in (1.2,4.0,6.8):
                before=(labels.center_times>=onset-0.15)&(labels.center_times<onset-0.06)
                peak=float(labels.rms[before].max())
                before_peaks.append(peak)
                if peak>1e-3:
                    raise ValueError("C0 has an audible tail before the next note")
            entries.append({"sample_id": sample_id, "split": split, "directory": sample_id,
                            "seed": song_seed, "amp": config.sources[0].amp.to_dict(),
                            "gain": config.sources[0].gain, "acoustic_overlap_ratio": ratio,
                            "minimum_note_gap_seconds": 2.8,
                            "max_sample_plus_release_seconds": 1.5 + config.sources[0].amp.release_ms/1000,
                            "pre_note_rms_peaks": before_peaks,
                            "mix_sha256": hashlib.sha256((directory/"mix.wav").read_bytes()).hexdigest(),
                            "envelope_sha256": hashlib.sha256((directory/"envelope.npz").read_bytes()).hexdigest(),
                            "stem_sum": rendered.report["stem_sum"]})
    index = {"version": "aat-curriculum-v1", "stage": "C0", "seed": seed,
             "split_policy": "held-out sequence/ADSR/gain; shared fixed voice intentional; not unseen timbre",
             "entries": entries}
    dump_json(out / "index.json", index)
    return index


def c1_config(seed: int, sample_id: str):
    """Two fixed timbres, alternating with acoustic gaps and A-B-A recurrence."""
    rng=np.random.default_rng(seed)
    initial=c0_config(seed,sample_id)
    base=initial.sources[0].to_dict()
    base['id']=base.pop('source_id')
    base.pop('index')
    sources=[copy.deepcopy(base),copy.deepcopy(base)]
    onsets=np.array([12,40,68,96,124])+rng.integers(-1,2,size=5)
    first=int(rng.integers(2))
    owners=[(first+i)%2 for i in range(5)]
    sources[1].update(id='s02',preset_ref='aat/c1/fixed-pluck-v1',sample_ref='generated/c1/fixed-pluck-v1',
                      sample={'type':'pluck','seed':47,'params':{'duration_seconds':1.5,'freq_hz':261.626,'decay_seconds':.8,'damping':.5}})
    sources[0]['preset_ref']='aat/c1/fixed-pad-v1'
    sources[0]['sample_ref']='generated/c1/fixed-pad-v1'
    for source_index,source in enumerate(sources):
        source['gain']=float(rng.uniform(.3,.65))
        source['amp']={'attack_ms':float(rng.choice([10,40,100])), 'decay_ms':100.,
                       'sustain':float(rng.choice([.35,.6,.9])),'release_ms':float(rng.choice([100,200,400]))}
        source['pattern']={'step_seconds':.1,'notes':[
            {'step':int(onset),'note':60,'velocity':int(rng.integers(65,116)),'length_steps':int(rng.integers(3,6))}
            for onset,owner in zip(onsets,owners) if owner==source_index]}
    config=config_from_dict({'render':{'sample_id':sample_id,'composition':sample_id,'seed':seed,
                                      'sample_rate':44100,'block_size':512,'bpm':100,'duration_seconds':14.,'tail_seconds':2.},
                             'sources':sources})
    return config,onsets*.1,owners


def prepare_c1(out: str | Path, *, seed=4610, counts=(4,2,2)):
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('C1 output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    entries=[]
    for split,count in zip(('train','val','test'),counts):
        for index in range(count):
            sample_id=f'c1-{split}-{index:02d}'
            song_seed=seed+len(entries)*101
            config,onsets,owners=c1_config(song_seed,sample_id)
            directory=out/sample_id
            rendered=render_sample(config,directory)
            labels=label_sample(directory)
            overlap=overlap_ratio(labels.rms[labels.valid])
            if overlap!=0:
                raise ValueError('C1 contains acoustic overlap')
            peaks=[]
            for onset in onsets:
                before=(labels.center_times>=onset-.15)&(labels.center_times<onset-.06)
                peaks.append(float(labels.rms[before].max()))
            if max(peaks)>1e-3:
                raise ValueError('C1 contains an audible tail before the next note')
            entries.append({'sample_id':sample_id,'split':split,'directory':sample_id,'seed':song_seed,
                            'onsets_seconds':onsets.tolist(),'owners':[config.sources[i].source_id for i in owners],
                            'minimum_note_gap_seconds':float(np.diff(onsets).min()),'acoustic_overlap_ratio':overlap,
                            'pre_note_rms_peaks':peaks,'amp':[s.amp.to_dict() for s in config.sources],
                            'gain':[s.gain for s in config.sources],
                            'mix_sha256':hashlib.sha256((directory/'mix.wav').read_bytes()).hexdigest(),
                            'envelope_sha256':hashlib.sha256((directory/'envelope.npz').read_bytes()).hexdigest(),
                            'stem_sum':rendered.report['stem_sum']})
    index={'version':'aat-curriculum-v1','stage':'C1','seed':seed,
           'split_policy':'held-out sequence/ADSR/gain; two fixed voices shared intentionally; not unseen timbre; alternating start/jittered onsets',
           'entries':entries}
    dump_json(out/'index.json',index)
    return index


def c1_local_config(seed: int,sample_id: str):
    """Dense, short alternating notes with true silence and local recurrence."""
    rng=np.random.default_rng(seed)
    old,_,_=c1_config(seed,sample_id)
    sources=[]
    first=int(rng.integers(2))
    steps=20+16*np.arange(36)+rng.integers(-1,2,size=36)  # .4 + .32*i +/- .02s
    owners=[(first+i)%2 for i in range(len(steps))]
    for i,old_source in enumerate(old.sources):
        source=old_source.to_dict()
        source['id']=source.pop('source_id');source.pop('index')
        source.update(preset_ref=f'aat/c1-local/{old_source.sample.type}-v1',
                      sample_ref=f'generated/c1-local/{old_source.sample.type}-v1',gain=float(rng.uniform(.35,.65)))
        source['sample']={'type':old_source.sample.type,'seed':46+i,'params':
            {'duration_seconds':.45,'freq_hz':261.626,'attack_seconds':.005,'release_seconds':.03,'detune_cents':0.,'stereo':False}
            if i==0 else {'duration_seconds':.45,'freq_hz':261.626,'decay_seconds':.3,'damping':.5}}
        source['amp']={'attack_ms':float(rng.choice([5,10,20])),'decay_ms':float(rng.choice([20,40])),
                       'sustain':float(rng.choice([.35,.6,.9])),'release_ms':float(rng.choice([40,60]))}
        source['pattern']={'step_seconds':.02,'notes':[
            {'step':int(step),'note':60,'velocity':int(rng.integers(90,116)),'length_steps':int(rng.choice([5,6]))}
            for step,owner in zip(steps,owners) if owner==i]}
        sources.append(source)
    config=config_from_dict({'render':{'sample_id':sample_id,'composition':sample_id,'seed':seed,
        'sample_rate':44100,'block_size':512,'bpm':100,'duration_seconds':12.,'tail_seconds':1.},'sources':sources})
    return config,steps*.02,owners


def prepare_c1_local(out: str | Path,*,seed=4640,counts=(4,2,2)):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('C1-local output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    entries=[]
    for split,count in zip(('train','val','test'),counts):
        for i in range(count):
            sample_id=f'c1-local-{split}-{i:02d}'
            song_seed=seed+len(entries)*101
            config,onsets,owners=c1_local_config(song_seed,sample_id)
            directory=out/sample_id
            rendered=render_sample(config,directory)
            labels=label_sample(directory)
            centers=labels.center_times[(labels.center_times>=1-1e-9)&(labels.center_times<=11+1e-9)]
            waveform=read_wav(directory/'mix.wav')
            report,arrays=audit_local_context(labels,centers,audio_frames=waveform.frames)
            ratio=overlap_ratio(labels.rms[labels.valid])
            if ratio!=0 or not report['passed']:
                raise ValueError(f'C1-local violates acoustic/local evidence constraints: overlap={ratio}, audit={report}')
            if any(s['event_count']!=18 or s['bridge_targets_exactly_zero']==0 for s in report['sources']):
                raise ValueError('C1-local must preserve 18 separate events/source and actual silence')
            np.savez_compressed(directory/'local-context.npz',**arrays)
            report['arrays_sha256']=hashlib.sha256((directory/'local-context.npz').read_bytes()).hexdigest()
            report['envelope_sha256']=hashlib.sha256((directory/'envelope.npz').read_bytes()).hexdigest()
            dump_json(directory/'local-context.json',report)
            entries.append({'sample_id':sample_id,'split':split,'directory':sample_id,'seed':song_seed,
                            'onsets_seconds':onsets.tolist(),'owners':[config.sources[j].source_id for j in owners],
                            'amp':[s.amp.to_dict() for s in config.sources],'gain':[s.gain for s in config.sources],
                            'scoring_start_seconds':1.,'scoring_end_seconds':11.,
                            'acoustic_overlap_ratio':ratio,'local_context':report,
                            'mix_sha256':hashlib.sha256((directory/'mix.wav').read_bytes()).hexdigest(),
                            'envelope_sha256':report['envelope_sha256'],'stem_sum':rendered.report['stem_sum']})
    index={'version':'aat-curriculum-local-v2','stage':'C1-local','constraint_version':CONSTRAINT_VERSION,'seed':seed,
           'split_policy':'held-out sequence/ADSR/gain; fixed pad/pluck; local note co-occurrence in every scored 2s input; 1..11s absolute scoring range',
           'entries':entries}
    dump_json(out/'index.json',index)
    return index
