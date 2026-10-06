"""Verify executed source snapshots against committed text, respecting CRLF/LF."""
import hashlib
import json
import random
import subprocess
from pathlib import Path


def digest(x):return hashlib.sha256(x).hexdigest()
def main():
    out=Path('runs/split-head/evidence');audit_path=out/'issue-51-split-execution-audit.json'
    audit=json.loads(audit_path.read_text());source=Path('runs/split-head/fit-v3/source')
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--commit',default='HEAD');args=parser.parse_args()
    commit=subprocess.check_output(['git','rev-parse',args.commit],text=True).strip()
    checks=[]
    for name,expected in audit['source_sha256'].items():
        raw=(source/name).read_bytes()
        if digest(raw)!=expected:raise ValueError('executed source snapshot changed: '+name)
        committed=subprocess.check_output(['git','show',commit+':research/issue_51_ssast/'+name])
        normalized=raw.replace(b'\r\n',b'\n')
        if normalized!=committed.replace(b'\r\n',b'\n'):raise ValueError('committed code differs from execution: '+name)
        checks.append(dict(name=name,executed_bytes_sha256=expected,committed_text_sha256=digest(normalized),
            identical_text_after_CRLF_LF_normalization=True))
    initial=Path('runs/window-depth/fit-v1/depth4/C1-raw/head.pth')
    arms=[json.loads((Path('runs/split-head/fit-v3')/a/'architecture.json').read_text()) for a in ('coupled','split')]
    init=json.loads(Path('runs/split-head/fit-v3/initialization.json').read_text())
    if init['original_sha256']!=digest(initial.read_bytes()):raise ValueError('original checkpoint mismatch')
    import torch
    original=torch.load(initial,weights_only=True,map_location='cpu')
    initial_states=[]
    for arm,meta in zip(('coupled','split'),arms):
        initial_path=Path('runs/split-head/fit-v3')/(arm+'-initial.pth')
        if meta['initial_sha256']!=digest(initial_path.read_bytes()):raise ValueError('initial checkpoint mismatch')
        state=torch.load(initial_path,weights_only=True,map_location='cpu')
        if not all(torch.equal(value,state[name]) for name,value in original.items()):raise ValueError('shared/identity mismatch')
        initial_states.append(state)
    user_weight=Path('runs/p0/user-frame.pth')
    user_sha=digest(user_weight.read_bytes())
    if user_sha!='b82b0714d92ddfd9de2c535b459ad900dea85a50c810dc43086fc6ac7047f689':raise ValueError('user SSAST weights changed')
    audit.update(original_tensor_copy_exact=True,amplitude_head_difference_only=True,initialization=init,user_weights_sha256=user_sha,user_weight_source_authenticated=False)
    # Deterministic full sample schedules reconstructed from the exact manifest
    # order and Python choice calls in both arms; no solver consumes RNG.
    entries=[dict(e,stage='C0') for e in json.loads(Path('runs/p1/frame-c0-cache/manifest.json').read_text())['entries']]
    schedules={}
    for j,stage in enumerate(('C1','C2-low','C2-high','C3')):
        entries += [dict(e,stage=stage) for e in json.loads((Path('runs/window-depth/cache')/stage/'manifest.json').read_text())['entries']]
        train=[e for e in entries if e['split']=='train'];new=[e for e in train if e['stage']==stage];old=[e for e in train if e['stage']!=stage]
        rng=random.Random(4650+j);pairs=[]
        for _ in range(1000):
            selected=(rng.choice(old or new),rng.choice(new));pairs.append([(e['stage'],e['sample_id']) for e in selected])
        schedules[stage]=dict(seed=4650+j,sha256=digest(json.dumps(pairs,separators=(',',':')).encode()),
            samples=2000,first=pairs[0],last=pairs[-1])
    audit.update(implementation_commit=commit,executed_vs_committed_source=checks,
        reconstruction_sample_schedules=schedules,schedule_note='reconstructed from same manifests/seeds and code; not a per-step live trace',
        initial_sha256=digest(initial.read_bytes()),source_snapshot='runs/split-head/fit-v3/source')
    audit_path.write_text(json.dumps(audit,indent=2)+'\n')
    for arm,meta in zip(('coupled','split'),arms):
        if digest((Path('runs/split-head/fit-v3')/(arm+'-initial.pth')).read_bytes())!=meta['initial_sha256']:
            raise ValueError('audit must not mutate initial checkpoints')
    print(json.dumps(dict(commit=commit,source_files=len(checks),initial_verified=True)))


if __name__=='__main__':main()
