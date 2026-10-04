"""Verify executed source snapshots against committed text, respecting CRLF/LF."""
import hashlib
import json
import random
import subprocess
from pathlib import Path


def digest(x):return hashlib.sha256(x).hexdigest()
def main():
    out=Path('runs/joint-teacher/evidence');path=out/'issue-51-joint-execution-audit.json'
    audit=json.loads(path.read_text());source=Path('runs/joint-teacher/fit-v2/source')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
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
    arms=[json.loads((Path('runs/joint-teacher/fit-v2')/a/'architecture.json').read_text()) for a in ('envelope-id','joint-id')]
    if not all(a['initial_sha256']==digest(initial.read_bytes()) for a in arms):raise ValueError('initial checkpoint mismatch')
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
        initial_sha256=digest(initial.read_bytes()),source_snapshot='runs/joint-teacher/fit-v2/source')
    path.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(dict(commit=commit,source_files=len(checks),initial_verified=True)))


if __name__=='__main__':main()
