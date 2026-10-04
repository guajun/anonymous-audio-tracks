"""Matched full-window depth curriculum, preserving historical evidence."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import train_courses as shared
from features import sha256
from window_head import WindowVHead
from fast_association import transport


def read(cache,stage):
    manifest=json.loads((cache/'manifest.json').read_text())
    rows=[]
    for entry in manifest['entries']:
        path=cache/entry['cache']
        if sha256(path)!=entry['cache_sha256']:raise ValueError('cache digest mismatch')
        z=np.load(path,allow_pickle=False)
        if z['tokens'].shape[1:]!=(197,768):raise ValueError('cropped cache forbidden')
        rows.append(dict(stage=stage,entry=entry,tokens=torch.tensor(z['tokens'],device='cuda'),
            rms=torch.tensor(z['token_rms'],device='cuda'),
            target=torch.tensor(z['targets'],device='cuda'),times=z['center_times'],mix=z['mix_rms_oracle'],
            context_valid_fraction=z['context_valid_fraction'] if 'context_valid_fraction' in z else np.ones(len(z['center_times']))))
    return rows,manifest


def seed(value):
    torch.manual_seed(value);np.random.seed(value);random.seed(value)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--c0-cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--depth',type=int,choices=(2,4,8),required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--c0-steps',type=int,default=2000)
    p.add_argument('--seconds',type=float,default=7200)
    p.add_argument('--benchmark',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    shared.transport=transport
    seed(46)
    args.out.mkdir(parents=True,exist_ok=True)
    rows,_=read(args.c0_cache,'C0')
    active=torch.cat([r['target'][r['target']>.001] for r in rows if r['entry']['split']=='train'])
    scale=float(torch.quantile(active,.95))
    raw=WindowVHead(args.depth).cuda()
    architecture=raw.architecture()
    architecture.update(training_seed=46,scale_r=scale,tf32_enabled=True,
        c0_cache_manifest_sha256=sha256(args.c0_cache/'manifest.json'),
        protocol='same full-window197-token inputs, same loss/optimizer/replay and sample schedule across depths')
    (args.out/'architecture.json').write_text(json.dumps(architecture,indent=2)+'\n')
    if args.benchmark:
        row=rows[0];optimizer=torch.optim.AdamW(raw.parameters(),lr=.001)
        for associated in (False,True):
            torch.cuda.synchronize();start=time.monotonic()
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                a,ta,c,meta,v=shared.forward(raw,row,scale,associated)
                value,_=shared.loss(ta/scale,row['target']/scale)
                value.backward();optimizer.step()
            torch.cuda.synchronize()
            print(json.dumps(dict(depth=args.depth,association=associated,
                seconds_per_sample_update=(time.monotonic()-start)/4,
                peak_gpu_bytes=torch.cuda.max_memory_allocated())),flush=True)
        return
    summary=[]
    seed(4600)
    c0=shared.fit(raw,rows,'C0',scale,False,args.c0_steps,args.seconds,
        args.out/'C0-raw',validation_interval=50)
    c0['architecture']=architecture
    summary.append(dict(stage='C0',raw=c0,actual_overlap=[0]*8))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=1)+'\n')
    local=None
    for index,stage in enumerate(('C1','C2-low','C2-high','C3')):
        newer,manifest=read(args.cache/stage,stage);rows+=newer
        # Identical complete-sample draw sequence for both arms/all depths.
        schedule_seed=4650+index
        seed(schedule_seed)
        raw_report=shared.fit(raw,rows,stage,scale,False,args.steps,args.seconds,
            args.out/(stage+'-raw'),validation_interval=50)
        if local is None:
            local=WindowVHead(args.depth).cuda();local.load_state_dict(raw.state_dict())
        seed(schedule_seed)
        local_report=shared.fit(local,rows,stage,scale,True,args.steps,args.seconds,
            args.out/(stage+'-local'),validation_interval=50)
        raw_report['architecture']=architecture;local_report['architecture']=architecture
        record=dict(stage=stage,raw=raw_report,local=local_report,
            actual_overlap=[e['actual_overlap_ratio_scored'] for e in manifest['entries']],
            cache_manifest_sha256=sha256(args.cache/stage/'manifest.json'),
            schedule_seed=schedule_seed,
            initialization='fresh full-window head C0; local begins selected raw C1 then continues local')
        summary.append(record)
        (args.out/'summary.json').write_text(json.dumps(summary,indent=1)+'\n')
    print('DEPTH COMPLETE',args.depth,flush=True)


if __name__=='__main__':main()
