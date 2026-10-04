"""Add explicit copy-window/E diagnostics to saved checkpoints; never train."""
import argparse
import json
from pathlib import Path
import torch
from head import TemporalVHead
from train_courses import read, evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--c0-cache',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    summary=json.loads((args.fit/'summary.json').read_text())
    rows,_=read(args.c0_cache,'C0')
    for stage in summary:
        current=stage['stage']
        newer,_=read(args.cache/current,current);rows+=newer
        for name,association in [('raw',False),('local',True)]:
            d=args.fit/(current+'-'+name)
            result=json.loads((d/'result.json').read_text())
            model=TemporalVHead().cuda()
            model.load_state_dict(torch.load(d/'head.pth',weights_only=True))
            result['validation']=evaluate(model,[r for r in rows if r['entry']['split']=='val'],result['scale_r'],association)
            result['test']=evaluate(model,[r for r in rows if r['entry']['split']=='test'],result['scale_r'],association,d)
            result['selection']='equal-sample mean val normalized MAE + .1*(1-mean source IoU), whole-PIT assignment, all seen stages; no test tuning'
            result['evaluation_revision']='adds explicit160ms copy-window, per-source silence false positives and activeE cosine diagnostics, corrects selection/boundary labels; same checkpoint, no training or test selection'
            stage[name]=result
            (d/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        (args.fit/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('evaluated',current,flush=True)


if __name__=='__main__':main()
