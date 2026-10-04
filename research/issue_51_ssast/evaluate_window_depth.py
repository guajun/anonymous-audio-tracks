"""Evaluate saved selected checkpoints with explicit boundary evidence."""
import argparse
import json
from pathlib import Path
import torch
import train_courses as shared
from train_window_depth import read
from window_head import WindowVHead
from fast_association import transport


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--c0-cache',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);shared.transport=transport
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    data={}
    data['C0'],_=read(args.c0_cache,'C0')
    for stage in ('C1','C2-low','C2-high','C3'):data[stage],_=read(args.cache/stage,stage)
    for depth in (2,4,8):
        d=args.fit/f'depth{depth}'
        summary=json.loads((d/'summary.json').read_text());rows=[]
        scale=json.loads((d/'architecture.json').read_text())['scale_r']
        for record in summary:
            stage=record['stage'];rows+=data[stage]
            for arm,association in (('raw',False),('local',True)):
                if arm not in record:continue
                directory=d/f'{stage}-{arm}'
                model=WindowVHead(depth).cuda()
                model.load_state_dict(torch.load(directory/'head.pth',weights_only=True));model.eval()
                r=record[arm]
                r['validation']=shared.evaluate(model,[x for x in rows if x['entry']['split']=='val'],scale,association)
                r['test']=shared.evaluate(model,[x for x in rows if x['entry']['split']=='test'],scale,association,directory)
                r['evaluation_revision']='same selected checkpoint; adds padded/complete boundary metrics; evaluation after suite, not concurrent training; no training or test selection'
                (directory/'result.json').write_text(json.dumps(r,indent=1)+'\n')
                del model
            (d/'summary.json').write_text(json.dumps(summary,indent=1)+'\n')
            print('EVALUATED',depth,stage,flush=True)


if __name__=='__main__':main()
