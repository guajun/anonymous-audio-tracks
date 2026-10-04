"""Original GPU selector audit of every selected local checkpoint; never train."""
import argparse
import gzip
import json
from pathlib import Path
import torch
import train_courses as shared
from train_window_depth import read
from window_head import WindowVHead
from local_association import transport
from summarize_courses import compact


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--c0-cache',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);shared.transport=transport
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    data={};data['C0'],_=read(args.c0_cache,'C0')
    for stage in ('C1','C2-low','C2-high','C3'):data[stage],_=read(args.cache/stage,stage)
    all_records=[];summary=[]
    for depth in (2,4,8):
        directory=args.fit/f'depth{depth}';rows=list(data['C0'])
        scale=json.loads((directory/'architecture.json').read_text())['scale_r']
        for stage in ('C1','C2-low','C2-high','C3'):
            rows+=data[stage]
            model=WindowVHead(depth).cuda()
            model.load_state_dict(torch.load(directory/f'{stage}-local'/'head.pth',weights_only=True));model.eval()
            r=dict(stage=stage,depth=depth,validation=shared.evaluate(model,[x for x in rows if x['entry']['split']=='val'],scale,True),
                test=shared.evaluate(model,[x for x in rows if x['entry']['split']=='test'],scale,True),
                evaluation='original GPU relation/selector at shared TF32 setting; same selected checkpoint, no training/test selection')
            all_records.append(r)
            summary.append(dict(depth=depth,stage=stage,current=compact(r,stage),
                forgetting={old:compact(r,old) for old in data if old in {x['stage'] for x in rows}}))
            (args.fit/'original-reference-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print('ORIGINAL GPU EVALUATED',depth,stage,flush=True)
    (args.fit/'original-reference-full.json.gz').write_bytes(gzip.compress(json.dumps(all_records,indent=1).encode(),mtime=0))


if __name__=='__main__':main()
