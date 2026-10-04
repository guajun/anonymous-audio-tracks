"""Real held-out checkpoint numerical audit of execution optimization."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from window_head import WindowVHead
from train_window_depth import read
from fast_association import transport as fast
from local_association import transport as original
from course_metrics import measure


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    rows,_=read(args.cache/'C3','C3');records=[]
    for depth in (2,4,8):
        d=args.fit/f'depth{depth}';scale=json.loads((d/'architecture.json').read_text())['scale_r']
        for arm in ('raw','local'):
            model=WindowVHead(depth).cuda();model.load_state_dict(torch.load(d/f'C3-{arm}'/'head.pth',weights_only=True));model.eval()
            with torch.no_grad():
                for row in rows:
                    if row['entry']['split']!='test':continue
                    v,_=model(row['tokens'],row['rms']/scale);v=v*scale
                    a,c,meta=original(v,row['times']);b,cc,mm=fast(v,row['times'])
                    original_metrics=measure(a.cpu().numpy(),row['target'].cpu().numpy(),row['times'],scale)
                    fast_metrics=measure(b.cpu().numpy(),row['target'].cpu().numpy(),row['times'],scale)
                    records.append(dict(depth=depth,arm=arm,sample_id=row['entry']['sample_id'],
                        max_amplitude_abs_difference=float((a-b).abs().max()),
                        max_relation_abs_difference=float((c-cc).abs().max()),
                        normalized_mae_delta=fast_metrics['normalized_mae']-original_metrics['normalized_mae'],
                        fragment_ids_equal=meta['fragment_ids_by_frame']==mm['fragment_ids_by_frame'],
                        original_source_iou=original_metrics['all']['per_source_iou'],
                        fast_source_iou=fast_metrics['all']['per_source_iou']))
    (args.fit/'trained-association-audit.json').write_text(json.dumps(dict(records=records,
        limitation='discrete near-tie references can change with CPU/GPU arithmetic; audit is numerical, not identity correctness'),indent=2)+'\n')
    print('TRAINED ASSOCIATION AUDIT COMPLETE',flush=True)


if __name__=='__main__':main()
