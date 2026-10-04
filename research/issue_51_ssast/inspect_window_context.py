"""Held-out counterfactual context sensitivity; no training/test selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import train_courses as shared
from train_window_depth import read
from window_head import WindowVHead
from fast_association import transport


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);shared.transport=transport
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    rows,_=read(args.cache/'C3','C3');records=[]
    for depth in (2,4,8):
        directory=args.fit/f'depth{depth}'
        architecture=json.loads((directory/'architecture.json').read_text())
        scale=architecture['scale_r']
        for arm,association in (('raw',False),('local',True)):
            model=WindowVHead(depth).cuda()
            model.load_state_dict(torch.load(directory/f'C3-{arm}'/'head.pth',weights_only=True))
            model.eval()
            for row in rows:
                if row['entry']['split']!='test':continue
                with torch.no_grad():
                    _,original,_,_,_=shared.forward(model,row,scale,association)
                    original=original.cpu().numpy()
                    base=shared.measure(original,row['target'].cpu().numpy(),row['times'],scale)
                    for kind in ('tokens','rms'):
                        altered=dict(row)
                        key='tokens' if kind=='tokens' else 'rms'
                        value=row[key].clone()
                        mean=value.float().mean(1,keepdim=True).to(value.dtype)
                        # Common protected range covers both98/99 centers +/-8
                        # at the deepest CNN; only statistical readout sees changes.
                        value[:,:90]=mean.expand_as(value)[:,:90]
                        value[:,108:]=mean.expand_as(value)[:,108:]
                        altered[key]=value
                        _,pred,_,_,_=shared.forward(model,altered,scale,association)
                        pred=pred.cpu().numpy()
                        metric=shared.measure(pred,row['target'].cpu().numpy(),row['times'],scale)
                        records.append(dict(depth=depth,arm=arm,sample_id=row['entry']['sample_id'],
                            intervention=f'outside90..107 {kind} replaced by per-window mean, central18 inputs and other input unchanged',
                            normalized_prediction_difference=float(np.abs(pred-original).mean()/scale),
                            baseline_normalized_mae=base['normalized_mae'],
                            altered_normalized_mae=metric['normalized_mae'],
                            mae_delta=metric['normalized_mae']-base['normalized_mae'],
                            source_iou=metric['all']['per_source_iou']))
    (args.fit/'context-sensitivity.json').write_text(json.dumps(dict(records=records,
        limitation='off-distribution counterfactual; sensitivity is not evidence of timbre identity, SSAST gain or generalization'),indent=2)+'\n')
    print('CONTEXT AUDIT COMPLETE',flush=True)


if __name__=='__main__':main()
