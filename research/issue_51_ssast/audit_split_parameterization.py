"""Post-fit train-only scalar-head/logit witnesses; no rule or checkpoint tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from window_head import WindowSplitHead
from train_window_depth import seed

def stats(x):
    a=x.detach().float().cpu().numpy().reshape(-1)
    return dict(mean=float(a.mean()),quantiles=np.quantile(a,[0,.01,.1,.5,.9,.99,1]).tolist(),exact_zero_fraction=float((a==0).mean()))

def main():
    torch.set_num_threads(4);seed(46);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('runs/split-head/fit-v3'))
    parser.add_argument('--transform',choices=('abs','softplus'),default='abs')
    parser.add_argument('--name',default='issue-51-split-parameterization.json');args=parser.parse_args()
    root=args.root;scale=.2591032087802887
    # Load only the same fixed C1 train clip, never val/test.
    cache=Path('runs/window-depth/cache/C1');manifest=json.loads((cache/'manifest.json').read_text())
    entry=next(e for e in manifest['entries'] if e['split']=='train')
    with np.load(cache/entry['cache']) as z:
        tokens=torch.tensor(z['tokens'],device='cuda');rms=torch.tensor(z['token_rms'],device='cuda')
    checkpoints=[root/'split-initial.pth']+[root/'split'/stage/'head.pth' for stage in ('C1','C2-low','C2-high','C3')]
    records=[]
    for path in checkpoints:
        model=WindowSplitHead(amplitude_transform=args.transform).cuda();model.load_state_dict(torch.load(path,weights_only=True));logits=[]
        hook=model.amplitude_output.register_forward_hook(lambda module,args,result:logits.append(result))
        z,a=model(tokens,rms/scale);hook.remove()
        gradient=torch.autograd.grad(a.mean(),model.amplitude_output.weight)[0]
        records.append(dict(checkpoint=str(path),sample_id=entry['sample_id'],A_normalized=stats(a),Z_norm=stats(z.norm(dim=-1)),
            logits=stats(logits[0]),nonnegative_transform_derivative=stats(torch.sign(logits[0].float()) if args.transform=='abs' else torch.sigmoid(logits[0].float())),
            amplitude_output_weight_norm=float(model.amplitude_output.weight.detach().norm()),
            identity_output_weight_norm=float(model.output.weight.detach().norm()),
            mean_A_gradient_amplitude_head_norm=float(gradient.norm())))
    out=Path('runs/split-head/evidence');out.mkdir(parents=True,exist_ok=True)
    (out/args.name).write_text(json.dumps(dict(records=records,amplitude_transform=args.transform,
        interpretation='train-only parameterization witness, not an intervention or explanation of all causal effects; no checkpoint/threshold tuning'),indent=2)+'\n')
    print(json.dumps(records),flush=True)

if __name__=='__main__':main()
