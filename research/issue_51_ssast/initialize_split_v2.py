"""Train-only scalar calibration; no shared/identity updates or GT fitting."""
import json
import time
from pathlib import Path
import torch
from window_head import WindowVHead, WindowSplitHead
from train_window_depth import read, seed
from features import sha256

ROOT=Path('runs/split-head/fit-v2')
SCALE=.2591032087802887

def main():
    torch.set_num_threads(4);seed(46)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    ROOT.mkdir(parents=True,exist_ok=True)
    initial=Path('runs/window-depth/fit-v1/depth4/C1-raw/head.pth')
    state=torch.load(initial,weights_only=True)
    old=WindowVHead(4).cuda();old.load_state_dict(state)
    split=WindowSplitHead(4).cuda();split.load_state_dict(state,strict=False)
    for name,p in split.named_parameters():p.requires_grad_(name.startswith('amplitude_output.'))
    # Avoid even loading val/test token arrays during initialization.
    rows=[]
    import numpy as np
    for cache,stage in [(Path('runs/p1/frame-c0-cache'),'C0'),(Path('runs/window-depth/cache/C1'),'C1')]:
        manifest=json.loads((cache/'manifest.json').read_text())
        for e in manifest['entries']:
            if e['split']!='train':continue
            path=cache/e['cache'];assert sha256(path)==e['cache_sha256']
            with np.load(path) as z:
                assert z['tokens'].shape[1:]==(197,768)
                rows.append((stage,e['sample_id'],torch.tensor(z['tokens'],device='cuda'),torch.tensor(z['token_rms'],device='cuda')))
    features=[];targets=[];start=time.monotonic()
    hook=old.output.register_forward_pre_hook(lambda module,args:features.append(args[0].detach().float()))
    with torch.no_grad():
        for _,_,tokens,rms in rows:targets.append(old(tokens,rms/SCALE)[1])
    hook.remove();x=torch.cat(features);target=torch.cat(targets)
    # Deterministic train-only least squares starts close to inverse-softplus A.
    xx=torch.cat((x,torch.ones_like(x[:,:1])),1).double()
    inverse=target.double()+torch.log(-torch.expm1(-target.double().clamp_min(1e-12)))
    candidates=[];solutions=[]
    # Predeclared train-only norm cap prevents the ill-conditioned 4943-norm fit.
    for strength in (.01,.1,1.,10.,100.):
        ridge=torch.eye(xx.shape[1],device='cuda',dtype=torch.float64)*strength
        solution=torch.linalg.solve(xx.T@xx+ridge,xx.T@inverse)
        prediction=torch.nn.functional.softplus((xx@solution).float())
        candidates.append(dict(ridge=strength,weight_norm=float(solution[:-1].norm()),train_mse=float((prediction-target).square().mean()),train_mae=float((prediction-target).abs().mean())))
        solutions.append(solution)
    print('TRAIN-ONLY RIDGE CANDIDATES',json.dumps(candidates),flush=True)
    eligible=[j for j,c in enumerate(candidates) if c['weight_norm']<=100.]
    chosen=min(eligible,key=lambda j:candidates[j]['train_mse']);solution=solutions[chosen]
    with torch.no_grad():
        split.amplitude_output.weight.copy_(solution[:-1].T.float());split.amplitude_output.bias.copy_(solution[-1].float())
    optimizer=torch.optim.Adam(split.amplitude_output.parameters(),lr=.001)
    history=[]
    for step in range(1000):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):raw=split.amplitude_output(x)
        a=torch.nn.functional.softplus(raw.float())
        loss=(a-target).square().mean();loss.backward();optimizer.step()
        with torch.no_grad():
            w=split.amplitude_output.weight;w.mul_(min(1.,100./float(w.norm())))
        if step%100==0:history.append(dict(step=step,train_mse=float(loss.detach())))
    torch.cuda.synchronize();split_seconds=time.monotonic()-start
    # Equal data-forward count for coupled control: no trainable A branch to fit,
    # zero residual already, so extra calibration passes cannot improve its init.
    start=time.monotonic()
    with torch.no_grad():
        for step in range(1000):
            with torch.autocast('cuda',dtype=torch.bfloat16):v=old.output(x)
            a=v.float().reshape(-1,8,128).norm(dim=-1)
            _=(a-target).square().mean()
    torch.cuda.synchronize();coupled_seconds=time.monotonic()-start
    differences=[]
    with torch.no_grad():
        for stage,sample,tokens,rms in rows:
            v,a=old(tokens,rms/SCALE);z,b=split(tokens,rms/SCALE)
            differences.append(dict(stage=stage,sample_id=sample,z_max_error=float((v-z).abs().max()),
                amplitude_mae=float((a-b).abs().mean()),amplitude_max_error=float((a-b).abs().max()),
                mean_reference_amplitude=float(a.mean()),relative_mae=float((a-b).abs().mean()/a.mean()),
                count_mae=float(((a*SCALE>.001).sum(1)-(b*SCALE>.001).sum(1)).abs().float().mean())))
    for name,p in old.state_dict().items():assert torch.equal(p,split.state_dict()[name])
    torch.save(old.state_dict(),ROOT/'coupled-initial.pth');torch.save(split.state_dict(),ROOT/'split-initial.pth')
    result=dict(method='train-only ridge chosen by MSE subject to amplitude weight norm<=100; 1000 projected Adam steps; shared/Z frozen',ridge_candidates=candidates,chosen_ridge=candidates[chosen]['ridge'],amplitude_weight_norm_cap=100.,
        training_samples=[(r[0],r[1]) for r in rows],data_used='C0/C1 train only, target old A, no GT/val/test',
        steps_per_arm=1000,coupled_control='same cached readout forward count, zero-residual control; no parameter updates',
        initialization_budget_limitation='equal pass budget, different wall time and optimizer work; scalar linear softplus cannot exactly represent norm of 128D affine output; added initialization norm cap, not a formal-training constraint',
        split_seconds=split_seconds,coupled_seconds=coupled_seconds,history=history,differences=differences,
        original_sha256=sha256(initial),coupled_sha256=sha256(ROOT/'coupled-initial.pth'),split_sha256=sha256(ROOT/'split-initial.pth'),
        shared_and_identity_exactly_equal=True,scale_r=SCALE)
    (ROOT/'initialization.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
