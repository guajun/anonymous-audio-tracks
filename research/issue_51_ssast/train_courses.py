"""Bounded stage learning, fixed validations, and 50/50 replay."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from head import TemporalVHead
from local_association import transport
from course_loss import loss
from course_metrics import measure, endpoint_evidence
from features import sha256


def read(cache,stage):
    m=json.loads((cache/'manifest.json').read_text())
    result=[]
    for e in m['entries']:
        p=cache/e['cache']
        if sha256(p)!=e['cache_sha256']:raise ValueError('cache digest mismatch')
        z=np.load(p,allow_pickle=False)
        token=z['tokens']
        rms=z['token_rms']
        if token.shape[1]==197:token=token[:,96:102];rms=rms[:,96:102]
        result.append(dict(stage=stage,entry=e,tokens=torch.tensor(token,device='cuda'),
            rms=torch.tensor(rms,device='cuda'),target=torch.tensor(z['targets'],device='cuda'),
            times=z['center_times'],mix=z['mix_rms_oracle']))
    return result,m


def forward(model,row,scale,association):
    v,a=model(row['tokens'],row['rms']/scale)
    if association:
        transported,c,meta=transport(v*scale,row['times'])
        return a*scale,transported,c,meta,v
    return a*scale,a*scale,None,None,v


@torch.no_grad()
def evaluate(model,rows,scale,association,out=None):
    records=[]
    for row in rows:
        start=time.monotonic()
        raw,ta,c,meta,v=forward(model,row,scale,association)
        raw=raw.cpu().numpy();ta=ta.cpu().numpy();b=row['target'].cpu().numpy()
        record=dict(sample_id=row['entry']['sample_id'],stage=row['stage'],
            raw=measure(raw,b,row['times'],scale),transported=measure(ta,b,row['times'],scale),
            raw_gated=measure(np.where(raw>.001,raw,0),b,row['times'],scale),
            transported_gated=measure(np.where(ta>.001,ta,0),b,row['times'],scale),
            elapsed_seconds=time.monotonic()-start)
        active=(torch.linalg.vector_norm(v,dim=-1)*scale)>.001
        unit=v/v.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        cosine=unit@unit.transpose(1,2)
        pair_mask=active[:,:,None]&active[:,None,:]&~torch.eye(8,device=v.device,dtype=torch.bool)[None]
        record['raw_active_E_pair_cosine_mean']=float(cosine[pair_mask].mean()) if pair_mask.any() else None
        if association:
            cc=c.cpu().numpy()
            record['endpoints']=endpoint_evidence(raw,ta,cc,b,row['times'],scale)
            record['association']=meta
            # Permutation invariance evaluated once on held-out complete sample.
            if out is not None:
                v,_=model(row['tokens'],row['rms']/scale)
                generator=torch.Generator(device='cuda').manual_seed(5151)
                perm=torch.stack([torch.randperm(8,device='cuda',generator=generator) for _ in range(len(v))])
                shuffled=v.gather(1,perm[:,:,None].expand_as(v))
                shuffled_a,_,_=transport(shuffled*scale,row['times'])
                shuffled_metric=measure(shuffled_a.cpu().numpy(),b,row['times'],scale)
                record['candidate_shuffle_normalized_mae_delta']=shuffled_metric['normalized_mae']-record['transported']['normalized_mae']
            record['amplitude_transport_total_max_error']=float(np.abs(ta.sum(1)-np.where(raw>.001,raw,0).sum(1)).max())
        if out is not None:
            np.savez_compressed(out/(row['stage']+'-'+row['entry']['sample_id']+'.npz'),
                times=row['times'],raw_a=raw,transported_a=ta,target=b,
                matrix=c.cpu().numpy() if c is not None else np.empty(0))
        records.append(record)
    return records


def score(records):
    return np.mean([r['transported']['normalized_mae']+.1*(1-np.mean(r['transported']['all']['per_source_iou'])) for r in records])


def fit(model,all_rows,current,scale,association,steps,seconds,out):
    out.mkdir(parents=True,exist_ok=True)
    train=[r for r in all_rows if r['entry']['split']=='train']
    new=[r for r in train if r['stage']==current]
    old=[r for r in train if r['stage']!=current]
    val=[r for r in all_rows if r['entry']['split']=='val']
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=1e30;meaningful=1e30;stale=0;best_state=None
    start=time.monotonic();history=[];reason='max_updates';update=0
    for step in range(1,steps+1):
        if time.monotonic()-start>=seconds:
            reason='wall_clock_budget';break
        selected=[random.choice(old),random.choice(new)]
        optimizer.zero_grad(set_to_none=True)
        parts=[]
        for row in selected:
            raw,ta,c,meta,v=forward(model,row,scale,association)
            if association:
                v.retain_grad()
            value,detail=loss(ta/scale,row['target']/scale)
            if not torch.isfinite(value):raise ValueError('nonfinite loss')
            (value/2).backward()
            direction_gradient=None
            if association:
                unit=v.detach()/v.detach().norm(dim=-1,keepdim=True).clamp_min(1e-12)
                radial=(v.grad*unit).sum(-1,keepdim=True)*unit
                direction_gradient=float((v.grad-radial).norm())
            parts.append(dict(stage=row['stage'],loss=float(value.detach()),
                shape=float(detail['shape'].detach()),silence=float(detail['silence'].detach()),
                unused=float(detail['unused'].detach()),
                vector_grad_norm=float(v.grad.norm()) if association else None,
                direction_grad_norm=direction_gradient))
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not np.isfinite(grad):
            (out/'failure.json').write_text(json.dumps(dict(stage=current,step=step,
                reason='nonfinite gradient; optimizer step refused',parts=parts),indent=2)+'\n')
            raise ValueError('nonfinite gradient')
        optimizer.step();update=step
        interval=10 if association else 100
        if step==1 or step%interval==0 or step==steps:
            results=evaluate(model,val,scale,association)
            current_score=float(score(results))
            if current_score<best:
                best=current_score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if current_score<meaningful*.99:
                meaningful=current_score;stale=0
            else:stale+=1
            h=dict(update=step,parts=parts,gradient_norm=grad,val_score=current_score,
                elapsed_seconds=time.monotonic()-start)
            history.append(h)
            print(json.dumps(dict(stage=current,association=association,**h)),flush=True)
            (out/'progress.json').write_text(json.dumps(history,indent=2)+'\n')
            if step>=1000 and stale>=10:reason='val_plateau_10_checks';break
    final_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(final_state,out/'last-head.pth')
    if best_state is not None:model.load_state_dict(best_state)
    torch.save(model.state_dict(),out/'head.pth')
    validation=evaluate(model,val,scale,association)
    test=evaluate(model,[r for r in all_rows if r['entry']['split']=='test'],scale,association,out)
    report=dict(stage=current,association_enabled=association,cycle_enabled=False,
        seed=46,updates=update,stop_reason=reason,seconds=time.monotonic()-start,
        budget=dict(updates=steps,seconds=seconds),scale_r=scale,replay_old_fraction=.5,
        validation_interval_updates=10 if association else 100,
        selection='equal-sample mean val normalized MAE + .1*(1-mean source IoU), whole-PIT assignment, all seen stages; no test tuning',
        loss='Huber(delta.1)/.1 + .1areaIoU + .1silence + .1unused, whole-sequence PIT',
        history=history,validation=validation,test=test)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--c0-cache',type=Path,required=True)
    p.add_argument('--initial',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--raw-steps',type=int,default=2000)
    p.add_argument('--assoc-steps',type=int,default=1000)
    p.add_argument('--assoc-seconds',type=int,default=120)
    p.add_argument('--stages',nargs='+',default=['C1','C2-low','C2-high','C3'])
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(46);np.random.seed(46);random.seed(46)
    args.out.mkdir(parents=True,exist_ok=True)
    c0,_=read(args.c0_cache,'C0')
    active=torch.cat([r['target'][r['target']>.001] for r in c0 if r['entry']['split']=='train'])
    scale=float(torch.quantile(active,.95))
    raw_model=TemporalVHead().cuda()
    raw_model.load_state_dict(torch.load(args.initial,weights_only=True))
    assoc_model=None
    rows=c0;summary=[]
    for stage in args.stages:
        newer,manifest=read(args.cache/stage,stage);rows=rows+newer
        baselines={}
        for kind in ('single_mix_rms','copy_mix_rms_8','zero'):
            records=[]
            for row in newer:
                if row['entry']['split']!='test':continue
                a=np.zeros((len(row['mix']),8),dtype=np.float32)
                if kind=='single_mix_rms':a[:,0]=row['mix']
                if kind=='copy_mix_rms_8':a[:]=row['mix'][:,None]
                records.append(dict(sample_id=row['entry']['sample_id'],metrics=measure(a,row['target'].cpu().numpy(),row['times'],scale)))
            baselines[kind]=records
        raw=fit(raw_model,rows,stage,scale,False,args.raw_steps,300,args.out/(stage+'-raw'))
        if assoc_model is None:
            assoc_model=TemporalVHead().cuda();assoc_model.load_state_dict(raw_model.state_dict())
        assoc=fit(assoc_model,rows,stage,scale,True,args.assoc_steps,args.assoc_seconds,args.out/(stage+'-local'))
        summary.append(dict(stage=stage,manifest_sha256=sha256(args.cache/stage/'manifest.json'),
            actual_overlap=[e['actual_overlap_ratio_scored'] for e in manifest['entries']],
            raw=raw,local=assoc,baselines=baselines,
            status='diagnostic; C0 collection acceptance not waived'))
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('COURSE RUN COMPLETE: C4 requires review of dual-source evidence',flush=True)


if __name__=='__main__':main()
