"""Matched stopped-gradient envelope teacher / direct identity curriculum.

Primary validation is prediction-only hard association; teacher is diagnostic.
"""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from window_head import WindowVHead, WindowSplitHead
from train_window_depth import read, seed
from features import sha256
from teacher_assignment import teacher as envelope_teacher, amplitude_loss
from joint_teacher import teacher as joint_teacher, frozen_confidence
TEACHER_MODE='joint'

def teacher(a,y,*,v=None,pairs=None):
    if TEACHER_MODE=='envelope':
        order,_,meta=envelope_teacher(a,y)
        return order,frozen_confidence(v,y),meta
    return joint_teacher(v,y,pairs=pairs,amplitudes=a)

from identity_supervision import make_pairs, identity_loss
from hard_identity_inference import connect
from course_metrics import measure


def quantiles(x):
    x=np.asarray(x).reshape(-1)
    return dict(count=len(x), mean=float(x.mean()), quantiles=np.quantile(x,[0,.1,.5,.9,1]).tolist()) if len(x) else dict(count=0,mean=None,quantiles=None)


def prepare(rows):
    for row in rows:
        y=row['target'].cpu().numpy()
        row['pairs']=make_pairs(y)


def relation(v,y,order,conf,pred,detail):
    n,k=order.shape;s=y.shape[1]
    inverse=np.empty((n,k),dtype=int)
    inverse[np.arange(n)[:,None],pred['orders']]=np.arange(k)[None,:]
    labels=inverse[np.arange(n)[:,None],order[:,:s]]
    active=y>0
    weights=y/(y+.01)*conf
    def result(left,right,source):
        w=np.sqrt(weights[left,source]*weights[right,source])
        valid=active[left,source]&active[right,source]
        correct=labels[left,source]==labels[right,source]
        return dict(pairs=int(valid.sum()),correct_fraction=float(correct[valid].mean()) if valid.any() else None,
            confidence_weight_sum=float(w[valid].sum()),weighted_correct_fraction=float((w[valid]*correct[valid]).sum()/w[valid].sum()) if w[valid].sum()>0 else None)
    t,i=np.nonzero(active[:-1]&active[1:]);boundary=(t+1)%16==0
    out=dict(adjacent=result(t,t+1,i),block_boundary=result(t[boundary],t[boundary]+1,i[boundary]),
        block_interior=result(t[~boundary],t[~boundary]+1,i[~boundary]))
    gap_left=[];gap_right=[];gap_source=[]
    for source in range(s):
        edges=np.diff(np.r_[False,active[:,source],False].astype(int))
        starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
        for left,right in zip(ends[:-1],starts[1:]):
            if right-left<=45:
                gap_left.append(left);gap_right.append(right);gap_source.append(source)
    out['across_silence']=result(np.array(gap_left,dtype=int),np.array(gap_right,dtype=int),np.array(gap_source,dtype=int))
    out['limitation']='GT-teacher candidate endpoints, evaluation only; within-block constant teacher is an easy index-continuity target, and ambiguous labels receive low weight'
    out['positive_cosine']=quantiles(detail['positive_cosine'].detach().cpu().numpy())
    out['negative_cosine']=quantiles(detail['negative_cosine'].detach().cpu().numpy())
    w=detail['weights'].detach().cpu().numpy()
    out['identity_pair_weights']=quantiles(w)
    if len(w):
        pos=detail['positive_cosine'].detach().cpu().numpy();neg=detail['negative_cosine'].detach().cpu().numpy()
        out['weighted_positive_cosine']=float((pos*w).sum()/w.sum()) if w.sum()>0 else None
        out['weighted_negative_cosine']=float((neg*w).sum()/w.sum()) if w.sum()>0 else None
        out['weighted_margin_violation']=float((detail['margin_violation'].detach().cpu().numpy()*w).sum()/w.sum()) if w.sum()>0 else None
    return out


@torch.no_grad()
def evaluate(model,rows,scale,out=None,extended=False):
    model.eval();records=[]
    for row in rows:
        v,a=model(row['tokens'],row['rms']/scale)
        y=row['target']/scale
        order,conf,tm=teacher(a,y,v=v,pairs=row['pairs'])
        lid,detail=identity_loss(v,y,order,conf,row['pairs'])
        vv=v.cpu().numpy()*scale;raw=a.cpu().numpy()*scale;b=row['target'].cpu().numpy()
        actual,pm=connect(vv,scale,diagnostics=extended,amplitudes=raw)
        ta=np.take_along_axis(raw,order,1)
        record=dict(stage=row['stage'],sample_id=row['entry']['sample_id'],
            actual=measure(actual,b,row['times'],scale),raw=measure(raw,b,row['times'],scale),
            teacher=measure(ta,b,row['times'],scale,order_override=np.arange(b.shape[1])),
            teacher_confidence=quantiles(conf[b>0]),teacher_zero_confidence_fraction=float((conf[b>0]==0).mean()) if (b>0).any() else None,
            teacher_block_candidate_change_fraction=float((np.diff(np.array(tm['block_source_orders']),axis=0)!=0).mean()),
            identity_loss=float(lid),identity_pair_count=detail['pair_count'],identity_effective_weight=detail['effective_weight'],
            amplitude_conservation_max_error=pm['total_amplitude_max_error'])
        if TEACHER_MODE=='joint':
            amp,_=amplitude_loss(a,y,order)
            record['joint_objective_audit']=dict(**{key:tm[key] for key in (
                'objective_before','objective_after','amplitude_before','amplitude_after','identity_before','identity_after',
                'accepted_moves','changed_blocks','amplitude_cap')},
                teacher_training_cost_abs_error=abs(float(amp+.2*lid)-tm['objective_after']))
            legacy,_,_=envelope_teacher(a,y)
            record['joint_vs_legacy_active_assignment_change']=float((order[:,:b.shape[1]]!=legacy[:,:b.shape[1]])[b>0].mean()) if (b>0).any() else None
            # Independent per-node envelope witness. This is label fit evidence,
            # never proof of physical source identity when envelopes are ambiguous.
            err=np.abs(raw[:,:,None]-b[:,None,:])/scale
            chosen=err[np.arange(len(b))[:,None],order[:,:b.shape[1]],np.arange(b.shape[1])[None]]
            record['teacher_label_fit_audit']=dict(active_chosen_error=quantiles(chosen[b>0]),
                active_excess_over_best_envelope=quantiles((chosen-err.min(1))[b>0]),
                limitation='GT RMS fit witnesses do not certify identity of envelope-identical physical sources')
        selected=record['actual']['order'];mask=np.ones(8,dtype=bool);mask[selected]=False
        record['actual']['unused_mean_normalized_amplitude']=float(actual[:,mask].mean()/scale)
        padded=np.concatenate((b,np.zeros((len(b),8-b.shape[1]))),1)
        aligned=np.concatenate((actual[:,selected],actual[:,mask]),1)
        record['actual']['full_eight_normalized_mae']=float(np.abs(aligned-padded).mean()/scale)
        record['relations']=relation(vv,b/scale,order,conf,pm,detail)
        p,_=row['pairs']
        if len(p):
            record['positive_cross_slot_fraction']=float((order[p[:,0],p[:,1]]!=order[p[:,2],p[:,3]]).mean())
            record['positive_cross_silence_fraction']=float(np.mean([np.any(b[t:u+1,i]==0) for t,i,u,_ in p]))
        if extended:
            record['teacher_context']=tm
            record['predicted_edge_gap']=quantiles(pm['edge_gaps'])
            record['actual_gated']=measure(np.where(actual>.001,actual,0),b,row['times'],scale)
            record['teacher_gated']=measure(np.where(ta>.001,ta,0),b,row['times'],scale,order_override=np.arange(b.shape[1]))
            record['output_gain_diagnostic']={}
            for gain in (.5,1.,2.):
                gain_a,_=connect(vv*gain,scale,diagnostics=False,amplitudes=raw*gain)
                record['output_gain_diagnostic'][str(gain)]=measure(gain_a,b*gain,row['times'],scale)
            generator=np.random.default_rng(5151)
            shuffle=np.array([generator.permutation(8) for _ in range(len(vv))])
            shuffled=vv[np.arange(len(vv))[:,None],shuffle]
            sh,_=connect(shuffled,scale,diagnostics=False,amplitudes=raw[np.arange(len(raw))[:,None],shuffle])
            record['candidate_shuffle_normalized_mae_delta']=measure(sh,b,row['times'],scale)['normalized_mae']-record['actual']['normalized_mae']
        if out is not None:
            np.savez_compressed(out/(row['stage']+'-'+row['entry']['sample_id']+'.npz'),
                times=row['times'],v=vv,raw_a=raw,actual_a=actual,teacher_a=ta,target=b,
                teacher_order=order,teacher_confidence=conf,predicted_order=pm['orders'])
        records.append(record)
    model.train();return records


def score(records):
    return float(np.mean([r['actual']['normalized_mae']+.1*(1-np.mean(r['actual']['all']['per_source_iou']))
        +.1*r['actual']['unused_mean_normalized_amplitude']+.01*r['actual']['source_count_mae'] for r in records]))


def gradients(model,v,amp,identity):
    av=torch.autograd.grad(amp,v,retain_graph=True,allow_unused=True)[0]
    if av is None:av=torch.zeros_like(v)
    iv=torch.autograd.grad(identity,v,retain_graph=True)[0]
    unit=v.detach()/v.detach().norm(dim=-1,keepdim=True).clamp_min(1e-8)
    def split(g):
        radial=(g*unit).sum(-1,keepdim=True)*unit
        return dict(norm=float(g.norm()),radial_norm=float(radial.norm()),tangent_norm=float((g-radial).norm()))
    ap=torch.autograd.grad(amp,model.project.weight,retain_graph=True)[0].float().flatten()
    ip=torch.autograd.grad(identity,model.project.weight,retain_graph=True)[0].float().flatten()
    return dict(amplitude_vector=split(av),identity_vector=split(iv),shared_projection_amplitude_norm=float(ap.norm()),
        shared_projection_identity_norm=float(ip.norm()),shared_projection_gradient_cosine=float(torch.dot(ap,ip)/(ap.norm()*ip.norm()).clamp_min(1e-20)))


def fit(model,rows,current,scale,weight,steps,seconds,out):
    out.mkdir(parents=True,exist_ok=True)
    train=[r for r in rows if r['entry']['split']=='train'];new=[r for r in train if r['stage']==current]
    old=[r for r in train if r['stage']!=current];val=[r for r in rows if r['entry']['split']=='val']
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf');best_state=None;best_step=0;meaningful=float('inf');stale=0;history=[];start=time.monotonic();update=0;reason='max_updates'
    for step in range(1,steps+1):
        if time.monotonic()-start>=seconds:reason='wall_clock_budget';break
        selected=[random.choice(old or new),random.choice(new)]
        optimizer.zero_grad(set_to_none=True);parts=[];log=step==1 or step%50==0 or step==steps
        for row in selected:
            v,a=model(row['tokens'],row['rms']/scale);y=row['target']/scale
            order,conf,tm=teacher(a,y,v=v,pairs=row['pairs']);amp,ad=amplitude_loss(a,y,order)
            identity,idd=identity_loss(v,y,order,conf,row['pairs']);value=amp+weight*identity
            if not torch.isfinite(value):raise ValueError('nonfinite teacher loss')
            gd=gradients(model,v,amp,identity) if log else None
            (value/2).backward()
            if log:parts.append(dict(stage=row['stage'],loss=float(value.detach()),amplitude_loss=float(amp.detach()),
                identity_loss=float(identity.detach()),weighted_identity_loss=float(weight*identity.detach()),
                point=float(ad['point'].detach()),area_iou_loss=float(ad['area_iou_loss'].detach()),
                silence=float(ad['silence'].detach()),zero_tracks=float(ad['zero_tracks'].detach()),
                identity_pair_count=idd['pair_count'],identity_effective_weight=idd['effective_weight'],gradient=gd,
                teacher_objective=tm if TEACHER_MODE=='joint' else None))
        gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not np.isfinite(gn):raise ValueError('nonfinite gradient; step refused')
        optimizer.step();update=step
        if log:
            records=evaluate(model,val,scale);sc=score(records)
            if sc<best:
                best=sc;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if sc<meaningful*.99:meaningful=sc;stale=0
            else:stale+=1
            h=dict(update=step,parts=parts,gradient_norm=gn,val_score=sc,seconds=time.monotonic()-start,
                validation_compact=[dict(stage=r['stage'],sample_id=r['sample_id'],actual=r['actual'],teacher=r['teacher'],teacher_confidence=r['teacher_confidence'],identity_loss=r['identity_loss']) for r in records])
            history.append(h);(out/'progress.json').write_text(json.dumps(history,indent=1)+'\n')
            print(json.dumps(dict(stage=current,weight=weight,update=step,val_score=sc,seconds=h['seconds'])),flush=True)
            if step<steps and step>=1000 and stale>=10:reason='val_plateau_10_checks';break
    torch.save(model.state_dict(),out/'last-head.pth')
    if best_state is not None:model.load_state_dict(best_state)
    torch.save(model.state_dict(),out/'head.pth')
    val_out=out/'validation';val_out.mkdir(exist_ok=True)
    validation=evaluate(model,val,scale,out=val_out,extended=True)
    test=evaluate(model,[r for r in rows if r['entry']['split']=='test'],scale,out=out,extended=True)
    report=dict(stage=current,identity_weight=weight,updates=update,selected_update=best_step,stop_reason=reason,
        plateau_condition_at_stop=bool(update>=1000 and stale>=10),seconds=time.monotonic()-start,
        budget=dict(updates=steps,seconds=seconds),scale_r=scale,history=history,validation=validation,test=test,
        selection='prediction-only actual MAE+.1(1-sourceIoU)+.1emptymeanA/r+.01countMAE; equal sample all seen val',
        training=TEACHER_MODE+' teacher; original norm amplitude + same direct ID .2; see joint objective scope',
        inference='predicted cosine continuous reliability adjacent Hungarian; no truth, predicted hard gate, soft amplitude/vector average or deployment history')
    (out/'result.json').write_text(json.dumps(report,indent=1)+'\n');return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher-mode',choices=('joint','envelope'),default='joint');p.add_argument('--cache',type=Path,required=True);p.add_argument('--c0-cache',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--identity-weight',type=float,choices=(0.,.2),required=True);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--amplitude-transform',choices=('abs','softplus'),default='softplus')
    p.add_argument('--head-mode',choices=('coupled','split'),default='coupled');p.add_argument('--seconds',type=int,default=7200);p.add_argument('--pilot',action='store_true');args=p.parse_args()
    global TEACHER_MODE;TEACHER_MODE=args.teacher_mode
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;seed(46)
    args.out.mkdir(parents=True,exist_ok=True);rows,manifest=read(args.c0_cache,'C0');prepare(rows)
    active=torch.cat([r['target'][r['target']>.001] for r in rows if r['entry']['split']=='train']);scale=float(torch.quantile(active,.95))
    model=(WindowSplitHead(4,amplitude_transform=args.amplitude_transform) if args.head_mode=='split' else WindowVHead(4)).cuda();model.load_state_dict(torch.load(args.initial,weights_only=True))
    metadata=dict(architecture=model.architecture(),initial=str(args.initial),initial_sha256=sha256(args.initial),
        head_mode=args.head_mode,teacher_mode=args.teacher_mode,identity_weight=args.identity_weight,scale_r=scale,seed=46,pilot=args.pilot,
        initial_description='same historical selected depth4 raw C1 head; full frozen SSAST197-token caches',c0_manifest_sha256=sha256(args.c0_cache/'manifest.json'))
    (args.out/'architecture.json').write_text(json.dumps(metadata,indent=2)+'\n')
    summary=[]
    for index,stage in enumerate(('C1','C2-low','C2-high','C3')):
        newer,m=read(args.cache/stage,stage);prepare(newer);rows+=newer;seed(4650+index)
        if index==0:
            initial=evaluate(model,[r for r in rows if r['entry']['split']=='train'],scale,extended=True)
            (args.out/'initial-train-diagnostics.json').write_text(json.dumps(initial,indent=1)+'\n')
        if args.pilot:
            # Train-only pipeline probe, no final weights or val/test selection.
            row=[r for r in newer if r['entry']['split']=='train'][0];optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[]
            for step in range(1,args.steps+1):
                optimizer.zero_grad(set_to_none=True);v,a=model(row['tokens'],row['rms']/scale);y=row['target']/scale
                order,conf,_=teacher(a,y,v=v,pairs=row['pairs']);amp,ad=amplitude_loss(a,y,order);identity,idd=identity_loss(v,y,order,conf,row['pairs'])
                value=amp+args.identity_weight*identity;gd=gradients(model,v,amp,identity) if step in (1,args.steps) else None
                value.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));
                if not np.isfinite(gn):raise ValueError('pilot nonfinite gradient')
                optimizer.step()
                if step==1 or step%20==0 or step==args.steps:
                    history.append(dict(update=step,amplitude_loss=float(amp.detach()),identity_loss=float(identity.detach()),gradient=gd,
                        confidence=quantiles(conf[y.cpu().numpy()>0]),zero_tracks=float(ad['zero_tracks'].detach())))
            (args.out/'pilot.json').write_text(json.dumps(dict(history=history,final_train=evaluate(model,[row],scale,extended=True)),indent=1)+'\n')
            print('TRAIN-ONLY PILOT COMPLETE',flush=True);return
        record=fit(model,rows,stage,scale,args.identity_weight,args.steps,args.seconds,args.out/stage)
        record['schedule_seed']=4650+index;record['cache_manifest_sha256']=sha256(args.cache/stage/'manifest.json')
        summary.append(record);(args.out/'summary.json').write_text(json.dumps(summary,indent=1)+'\n')
    print('TEACHER ARM COMPLETE',args.identity_weight,flush=True)


if __name__=='__main__':main()
