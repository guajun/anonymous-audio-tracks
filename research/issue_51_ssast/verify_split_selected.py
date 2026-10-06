"""Reload all selected checkpoints and audit prediction artifacts/endpoint labels."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from window_head import WindowVHead, WindowSplitHead
from train_window_depth import read,seed
from features import sha256
from joint_teacher import teacher
from teacher_assignment import adjacency
from train_joint_teacher import prepare


def endpoints(z,source_rows):
    y=z['target'];order=z['teacher_order'];po=z['predicted_order'];n,k=po.shape;s=y.shape[1]
    inverse=np.empty_like(po);inverse[np.arange(n)[:,None],po]=np.arange(k)[None]
    labels=inverse[np.arange(n)[:,None],order[:,:s]];active=y>0
    out=[]
    for i in range(s):
        edges=np.diff(np.r_[False,active[:,i],False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
        pairs=[(l,r) for l,r in zip(ends[:-1],starts[1:]) if r-l<=45]
        out.append(dict(source=i,pairs=len(pairs),same_any_predicted_row=int(sum(labels[l,i]==labels[r,i] for l,r in pairs)),
            both_in_whole_sequence_source_row=int(sum(labels[l,i]==source_rows[i] and labels[r,i]==source_rows[i] for l,r in pairs)),
            incorrect=[dict(left=int(l),right=int(r),left_row=int(labels[l,i]),right_row=int(labels[r,i]),
                expected_row=int(source_rows[i])) for l,r in pairs if labels[l,i]!=source_rows[i] or labels[r,i]!=source_rows[i]]))
    return out


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;seed(46)
    root=Path('runs/split-head/fit-v1');out=Path('runs/split-head/evidence');out.mkdir(parents=True,exist_ok=True)
    rows,_=read(Path('runs/p1/frame-c0-cache'),'C0')
    for stage in ('C1','C2-low','C2-high','C3'):
        new,_=read(Path('runs/window-depth/cache')/stage,stage);rows+=new
    prepare(rows);by_id={(r['stage'],r['entry']['sample_id']):r for r in rows};model=WindowVHead(4).cuda();records=[];start=time.monotonic()
    for arm in ('coupled','split'):
        model=(WindowSplitHead(4) if arm=='split' else WindowVHead(4)).cuda()
        scale=json.loads((root/arm/'architecture.json').read_text())['scale_r']
        summary=json.loads((root/arm/'summary.json').read_text())
        for result in summary:
            stage=result['stage'];checkpoint=root/arm/stage/'head.pth';model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval()
            for split_name in ('validation','test'):
              for r in result[split_name]:
                row=by_id[(r['stage'],r['sample_id'])];path=root/arm/stage/('validation' if split_name=='validation' else '')/(r['stage']+'-'+r['sample_id']+'.npz')
                z=np.load(path)
                with torch.no_grad():v,a=model(row['tokens'],row['rms']/scale)
                delta=float(np.abs(v.cpu().numpy()*scale-z['v']).max())
                if delta>1e-6:raise ValueError('selected checkpoint does not reproduce saved V: '+str(delta))
                adelta=float(np.abs(a.cpu().numpy()*scale-z['raw_a']).max())
                if adelta>1e-6:raise ValueError('independent amplitude reload failed')
                from hard_identity_inference import connect
                from teacher_assignment import amplitude_loss
                from identity_supervision import identity_loss
                from course_metrics import measure
                actual,pm=connect(v.cpu().numpy()*scale,scale,diagnostics=False,amplitudes=a.cpu().numpy()*scale)
                np.testing.assert_allclose(actual,z['actual_a'],atol=1e-6)
                o,c,tm=teacher(v,row['target']/scale,amplitudes=a,pairs=row['pairs'])
                np.testing.assert_array_equal(o,z['teacher_order'])
                la,_=amplitude_loss(a,row['target']/scale,o)
                li,_=identity_loss(v,row['target']/scale,o,c,row['pairs'])
                objective_error=abs(float(la+.2*li)-tm['objective_after'])
                if objective_error>1e-6:raise ValueError('teacher/torch objective mismatch')
                g=adjacency(z['teacher_order']);one_to_one=bool(np.all(g.sum(1)==1)&np.all(g.sum(2)==1))
                if not one_to_one:raise ValueError('G not a permutation')
                records.append(dict(arm=arm,checkpoint_stage=stage,evaluation_stage=r['stage'],sample_id=r['sample_id'],
                    checkpoint_sha256=sha256(checkpoint),split=split_name,max_a_error=adelta,objective_error=objective_error,max_v_error=delta,g_one_to_one=one_to_one,
                    raw_actual_count_max_error=int(np.abs((z['raw_a']>.001).sum(1)-(z['actual_a']>.001).sum(1)).max()),
                    endpoint_labels=endpoints(z,r['actual']['order']),
                    teacher_label_confidence_limitation='same-row endpoint measure is weaker than both endpoints in whole-sequence GT source row; RMS cannot certify identical physical sources'))
    sensitivity=[];evidence_reactions=[]
    row=next(r for r in rows if r['stage']=='C3' and r['entry']['split']=='train')
    for arm in ('coupled','split'):
        model=(WindowSplitHead(4) if arm=='split' else WindowVHead(4)).cuda()
        model.load_state_dict(torch.load(root/arm/'C3'/'head.pth',weights_only=True))
        with torch.no_grad():v,a=model(row['tokens'],row['rms']/scale)
        yy=(row['target']/scale).cpu().numpy();base_order=None
        for block,sweeps in ((16,1),(16,3),(1,1)):
            st=time.monotonic();o,c,m=teacher(v,row['target']/scale,amplitudes=a,pairs=row['pairs'],block=block,sweeps=sweeps)
            sensitivity.append(dict(arm=arm,block=block,sweeps=sweeps,seconds=time.monotonic()-st,metadata=m))
            if block==16 and sweeps==1:base_order=o.copy()
        for name,x,weight in [('no_identity_objective',v,0.),('same_A_same_reliability_time_direction_flip',v*torch.tensor(np.where(np.arange(len(v))%2,1.,-1.),device=v.device,dtype=v.dtype)[:,None,None],.2)]:
            o,c,m=teacher(x,row['target']/scale,amplitudes=a,pairs=row['pairs'],weight=weight)
            evidence_reactions.append(dict(arm=arm,name=name,metadata=m,active_assignment_change=float((o[:,:yy.shape[1]]!=base_order[:,:yy.shape[1]])[yy>0].mean())))
    result=dict(records=records,npz_rechecks=len(records),max_v_error=max(r['max_v_error'] for r in records),
        max_a_error=max(r['max_a_error'] for r in records),max_objective_error=max(r['objective_error'] for r in records),count_conservation_max_error=max(r['raw_actual_count_max_error'] for r in records),train_only_search_sensitivity=sensitivity,
        real_train_identity_evidence_reactions=evidence_reactions,
        seconds=time.monotonic()-start,no_test_tuned_thresholds=True)
    (out/'issue-51-split-selected-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rechecks=len(records),max_v_error=result['max_v_error'],seconds=result['seconds'],
        sensitivity=[dict(block=r['block'],sweeps=r['sweeps'],cost=r['metadata']['objective_after']) for r in sensitivity])))


if __name__=='__main__':main()
