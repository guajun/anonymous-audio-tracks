"""Save enumerable truth witnesses and measured optimization approximation gaps."""
import json
from pathlib import Path
import numpy as np
import torch
from joint_teacher import Objective,teacher
from teacher_assignment import amplitude_loss
from identity_supervision import identity_loss
from test_joint_teacher import example


def main():
    out=Path('runs/split-head/evidence');out.mkdir(parents=True,exist_ok=True)
    v,y=example();cases=[]
    for name,x,b in [('identity_swap',v,y),('stable_identity',v.copy(),y),('strong_conflict',v*5,y*5),
                     ('all_zero_gt',v,np.zeros_like(y)),('near_zero_gt',v*1e-5,y*1e-5),
                     ('copied_outputs',np.repeat(v[:,0:1],3,axis=1),y)]:
        if name=='stable_identity':x[2:]=x[2:,:,::-1]
        a=np.linalg.norm(x,axis=-1)
        # Independent Z radius differs; A is explicitly retained.
        x=x*7.
        o,c,m=teacher(x,b,block=1,sweeps=1,amplitudes=a)
        eo,_,em=teacher(x,b,block=1,exhaustive=True,amplitudes=a)
        obj=Objective(x,b,block=1,amplitudes=a)
        amp,_=amplitude_loss(torch.tensor(obj.a),torch.tensor(b),o)
        lid,_=identity_loss(torch.tensor(x),torch.tensor(b),o,c,(obj.p,obj.neg))
        cases.append(dict(name=name,coordinate=m,exhaustive=em,order=o.tolist(),exhaustive_order=eo.tolist(),
            optimality_gap=m['objective_after']-em['objective_after'],
            torch_full_cost=float(amp+.2*lid),full_cost_abs_error=abs(float(amp+.2*lid)-m['objective_after'])))
    rng=np.random.default_rng(5152);random=[]
    for index in range(24):
        x=rng.normal(size=(4,3,3)).astype(np.float32)*.15
        b=rng.uniform(0,.3,size=(4,2)).astype(np.float32)
        a=np.linalg.norm(x,axis=-1)*np.random.default_rng(6152+index).uniform(.5,1.5,(4,3)).astype(np.float32)
        _,_,m=teacher(x,b,block=1,amplitudes=a);_,_,em=teacher(x,b,block=1,exhaustive=True,amplitudes=a)
        random.append(dict(index=index,before=m['objective_before'],coordinate=m['objective_after'],
            exact=em['objective_after'],gap=m['objective_after']-em['objective_after']))
    result=dict(cases=cases,random_scope='24 seed5152 four-frame 3P2 independent A random factor seed6152+index, all 6^4 combinations within amplitude/node trust region',
        random=random,max_coordinate_optimality_gap=max(r['gap'] for r in random),
        cases_with_nonzero_gap=sum(r['gap']>1e-7 for r in random),tests='40 passed',
        inference='unchanged prediction-only; teacher exhaustive scope is diagnostic, not deployed')
    (out/'issue-51-split-small.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(max_gap=result['max_coordinate_optimality_gap'],nonzero=result['cases_with_nonzero_gap'],
        examples=[dict(name=c['name'],before=c['coordinate']['objective_before'],after=c['coordinate']['objective_after'],exact=c['exhaustive']['objective_after'],error=c['full_cost_abs_error']) for c in cases])))


if __name__=='__main__':main()
