"""Full-loss joint teacher. Exact cost, restricted block coordinate search, no index prior.

All confidence is frozen BEFORE search; assignments cannot lower their own weight.
The amplitude trust region bounds exploitation of unreliable predicted directions.
"""
import itertools
import numpy as np
import torch
from teacher_assignment import teacher as envelope_teacher, states
from identity_supervision import make_pairs


def array(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x,dtype=np.float32)


def frozen_confidence(v,y):
    v=array(v);y=array(y);a=np.linalg.norm(v,axis=-1);s=y.shape[1]
    distinct=0. if s==1 else float(np.mean(np.abs(y[:,0]-y[:,1])))
    distinct=distinct/(distinct+.01)
    unit=v/np.maximum(a[:,:,None],1e-8)
    gram=np.einsum('nkd,nld->nkl',unit,unit)
    valid=(a>0)[:,:,None]&(a>0)[:,None,:]
    spread=np.max(np.where(valid,1-gram,0),axis=(1,2)).clip(0,1)
    return np.repeat((distinct*spread)[:,None],s,axis=1).astype(np.float32)


class Objective:
    def __init__(self,v,y,pairs=None,block=16,weight=.2):
        self.v=array(v);self.y=array(y);self.a=np.linalg.norm(self.v,axis=-1)
        self.n,self.k=self.a.shape;self.s=self.y.shape[1];self.block=block;self.weight=weight
        self.states=states(self.k,self.s);self.b=(self.n+block-1)//block
        self.unit=self.v/np.maximum(self.a[:,:,None],1e-8)
        self.p,self.neg=make_pairs(self.y) if pairs is None else pairs
        # GT source names cannot be identified when their entire envelopes agree.
        # Identical candidate directions offer no identity evidence. Continuous,
        # prediction-derived but assignment-independent reliability, frozen here.
        self.conf=frozen_confidence(self.v,self.y)
        activity=self.y/(self.y+.01)
        if len(self.p):
            p=self.p;q=self.neg
            self.base=np.sqrt(activity[p[:,0],p[:,1]]*activity[p[:,2],p[:,3]]*
                activity[q[:,0],q[:,1]]*activity[q[:,2],q[:,3]])
            c=self.conf
            self.pw=self.base*np.sqrt(c[p[:,0],p[:,1]]*c[p[:,2],p[:,3]]*c[q[:,0],q[:,1]]*c[q[:,2],q[:,3]])
            self.denom=max(float(self.base.sum()),1e-12)
            self.cp=np.einsum('pkd,pld->pkl',self.unit[p[:,0]],self.unit[p[:,2]])
            self.cf=np.einsum('pkd,pld->pkl',self.unit[p[:,0]],self.unit[q[:,0]])
            self.cr=np.einsum('pkd,pld->pkl',self.unit[q[:,2]],self.unit[p[:,2]])
        self.tables=[]
        silent=self.y==0;self.silent_count=max(1,int(silent.sum()));self.present=self.y.sum(0)>0
        for j in range(self.b):
            lo=j*block;hi=min(self.n,lo+block);a=self.a[lo:hi];y=self.y[lo:hi]
            d=np.abs(a[:,:,None]-y[:,None,:])
            point=np.where(d<.1,5*d*d,d-.05).sum(0)/(self.n*self.s)
            silence=.1*(a[:,:,None]*silent[lo:hi,None,:]).sum(0)/self.silent_count
            inter=np.minimum(a[:,:,None],y[:,None,:]).sum(0)
            union=np.maximum(a[:,:,None],y[:,None,:]).sum(0)
            pp=self.states;ix=np.arange(self.s)
            zero=.1*(a.sum()-a.sum(0)[pp].sum(1))/(self.n*(self.k-self.s))
            self.tables.append((point[pp,ix].sum(1)+silence[pp,ix].sum(1)+zero,inter[pp,ix],union[pp,ix]))

    def orders(self,path):
        selected=np.repeat(self.states[np.asarray(path)],self.block,axis=0)[:self.n]
        return np.array([list(row)+[k for k in range(self.k) if k not in row] for row in selected])

    def amplitude(self,paths):
        paths=np.atleast_2d(paths);b=len(paths);point=np.zeros(b);inter=np.zeros((b,self.s));union=inter.copy()
        for j,(c,i,u) in enumerate(self.tables):
            point+=c[paths[:,j]];inter+=i[paths[:,j]];union+=u[paths[:,j]]
        if self.present.any():point+=.1*(1-inter[:,self.present]/np.maximum(union[:,self.present],1e-12)).mean(1)
        return point

    def pair_cost(self,paths,subset=None):
        paths=np.atleast_2d(paths)
        if not len(self.p):return np.zeros((len(paths),0))
        ix=np.arange(len(self.p)) if subset is None else subset;p=self.p[ix];q=self.neg[ix]
        selected=self.states[paths]
        x=selected[:,p[:,0]//self.block,p[:,1]];z=selected[:,p[:,2]//self.block,p[:,3]]
        f=selected[:,q[:,0]//self.block,q[:,1]];r=selected[:,q[:,2]//self.block,q[:,3]]
        pc=self.cp[ix][np.arange(len(ix))[None],x,z];fc=self.cf[ix][np.arange(len(ix))[None],x,f]
        rc=self.cr[ix][np.arange(len(ix))[None],r,z]
        nonzero=(self.a[p[:,0][None],x]>0)&(self.a[p[:,2][None],z]>0)&(self.a[q[:,0][None],f]>0)&(self.a[q[:,2][None],r]>0)
        return (.5*(np.maximum(.2+fc-pc,0)+np.maximum(.2+rc-pc,0))+.05*(1-pc))*self.pw[ix]*nonzero/self.denom

    def cost(self,paths):
        amp=self.amplitude(paths);identity=self.pair_cost(paths).sum(1)
        return amp+self.weight*identity,amp,identity


def teacher(v,y,*,pairs=None,block=16,weight=.2,sweeps=1,amplitude_slack=.02,node_slack=.02,exhaustive=False):
    obj=Objective(v,y,pairs,block,weight)
    old,_,_=envelope_teacher(obj.a,obj.y,block=block,switch=0.)
    initial=np.array([np.flatnonzero((obj.states==old[j*block,:obj.s]).all(1))[0] for j in range(obj.b)])
    # A global mean bound alone permits sacrificing individual genuine nodes.
    # Freeze a per-source, per-active-node GT envelope trust box around the
    # envelope initialization. This remains feasible by construction and cannot
    # be weakened by E or by an assignment's own confidence.
    allowed=[]
    for j in range(obj.b):
        lo=j*block;hi=min(obj.n,lo+block);yy=obj.y[lo:hi]
        error=np.abs(obj.a[lo:hi,:,None]-yy[:,None,:])
        baseline=error[:,obj.states[initial[j]],np.arange(obj.s)]
        candidate=error[:,obj.states,np.arange(obj.s)]
        allowed.append(np.all((candidate<=baseline[:,None,:]+node_slack+1e-9)|(yy[:,None,:]==0),axis=(0,2)))
    allowed=np.array(allowed)
    path=initial.copy();before,base_amp,base_id=obj.cost(path);cap=float(base_amp[0])+amplitude_slack
    history=[float(before[0])];moves=0;gaps=[]
    if exhaustive:
        if len(obj.states)**obj.b>200000:raise ValueError('exhaustive scope too large')
        paths=np.array(list(itertools.product(range(len(obj.states)),repeat=obj.b)))
        costs,amps,_=obj.cost(paths)
        feasible=allowed[np.arange(obj.b)[None],paths].all(1)
        costs=np.where((amps<=cap+1e-9)&feasible,costs,np.inf)
        path=paths[costs.argmin()];history.append(float(costs.min()));moves=int((path!=initial).sum())
    else:
        current_pairs=obj.pair_cost(path)[0];current_id=float(current_pairs.sum())
        for sweep in range(sweeps):
            for j in range(obj.b):
                paths=np.tile(path,(len(obj.states),1));paths[:,j]=np.arange(len(obj.states))
                amp=obj.amplitude(paths)
                if len(obj.p):
                    nodes=np.column_stack((obj.p[:,0],obj.p[:,2],obj.neg[:,0],obj.neg[:,2]))//block
                    affected=np.flatnonzero((nodes==j).any(1));new_pairs=obj.pair_cost(paths,affected)
                    ids=current_id-current_pairs[affected].sum()+new_pairs.sum(1)
                else:affected=np.array([],dtype=int);new_pairs=np.zeros((len(paths),0));ids=np.zeros(len(paths))
                costs=amp+weight*ids;costs=np.where((amp<=cap+1e-9)&allowed[j],costs,np.inf)
                best=int(costs.argmin());prior=path[j]
                # Strict descent preserves ties rather than fabricating evidence.
                if costs[best]<costs[prior]-1e-8:
                    path[j]=best;moves+=1;current_pairs[affected]=new_pairs[best];current_id=float(current_pairs.sum())
                    history.append(float(costs[best]))
                finite=np.sort(costs[np.isfinite(costs)]);gaps.append(float(finite[1]-finite[0]) if len(finite)>1 else None)
    after,amp,lid=obj.cost(path);orders=obj.orders(path)
    meta=dict(teacher='full amplitude + same direct bidirectional ID; frozen GT/directional reliability; no index continuity',
        block_frames=block,block_source_orders=obj.states[path].tolist(),sweeps=sweeps,exhaustive=exhaustive,
        search_scope='all one-to-one block states; coordinate descent, NOT global optimum',
        amplitude_slack=amplitude_slack,amplitude_cap=cap,objective_before=float(before[0]),objective_after=float(after[0]),
        node_envelope_slack=node_slack,admissible_states_per_block=allowed.sum(1).tolist(),
        amplitude_before=float(base_amp[0]),amplitude_after=float(amp[0]),identity_before=float(base_id[0]),identity_after=float(lid[0]),
        objective_history=history,accepted_moves=moves,changed_blocks=int((path!=initial).sum()),
        conditional_state_gaps=gaps,confidence_definition='frozen GT envelope distinctness times prediction direction spread; reliability, not posterior label correctness',
        confidence_mean=float(obj.conf.mean()),zero_rows='undefined identity; original norms still supervised')
    return orders,obj.conf,meta
