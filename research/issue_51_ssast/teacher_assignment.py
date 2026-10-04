"""Stopped-gradient contextual hard source/zero-track permutation teacher."""
from functools import lru_cache
import itertools
import numpy as np
import torch


@lru_cache(None)
def states(k,s):
    if s not in (1,2):raise ValueError('current experiment only C0/two sources')
    return np.array(list(itertools.permutations(range(k),s)),dtype=np.int64)


def teacher(a,y, *, block=16,switch=.01,confidence_scale=.02):
    # Normalized amplitudes. No predicted gate and no identity input to teacher.
    aa=a.detach().float().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    yy=y.detach().float().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    n,k=aa.shape;s=yy.shape[1];pp=states(k,s);emissions=[];strength=[]
    for start in range(0,n,block):
        x=aa[start:start+block];b=yy[start:start+block]
        delta=np.abs(x[:,:,None]-b[:,None,:])
        huber=np.where(delta<.1,5*delta**2,delta-.05).mean(0)/s
        inter=np.minimum(x[:,:,None],b[:,None,:]).sum(0)
        union=np.maximum(x[:,:,None],b[:,None,:]).sum(0).clip(1e-12)
        iou=.1*(1-inter/union)/s*(b.sum(0)>0)[None]
        silent=b==0
        silence=.1*(x[:,:,None]*silent[:,None,:]).sum(0)/max(1,silent.sum())
        real=(huber+iou+silence)[pp,np.arange(s)].sum(1)
        zero=.1*(x.mean(0).sum()-x.mean(0)[pp].sum(1))/(k-s)
        emissions.append(real+zero)
        peak=b.max(0);strength.append(peak/(peak+.01))
    emissions=np.array(emissions);strength=np.array(strength)
    mismatch=pp[:,None,:]!=pp[None,:,:]
    values=emissions[0].copy();back=[]
    for j in range(1,len(emissions)):
        weight=np.sqrt(strength[j-1]*strength[j])
        transition=switch*(mismatch*weight).sum(-1)/s
        total=values[:,None]+transition
        parent=total.argmin(0);back.append(parent)
        values=total[parent,np.arange(len(pp))]+emissions[j]
    path=[int(values.argmin())]
    for parent in reversed(back):path.append(int(parent[path[-1]]))
    path=path[::-1]
    orders=[];conf=[];gaps=[]
    for j,state in enumerate(path):
        selected=pp[state]
        order=list(selected)+[c for c in range(k) if c not in selected]
        gap=[]
        for i,candidate in enumerate(selected):
            # Emission-only marginal, not artificial confidence from switch prior.
            same=pp[:,i]==candidate
            gap.append(max(0.,float(emissions[j,~same].min()-emissions[j,same].min())))
        gap=np.array(gap);confidence=gap/(gap+confidence_scale)
        length=min(block,n-j*block)
        orders.extend([order]*length);conf.extend([confidence.tolist()]*length);gaps.append(gap.tolist())
    return np.array(orders),np.array(conf),dict(block_frames=block,block_seconds=block*.02,
        switch_penalty=switch,confidence_scale=confidence_scale,
        block_source_orders=pp[path].tolist(),emission_marginal_gaps=gaps,
        zero_rows='remaining candidate order arbitrary; zero truth has no identity target',
        teacher='context envelope cost + amplitude-weighted candidate-index continuity; stopped gradient; no predicted E')


def permutation(order,k=8):
    order=np.asarray(order)
    p=np.zeros((len(order),k,k),dtype=np.float32)
    p[np.arange(len(order))[:,None],np.arange(k)[None],order]=1
    return p


def adjacency(order,k=8):
    p=permutation(order,k)
    return np.swapaxes(p[:-1],1,2)@p[1:]


def amplitude_loss(a,y,order):
    # Reorder original scalar norms, never average128-dimensional vectors.
    index=torch.as_tensor(order,device=a.device)
    aligned=a.gather(1,index);s=y.shape[1]
    real=aligned[:,:s]
    point=torch.nn.functional.huber_loss(real,y,delta=.1,reduction='mean')/.1
    inter=torch.minimum(real,y).sum(0)
    union=torch.maximum(real,y).sum(0).clamp_min(1e-12)
    present=y.sum(0)>0
    area=(1-inter[present]/union[present]).mean() if present.any() else a.sum()*0
    mask=y==0;silence=real[mask].mean() if mask.any() else a.sum()*0
    zero=aligned[:,s:].mean()
    value=point+.1*area+.1*silence+.1*zero
    return value,dict(point=point,area_iou_loss=area,silence=silence,zero_tracks=zero,
        padded_targets=torch.cat((y,torch.zeros_like(aligned[:,s:])),1),aligned_amplitudes=aligned)
