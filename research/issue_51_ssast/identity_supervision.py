"""Truth-weighted positive/negative temporal cosine ranking, two-way competition."""
import numpy as np
import torch


def make_pairs(target, offsets=(1,2,5,15,30,45),radius=45):
    y=np.asarray(target);n,s=y.shape;positive=[];negative=[]
    for i in range(s):
        active=np.flatnonzero(y[:,i]>0)
        for k in range(s):
            if k==i:continue
            other=np.flatnonzero(y[:,k]>0)
            if not len(other):continue
            def nearest(time):
                pos=np.searchsorted(other,time)
                candidates=[other[q] for q in (pos-1,pos) if 0<=q<len(other)]
                return min(candidates,key=lambda q:abs(q-time))
            for lag in offsets:
                for t in active:
                    u=t+lag
                    if u>=n or y[u,i]<=0:continue
                    forward=nearest(u);reverse=nearest(t)
                    if max(abs(forward-u),abs(reverse-t))>radius:continue
                    positive.append((t,i,u,i));negative.append((forward,k,reverse,k))
    return np.array(positive,dtype=np.int64).reshape(-1,4),np.array(negative,dtype=np.int64).reshape(-1,4)


def identity_loss(v,y,order,confidence,pairs, *, margin=.2,pull=.05):
    p,neg=pairs;s=y.shape[1];index=torch.as_tensor(order[:,:s],device=v.device)
    real=v.gather(1,index[:,:,None].expand(-1,-1,v.shape[-1]))
    a=real.norm(dim=-1)
    unit=torch.where((a>0)[:,:,None],real/a.clamp_min(1e-8)[:,:,None],torch.zeros_like(real))
    conf=torch.as_tensor(confidence,device=v.device,dtype=v.dtype)
    activity=y/(y+.01)
    weights=activity*conf*(a.detach()>0)
    if not len(p):
        return v.sum()*0,dict(pair_count=0,effective_weight=0.,positive=None,negative=None,
            weights=v.new_empty(0),positive_cosine=v.new_empty(0),negative_cosine=v.new_empty(0))
    p=torch.as_tensor(p,device=v.device);neg=torch.as_tensor(neg,device=v.device)
    x=unit[p[:,0],p[:,1]];z=unit[p[:,2],p[:,3]]
    nf=unit[neg[:,0],neg[:,1]];nr=unit[neg[:,2],neg[:,3]]
    positive=(x*z).sum(-1);forward=(x*nf).sum(-1);reverse=(nr*z).sum(-1)
    weight=(weights[p[:,0],p[:,1]]*weights[p[:,2],p[:,3]]*
        weights[neg[:,0],neg[:,1]]*weights[neg[:,2],neg[:,3]]).sqrt()
    ranking=.5*(torch.relu(margin+forward-positive)+torch.relu(margin+reverse-positive))
    # Normalize by GT activity alone. Confidence must reduce total supervision,
    # not disappear by dividing by the same small confidence-weight sum.
    base=(activity[p[:,0],p[:,1]]*activity[p[:,2],p[:,3]]*
        activity[neg[:,0],neg[:,1]]*activity[neg[:,2],neg[:,3]]).sqrt()
    value=((ranking+pull*(1-positive))*weight).sum()/base.sum().clamp_min(1e-12)
    return value,dict(pair_count=len(p),effective_weight=float(weight.sum().detach()),
        positive_cosine=positive,negative_cosine=.5*(forward+reverse),weights=weight,
        margin_violation=(.5*((forward+margin>positive).float()+(reverse+margin>positive).float())))
