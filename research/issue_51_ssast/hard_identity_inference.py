"""Prediction-only adjacent one-to-one identity edges; no amplitude mixing."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def connect(v,scale, *, diagnostics=True):
    v=np.asarray(v);a=np.linalg.norm(v,axis=-1);n,k=a.shape
    e=np.divide(v,a[:,:,None],out=np.zeros_like(v),where=a[:,:,None]>0)
    reliability=a/(a+.01*scale)
    scores=e[:-1]@np.swapaxes(e[1:],1,2)
    scores*=np.sqrt(reliability[:-1,:,None]*reliability[1:,None,:])
    defined=(a[:-1,:,None]>0)&(a[1:,None,:]>0)
    scores=np.where(defined,scores,0)
    links=[];gaps=[];orders=[np.arange(k)]
    for matrix in scores:
        r,c=linear_sum_assignment(-matrix);mapping=np.empty(k,dtype=int);mapping[r]=c
        links.append(mapping);orders.append(mapping[orders[-1]])
        if diagnostics:
            best=matrix[r,c].sum();edge=[]
            for row,column in zip(r,c):
                alt=matrix.copy();alt[row,column]=-1e6
                rr,cc=linear_sum_assignment(-alt)
                edge.append(max(0.,float(best-alt[rr,cc].sum())))
            gaps.append(edge)
    orders=np.array(orders);result=np.take_along_axis(a,orders,axis=1)
    return result,dict(orders=orders,links=np.array(links,dtype=int).reshape(-1,k),
        edge_gaps=np.array(gaps),reliability_scale=.01*scale,
        inference='adjacent Hungarian weighted predicted cosine, no ground truth/gate/history replacement',
        identity_zero='exact-zero candidate has no descriptor; dummy zero scores, no stable zero identity claim',
        total_amplitude_max_error=float(np.abs(result.sum(1)-a.sum(1)).max()))
