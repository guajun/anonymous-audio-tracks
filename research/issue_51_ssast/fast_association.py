"""CPU discrete selection, batched differentiable GPU relation computation.

Reference selection was already discrete. Preserve original selector and all
referenced real E gradients; do not introduce truncated BPTT or a new loss.
CPU/GPU floating-point tie decisions can differ; report this numerical limit.
"""
import torch
from local_association import transport as reference_transport, sinkhorn


def transport(v,times,*,gate=.001,horizon=.9,temperature=.1):
    with torch.no_grad():
        _,_,meta=reference_transport(v.detach().cpu(),times,gate=gate,
            horizon=horizon,temperature=temperature,record_selection=True)
    records=meta.pop('selection')
    t,k,d=v.shape
    rows=torch.tensor([r['rows'] for r in records],device=v.device)
    columns=torch.tensor([r['columns'] for r in records],device=v.device)
    indices=torch.tensor([r['references'] for r in records],device=v.device)
    has_ref=indices[:,:,0]>=0
    flattened=(indices[:,:,0]*k+indices[:,:,1]).clamp_min(0)
    a=torch.linalg.vector_norm(v,dim=-1)
    valid=a.detach()>gate
    e=torch.where(valid[...,None],v/a.clamp_min(1e-12)[...,None],torch.zeros_like(v))
    refs=e.reshape(t*k,d)[flattened]
    refs=refs/refs.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    current=e.gather(1,columns[...,None].expand(t,k,d))
    active=valid.gather(1,columns)
    scores=(refs@current.transpose(1,2))/temperature
    scores=torch.where(has_ref[:,:,None]&active[:,None,:],scores,torch.zeros_like(scores))
    weights=sinkhorn(scores)
    weights=weights*active[:,None,:]
    c=torch.zeros_like(weights)
    batch=torch.arange(t,device=v.device)[:,None,None]
    c[batch,rows[:,:,None],columns[:,None,:]]=weights
    direct=torch.tensor([r['direct'] for r in records],device=v.device)
    anchors=torch.diag_embed(valid.to(v.dtype))
    c=torch.where(direct[:,None,None],anchors,c)
    meta['execution']='original discrete selector on CPU; same referenced genuine E in batched differentiable relations'
    meta['numerical_limit']='CPU/GPU near-tie discrete choices may differ; equivalence tests record tolerance'
    return (c@a[...,None]).squeeze(-1),c,meta
