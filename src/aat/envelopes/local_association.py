"""Bounded endpoint distributions, direct anchors, and honest track fragments.

No persistent normalized prototype. Endpoints are actual nonzero observations
and differentiable correspondence weights, replaced without confidence gating.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch.nn import functional as F
from .models import magnitude_gate

VERSION='aat-local-endpoint-fragments-v3'


@dataclass(frozen=True)
class LocalAssociationConfig:
    temperature: float=.1
    similarity_gate: float=.5  # relative bias of evidence versus unused lanes
    iterations: int=12
    context_seconds: float=2.
    hop_seconds: float=.04
    edge_guard_seconds: float=.05
    evidence_seconds: float=.04
    transport_gate: float=.002

    def __post_init__(self):
        values=(self.temperature,self.similarity_gate,self.context_seconds,self.hop_seconds,self.edge_guard_seconds,self.evidence_seconds,self.transport_gate)
        if not all(math.isfinite(v) for v in values) or min(self.temperature,self.context_seconds,self.hop_seconds)<=0:
            raise ValueError('finite positive local settings required')
        if self.iterations<1 or min(self.edge_guard_seconds,self.evidence_seconds)<0 or self.maximum_endpoint_distance<=0:
            raise ValueError('invalid local evidence margin')
        if not 0<=self.transport_gate<1:raise ValueError('invalid transport magnitude gate')

    @property
    def maximum_endpoint_distance(self):
        return self.context_seconds/2-self.edge_guard_seconds-self.evidence_seconds


def balanced(logits,iterations):
    """Unit-capacity soft allocation, columns exact, rows approximate."""
    log_a=logits
    for _ in range(iterations):
        log_a=log_a-log_a.logsumexp(1,keepdim=True)
        log_a=log_a-log_a.logsumexp(0,keepdim=True)
    a=log_a.exp()
    return a/a.sum(0,keepdim=True).clamp_min(1e-12)


def rollout(directions,intensity,config=LocalAssociationConfig(),*,center_times=None,context_bounds=None):
    if directions.ndim!=3 or intensity.shape!=directions.shape[:2] or not len(intensity):
        raise ValueError('expected [T,K,D] and [T,K]')
    if not 1<=intensity.shape[1]<=8 or not torch.isfinite(directions).all() or not torch.isfinite(intensity).all() or (intensity<0).any():
        raise ValueError('finite nonnegative candidates required')
    count,k=intensity.shape
    times=np.arange(count)*config.hop_seconds if center_times is None else np.asarray(torch.as_tensor(center_times).cpu(),dtype=float)
    if times.shape!=(count,) or not np.isfinite(times).all() or (np.diff(times)<=0).any():
        raise ValueError('increasing actual inference times required')
    bounds=np.stack((times-config.context_seconds/2,times+config.context_seconds/2),1) if context_bounds is None else np.asarray(torch.as_tensor(context_bounds).cpu(),dtype=float)
    if bounds.shape!=(count,2) or not np.isfinite(bounds).all() or not np.allclose(bounds.mean(1),times,atol=1e-6) or not np.allclose(np.diff(bounds,axis=1),config.context_seconds,atol=1e-6):
        raise ValueError('actual context bounds must agree with centers and width')
    # Zero observations have no E. A normalized endpoint is never fabricated.
    nonzero=intensity.detach()>0
    norms=directions.norm(dim=-1)
    if ((norms.detach()<=1e-8)&nonzero).any():raise ValueError('nonzero activity requires a valid direction')
    e=F.normalize(directions,dim=-1)*nonzero[...,None]
    endpoint_e=[None]*k;endpoint_weights=[None]*k
    last=np.full(k,-1,dtype=int);fragments=np.full(k,-1,dtype=int)
    next_fragment=k  # K zero lanes let closed magnitude gates receive A gradients.
    used_initial=np.zeros(k,dtype=bool)
    def new_fragment(row):
        nonlocal next_fragment
        if not used_initial[row]:
            used_initial[row]=True
            return row
        value=next_fragment;next_fragment+=1
        return value
    values,identities,links,entropies,nulls,births,residuals,anchors,expiries,pretransport=[],[],[],[],[],[],[],[],[],[]
    for frame,p in enumerate(intensity):
        active=nonzero[frame]
        live=last>=0
        for row in np.flatnonzero(live):
            old=last[row];margin=config.edge_guard_seconds+config.evidence_seconds
            common=(times[frame]-times[old]<config.maximum_endpoint_distance and
                    times[old]>bounds[frame,0]+margin and times[frame]<bounds[old,1]-margin)
            if not common:live[row]=False
        expired=(last>=0)&~live
        last[expired]=-1;fragments[expired]=-1
        for row in np.flatnonzero(expired):endpoint_e[row]=None;endpoint_weights[row]=None
        prior_fragments=fragments.copy();prior_last=last.copy()
        live_tensor=torch.as_tensor(live,device=p.device)
        if not active.any():
            # No seed, no matching through zero E. Preserve only finite past
            # event records; zero-valued A keeps the straight-through gate path.
            a=torch.eye(k,device=p.device,dtype=p.dtype)
            b=a
            tracked=p
            before_gate=p
            entropy=p*0;null=live_tensor.to(p.dtype);birth=p*0
            direct=False
        elif not live.any():
            # First effective activity is self-corresponding by construction.
            a=torch.eye(k,device=p.device,dtype=p.dtype);b=a
            tracked=p;entropy=p*0;null=p*0;birth=active.to(p.dtype)
            before_gate=p
            direct=True
            for row in torch.nonzero(active).flatten().cpu().tolist():
                fragments[row]=new_fragment(row)
        else:
            scores=[]
            for row in range(k):
                if live[row]:
                    # Compare a distribution of latest REAL event observations,
                    # not the direction of a confidence-updated mean prototype.
                    cosine=endpoint_e[row]@e[frame].T
                    scores.append(endpoint_weights[row]@cosine)
                else:scores.append(p.new_full((k,),config.similarity_gate))
            logits=(torch.stack(scores)-config.similarity_gate)/config.temperature
            # Inactive current columns are anonymous capacity, never E matches.
            logits=torch.where(active[None,:],logits,torch.zeros_like(logits))
            a=balanced(logits,config.iterations)
            b=balanced(logits.T,config.iterations)  # independently normalized
            before_gate=a@p
            tracked=magnitude_gate(before_gate,config.transport_gate,training=torch.is_grad_enabled())
            observed=a*active[None,:]
            conditional=observed/observed.sum(1,keepdim=True).clamp_min(1e-12)
            entropy=-(conditional*conditional.clamp_min(1e-12).log()).sum(1)
            null=(a*~active[None,:]).sum(1)
            birth=tracked.new_zeros(k)
            for row in range(k):
                if tracked[row].detach()>0 and fragments[row]<0:
                    fragments[row]=new_fragment(row)
                    birth[row]=observed[row].sum()
            direct=False
            links.append({'frame':frame,'endpoint_frames':prior_last,'fragment_ids':prior_fragments,
                          'eligible':live_tensor,'candidate_nonzero':active,'forward':a,'backward':b})
        for row in range(k):
            if tracked[row].detach()>0:
                weights=a[row]*active
                endpoint_weights[row]=weights/weights.sum().clamp_min(1e-12)
                endpoint_e[row]=e[frame]
                last[row]=frame
        values.append(tracked);identities.append(np.where(tracked.detach().cpu().numpy()>0,fragments,-1))
        pretransport.append(before_gate)
        entropies.append(entropy);nulls.append(null);births.append(birth)
        residuals.append((a.sum(1)-1).abs().max())
        anchors.append(direct);expiries.append(int(expired.sum()))
    # A lane reused after expiry becomes a NEW column. PIT cannot silently join
    # two disconnected fragments simply because their local lane index agrees.
    columns=max(k,next_fragment)
    expanded=[]
    for p,ids in zip(values,identities):
        idx=torch.as_tensor(np.where(ids>=0,ids,np.arange(k)),device=p.device)
        expanded.append(p.new_zeros(columns).scatter_add(0,idx,p))
    return {'version':VERSION,'tracked':torch.stack(expanded),'lane_activity':torch.stack(values),
            'fragment_ids':np.stack(identities),'cycle_links':links,
            'pregate_lane_activity':torch.stack(pretransport),
            'entropy':torch.stack(entropies),'null_mass':torch.stack(nulls),'birth_mass':torch.stack(births),
            'capacity_residual':torch.stack(residuals),'direct_anchor_frames':np.flatnonzero(anchors).tolist(),
            'expired_endpoints':expiries,'center_times':times,'context_bounds':bounds,
            'maximum_endpoint_distance_seconds':config.maximum_endpoint_distance}


def cycle_loss(result,target,assignment,*,threshold=.002):
    """Cycle on exactly the same finite endpoint/window pairs as main matching."""
    loss=result['tracked'].sum()*0;total=0
    if not assignment.unique:return loss,0.
    selected=assignment.assignments[0]
    for link in result['cycle_links']:
        a,b=link['forward'],link['backward']
        # Capacity placeholders are not identity states. Condition each
        # independent direction on the same VALID nonzero endpoint pair set.
        a=a*link['eligible'][:,None]*link['candidate_nonzero'][None,:]
        b=b*link['candidate_nonzero'][:,None]*link['eligible'][None,:]
        a=a/a.sum(1,keepdim=True).clamp_min(1e-12)
        b=b/b.sum(1,keepdim=True).clamp_min(1e-12)
        diagonal=torch.diagonal(a@b).clamp_min(1e-8)
        mask=torch.zeros_like(diagonal)
        for source,fragment in enumerate(selected):
            for row in np.flatnonzero(link['fragment_ids']==fragment):
                old=link['endpoint_frames'][row];now=link['frame']
                if old<0 or not link['eligible'][row]:continue
                endpoints=target[[old,now]]
                valid=bool((endpoints[:,source]>=threshold).all())
                for other in range(target.shape[1]):
                    if source==other:continue
                    equal=(endpoints[:,source]-endpoints[:,other]).abs()<=.001+.05*torch.maximum(endpoints[:,source],endpoints[:,other])
                    valid=valid and not bool((equal&(endpoints[:,other]>=threshold)).any())
                if valid:mask[row]=1
        loss=loss+(-diagonal.log()*mask).sum();total+=float(mask.sum())
    denominator=max(1,len(result['cycle_links'])*result['lane_activity'].shape[1])
    return loss/max(1,total),total/denominator
