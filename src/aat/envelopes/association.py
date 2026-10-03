"""Differentiable partial transport, persistent memory, and optional cycle loss.

Track rows are anonymous capacity, not candidate channel IDs. A dustbin row and
column permit unmatched observations. No descriptor or transport is detached.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class AssociationConfig:
    temperature: float = .1
    similarity_gate: float = .5
    iterations: int = 12
    activity_threshold: float = .002
    update_scale: float = .03
    max_gap_frames: int = 400

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.temperature,self.similarity_gate,self.activity_threshold,self.update_scale)):
            raise ValueError("finite association settings required")
        if min(self.temperature,self.activity_threshold,self.update_scale)<=0 or self.iterations<1 or self.max_gap_frames<1:
            raise ValueError("positive association settings required")


def _transport(logits, iterations):
    """Balanced joint with K unit marginals and one K-capacity dustbin."""
    k=logits.shape[0]-1
    marginals=logits.new_ones(k+1)
    marginals[-1]=k
    log_mass=marginals.log()
    u,v=torch.zeros_like(log_mass),torch.zeros_like(log_mass)
    for _ in range(iterations):
        u=log_mass-torch.logsumexp(logits+v[None,:],dim=1)
        v=log_mass-torch.logsumexp(logits+u[:,None],dim=0)
    joint=(logits+u[:,None]+v[None,:]).exp()
    # Use row-conditional mass to enforce the capacity of each track exactly.
    # Column capacity is approximate at finite Sinkhorn iterations; expose its
    # residual instead of claiming an exact discrete one-to-one assignment.
    return joint/joint.sum(-1,keepdim=True).clamp_min(1e-12)*marginals[:,None]


def rollout(directions,intensity,config=AssociationConfig()):
    """Transport A using the same soft weights computed from E.

    Initial row ordering inherits the first window only, which is immaterial
    under segment PIT. Initial descriptors are *untrusted seeds*, including
    silent ones. Unused/expired rows may acquire observations (soft birth).
    Real memory updates require audible, unambiguous evidence; silence cannot
    overwrite the last trusted prototype. Expiry returns a row to its initial
    seed. This is a bounded research lifecycle, not unrestricted track growth.
    """
    if directions.ndim!=3 or intensity.shape!=directions.shape[:2] or len(directions)<1:
        raise ValueError("expected [T,K,D] and [T,K]")
    if not 1<=directions.shape[1]<=8 or not torch.isfinite(directions).all() or not torch.isfinite(intensity).all() or (intensity<0).any():
        raise ValueError("finite descriptors and nonnegative candidate intensity required")
    k=directions.shape[1]
    seeds=F.normalize(directions[0],dim=-1)
    prototype=seeds
    trusted=intensity.new_zeros(k)
    age=torch.zeros(k,dtype=torch.long,device=intensity.device)
    curves,forwards,backwards,uncertainties,nulls,births,memories,residuals=[],[],[],[],[],[],[],[]
    for frame,(e,p) in enumerate(zip(directions,intensity)):
        expired=age>=config.max_gap_frames
        prototype=torch.where(expired[:,None],seeds,prototype)
        trusted=torch.where(expired,torch.zeros_like(trusted),trusted)
        similarity=prototype@e.T
        # Soft visibility retains a learning signal at small positive A. Only
        # memory writes use the audible threshold, so training cannot get stuck
        # behind a hard "all candidates inactive" matching gate.
        visibility=p/(p+config.activity_threshold)
        real=(similarity-config.similarity_gate)/config.temperature+visibility.clamp_min(1e-12).log()[None,:]
        logits=F.pad(real,(0,1,0,1))
        joint=_transport(logits,config.iterations)
        a=joint[:k,:k]
        reverse=_transport(logits.T,config.iterations)[:k,:k]
        tracked=a@p
        conditional=a/a.sum(-1,keepdim=True).clamp_min(1e-12)
        entropy=-(conditional*conditional.clamp_min(1e-12).log()).sum(-1)
        certainty=(1-entropy/math.log(k)).clamp(0,1) if k>1 else torch.ones_like(tracked)
        audible=F.relu(p-config.activity_threshold)
        support=a@audible
        evidence=F.normalize(a@(audible[:,None]*e),dim=-1)
        update=(support/(support+config.update_scale))*certainty
        prototype=F.normalize((1-update[:,None])*prototype+update[:,None]*evidence,dim=-1)
        birth=(1-trusted)*update
        trusted=trusted+(1-trusted)*update
        confident=(support>config.activity_threshold)&(certainty>.2)
        age=torch.where(confident,torch.zeros_like(age),age+1)
        curves.append(tracked)
        if frame:
            forwards.append(a)
            backwards.append(reverse)
        uncertainties.append(entropy)
        nulls.append(joint[:k,k])
        births.append(birth)
        memories.append(prototype)
        desired=p.new_ones(k+1);desired[-1]=k
        residuals.append((joint.sum(0)-desired).abs().max())
    return {"tracked":torch.stack(curves),"forward":forwards,"backward":backwards,
            "entropy":torch.stack(uncertainties),"null_mass":torch.stack(nulls),
            "birth_mass":torch.stack(births),"prototypes":torch.stack(memories),
            "capacity_residual":torch.stack(residuals)}


def reference_cycle_mask(target,assignment,slots,*,threshold=.002,ambiguity_relative=.05,ambiguity_absolute=.001):
    """Mask anonymous, silent/birth, and equal-envelope regions in supervised cycle.

    This mask uses labels only in the supervised training loss. Ambiguous PIT
    optima receive no diagonal identity demand. Main shape loss remains active.
    """
    mask=target.new_zeros((len(target)-1,slots))
    if not assignment.unique:
        return mask
    active=target>=threshold
    ambiguous=torch.zeros_like(active)
    for i in range(target.shape[1]):
        for j in range(i):
            equal=(target[:,i]-target[:,j]).abs()<=ambiguity_absolute+ambiguity_relative*torch.maximum(target[:,i],target[:,j])
            both=equal&active[:,i]&active[:,j]
            ambiguous[:,i]|=both
            ambiguous[:,j]|=both
    for source,track in enumerate(assignment.assignments[0]):
        mask[:,track]=(active[1:,source]&active[:-1,source]&~ambiguous[1:,source]&~ambiguous[:-1,source]).to(mask.dtype)
    return mask


def cycle_loss(result,mask):
    """Independently normalized soft forward/backward conditional return loss.

    Reciprocal consistency alone cannot establish correct correspondence. Never
    construct an inverse hard permutation and call its identity a learned cycle.
    """
    if len(result["forward"])!=len(mask):
        raise ValueError("cycle mask must cover transitions")
    terms=[]
    for a,b in zip(result["forward"],result["backward"]):
        a=a/a.sum(-1,keepdim=True).clamp_min(1e-12)
        b=b/b.sum(-1,keepdim=True).clamp_min(1e-12)
        terms.append(-torch.diagonal(a@b).clamp_min(1e-8).log())
    if not terms:
        return result["tracked"].sum()*0
    return (torch.stack(terms)*mask).sum()/mask.sum().clamp_min(1)
