"""Differentiable envelope losses and fixed-per-segment permutation matching."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from aat.losses.matching import enumerate_optimal_assignments


@dataclass(frozen=True)
class ShapeLossConfig:
    family: str = "huber_iou"
    delta: float = 0.05
    iou_weight: float = 0.1
    empty_weight: float = 1.0
    scales_seconds: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)
    hop_seconds: float = 0.02
    normalize_huber: bool = True
    null_weight: float | None = None

    def __post_init__(self):
        if self.family not in ("l1", "huber", "area_iou", "huber_iou", "multiscale"):
            raise ValueError("unsupported shape loss family")
        values = (self.delta, self.iou_weight, self.empty_weight, self.hop_seconds, *self.scales_seconds)
        if self.null_weight is not None:
            values=(*values,self.null_weight)
            if self.null_weight<0:
                raise ValueError("nonnegative null weight required")
        if not all(math.isfinite(v) for v in values) or min(self.delta, self.hop_seconds) <= 0:
            raise ValueError("finite positive delta/hop required")
        if min(self.iou_weight, self.empty_weight, *self.scales_seconds) < 0 or not self.scales_seconds:
            raise ValueError("invalid shape weights/scales")


def _smooth(x: Tensor, sigma_samples: float):
    if sigma_samples == 0:
        return x
    radius = max(1, math.ceil(3 * sigma_samples))
    offsets = torch.arange(-radius, radius+1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma_samples).square())
    kernel = kernel / kernel.sum()
    flat = x.reshape(-1, 1, x.shape[-1])
    return F.conv1d(F.pad(flat, (radius, radius)), kernel[None, None]).reshape_as(x)


def curve_components(predicted: Tensor, target: Tensor, valid: Tensor, config: ShapeLossConfig):
    """Return loss per broadcasted curve, time on the last axis.

    Empty curves use explicit output suppression, not epsilon-normalized IoU.
    All families share the same pointwise penalty on silent target samples.
    """
    predicted, target = torch.broadcast_tensors(predicted, target)
    if predicted.shape[-1] != valid.shape[-1] or valid.dtype != torch.bool or not valid.any():
        raise ValueError("nonempty valid time mask required")
    p, y = predicted.float(), target.float()
    if not torch.isfinite(p).all() or not torch.isfinite(y).all() or (p < 0).any() or (y < 0).any():
        raise ValueError("finite nonnegative envelopes required")
    mask = valid.to(p.dtype)
    denom = mask.sum(-1).clamp_min(1)
    residual = p-y
    l1 = (residual.abs()*mask).sum(-1)/denom
    huber = (F.huber_loss(p, y, reduction="none", delta=config.delta)*mask).sum(-1)/denom
    if config.normalize_huber:
        # Match L1's unit slope for large residuals; otherwise the null-track
        # L1 penalty can dominate merely because delta is expressed in RMS.
        huber = huber/config.delta
    intersection = (torch.minimum(p,y)*mask).sum(-1)
    union = (torch.maximum(p,y)*mask).sum(-1)
    nonempty = (y*mask).sum(-1) > 0
    iou = 1 - intersection / union.clamp_min(1e-8)
    if config.family == "l1":
        main = l1
    elif config.family == "huber":
        main = huber
    elif config.family == "area_iou":
        main = iou
    elif config.family == "huber_iou":
        main = huber + config.iou_weight*iou
    else:
        terms = []
        for sigma in config.scales_seconds:
            smooth_p, smooth_y = _smooth(p*mask, sigma/config.hop_seconds), _smooth(y*mask, sigma/config.hop_seconds)
            value=(F.huber_loss(smooth_p, smooth_y, reduction="none", delta=config.delta)*mask).sum(-1)/denom
            terms.append(value/config.delta if config.normalize_huber else value)
        main = torch.stack(terms).mean(0)
    silent = ((y == 0)*mask).to(p.dtype)
    silence_loss = (p*silent).sum(-1)/silent.sum(-1).clamp_min(1)
    empty_loss = (p*mask).sum(-1)/denom
    zero=torch.zeros_like(main)
    null_weight=config.empty_weight if config.null_weight is None else config.null_weight
    return {"shape":torch.where(nonempty,main,zero),
            "silence":torch.where(nonempty,config.empty_weight*silence_loss,zero),
            "null":torch.where(nonempty,zero,null_weight*empty_loss)}


def curve_loss(predicted: Tensor, target: Tensor, valid: Tensor, config: ShapeLossConfig):
    return sum(curve_components(predicted,target,valid,config).values())


def segment_shape_loss(predicted: Tensor, target: Tensor, valid: Tensor, config: ShapeLossConfig):
    """One permutation for [time, tracks] over the entire segment.

    Assignment includes the costs of unused tracks. Tied optimal assignments
    are averaged symmetrically. Truncated tie sets fail rather than invent IDs.
    """
    components,assignment=segment_shape_components(predicted,target,valid,config)
    return sum(components.values()),assignment


def segment_shape_components(predicted: Tensor, target: Tensor, valid: Tensor, config: ShapeLossConfig, *, fragment_capacity=None):
    """PIT-selected shape/silence/null terms on one common segment mapping."""
    if predicted.ndim != 2 or target.ndim != 2 or predicted.shape[0] != target.shape[0]:
        raise ValueError("expected matching [time, tracks/sources] arrays")
    slots, sources = predicted.shape[1], target.shape[1]
    if not 1 <= sources <= slots or (slots>8 and fragment_capacity is None):
        raise ValueError("expected 1 <= sources <= tracks <= 8")
    capacity=slots if fragment_capacity is None else fragment_capacity
    if not sources<=capacity<=8:raise ValueError('invalid local candidate capacity')
    null_denominator=max(1,capacity-sources)
    paired_parts=curve_components(predicted.T[None], target.T[:,None], valid, config)
    empty_parts=curve_components(predicted.T, torch.zeros_like(predicted.T), valid, config)
    paired=sum(paired_parts.values())
    empty=sum(empty_parts.values())
    # Replacing a null target by a source also replaces that track's empty cost.
    cost = paired / sources
    if slots > sources:
        cost = cost - empty[None] / null_denominator
    detached=cost.detach().cpu().numpy()
    assignment = fragment_assignment(detached) if slots>8 else enumerate_optimal_assignments(detached, max_optimal=40320)
    if assignment.truncated:
        raise ValueError("unresolved truncated trajectory matching")
    marginal = np.zeros((sources,slots),dtype=np.float64)
    for perm in assignment.assignments:
        marginal[np.arange(sources),np.asarray(perm)] += 1/len(assignment.assignments)
    weights = torch.as_tensor(marginal,device=predicted.device,dtype=paired.dtype)
    values={name:(part*weights).sum()/sources for name,part in paired_parts.items()}
    if slots>sources:
        values["null"]=values["null"]+(empty*(1-weights.sum(0))).sum()/null_denominator
    return values, assignment


def fragment_assignment(cost):
    """Exact rectangular PIT for this 1/2-source curriculum, without 2**F DP."""
    from aat.losses.matching import OptimalAssignments
    sources,fragments=cost.shape
    if sources==1:
        totals=cost[0];best=totals.min()
        found=np.flatnonzero(totals<=best+1e-9+1e-9*abs(best))
        assignments=tuple((int(i),) for i in found)
    elif sources==2:
        totals=cost[0,:,None]+cost[1,None,:]
        np.fill_diagonal(totals,np.inf);best=totals.min()
        found=np.argwhere(totals<=best+1e-9+1e-9*abs(best))
        assignments=tuple(map(tuple,found.tolist()))
    else:raise ValueError('wide fragment PIT currently supports at most two sources')
    if len(assignments)>40320:raise ValueError('unresolved fragment PIT ties')
    return OptimalAssignments(assignments=assignments,num_optimal=len(assignments),truncated=False,optimal_cost=float(best))


def area_iou_numpy(predicted, target):
    p, y = np.asarray(predicted), np.asarray(target)
    union = np.maximum(p,y).sum()
    return float(np.minimum(p,y).sum()/union) if union > 0 else 1.0
