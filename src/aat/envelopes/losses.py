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

    def __post_init__(self):
        if self.family not in ("l1", "huber", "area_iou", "huber_iou", "multiscale"):
            raise ValueError("unsupported shape loss family")
        values = (self.delta, self.iou_weight, self.empty_weight, self.hop_seconds, *self.scales_seconds)
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


def curve_loss(predicted: Tensor, target: Tensor, valid: Tensor, config: ShapeLossConfig):
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
            terms.append((F.huber_loss(smooth_p, smooth_y, reduction="none", delta=config.delta)*mask).sum(-1)/denom)
        main = torch.stack(terms).mean(0)
    silent = ((y == 0)*mask).to(p.dtype)
    silence_loss = (p*silent).sum(-1)/silent.sum(-1).clamp_min(1)
    empty_loss = (p*mask).sum(-1)/denom
    return torch.where(nonempty, main + config.empty_weight*silence_loss,
                       config.empty_weight*empty_loss)


def segment_shape_loss(predicted: Tensor, target: Tensor, valid: Tensor, config: ShapeLossConfig):
    """One permutation for [time, tracks] over the entire segment.

    Assignment includes the costs of unused tracks. Tied optimal assignments
    are averaged symmetrically. Truncated tie sets fail rather than invent IDs.
    """
    if predicted.ndim != 2 or target.ndim != 2 or predicted.shape[0] != target.shape[0]:
        raise ValueError("expected matching [time, tracks/sources] arrays")
    slots, sources = predicted.shape[1], target.shape[1]
    if not 1 <= sources <= slots <= 8:
        raise ValueError("expected 1 <= sources <= tracks <= 8")
    paired = curve_loss(predicted.T[None], target.T[:,None], valid, config)
    empty = curve_loss(predicted.T, torch.zeros_like(predicted.T), valid, config)
    # Replacing a null target by a source also replaces that track's empty cost.
    cost = paired / sources
    if slots > sources:
        cost = cost - empty[None] / (slots-sources)
    assignment = enumerate_optimal_assignments(cost.detach().cpu().numpy(), max_optimal=40320)
    if assignment.truncated:
        raise ValueError("unresolved truncated trajectory matching")
    values = []
    for perm in assignment.assignments:
        chosen = torch.tensor(perm, device=predicted.device)
        unused = [k for k in range(slots) if k not in perm]
        # Source curves and unused tracks have separate averages; K cannot dilute the main loss.
        value = paired[torch.arange(sources, device=predicted.device), chosen].mean()
        if unused:
            value = value + empty[unused].mean()
        values.append(value)
    return torch.stack(values).mean(), assignment


def area_iou_numpy(predicted, target):
    p, y = np.asarray(predicted), np.asarray(target)
    union = np.maximum(p,y).sum()
    return float(np.minimum(p,y).sum()/union) if union > 0 else 1.0
