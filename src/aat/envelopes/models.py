"""Query-free competitive convolutional vector head and soft association."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EnvelopeVectorHead(nn.Module):
    """Shared vector mapping; vector radius represents linear RMS intensity.

    Slot allocation is a convolution over observed features. There are no
    learned query vectors or independent per-slot descriptor projection heads.
    Absolute mixture RMS is supplied because Demucs normalizes each input.
    """
    def __init__(self, in_channels: int, slots=8, hidden=128):
        super().__init__()
        self.slots = slots
        self.project = nn.Sequential(nn.Conv2d(in_channels, hidden, 1), nn.GELU(),
                                     nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU())
        self.allocate = nn.Conv2d(hidden, slots+1, 1)  # extra background channel
        self.vector = nn.Sequential(nn.Linear(2*hidden+3, hidden), nn.GELU(), nn.Linear(hidden,128))
        nn.init.normal_(self.vector[-1].weight, std=0.001)
        nn.init.zeros_(self.vector[-1].bias)

    def forward(self, features: Tensor, relative_times: Tensor, mixture_rms: Tensor):
        x = self.project(features.float())
        masks = self.allocate(x).softmax(1)[:, :self.slots]
        mass = masks.sum((2,3)).clamp_min(1e-6)
        context = torch.einsum('bkft,bcft->bkc',masks,x)/mass[...,None]
        center_weight = torch.exp(-0.5*(relative_times/0.05).square())[:,None,None,:]
        cmask = masks*center_weight
        cmass = cmask.sum((2,3)).clamp_min(1e-6)
        center = torch.einsum('bkft,bcft->bkc',cmask,x)/cmass[...,None]
        occupancy = (cmass/(center_weight.sum((2,3))*features.shape[2]).clamp_min(1e-6))[...,None]
        level = mixture_rms[:,None,None].expand(-1,self.slots,1)
        z = self.vector(torch.cat((context,center,occupancy,level,torch.log1p(level*100)),dim=-1))
        radius = z.norm(dim=-1)
        intensity = radius/(1+radius)
        direction = F.normalize(z,dim=-1,eps=1e-8)
        return direction, intensity


def associate_soft(directions: Tensor, intensity: Tensor, *, temperature=0.05, iterations=12):
    """Differentiable K-track rollout with a doubly stochastic allocation.

    Local capacity is fixed for this experiment. All tracks persist over the
    segment; activity controls prototype updates. Birth/death is not modeled.
    """
    if directions.ndim != 3 or intensity.shape != directions.shape[:2] or temperature <= 0:
        raise ValueError("expected [time,K,D] directions and [time,K] intensity")
    prototype = directions[0]
    curves, matrices = [intensity[0]], []
    for e,p in zip(directions[1:],intensity[1:]):
        log_a = prototype @ e.T / temperature
        for _ in range(iterations):
            log_a = log_a - log_a.logsumexp(-1,keepdim=True)
            log_a = log_a - log_a.logsumexp(-2,keepdim=True)
        a = log_a.exp()
        tracked = a @ p
        gate = (tracked/(tracked+0.03))[:,None]
        prototype = F.normalize((1-gate)*prototype+gate*(a@e),dim=-1,eps=1e-8)
        curves.append(tracked)
        matrices.append(a)
    return torch.stack(curves), matrices
