"""Anonymous whole-sequence assignment; local transport remains upstream."""
import itertools
import torch


def cost_matrix(a, b):
    distance = a[:,:,None]-b[:,None,:]
    point = torch.nn.functional.huber_loss(a[:,:,None].expand_as(distance),
        b[:,None,:].expand_as(distance), delta=.1, reduction='none').mean(0)/.1
    inter = torch.minimum(a[:,:,None],b[:,None,:]).sum(0)
    union = torch.maximum(a[:,:,None],b[:,None,:]).sum(0).clamp_min(1e-12)
    return point+.1*(1-inter/union)


def pit(a,b):
    permutations = torch.tensor(list(itertools.permutations(range(a.shape[1]), b.shape[1])),device=a.device)
    cost = cost_matrix(a,b)
    values = cost[permutations,torch.arange(b.shape[1],device=a.device)].mean(1)
    best = values.argmin()
    order = permutations[best]
    sorted_cost = values.sort().values
    return order, values[best], sorted_cost[1]-sorted_cost[0]


def loss(a,b, *, unused_weight=.1):
    order, shape, gap = pit(a,b)
    matched = a[:,order]
    silent = b==0
    silence = matched[silent].mean() if silent.any() else a.sum()*0
    mask = torch.ones(a.shape[1],device=a.device,dtype=torch.bool)
    mask[order]=False
    unused = a[:,mask].mean()
    return shape+.1*silence+unused_weight*unused, dict(order=order,shape=shape,
        silence=silence,unused=unused,pit_gap=gap)
