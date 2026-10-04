"""C0 entire-sample PIT and separate silence/unused contributions."""
import torch


def area_iou(a, b):
    return torch.minimum(a, b).sum(0) / torch.maximum(a, b).sum(0).clamp_min(1e-12)


def c0_loss(a, target, kind="huber_iou"):
    # a[T,K], target[T,1], one fixed slot for the whole sample.
    b = target[:, :1].expand_as(a)
    if kind == "l1":
        point = (a - b).abs().mean(0)
    elif kind == "huber_iou":
        point = torch.nn.functional.huber_loss(a, b, delta=.1, reduction="none").mean(0) / .1
    else:
        raise ValueError(kind)
    shape = point + (.1 * (1 - area_iou(a, b)) if kind == "huber_iou" else 0)
    slot = shape.argmin()
    silence_mask = target[:, 0] == 0
    silence = a[silence_mask, slot].mean() if silence_mask.any() else a.sum() * 0
    unused_mask = torch.arange(a.shape[1], device=a.device) != slot
    unused = a[:, unused_mask].mean()
    return shape[slot] + .1 * silence + .1 * unused, dict(
        slot=int(slot.detach()), shape=shape[slot], silence=silence, unused=unused)
