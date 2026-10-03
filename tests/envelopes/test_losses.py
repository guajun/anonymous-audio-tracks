"""Loss geometry, silent gradients, and segment-level identity constraints."""
import itertools
import numpy as np
import pytest
torch=pytest.importorskip("torch")
from aat.envelopes.losses import ShapeLossConfig,curve_loss,segment_shape_loss
from aat.envelopes.models import EnvelopeVectorHead,associate_soft


@pytest.mark.parametrize("family",["l1","huber","area_iou","huber_iou","multiscale"])
def test_empty_target_has_finite_suppression_gradient(family):
    p=torch.full((20,),.02,requires_grad=True)
    loss=curve_loss(p,torch.zeros_like(p),torch.ones(20,dtype=torch.bool),ShapeLossConfig(family=family))
    loss.backward()
    assert torch.isfinite(loss)
    assert (p.grad>0).all()


def test_area_iou_is_normalized_l1_and_not_shift_invariant():
    y=torch.tensor([0.,0.,.2,.5,.2,0.,0.])
    p=torch.roll(y,1)
    config=ShapeLossConfig(family="area_iou",empty_weight=0)
    loss=curve_loss(p,y,torch.ones(7,dtype=torch.bool),config)
    difference=(p-y).abs().sum()
    assert loss.item()==pytest.approx((2*difference/((p+y).sum()+difference)).item())
    assert loss>0


def test_matching_cannot_hide_a_mid_segment_switch():
    y=torch.tensor([[.2,0.],[.2,0.],[0.,.2],[0.,.2]])
    valid=torch.ones(4,dtype=torch.bool)
    config=ShapeLossConfig(family="l1")
    correct,_=segment_shape_loss(y,y,valid,config)
    switched=torch.tensor([[.2,0.],[.2,0.],[.2,0.],[.2,0.]])
    wrong,_=segment_shape_loss(switched,y,valid,config)
    reordered,_=segment_shape_loss(y.flip(1),y,valid,config)
    assert correct==0 and reordered==0 and wrong>0


def test_assignment_optimizes_the_actual_empty_and_source_objective():
    torch.manual_seed(46)
    p=torch.rand(12,5)*.3
    y=torch.rand(12,2)*.3
    valid=torch.ones(12,dtype=torch.bool)
    config=ShapeLossConfig(family="huber_iou")
    actual,_=segment_shape_loss(p,y,valid,config)
    values=[]
    for perm in itertools.permutations(range(5),2):
        paired=torch.stack([curve_loss(p[:,k],y[:,s],valid,config) for s,k in enumerate(perm)]).mean()
        empty=torch.stack([curve_loss(p[:,k],torch.zeros(12),valid,config) for k in range(5) if k not in perm]).mean()
        values.append(float(paired+empty))
    assert float(actual)==pytest.approx(min(values))


def test_multiscale_propagates_a_signal_toward_nearby_nonoverlapping_event():
    y=torch.zeros(40);y[15]=.2
    p=torch.zeros(40,requires_grad=True)
    with torch.no_grad():p[18]=.2
    valid=torch.ones(40,dtype=torch.bool)
    config=ShapeLossConfig(family="multiscale",empty_weight=0)
    curve_loss(p,y,valid,config).backward()
    assert torch.isfinite(p.grad).all()
    assert p.grad[15]<0


def test_shared_head_outputs_direction_and_intensity_without_query_parameters():
    torch.manual_seed(46)
    model=EnvelopeVectorHead(12)
    e,p=model(torch.randn(3,12,8,20),torch.linspace(-1,1,20).expand(3,-1),torch.ones(3)*.1)
    assert e.shape==(3,8,128) and p.shape==(3,8)
    torch.testing.assert_close(e.norm(dim=-1),torch.ones(3,8))
    assert ((p>=0)&(p<1)).all()
    assert not any("query" in name for name,_ in model.named_parameters())
    assert sum(v.numel() for v in model.parameters())<2_000_000


def test_shape_error_reaches_descriptors_through_soft_association():
    torch.manual_seed(9)
    raw=torch.randn(5,2,4,requires_grad=True)
    e=torch.nn.functional.normalize(raw,dim=-1)
    p=torch.tensor([[.2,.05],[.15,.1],[.1,.15],[.05,.2],[.2,.05]])
    tracked,matrices=associate_soft(e,p,temperature=.4,iterations=30)
    (tracked-torch.tensor([[.2,.05]]).expand(5,-1)).square().sum().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>1e-5
    for a in matrices:
        torch.testing.assert_close(a.sum(0),torch.ones(2),atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(a.sum(1),torch.ones(2),atol=1e-4,rtol=1e-4)
