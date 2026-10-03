import numpy as np
import pytest
torch=pytest.importorskip('torch')
from aat.envelopes.local_association import LocalAssociationConfig,rollout,cycle_loss
from aat.envelopes.models import magnitude_gate,EnvelopeVectorHead
from aat.envelopes.losses import ShapeLossConfig,segment_shape_components,fragment_assignment


def test_first_effective_frame_is_direct_and_not_soft_self_match():
    e=torch.ones(1,3,4)
    p=torch.tensor([[.3,.1,0.]])
    result=rollout(e,p)
    torch.testing.assert_close(result['tracked'],p,rtol=0,atol=0)
    assert result['direct_anchor_frames']==[0]
    assert result['fragment_ids'].tolist()==[[0,1,-1]]
    assert result['cycle_links']==[]


def test_zero_first_windows_have_no_seed_and_first_activity_anchors_directly():
    e=torch.zeros(4,2,3);e[3]=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
    p=torch.tensor([[0.,0.]]*3+[[.2,.1]])
    result=rollout(e,p)
    assert (result['fragment_ids'][:3]==-1).all()
    torch.testing.assert_close(result['tracked'],p)
    assert result['direct_anchor_frames']==[3]


def test_short_zero_gap_links_nonzero_endpoints_in_shared_real_context():
    e=torch.zeros(16,1,2);e[0,0,0]=1;e[15,0,0]=1
    p=torch.zeros(16,1);p[0]=.2;p[15]=.3
    result=rollout(e,p)
    assert result['tracked'].shape==(16,1)
    assert result['fragment_ids'][0,0]==result['fragment_ids'][15,0]==0
    assert (result['fragment_ids'][1:15]==-1).all()
    link=result['cycle_links'][0]
    assert link['endpoint_frames'].tolist()==[0] and link['frame']==15
    assert .04*15<result['maximum_endpoint_distance_seconds']
    assert result['tracked'][1:15].sum()==0


@pytest.mark.parametrize('distance,expected_same',[(.6,True),(1.2,False)])
def test_one_source_zero_gap_while_another_source_remains_audible(distance,expected_same):
    n=round(distance/.04)+1
    e=torch.eye(2)[None].expand(n,-1,-1).clone()
    p=torch.zeros(n,2);p[:,1]=.2;p[0,0]=.2;p[-1,0]=.2
    e[1:-1,0]=0
    result=rollout(e,p)
    assert (result['tracked'][1:-1,0]==0).all()
    if expected_same:assert (result['fragment_ids'][1:-1,0]==-1).all()
    same=result['fragment_ids'][0,0]==result['fragment_ids'][-1,0]
    assert same==expected_same
    assert result['fragment_ids'][0,1]==result['fragment_ids'][-1,1]
    if expected_same:
        link=result['cycle_links'][-1]
        assert link['endpoint_frames'][0]==0  # no zero or other-source overwrite
    else:
        assert result['tracked'][-1,0]==0


@pytest.mark.parametrize('hop',[.04,.05])
def test_long_gap_creates_real_fragment_and_does_not_reuse_old_track_id(hop):
    e=torch.ones(30,1,2);p=torch.zeros(30,1);p[0]=.2;p[-1]=.2
    result=rollout(e,p,LocalAssociationConfig(hop_seconds=hop))
    assert result['tracked'].shape==(30,2)
    assert result['fragment_ids'][0,0]!=result['fragment_ids'][-1,0]
    assert len(result['direct_anchor_frames'])==2
    assert result['cycle_links']==[]
    assert result['tracked'][-1,0]==0 and result['tracked'][0,1]==0


def test_window_edges_and_evidence_margin_prevent_boundary_virtual_link():
    result=rollout(torch.ones(2,1,2),torch.ones(2,1)*.2,
                   center_times=torch.tensor([100.,100.95]),
                   context_bounds=torch.tensor([[99.,101.],[99.95,101.95]]))
    assert result['tracked'].shape[1]==2


def test_per_window_permutations_and_mass_conservation():
    torch.manual_seed(7)
    e=torch.nn.functional.normalize(torch.randn(9,3,4),dim=-1)
    p=torch.rand(9,3)*.2+.01
    expected=rollout(e,p)
    perm=torch.stack([torch.arange(3)]+[torch.randperm(3) for _ in range(8)])
    actual=rollout(e.gather(1,perm[...,None].expand_as(e)),p.gather(1,perm))
    torch.testing.assert_close(expected['tracked'],actual['tracked'],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(actual['tracked'].sum(1),p.sum(1),atol=1e-6,rtol=1e-5)


def test_later_shape_divergence_reaches_early_e_without_cycle():
    torch.manual_seed(46)
    raw=torch.randn(7,2,4,requires_grad=True)
    e=torch.nn.functional.normalize(raw,dim=-1)
    p=torch.tensor([[.1,.1]]*3+[[.2,.05],[.05,.2],[.2,.05],[.05,.2]])
    result=rollout(e,p,LocalAssociationConfig(temperature=.4))
    target=torch.tensor([[.1,.1]]*3+[[.2,.05]]*4)
    parts,_=segment_shape_components(result['tracked'],target,torch.ones(7,dtype=torch.bool),ShapeLossConfig(family='l1'),fragment_capacity=2)
    sum(parts.values()).backward()
    assert torch.isfinite(raw.grad).all() and raw.grad[:3].abs().sum()>1e-5


def test_cycle_uses_independent_local_pairs_is_optional_and_nontrivial():
    raw=torch.ones(5,2,3,requires_grad=True)
    e=torch.nn.functional.normalize(raw,dim=-1)
    p=torch.ones(5,2)*.1
    result=rollout(e,p)
    # Unique source mapping and distinct targets enable this diagnostic mask.
    target=torch.tensor([[.2,.05]]*5)
    _,assignment=segment_shape_components(torch.tensor([[.2,.05]]*5),target,torch.ones(5,dtype=torch.bool),ShapeLossConfig(family='l1'))
    value,fraction=cycle_loss(result,target,assignment)
    assert value.item()==pytest.approx(np.log(2),abs=1e-5) and fraction>0
    value.backward();assert torch.isfinite(raw.grad).all()
    assert result['tracked'].shape==(5,2)  # rollout has no cycle switch dependency


def test_closed_gate_has_exact_zero_no_e_but_training_amplitude_gradient():
    activity=torch.tensor([.0001,.003],requires_grad=True)
    selected=magnitude_gate(activity,.002,training=True)
    assert selected.tolist()==pytest.approx([0.,.003])
    selected.sum().backward();torch.testing.assert_close(activity.grad,torch.ones(2))
    inferred=magnitude_gate(activity,.002,training=False)
    gradient=torch.autograd.grad(inferred.sum(),activity)[0]
    assert gradient[0]==0 and gradient[1]==1
    torch.manual_seed(46)
    model=EnvelopeVectorHead(12,magnitude_gate=.9)
    features=torch.randn(3,12,8,20);times=torch.linspace(-1,1,20).expand(3,-1)
    e,a,pre=model(features,times,torch.ones(3)*.1,return_raw=True)
    assert a.sum()==0 and e.sum()==0 and pre.min()>0
    a.sum().backward();assert model.vector[-1].weight.grad.abs().sum()>0


def test_cycle_excludes_zero_capacity_columns_and_unseeded_rows():
    e=torch.ones(2,3,4);e[:,2]=0
    p=torch.tensor([[.1,.1,0.]]*2)
    result=rollout(e,p)
    target=torch.tensor([[.2,.05]]*2)
    _,assignment=segment_shape_components(torch.tensor([[.2,.05,0.]]*2),target,
                                         torch.ones(2,dtype=torch.bool),ShapeLossConfig(family='l1'))
    value,fraction=cycle_loss(result,target,assignment)
    assert value.item()==pytest.approx(np.log(2),abs=1e-5)
    assert fraction>0 and result['cycle_links'][0]['candidate_nonzero'].tolist()==[True,True,False]


def test_whole_segment_pit_penalizes_fragmentation_and_wide_assignment_is_exact():
    truth=torch.zeros(30,1);truth[0]=.2;truth[-1]=.2
    result=rollout(torch.ones(30,1,2),truth)
    parts,_=segment_shape_components(result['tracked'],truth,torch.ones(30,dtype=torch.bool),ShapeLossConfig(family='l1'),fragment_capacity=1)
    assert parts['shape']>0 and parts['null']>0
    cost=np.full((2,12),5.);cost[0,10]=0;cost[1,11]=0
    optimal=fragment_assignment(cost)
    assert optimal.unique and optimal.assignments==((10,11),)
