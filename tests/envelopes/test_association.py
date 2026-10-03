import numpy as np
import pytest
torch=pytest.importorskip("torch")
from aat.envelopes.association import AssociationConfig,rollout,reference_cycle_mask,cycle_loss
from aat.envelopes.losses import ShapeLossConfig,segment_shape_components
from aat.envelopes.curriculum import c1_config


def test_per_window_candidate_permutations_do_not_change_tracks():
    torch.manual_seed(12)
    e=torch.nn.functional.normalize(torch.eye(3)[None].expand(7,-1,-1)+.02*torch.randn(7,3,3),dim=-1)
    p=torch.rand(7,3)*.2
    config=AssociationConfig(iterations=40)
    expected=rollout(e,p,config)["tracked"]
    perm=torch.stack([torch.arange(3)]+[torch.randperm(3) for _ in range(6)])
    actual=rollout(e.gather(1,perm[...,None].expand_as(e)),p.gather(1,perm),config)["tracked"]
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_silent_observations_preserve_memory_and_use_null_matches():
    e=torch.eye(2)[None].expand(6,-1,-1).clone()
    e[2:4]=e[2:4].flip(1)
    p=torch.tensor([[.2,.1],[.1,.2],[0.,0.],[0.,0.],[.2,.1],[.1,.2]])
    result=rollout(e,p,AssociationConfig(iterations=40))
    torch.testing.assert_close(result["prototypes"][1],result["prototypes"][3])
    assert result["tracked"][2:4].abs().max()==0
    assert result["null_mass"][2:4].min()>.99


def test_future_shape_error_reaches_earlier_descriptors_without_cycle():
    torch.manual_seed(46)
    raw=torch.randn(8,2,4,requires_grad=True)
    e=torch.nn.functional.normalize(raw,dim=-1)
    # Earlier shapes coincide; only later divergence supplies shape evidence.
    p=torch.tensor([[.1,.1]]*4+[[.2,.05],[.15,.1],[.05,.2],[.2,.05]])
    result=rollout(e,p,AssociationConfig(temperature=.4,similarity_gate=0,iterations=30))
    target=torch.tensor([[.1,.1]]*4+[[.2,.05],[.2,.05],[.2,.05],[.2,.05]])
    parts,_=segment_shape_components(result["tracked"],target,torch.ones(8,dtype=torch.bool),ShapeLossConfig(family="l1"))
    sum(parts.values()).backward()
    assert torch.isfinite(raw.grad).all()
    assert raw.grad[:4].abs().sum()>1e-5


def test_equal_active_envelopes_are_masked_in_cycle_but_shape_stays_active():
    target=torch.tensor([[.1,.1],[.1,.1],[.2,.05],[.2,.05]])
    pred=target.clone();pred[2:,0]=.15
    parts,assignment=segment_shape_components(pred,target,torch.ones(4,dtype=torch.bool),ShapeLossConfig(family="l1"))
    mask=reference_cycle_mask(target,assignment,2)
    assert mask[:2].sum()==0 and mask[2].sum()==2
    assert sum(parts.values())>0


def test_cycle_is_nontrivial_for_soft_mixture_and_not_a_hard_inverse():
    raw=torch.ones(5,2,3,requires_grad=True)
    e=torch.nn.functional.normalize(raw,dim=-1)
    result=rollout(e,torch.ones(5,2)*.1)
    value=cycle_loss(result,torch.ones(4,2))
    assert value.item()==pytest.approx(np.log(2),abs=1e-5)
    value.backward()
    assert torch.isfinite(raw.grad).all()


def test_null_and_source_silence_coefficients_are_independent():
    pred=torch.tensor([[.1,.03],[.02,.03]])
    target=torch.tensor([[.1],[0.]])
    a,_=segment_shape_components(pred,target,torch.ones(2,dtype=torch.bool),ShapeLossConfig(family="l1",null_weight=1))
    b,_=segment_shape_components(pred,target,torch.ones(2,dtype=torch.bool),ShapeLossConfig(family="l1",null_weight=3))
    torch.testing.assert_close(a["shape"],b["shape"])
    torch.testing.assert_close(a["silence"],b["silence"])
    torch.testing.assert_close(a["null"]*3,b["null"])


def test_c1_contains_recurrence_and_safe_acoustic_gaps():
    starts=set()
    for seed in range(4610,4620):
        config,onsets,owners=c1_config(seed,f'c1-{seed}')
        assert len(config.sources)==2 and owners[0]==owners[2]==owners[4]
        assert owners[0]!=owners[1]
        assert np.diff(onsets).min()>1.5+max(s.amp.release_ms for s in config.sources)/1000
        starts.add(owners[0])
    assert starts=={0,1}
