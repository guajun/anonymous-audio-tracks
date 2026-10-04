"""Independent torch loss and enumerable small-scope checks."""
import itertools
import unittest
import numpy as np
import torch
from joint_teacher import Objective,teacher,frozen_confidence
from teacher_assignment import amplitude_loss,permutation,adjacency
from identity_supervision import make_pairs,identity_loss


def example():
    y=np.array([[.20,.10],[.151,.149],[.149,.151],[.10,.20]],np.float32)
    v=np.zeros((4,3,2),np.float32);v[:,0,0]=y[:,0];v[:,1,1]=y[:,1]
    v[2:]=v[2:,:,::-1];v[:,2]=1e-5
    return v,y


class JointTests(unittest.TestCase):
    def test_numpy_full_objective_matches_independent_torch_training_loss(self):
        v,y=example();obj=Objective(v,y,block=1)
        for path in itertools.islice(itertools.product(range(6),repeat=4),0,300,23):
            o=obj.orders(path);amp,_=amplitude_loss(torch.tensor(obj.a),torch.tensor(y),o)
            lid,_=identity_loss(torch.tensor(v),torch.tensor(y),o,obj.conf,(obj.p,obj.neg))
            c,a,i=obj.cost(path)
            self.assertAlmostEqual(float(amp),float(a[0]),places=6)
            self.assertAlmostEqual(float(lid),float(i[0]),places=6)
            self.assertAlmostEqual(float(amp+.2*lid),float(c[0]),places=6)

    def test_identity_changes_assignment_and_exact_enumeration(self):
        v,y=example();o,c,m=teacher(v,y,block=1,sweeps=3)
        exact,_,em=teacher(v,y,block=1,exhaustive=True)
        self.assertGreater(m['changed_blocks'],0)
        self.assertLess(m['objective_after'],m['objective_before'])
        self.assertLessEqual(em['objective_after'],m['objective_after']+1e-7)
        stable=v.copy();stable[2:]=stable[2:,:,::-1]
        other,_,_=teacher(stable,y,block=1,sweeps=3)
        self.assertTrue(np.any(other[:,:2]!=o[:,:2]))
        self.assertLessEqual(m['amplitude_after'],m['amplitude_cap']+1e-7)

    def test_strong_envelope_conflict_cannot_escape_trust_region(self):
        v,y=example();y*=5;v*=5
        _,_,m=teacher(v,y,block=1,amplitude_slack=.0001,sweeps=3)
        self.assertLessEqual(m['amplitude_after'],m['amplitude_cap']+1e-7)

    def test_node_guard_forbids_sacrificing_genuine_source_for_identity(self):
        v,y=example();o,_,m=teacher(v,y,block=1,exhaustive=True)
        self.assertTrue(np.all(o[:,:2]<2))
        a=np.linalg.norm(v,axis=-1);selected=a[np.arange(4)[:,None],o[:,:2]]
        self.assertLessEqual(float(np.abs(selected-y).max()),.020001)

    def test_frame_shuffle_exact_scope_equivariance(self):
        v,y=example();o,_,m=teacher(v,y,block=1,exhaustive=True)
        rng=np.random.default_rng(52);sh=np.array([rng.permutation(3) for _ in v]);vs=v[np.arange(4)[:,None],sh]
        so,_,sm=teacher(vs,y,block=1,exhaustive=True)
        self.assertAlmostEqual(m['objective_after'],sm['objective_after'],places=6)
        # Equal-cost symmetric minimizers need not choose the same labels. Check
        # the shuffled optimum maps to a minimizer of the original full objective.
        mapped=sh[np.arange(4)[:,None],so[:,:2]];obj=Objective(v,y,block=1)
        path=[np.flatnonzero((obj.states==row).all(1))[0] for row in mapped]
        self.assertAlmostEqual(float(obj.cost(path)[0][0]),m['objective_after'],places=6)

    def test_copy_ambiguity_zero_gt_and_one_to_one(self):
        v,y=example();v[:]=v[:,0,None,:]
        self.assertLess(float(frozen_confidence(v,y).max()),1e-5)
        o,c,m=teacher(v,np.zeros_like(y),block=1)
        self.assertEqual(m['identity_after'],0.)
        p=permutation(o,k=3);g=adjacency(o,k=3)
        np.testing.assert_equal(p.sum(1),1);np.testing.assert_equal(p.sum(2),1)
        np.testing.assert_equal(g.sum(1),1);np.testing.assert_equal(g.sum(2),1)

    def test_cross_silence_near_zero_gradient_and_no_vector_cancellation(self):
        v,y=example();y[1:3,0]=0;v[1:3,0]=0
        pairs=make_pairs(y,offsets=(3,));self.assertTrue(len(pairs[0])>0)
        o,c,m=teacher(v,y,pairs=pairs,block=1)
        vv=torch.tensor(v*1e-4,requires_grad=True)
        lid,_=identity_loss(vv,torch.tensor(y),o,c,pairs);lid.backward()
        self.assertTrue(torch.isfinite(vv.grad).all())
        self.assertGreater(float(vv.grad.norm()),0)
        opp=torch.tensor([[[1.,0],[-1.,0],[.1,0]]]*4,requires_grad=True)
        amp,d=amplitude_loss(opp.norm(dim=-1),torch.zeros(4,2),np.tile([0,1,2],(4,1)))
        self.assertGreater(float(amp.detach()),0);self.assertEqual(float(d['aligned_amplitudes'][0,0].detach()),1.)
        amp.backward();self.assertGreater(float(opp.grad.norm()),0)


if __name__=='__main__':unittest.main()
