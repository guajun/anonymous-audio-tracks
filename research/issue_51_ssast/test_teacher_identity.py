import unittest
import numpy as np
import torch
from teacher_assignment import teacher,permutation,adjacency,amplitude_loss
from identity_supervision import identity_loss,make_pairs
from hard_identity_inference import connect


class TeacherIdentityTests(unittest.TestCase):
    def test_context_resolves_equal_instantaneous_amplitudes(self):
        y=np.array([[.2,.0],[.1,.1],[.0,.2],[.1,.1]]*4,dtype=np.float32)
        a=np.zeros((16,8),dtype=np.float32);a[:,3]=y[:,0];a[:,5]=y[:,1]
        order,conf,_=teacher(a,y)
        np.testing.assert_array_equal(order[:,:2],np.tile([3,5],(16,1)))
        self.assertGreater(conf.min(),0)
        p=permutation(order);np.testing.assert_array_equal(p.sum(1),np.ones((16,8)))
        np.testing.assert_array_equal(p.sum(2),np.ones((16,8)))

    def test_copied_and_identical_truth_has_no_reliable_identity(self):
        y=np.ones((16,2),dtype=np.float32)*.2;a=np.ones((16,8),dtype=np.float32)*.2
        _,conf,_=teacher(a,y)
        self.assertEqual(float(conf.max()),0)

    def test_graph_is_p_transpose_p_next(self):
        order=np.tile(np.arange(8),(2,1));order[1,[1,4]]=order[1,[4,1]]
        g=adjacency(order)
        self.assertEqual(g[0,1,4],1);self.assertEqual(g[0,4,1],1)
        np.testing.assert_array_equal(g.sum(1),np.ones((1,8)))

    def test_zero_tracks_have_linear_gradient_even_near_zero(self):
        for amplitude in (1e-2,1e-5):
            a=torch.full((8,8),amplitude,requires_grad=True);y=torch.zeros(8,2)
            order=np.tile(np.arange(8),(8,1));value,_=amplitude_loss(a,y,order)
            value.backward()
            torch.testing.assert_close(a.grad[:,2:],torch.full((8,6),.1/(8*6)))

    def test_opposite_vectors_cannot_average_to_false_zero(self):
        v=torch.zeros(16,8,2);v[:,0,0]=1;v[:,1,0]=-1
        a=v.norm(dim=-1);y=torch.zeros(16,2);order=np.tile(np.arange(8),(16,1))
        value,details=amplitude_loss(a,y,order)
        self.assertGreater(float(value),0)
        self.assertEqual(float(details['aligned_amplitudes'][:,:2].min()),1.)

    def test_direct_identity_gradient_is_nonzero_below_eval_gate(self):
        torch.manual_seed(5153)
        y=torch.full((12,2),.2);order=np.tile(np.arange(8),(12,1));conf=np.ones((12,2))
        pairs=make_pairs(y.numpy(),offsets=(1,2))
        v=(torch.randn(12,8,6)*1e-5).requires_grad_(True)
        loss,details=identity_loss(v,y,order,conf,pairs)
        loss.backward()
        self.assertTrue(torch.isfinite(v.grad).all())
        self.assertGreater(float(v.grad[:,:2].norm()),0)
        self.assertEqual(float(v.grad[:,2:].norm()),0)
        unit=v.detach()/v.detach().norm(dim=-1,keepdim=True)
        self.assertLess(float((v.grad*unit).sum(-1).abs().max()),.01)
        self.assertGreater(details['effective_weight'],0)

    def test_zero_truth_identity_is_undefined_and_has_no_gradient(self):
        v=torch.randn(16,8,4,requires_grad=True);y=torch.zeros(16,2)
        order=np.tile(np.arange(8),(16,1));conf=np.ones((16,2))
        loss,_=identity_loss(v,y,order,conf,make_pairs(y.numpy()))
        loss.backward();self.assertEqual(float(v.grad.norm()),0)

    def test_ambiguity_reduces_total_identity_weight(self):
        torch.manual_seed(5155);v=torch.randn(12,8,6);y=torch.full((12,2),.2)
        order=np.tile(np.arange(8),(12,1));pairs=make_pairs(y.numpy(),offsets=(1,2))
        strong,_=identity_loss(v,y,order,np.ones((12,2)),pairs)
        weak,_=identity_loss(v,y,order,np.full((12,2),.1),pairs)
        torch.testing.assert_close(weak,strong*.01)

    def test_prediction_only_hard_links_preserve_mass_and_shuffle(self):
        v=np.zeros((4,8,3),dtype=np.float32);v[:,3,0]=.2;v[:,5,1]=.1
        a,m=connect(v,1.)
        rng=np.random.default_rng(51);shuffled=np.stack([x[rng.permutation(8)] for x in v])
        b,mm=connect(shuffled,1.)
        np.testing.assert_allclose(np.sort(a,axis=1),np.sort(b,axis=1))
        np.testing.assert_allclose(a.sum(1),np.linalg.norm(v,axis=-1).sum(1))
        self.assertLess(m['total_amplitude_max_error'],1e-7)
        self.assertTrue(np.all(np.ptp(a,axis=0)==0));self.assertTrue(np.all(np.ptp(b,axis=0)==0))


if __name__=='__main__':unittest.main()
