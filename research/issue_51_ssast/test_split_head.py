"""Independent A/E contracts and full-objective checks for split heads."""
import unittest
import numpy as np
import torch
from window_head import WindowSplitHead
from joint_teacher import Objective,teacher
from teacher_assignment import amplitude_loss,permutation,adjacency
from identity_supervision import identity_loss,make_pairs
from hard_identity_inference import connect
from test_joint_teacher import example

class SplitTests(unittest.TestCase):
    def test_full_objective_uses_independent_A_and_matches_torch(self):
        z,y=example();a=np.linalg.norm(z,axis=-1)*1.3
        order,c,m=teacher(z,y,amplitudes=a,block=1)
        amp,_=amplitude_loss(torch.tensor(a),torch.tensor(y),order)
        lid,_=identity_loss(torch.tensor(z),torch.tensor(y),order,c,make_pairs(y))
        self.assertAlmostEqual(float(amp+.2*lid),m['objective_after'],places=6)
        obj=Objective(z,y,block=1,amplitudes=a)
        np.testing.assert_array_equal(obj.a,a)
        self.assertGreater(float(np.abs(obj.a-np.linalg.norm(z,axis=-1)).max()),.01)

    def test_Z_radius_does_not_change_teacher_or_inference_A(self):
        z,y=example();a=np.linalg.norm(z,axis=-1)
        factor=np.random.default_rng(9).uniform(.1,10,z.shape[:2]).astype(np.float32)
        o,c,m=teacher(z,y,amplitudes=a,block=1)
        oo,cc,mm=teacher(z*factor[:,:,None],y,amplitudes=a,block=1)
        np.testing.assert_array_equal(o,oo);np.testing.assert_allclose(c,cc,atol=1e-6)
        self.assertAlmostEqual(m['objective_after'],mm['objective_after'],places=6)
        p,meta=connect(z,1.,amplitudes=a);q,qm=connect(z*factor[:,:,None],1.,amplitudes=a)
        np.testing.assert_array_equal(p,q)
        np.testing.assert_allclose(p.sum(1),a.sum(1),atol=1e-7)
        np.testing.assert_array_equal((p>.001).sum(1),(a>.001).sum(1))
        for arr in (permutation(o,k=3),adjacency(o,k=3)):
            np.testing.assert_array_equal(arr.sum(1),1);np.testing.assert_array_equal(arr.sum(2),1)

    def test_output_paths_disjoint_but_shared_features_receive_both_gradients(self):
        torch.manual_seed(51);model=WindowSplitHead();x=torch.randn(5,384,requires_grad=True)
        with torch.no_grad():model.amplitude_output.weight.normal_(std=.01)
        z=model.output(x).reshape(5,8,128);a=torch.nn.functional.softplus(model.amplitude_output(x))
        amp=a.mean();unit=torch.nn.functional.normalize(z,dim=-1)
        lid=(unit[:,0]*unit[:,1]).sum(-1).mean()
        self.assertIsNone(torch.autograd.grad(lid,model.amplitude_output.weight,allow_unused=True,retain_graph=True)[0])
        self.assertIsNone(torch.autograd.grad(amp,model.output.weight,allow_unused=True,retain_graph=True)[0])
        self.assertGreater(float(torch.autograd.grad(amp,x,retain_graph=True)[0].norm()),0)
        self.assertGreater(float(torch.autograd.grad(lid,x)[0].norm()),0)
        before=a.detach().clone()
        with torch.no_grad():model.output.weight.mul_(7);model.output.bias.mul_(7)
        torch.testing.assert_close(before,torch.nn.functional.softplus(model.amplitude_output(x)))
        self.assertEqual(sum(p.numel() for p in model.parameters()),693000)

    def test_GT_zero_has_no_ID_but_independent_amplitude_is_supervised(self):
        z,y=example();y*=0;a=torch.full((4,3),1e-6,requires_grad=True)
        order,c,m=teacher(z,y,amplitudes=a,block=1)
        lid,_=identity_loss(torch.tensor(z,requires_grad=True),torch.tensor(y),order,c,make_pairs(y))
        self.assertEqual(float(lid),0)
        loss,_=amplitude_loss(a,torch.tensor(y),order);loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all());self.assertGreater(float(a.grad.norm()),0)

if __name__=='__main__':unittest.main()
