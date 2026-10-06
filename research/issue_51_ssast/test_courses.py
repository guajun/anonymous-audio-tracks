"""Meaningful association contracts; cycle is absent."""
import unittest
import numpy as np
import torch
from local_association import transport
from course_loss import loss, pit
from course_metrics import measure
from head import TemporalVHead


class CourseTests(unittest.TestCase):
    def test_zero_has_no_anchor_and_first_actual_frame_direct(self):
        v=torch.zeros(4,3,4)
        v[2,1,0]=.2
        a,c,m=transport(v,np.arange(4)*.02)
        self.assertEqual(m['anchors'],[[2,1]])
        self.assertAlmostEqual(float(a[2,1]),.2)
        self.assertEqual(float(c[:2].sum()),0.)
        self.assertEqual(float(c[2,1,1]),1.)
        self.assertFalse(m['cycle_enabled'])

    def test_long_gap_expires_descriptor_and_makes_fragment(self):
        v=torch.zeros(3,2,2);v[0,0,0]=.2;v[2,1,1]=.3
        a,c,m=transport(v,np.array([0.,.2,1.5]))
        self.assertEqual(m['expired_fragments'],[[2,0]])
        self.assertEqual(m['anchors'],[[0,0],[2,1]])
        self.assertNotEqual(m['fragment_ids_by_frame'][0][0],m['fragment_ids_by_frame'][2][1])
        self.assertEqual(float(c[2,1,1]),1.)

    def test_shape_gradient_to_direction_and_same_transport_weights(self):
        direction=torch.tensor([[[1.,.2],[.2,1.]],[[.8,.5],[.1,1.]]],requires_grad=True)
        unit=direction/direction.norm(dim=-1,keepdim=True)
        amplitude=torch.tensor([[.3,.1],[.1,.3]])
        v=unit*amplitude[...,None]
        a,c,_=transport(v,np.array([0.,.02]),temperature=.3)
        torch.testing.assert_close(a[1],c[1]@amplitude[1])
        b=torch.tensor([[.3,.1],[.3,.1]])
        value,_=loss(a,b)
        value.backward()
        self.assertTrue(torch.isfinite(direction.grad).all())
        self.assertGreater(float(direction.grad.norm()),1e-7)

    def test_pit_single_permutation_for_two_sources(self):
        b=torch.tensor([[1.,0.],[0.,1.],[.5,.2]])
        a=torch.zeros(3,8);a[:,3]=b[:,0];a[:,5]=b[:,1]
        order,shape,gap=pit(a,b)
        self.assertEqual(order.tolist(),[3,5])
        self.assertEqual(float(shape),0.)
        self.assertGreater(float(gap),0.)

    def test_mixture_copy_is_reported_not_claimed_as_separation(self):
        b=np.array([[1.,0.],[.5,.4],[0.,.8]],dtype=np.float32)
        a=np.tile(b.sum(1)[:,None],(1,8))
        m=measure(a,b,np.arange(3)*.02,1.)
        self.assertEqual(m['source_count_mae'],float(np.mean([7,6,7])))
        self.assertGreater(len(m['global_copy_pairs']),0)
        self.assertLess(np.mean(m['overlap']['per_source_iou']),1)

    def test_cache_subset_head_matches_full(self):
        model=TemporalVHead()
        x=torch.randn(2,197,768);r=torch.randn(2,197)
        full,_=model(x,r);short,_=model(x[:,96:102],r[:,96:102])
        torch.testing.assert_close(full,short)

    def test_long_sequence_gradient_is_finite(self):
        torch.manual_seed(5151)
        v=(torch.randn(120,3,4)*.1).requires_grad_(True)
        a,c,m=transport(v,np.arange(120)*.02)
        value=a[:,0].mean()
        value.backward()
        self.assertTrue(torch.isfinite(v.grad).all())
        self.assertGreater(float(v.grad.norm()),0.)


if __name__=='__main__':unittest.main()
