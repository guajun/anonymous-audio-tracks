import unittest
import numpy as np
import torch
from window_head import WindowVHead
from fast_association import transport as fast
from local_association import transport as reference


class WindowDepthTests(unittest.TestCase):
    def test_every_token_affects_output_at_all_depths(self):
        torch.manual_seed(5152)
        counts=[]
        for depth in (2,4,8):
            model=WindowVHead(depth)
            tokens=torch.randn(1,197,768,requires_grad=True)
            v,_=model(tokens,torch.ones(1,197))
            v.square().sum().backward()
            self.assertTrue((tokens.grad.abs().sum(-1)>0).all())
            counts.append(model.architecture()['parameters'])
            with self.assertRaises(ValueError):model(tokens[:,96:102],torch.ones(1,6))
        self.assertTrue(counts[0]<counts[1]<counts[2])

    def test_batched_transport_and_gradients_match_reference(self):
        self.check_transport('cpu')

    def test_far_intervention_preserves_center_path_at_every_depth(self):
        torch.manual_seed(5154)
        original=torch.randn(1,197,769)
        altered=original.clone()
        altered[:,:90]=original.mean(1,keepdim=True)
        altered[:,108:]=original.mean(1,keepdim=True)
        for depth in (2,4,8):
            model=WindowVHead(depth)
            with torch.no_grad():
                a=model.temporal(model.project(original).transpose(1,2))
                b=model.temporal(model.project(altered).transpose(1,2))
            torch.testing.assert_close(a[:,:,98:100],b[:,:,98:100],rtol=0,atol=0)
            self.assertGreater(float((a.mean(-1)-b.mean(-1)).abs().sum()),0.)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA equivalence checked on experiment GPU')
    def test_batched_cuda_transport_and_gradients_match_reference(self):
        self.check_transport('cuda')

    def check_transport(self,device):
        torch.manual_seed(5152)
        v=torch.randn(55,4,7)*.07
        v[8:14]=0;v[20:24,1]=0;v[40:]=0;v[54,2]=.1
        times=np.r_[np.arange(54)*.02,2.4]
        x=v.to(device).clone().requires_grad_(True);y=v.to(device).clone().requires_grad_(True)
        a,c,m=reference(x,times);b,cc,mm=fast(y,times)
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(c,cc,atol=1e-6,rtol=1e-5)
        self.assertEqual(m['anchors'],mm['anchors'])
        self.assertEqual(m['fragment_ids_by_frame'],mm['fragment_ids_by_frame'])
        weights=torch.randn_like(a)
        (a*weights).sum().backward();(b*weights).sum().backward()
        torch.testing.assert_close(x.grad,y.grad,atol=2e-5,rtol=2e-4)


if __name__=='__main__':unittest.main()
