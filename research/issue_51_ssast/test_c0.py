"""P1 contracts: sample PIT, gradient path, and time-boundary metrics."""
import unittest

import numpy as np
import torch

from dataset_adapter import crop_two_seconds
from head import TemporalVHead
from shape_loss import c0_loss
from train_c0 import metrics


class C0Tests(unittest.TestCase):
    def test_crop_has_explicit_edge_padding(self):
        crop, fraction = crop_two_seconds(torch.ones(48000), 0.)
        self.assertEqual(fraction, .5)
        self.assertEqual(float(crop[:, :16000].sum()), 0.)
        self.assertEqual(float(crop[:, 16000:].sum()), 16000.)

    def test_entire_sample_slot_not_frame_pit(self):
        truth = torch.ones(6, 1)
        a = torch.tensor([[1., 0.]] * 3 + [[0., 1.]] * 3, requires_grad=True)
        loss, detail = c0_loss(a, truth)
        self.assertGreater(float(detail['shape'].detach()), .4)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_head_norm_and_parameter_gradients(self):
        torch.manual_seed(51)
        head = TemporalVHead()
        v, a = head(torch.randn(2, 197, 768), torch.ones(2, 197))
        self.assertEqual(tuple(v.shape), (2, 8, 128))
        self.assertTrue(torch.equal(a, torch.linalg.vector_norm(v, dim=-1)))
        a.sum().backward()
        self.assertGreater(float(head.temporal[0].weight.grad.norm()), 0.)

    def test_receptive_field_optimization_matches_full_sequence(self):
        torch.manual_seed(51)
        head = TemporalVHead()
        tokens, rms = torch.randn(3, 197, 768), torch.randn(3, 197)
        optimized, _ = head(tokens, rms)
        h = head.project(torch.cat([tokens, rms.unsqueeze(-1)], dim=-1))
        h = head.temporal(h.transpose(1, 2)).transpose(1, 2)
        full = head.output(.75 * h[:, 98] + .25 * h[:, 99]).reshape(3, 8, 128)
        torch.testing.assert_close(optimized, full, rtol=1e-5, atol=1e-7)

    def test_last_active_frame_and_matching_cost(self):
        truth = np.array([[0.], [1.], [1.]])
        a = np.zeros((3, 8))
        a[:, 4] = truth[:, 0]
        m = metrics(a, truth, np.arange(3) * .02, 1.)
        self.assertEqual(m['slot'], 4)
        self.assertEqual(m['area_iou'], 1.)
        self.assertEqual(m['boundaries']['offset']['median_abs_seconds'], 0.)
        _, detail = c0_loss(torch.tensor(a), torch.tensor(truth))
        self.assertEqual(detail['slot'], m['slot'])


if __name__ == '__main__':
    unittest.main()
