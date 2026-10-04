"""Small tests of real frontend/grid/type contract risks; no model training."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from features import FrozenSSAST, grid, log_fbank


class FrontendTests(unittest.TestCase):
    def test_real_two_second_frontend_and_fixed_stats(self):
        silent = log_fbank(torch.zeros(1, 32000))
        self.assertEqual(tuple(silent.shape), (198, 128))
        self.assertTrue(torch.isfinite(silent).all())
        self.assertGreater(float(silent.abs().sum()), 0.)
        with self.assertRaises(ValueError):
            log_fbank(torch.zeros(2, 32000))

    def test_frame_and_patch_coordinates(self):
        f, t, c, start, end = grid(198, 128, 2, 128, 1)
        self.assertEqual((f, t), (1, 197))
        self.assertAlmostEqual(c[0], .0175)
        np.testing.assert_allclose(np.diff(c), .01, atol=1e-14)
        np.testing.assert_allclose(end - start, .035)
        f, t, c, _, _ = grid(198, 16, 16, 10, 10, origin=1.)
        self.assertEqual((f, t), (12, 19))
        self.assertAlmostEqual(c[0], 1.0875)
        self.assertAlmostEqual(c[-1], 2.8875)

    def test_actual_conv_flatten_order(self):
        # A real convolution exposes F-major/T-minor flatten ordering.
        conv = torch.nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            conv.weight.fill_(1.)
        image = torch.tensor([[[[10., 11., 12.], [20., 21., 22.]]]])
        flat = conv(image).flatten(2).transpose(1, 2)
        self.assertEqual(flat[0, :, 0].tolist(), [10., 11., 12., 20., 21., 22.])
        self.assertTrue(torch.equal(flat.reshape(1, 2, 3, 1)[0, :, :, 0], image[0, 0]))

    def test_patch_checkpoint_cannot_silently_be_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'model.py'
            source.write_text('# no class needed: reject before constructing model\n')
            checkpoint = Path(directory) / 'patch.pth'
            torch.save({'module.v.patch_embed.proj.weight': torch.zeros(768, 1, 16, 16)}, checkpoint)
            with self.assertRaisesRegex(ValueError, 'checkpoint type mismatch'):
                FrozenSSAST(source, checkpoint, model_type='frame', device='cpu')


if __name__ == '__main__':
    unittest.main()
