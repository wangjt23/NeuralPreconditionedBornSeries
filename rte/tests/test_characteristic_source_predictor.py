import unittest

import torch

from rte.characteristic_source_predictor import CharacteristicDecoder, CharacteristicSourcePredictor
from rte.global_source_predictor import GlobalSourcePredictor


class CharacteristicSourceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)

    def test_upgrade_exact_and_gradients(self):
        base = GlobalSourcePredictor(width=8, blocks=1, modes=2)
        torch.nn.init.normal_(base.output.weight, std=.01)
        model = CharacteristicSourcePredictor(width=8, blocks=1, modes=2)
        model.load_state_dict(base.state_dict())
        t, s = torch.ones(2, 1, 16, 16)*10, torch.ones(2, 1, 16, 16)*9
        q, ref = [torch.randn(2, 12, 16, 16, dtype=torch.complex64) for _ in range(2)]
        torch.testing.assert_close(model(t, s, q, ref), base(t, s, q, ref), atol=0, rtol=0)
        # Positive source-amplitude homogeneity is an exact architectural
        # property (away from normalization floors), not something to learn.
        torch.testing.assert_close(model(t, s, 3*q, 3*ref), 3*model(t, s, q, ref))
        model(t, s, q, ref).abs().square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.characteristic.combine.weight.grad.abs().sum()), 0.)

    def test_transverse_modes_and_stability(self):
        decoder = CharacteristicDecoder(width=8)
        n = 32
        fields = torch.randn(2, 3, 12, n, n, dtype=torch.complex64)
        # Angle zero travels in x. An arbitrary y-dependent field must survive
        # unchanged, even at the finest transverse grid frequency.
        fields[:, :, 0] = fields[:, :, 0, :1].expand(-1, -1, n, -1)
        t = torch.ones(2, 1, n, n)*4
        result = decoder.filter(fields, t)
        torch.testing.assert_close(result[:, :, 0], fields[:, :, 0], atol=8e-7, rtol=8e-7)
        self.assertLessEqual(float(result.abs().square().sum()), float(fields.abs().square().sum()))
        shifted = torch.roll(fields, (3, 5), (-2, -1))
        torch.testing.assert_close(decoder.filter(shifted, t), torch.roll(result, (3, 5), (-2, -1)))


if __name__ == '__main__':
    unittest.main()
