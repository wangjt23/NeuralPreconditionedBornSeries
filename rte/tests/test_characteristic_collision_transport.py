import unittest

import torch

from rte.characteristic_collision_transport import CharacteristicCollisionTransport
from rte.mixed_precision_transport import high_precision_copy
from rte.neural_transport_operator import NeuralTransportOperator
from rte.run_hard_case_experiments import make_case
from rte.train_krylov_transport import training_loss


class CharacteristicCTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_head_and_conjugate_symbol(self):
        op = make_case('mixed', 8, 6, torch.float64, .1)[0]
        model = CharacteristicCollisionTransport(op, 2).double()
        r = torch.randn_like(op.source)
        torch.testing.assert_close(model(r), model.backbone(r), atol=1e-11, rtol=1e-11)
        with torch.no_grad():
            model.symbol_net[-1].weight.normal_(std=.03)
        _, summary = NeuralTransportOperator._features(model)
        scale = op.sigma_t.mean((1, 2, 3))[:, None, None, None]
        kx, ky = torch.randn(5, 1, dtype=torch.float64), torch.randn(1, 5, dtype=torch.float64)
        positive = model.modulation(kx, ky, summary, scale)
        negative = model.modulation(-kx, -ky, summary, scale)
        torch.testing.assert_close(negative, positive.conj(), atol=1e-12, rtol=1e-12)
        dc = model.modulation(torch.zeros(1, 1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64), summary, scale)
        torch.testing.assert_close(dc, dc[:, :, :1].expand_as(dc))

    def test_train_cache_and_gradient(self):
        op = make_case('mixed', 8, 6, torch.float32, .1)[0]
        model = CharacteristicCollisionTransport(op, 2)
        loss, _ = training_loss(high_precision_copy(op), model, op.source, 2)
        loss.backward()
        self.assertTrue(torch.isfinite(model.symbol_net[-1].weight.grad).all())
        with torch.no_grad():
            value = model(op.source)
            model.eval()
            torch.testing.assert_close(model(op.source), value)


if __name__ == '__main__':
    unittest.main()
