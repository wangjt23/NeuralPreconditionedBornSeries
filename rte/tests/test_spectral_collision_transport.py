import unittest

import torch

from rte.mixed_precision_transport import high_precision_copy
from rte.run_hard_case_experiments import make_case
from rte.spectral_collision_transport import SpectralCollisionTransport
from rte.train_krylov_transport import training_loss
from rte.repeated_transport import RepeatedTransportCorrection
from rte.compiled_hss import CompiledHSS


class SpectralCTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_spectral_head_equals_ct_and_trained_linearity(self):
        op = make_case('mixed', 8, 6, torch.float64, .1)[0]
        model = SpectralCollisionTransport(op, 2).double()
        r, s = torch.randn_like(op.source), torch.randn_like(op.source)
        torch.testing.assert_close(model(r), model.backbone(r), atol=1e-11, rtol=1e-11)
        with torch.no_grad():
            model.symbol_net[-1].weight.normal_(std=.03)
        torch.testing.assert_close(model((.2+.3j)*r+s), (.2+.3j)*model(r)+model(s), atol=1e-11, rtol=1e-11)

    def test_cache_and_krylov_gradients(self):
        op = make_case('mixed', 8, 6, torch.float32, .1)[0]
        model = SpectralCollisionTransport(op, 2)
        loss, _ = training_loss(high_precision_copy(op), model, op.source, 2)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        calls = []
        hook = model.symbol_net.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            direct = model(op.source)
            model.eval()
            compiled = model(op.source)
            torch.testing.assert_close(compiled, direct)
            model(op.source)
            self.assertEqual(len(calls), 2)
            model.symbol_net[-1].bias.add_(.01)
            model(op.source)
            self.assertEqual(len(calls), 3)
            op.setup(op.sigma_t+.2, op.sigma_s, op.source)
            model(op.source)
            self.assertEqual(len(calls), 4)
        hook.remove()

    def test_repeated_analytic_block_matches_deeper_hss(self):
        op = make_case('mixed', 8, 6, torch.float64, .1)[0]
        repeated = RepeatedTransportCorrection(op, CompiledHSS(op, 2), 2)
        r = torch.randn_like(op.source)
        torch.testing.assert_close(repeated(r), CompiledHSS(op, 4)(r), atol=1e-11, rtol=1e-11)


if __name__ == '__main__':
    unittest.main()
