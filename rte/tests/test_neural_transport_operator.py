import unittest
import weakref

import torch

from rte.neural_transport_operator import NeuralTransportOperator
from rte.collision_transport import CollisionTransportPreconditioner
from rte.compiled_hss import CompiledHSS
from rte.run_hard_case_experiments import make_case
from rte.transport_baselines import ClassicalTransportStack


class NeuralTransportOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make(self, dtype=torch.float64, device='cpu'):
        op = make_case('mixed', 8, 6, dtype, .1)[0].to(device)
        op.setup(op.sigma_t.to(device), op.sigma_s.to(device), op.source.to(device))
        model = NeuralTransportOperator(op, width=8).to(device=device, dtype=dtype)
        return op, model

    def test_zero_heads_equal_two_reference_corrections(self):
        op, model = self.make()
        reference = ClassicalTransportStack(op, 'reference', 2, 0.)
        r = torch.randn_like(op.source)
        torch.testing.assert_close(model(r), reference(r), atol=1e-11, rtol=1e-11)

    def test_compiled_classical_hss_matches_fft_control_and_refreshes(self):
        op, _ = self.make()
        compiled = CompiledHSS(op, depth=4, shift_factor=.5)
        original = CollisionTransportPreconditioner(op, depth=4, shift_factor=.5)
        for _ in range(2):
            r = torch.randn_like(op.source)
            torch.testing.assert_close(compiled(r), original(r), atol=1e-11, rtol=1e-11)
            op.setup(op.sigma_t+.2, op.sigma_s, op.source)

    def test_hss_anchor_and_trained_linearity(self):
        op, _ = self.make()
        model = NeuralTransportOperator(op, width=8, base='hss').double()
        reference = CollisionTransportPreconditioner(op, depth=2)
        r, s = torch.randn_like(op.source), torch.randn_like(op.source)
        torch.testing.assert_close(model(r), reference(r), atol=1e-11, rtol=1e-11)
        with torch.no_grad():
            model.matrix_head.weight.normal_(std=.03)
            model.symbol_net[-1].weight.normal_(std=.03)
            model.shift_net[-1].weight.normal_(std=.03)
        torch.testing.assert_close(model((.2+.3j)*r+s), (.2+.3j)*model(r)+model(s), atol=1e-11, rtol=1e-11)
        (r-op.A_op(model(r))).abs().square().mean().backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_real_compiled_maps_match_complex_training_path(self):
        for device in ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []):
            op, _ = self.make(torch.float32, device)
            model = NeuralTransportOperator(op, depth=4, width=8, base='hss').to(device)
            with torch.no_grad():
                model.matrix_head.weight.normal_(std=.03)
                model.symbol_net[-1].weight.normal_(std=.03)
                residual = torch.randn_like(op.source)
                direct = model(residual)
                model.eval()
                compiled = model(residual)
                self.assertFalse(model.fields()[0].is_complex())
                torch.testing.assert_close(compiled, direct, atol=1e-6, rtol=1e-5)

    def test_linearity_zero_and_gradients(self):
        op, model = self.make()
        with torch.no_grad():
            model.matrix_head.weight.normal_(std=.03)
            model.symbol_net[-1].weight.normal_(std=.03)
        r, s = torch.randn_like(op.source), torch.randn_like(op.source)
        alpha = .25 + .6j
        torch.testing.assert_close(model(alpha*r+s), alpha*model(r)+model(s), atol=1e-11, rtol=1e-11)
        self.assertEqual(model(torch.zeros_like(r)).abs().max().item(), 0.)
        loss = (r-op.A_op(model(r))).abs().square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.stem.weight.grad.norm().item(), 0.)

    def test_scalar_protection_and_full_rank_local_maps(self):
        _, model = self.make()
        with torch.no_grad():
            model.matrix_head.bias.normal_(std=8.)
            maps, _ = model.fields()
            torch.testing.assert_close(maps.sum(-1), torch.ones_like(maps[..., 0]), atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(maps.sum(-2), torch.ones_like(maps[..., 0]), atol=1e-12, rtol=1e-12)
            self.assertGreater(torch.linalg.svdvals(maps).min().item(), .5)

    def test_cache_lifetime_parameter_and_coefficient_refresh(self):
        op, model = self.make()
        model.eval()
        calls = []
        hook = model.stem.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            a = model(op.source)
            torch.testing.assert_close(a, model(op.source))
            self.assertEqual(len(calls), 1)
            ref = weakref.ref(op.sigma_t)
            for _ in range(3):
                op.setup(op.sigma_t+.2, op.sigma_s, op.source)
            self.assertIsNotNone(ref())
            cached = model(op.source)
            self.assertIsNone(ref())
            self.assertEqual(len(calls), 2)
            model.matrix_head.bias.add_(.1)
            cached = model(op.source)
            self.assertEqual(len(calls), 3)
            model.train()
            torch.testing.assert_close(cached, model(op.source), atol=1e-11, rtol=1e-11)
        hook.remove()

    def test_single_medium_multiple_rhs(self):
        op, model = self.make()
        r = torch.randn(3, *op.source.shape[1:], dtype=op.source.dtype)
        torch.testing.assert_close(model(r), torch.cat([model(x) for x in r.split(1)]), atol=1e-11, rtol=1e-11)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_forward_backward_and_cache_agreement(self):
        op, model = self.make(torch.float32, 'cuda:0')
        r = torch.randn_like(op.source)
        loss = (r-op.A_op(model(r))).abs().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.matrix_head.weight.grad).all())
        with torch.no_grad():
            direct = model(r)
            model.eval()
            torch.testing.assert_close(model(r), direct, atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
