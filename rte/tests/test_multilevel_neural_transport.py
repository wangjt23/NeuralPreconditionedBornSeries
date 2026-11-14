import unittest

import torch

from rte.compiled_hss import CompiledHSS
from rte.multilevel_neural_transport import MultilevelNeuralTransport, prolong_field, restrict_field
from rte.neural_transport_operator import NeuralTransportOperator
from rte.run_hard_case_experiments import make_case


class MultilevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_transfer_constant_and_band_projection(self):
        value = torch.randn(2, 6, 8, 8, dtype=torch.complex128)
        projected = restrict_field(value, 8)
        fine = prolong_field(value, 16)
        torch.testing.assert_close(restrict_field(fine, 8), projected)
        torch.testing.assert_close(prolong_field(projected, 16), fine)
        ones = torch.ones_like(value)
        torch.testing.assert_close(prolong_field(ones, 16), torch.ones(2, 6, 16, 16, dtype=value.dtype))
        torch.testing.assert_close(restrict_field(torch.ones_like(fine), 8), ones)

    def test_linearity_gradients_and_refresh(self):
        op = make_case('mixed', 16, 6, torch.float64, .1)[0]
        model = MultilevelNeuralTransport(op, lambda coarse: NeuralTransportOperator(coarse, width=8, base='hss').double(), 8)
        r, s = torch.randn_like(op.source), torch.randn_like(op.source)
        torch.testing.assert_close(model((.2+.3j)*r+s), (.2+.3j)*model(r)+model(s))
        self.assertEqual(model(torch.zeros_like(r)).abs().max().item(), 0.)
        model(r).abs().square().mean().backward()
        self.assertTrue(torch.isfinite(model.coarse_model.matrix_head.weight.grad).all())
        with torch.no_grad():
            model.prepare()
            first = model(r)
            op.setup(op.sigma_t+.3, op.sigma_s, op.source)
            refreshed = model(r)
            self.assertFalse(torch.allclose(first, refreshed))
            self.assertTrue((model.coarse_operator.sigma_t > model.coarse_operator.sigma_s).all())

    def test_analytic_anchor_equal_classical_two_level(self):
        op = make_case('mixed', 16, 6, torch.float64, .1)[0]
        neural = MultilevelNeuralTransport(op, lambda coarse: NeuralTransportOperator(coarse, depth=2, width=8, base='hss').double(), 8, pre=1, post=1)
        classical = MultilevelNeuralTransport(op, lambda coarse: CompiledHSS(coarse, 2), 8, pre=1, post=1)
        r = torch.randn_like(op.source)
        torch.testing.assert_close(neural(r), classical(r), atol=1e-11, rtol=1e-11)


if __name__ == '__main__':
    unittest.main()
