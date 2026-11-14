import unittest

import torch

from rte.global_source_predictor import FrozenTSA, GlobalSourcePredictor, balance_integrated_source
from rte.mixed_precision_transport import high_precision_copy
from rte.run_hard_case_experiments import make_case
from rte.transport_baselines import ClassicalTransportStack


class GlobalSourcePredictorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)

    def test_zero_start_multiresolution_and_gradients(self):
        model = GlobalSourcePredictor(width=8, blocks=2, modes=2)
        for n in (16, 32):
            t, s = torch.ones(2, 1, n, n)*10, torch.ones(2, 1, n, n)*9
            q, ref = [torch.randn(2, 12, n, n, dtype=torch.complex64) for _ in range(2)]
            result = model(t, s, q, ref)
            torch.testing.assert_close(result, ref)
            result.abs().square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_mass_balance_and_cached_tsa(self):
        low = make_case('mixed', 8, 12, torch.float32, 0.)[0]
        high = high_precision_copy(low)
        q = torch.randn_like(high.source)
        z = balance_integrated_source(high, torch.randn_like(q), q)
        torch.testing.assert_close(high.A_op(z).sum(), q.sum(), atol=1e-10, rtol=1e-10)
        baseline = ClassicalTransportStack(low, 'tsa', 3, 0.)
        ref = baseline.reference
        stream_inverse = 1/(ref._stream_symbol(1, low.complex_dtype, low.source.device)+ref.sigma0)
        frozen = FrozenTSA(low, ref.reference_transport_inverse, stream_inverse, 3)
        rhs = torch.randn_like(low.source).requires_grad_()
        torch.testing.assert_close(frozen(rhs), baseline(rhs), atol=2e-6, rtol=2e-6)
        frozen(rhs).abs().square().mean().backward()
        self.assertTrue(torch.isfinite(rhs.grad).all())

    def test_nested_materials_keep_physical_rotations(self):
        from rte.train_source_predictor import source_family
        _, coarse = source_family(64, 3, 917)
        _, fine = source_family(128, 3, 917)
        for key in ('sigma_t', 'sigma_s'):
            torch.testing.assert_close(coarse[key], fine[key][..., ::2, ::2], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
