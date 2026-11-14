import unittest

import torch

from rte.coarse_response_transport import CachedCoarseResponse, CoarseResponseTransport
from rte.compiled_hss import CompiledHSS
from rte.hierarchical_response import HierarchicalResponse
from rte.mixed_precision_transport import high_precision_copy
from rte.parity_coarse import GalerkinSchurCorrection
from rte.parity_transport import ParitySchur
from rte.run_hard_case_experiments import make_case


class CoarseResponseTransportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(15)
        self.low = make_case('mixed', 8, 12, torch.float32, 0.)[0]
        self.fine = high_precision_copy(self.low)
        self.parity = ParitySchur(self.fine)
        self.coarse = GalerkinSchurCorrection(self.parity, size=4)

    def test_cached_dense_and_hierarchy_equal_cholesky(self):
        inverse = torch.cholesky_inverse(self.coarse.factor)
        response = HierarchicalResponse.from_dense(inverse, 6, 4, tolerance=0)
        rhs = torch.randn(2, 6, 8, 8, dtype=torch.complex128)
        dense = CachedCoarseResponse(self.coarse, dtype=torch.complex128)
        hierarchy = CachedCoarseResponse(self.coarse, response, dtype=torch.complex128)
        torch.testing.assert_close(dense(rhs), self.coarse(rhs), atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(hierarchy(rhs), self.coarse(rhs), atol=1e-11, rtol=1e-11)
        self.assertFalse(hasattr(dense, 'coarse'))  # no hidden dense teacher cache

    def test_action_is_full_angle_linear_and_preserves_physics(self):
        cache = CachedCoarseResponse(self.coarse)
        action = CoarseResponseTransport(self.fine, self.parity, cache,
            CompiledHSS(self.low, 4, 2.), post=True)
        r, s = [torch.randn_like(self.fine.source) for _ in range(2)]
        before = self.fine.A_op(r).clone()
        z = action(r)
        self.assertEqual(z.shape, r.shape)
        torch.testing.assert_close(action(r+2j*s), z+2j*action(s), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(self.fine.A_op(r), before, atol=0, rtol=0)
        torch.testing.assert_close(action(torch.zeros_like(r)), torch.zeros_like(r))
        self.fine.setup(self.fine.sigma_t, self.fine.sigma_s, self.fine.source)
        with self.assertRaisesRegex(RuntimeError, 'reconstruct'):
            action(r)

    def test_harmonic_only_correction_preserves_odd_image(self):
        cache = CachedCoarseResponse(self.coarse)
        hss = CompiledHSS(self.low, 4, 2.)
        action = CoarseResponseTransport(self.fine, self.parity, cache, hss,
            recover_odd_rhs=False)
        rhs = torch.randn_like(self.fine.source)
        fine_value = hss(rhs.to(torch.complex64)).to(rhs.dtype)
        correction = action(rhs)-fine_value
        _, odd = self.parity.split(self.parity.full_reference_action(correction))
        torch.testing.assert_close(odd, torch.zeros_like(odd), atol=1e-11, rtol=0)

    def test_physical_minimum_residual_guard(self):
        cache = CachedCoarseResponse(self.coarse)
        hss = CompiledHSS(self.low, 4, 2.)
        action = CoarseResponseTransport(self.fine, self.parity, lambda r: -100*cache(r), hss,
            recover_odd_rhs=False, safeguard=True)
        rhs = torch.randn(3, 12, 8, 8, dtype=torch.complex128)
        base = hss(rhs.to(torch.complex64)).to(rhs.dtype)
        before = (rhs-self.fine.A_op(base)).abs().square().flatten(1).sum(1)
        after = (rhs-self.fine.A_op(action(rhs))).abs().square().flatten(1).sum(1)
        self.assertTrue(bool((after <= before*(1+1e-12)).all()))
        torch.testing.assert_close(action(torch.zeros_like(rhs)), torch.zeros_like(rhs))


if __name__ == '__main__':
    unittest.main()
