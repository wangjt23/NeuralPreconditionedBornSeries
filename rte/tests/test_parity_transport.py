import unittest

import torch

from rte.parity_transport import ParitySchur, pcg
from rte.run_hard_case_experiments import make_case


class ParityTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_schur_hermitian_positive_and_block_recovery(self):
        op = make_case('mixed', 8, 12, torch.float64, 0.)[0]
        p = ParitySchur(op)
        x, y = [torch.randn(1, 6, 8, 8, dtype=torch.complex128) for _ in range(2)]
        xy = (x.conj()*p.schur(y)).sum()
        yx = (p.schur(x).conj()*y).sum()
        torch.testing.assert_close(xy, yx, atol=1e-10, rtol=1e-11)
        self.assertGreater(float((x.conj()*p.schur(x)).sum().real), 0.)
        q = torch.randn_like(op.source)
        recovered = p.recover(x, q)
        e, o = p.split(p.full_reference_action(recovered)-q)
        torch.testing.assert_close(o, torch.zeros_like(o), atol=1e-11, rtol=0)
        torch.testing.assert_close(e, p.schur(x)-p.reduced_rhs(q), atol=1e-11, rtol=1e-11)

    def test_surrogate_close_but_does_not_change_physics(self):
        op = make_case('mixed', 8, 12, torch.float64, 0.)[0]
        p = ParitySchur(op)
        q = torch.randn_like(op.source)
        torch.testing.assert_close(p.full_reference_action(q), op.A_op(q), atol=2e-12, rtol=2e-12)
        before = op.A_op(q).clone()
        p.solve(q, 3)
        torch.testing.assert_close(before, op.A_op(q), atol=0, rtol=0)

    def test_homogeneous_reference_exact_and_zero_rhs(self):
        op = make_case('diffusive', 8, 12, torch.float64, 0.)[0]
        p = ParitySchur(op)
        q = torch.randn_like(op.source)
        x, _, used = p.solve(q, 4, 1e-10)
        self.assertEqual(used, 1)
        torch.testing.assert_close(op.A_op(x), q, atol=1e-11, rtol=1e-11)
        x, r, used = p.solve(torch.zeros_like(q), 4)
        self.assertEqual(used, 0)
        self.assertTrue(bool(torch.isfinite(x).all()))
        self.assertEqual(float(r.max()), 0.)

    def test_cache_rejects_material_change(self):
        op = make_case('mixed', 8, 12, torch.float64, 0.)[0]
        p = ParitySchur(op)
        op.setup(op.sigma_t, op.sigma_s, op.source)
        with self.assertRaisesRegex(RuntimeError, 'reconstruct'):
            p.solve(op.source, 2)


if __name__ == '__main__':
    unittest.main()
