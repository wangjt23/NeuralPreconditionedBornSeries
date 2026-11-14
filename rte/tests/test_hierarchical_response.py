import unittest

import torch

from rte.hierarchical_response import Cluster, HierarchicalParityPreconditioner, HierarchicalResponse, retained_rank, separated


class HierarchicalResponseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def test_periodic_admissibility(self):
        self.assertFalse(separated(Cluster(3, 0, 0), Cluster(3, 7, 7)))
        self.assertTrue(separated(Cluster(3, 0, 0), Cluster(3, 6, 0)))
        self.assertFalse(separated(Cluster(1, 0, 0), Cluster(1, 1, 1)))

    def test_exact_partition_and_batch_action(self):
        size, channels = 8, 2
        d = channels*size**2
        matrix = torch.randn(d, d, dtype=torch.complex128)
        response = HierarchicalResponse.from_dense(matrix, channels, size, tolerance=0, leaf_size=2)
        count = torch.zeros(d, d, dtype=torch.int32)
        for group in response.groups:
            for rows, cols in zip(group['rows'], group['cols']):
                count[rows[:, None], cols[None, :]] += 1
        torch.testing.assert_close(count, torch.ones_like(count))
        rhs = torch.randn(3, d, dtype=matrix.dtype)
        torch.testing.assert_close(response(rhs), rhs@matrix.T)
        torch.testing.assert_close(response.dense(), matrix)
        self.assertEqual(response.diagnostics()['coefficient_scalars'], d*d)

    def test_factored_action_and_complex_linearity(self):
        d = 256
        u = torch.randn(d, 2, dtype=torch.complex128)
        v = torch.randn(2, d, dtype=torch.complex128)
        matrix = u@v + torch.eye(d, dtype=u.dtype)
        response = HierarchicalResponse.from_dense(matrix, 1, 16, leaf_size=2,
            tolerance=1e-6, rank_multiple=1)
        self.assertGreater(response.diagnostics()['low_rank_blocks'], 0)
        self.assertLess(response.diagnostics()['stored_scalar_fraction'], 1)
        torch.testing.assert_close(response.dense(), matrix, atol=1e-10, rtol=1e-10)
        rhs = torch.randn(2, d, dtype=u.dtype)
        torch.testing.assert_close(response(rhs[:1]+2j*rhs[1:]), response(rhs[:1])+2j*response(rhs[1:]))
        rhs.requires_grad_()
        response(rhs).abs().square().sum().backward()
        self.assertTrue(torch.isfinite(rhs.grad).all())

    def test_weighted_and_zero(self):
        d = 64
        diagonal = torch.logspace(0, 4, d, dtype=torch.float64).to(torch.complex128)
        metric = torch.diag(diagonal)
        teacher = torch.diag(1/diagonal) + torch.ones(d, d, dtype=diagonal.dtype)*1e-5
        response = HierarchicalResponse.from_dense(teacher, 1, 8, matrix=metric,
            tolerance=1e-5, leaf_size=1, rank_multiple=1)
        torch.testing.assert_close(response.dense(), teacher, atol=1e-12, rtol=1e-10)
        zero = HierarchicalResponse.from_dense(torch.zeros_like(teacher), 1, 8,
            matrix=metric, tolerance=.01, leaf_size=1, rank_multiple=1)
        torch.testing.assert_close(zero(torch.ones(1, d, dtype=teacher.dtype)), torch.zeros(1, d, dtype=teacher.dtype))
        self.assertEqual(retained_rank(torch.zeros(3), .1), 0)
        self.assertEqual(retained_rank(torch.ones(3), 0), 3)

    def test_validation(self):
        with self.assertRaises(ValueError):
            HierarchicalResponse.from_dense(torch.eye(36), 1, 6)
        with self.assertRaises(ValueError):
            retained_rank(torch.ones(3), -1)

    def test_two_sided_energy_weighting_and_zero(self):
        d = 64
        diagonal = torch.logspace(0, 3, d, dtype=torch.float64).to(torch.complex128)
        metric = torch.diag(diagonal)
        teacher = torch.diag(1/diagonal)+torch.ones(d, d, dtype=diagonal.dtype)*1e-5
        response = HierarchicalResponse.from_dense(teacher, 1, 8, matrix=metric,
            metric='energy', tolerance=1e-5, leaf_size=1, rank_multiple=1)
        torch.testing.assert_close(response.dense(), teacher, atol=1e-12, rtol=1e-10)
        # ||H^.5 E_IJ H^.5||_F = ||R_I E R_J^*||_F, R_I^* R_I=H_II.
        a = torch.randn(12, 12, dtype=torch.complex128)
        h = a@a.mH+torch.eye(12, dtype=a.dtype)
        eig, vectors = torch.linalg.eigh(h)
        square_root = (vectors*eig.sqrt())@vectors.mH
        rows, cols = torch.tensor([0, 4, 7]), torch.tensor([1, 2, 8, 9])
        e = torch.randn(3, 4, dtype=a.dtype)
        ri = torch.linalg.cholesky(h[rows[:, None], rows[None, :]]).mH
        rj = torch.linalg.cholesky(h[cols[:, None], cols[None, :]]).mH
        torch.testing.assert_close(torch.linalg.vector_norm(square_root[:, rows]@e@square_root[cols]),
            torch.linalg.vector_norm(ri@e@rj.mH))

    def test_full_angle_recovery_precision_and_cache_guard(self):
        from rte.audit_parity_response_blocks import dense_action
        from rte.parity_transport import ParitySchur
        from rte.run_hard_case_experiments import make_case
        op = make_case('mixed', 4, 12, torch.float64, 0.)[0]
        parity = ParitySchur(op)
        matrix = dense_action(parity.schur, 6, 4, torch.device('cpu'))
        inverse = torch.linalg.inv(matrix)
        response = HierarchicalResponse.from_dense(inverse, 6, 4, tolerance=0, leaf_size=2)
        precondition = HierarchicalParityPreconditioner(parity, response)
        rhs = torch.randn_like(op.source)
        torch.testing.assert_close(op.A_op(precondition(rhs)), rhs, atol=1e-10, rtol=1e-10)
        single = HierarchicalParityPreconditioner(parity, response.to(dtype=torch.complex64))
        relative = torch.linalg.vector_norm(op.A_op(single(rhs))-rhs)/torch.linalg.vector_norm(rhs)
        self.assertLess(float(relative), 1e-4)
        before = op.A_op(rhs).clone()
        single(rhs)
        torch.testing.assert_close(before, op.A_op(rhs), atol=0, rtol=0)
        op.setup(op.sigma_t, op.sigma_s, op.source)
        with self.assertRaisesRegex(RuntimeError, 'reconstruct'):
            precondition(rhs)


if __name__ == '__main__':
    unittest.main()
