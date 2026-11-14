import unittest

import torch

from rte.hierarchical_response import HierarchicalResponse
from rte.response_residual_repair import RepairedResponse, residual_repair_basis


class ResidualRepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(191)

    def test_hierarchical_adjoint(self):
        d = 64
        u = torch.randn(d, 2, dtype=torch.complex128)
        matrix = u@torch.randn(2, d, dtype=u.dtype)+torch.eye(d, dtype=u.dtype)
        response = HierarchicalResponse.from_dense(matrix, 1, 8, leaf_size=1,
            tolerance=1e-5, rank_multiple=1)
        rhs = torch.randn(3, d, dtype=u.dtype)
        torch.testing.assert_close(response.adjoint(rhs), rhs@response.dense().conj())

    def test_repair_identity_orthogonality_and_error_reduction(self):
        d = 64
        a = torch.randn(d, d, dtype=torch.complex128)
        matrix = a@a.mH+torch.eye(d, dtype=a.dtype)
        inverse = torch.linalg.inv(matrix)
        perturbation = torch.randn(d, 3, dtype=a.dtype)@torch.randn(3, d, dtype=a.dtype)*.01
        approximate = inverse+perturbation
        base = HierarchicalResponse.from_dense(approximate, 1, 8, tolerance=0, leaf_size=2)
        correction, basis, singular = residual_repair_basis(matrix, inverse, base, rank=3)
        response = RepairedResponse(base, correction, basis)
        identity = torch.eye(d, dtype=a.dtype)
        torch.testing.assert_close(basis.mH@basis, identity[:3, :3])
        torch.testing.assert_close(response(basis.T), (inverse@basis).T, atol=1e-12, rtol=1e-11)
        result = response(identity).T
        before = identity-matrix@approximate
        after = identity-matrix@result
        torch.testing.assert_close(after, before@(identity-basis@basis.mH), atol=1e-11, rtol=1e-10)
        self.assertLess(float(torch.linalg.vector_norm(after)), 1e-10)
        self.assertLess(float(singular[3]), 1e-10)
        self.assertEqual(response.to(dtype=torch.complex64).diagnostics()['repair_rank'], 3)


if __name__ == '__main__':
    unittest.main()
