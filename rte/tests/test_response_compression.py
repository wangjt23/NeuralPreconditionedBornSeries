import unittest

import torch

from rte.audit_hierarchical_response_oracle import reconstruct_block
from rte.audit_parity_response_blocks import rank_for_frobenius


class ResponseCompressionTests(unittest.TestCase):
    def test_left_weighted_truncation_preserves_residual_metric(self):
        torch.manual_seed(4)
        a = torch.randn(20, 8, dtype=torch.complex128)*torch.logspace(0, 3, 8)
        block = torch.randn(8, 7, dtype=torch.complex128)
        _, r = torch.linalg.qr(a, mode='reduced')
        u, s, vh = torch.linalg.svd(r @ block, full_matrices=False)
        compressed = reconstruct_block(u, s, vh, 3, r)
        error = torch.linalg.vector_norm(a @ (block-compressed))
        torch.testing.assert_close(error, torch.linalg.vector_norm(s[3:]), atol=1e-9, rtol=1e-12)
        u0, s0, vh0 = torch.linalg.svd(block, full_matrices=False)
        euclidean = reconstruct_block(u0, s0, vh0, 3)
        self.assertLessEqual(float(error), float(torch.linalg.vector_norm(a @ (block-euclidean))))

    def test_rank_criterion_and_zero_block(self):
        self.assertEqual(rank_for_frobenius(torch.tensor([1., .01, .0001]), .02), 1)
        self.assertEqual(rank_for_frobenius(torch.zeros(3), .01), 0)


if __name__ == '__main__':
    unittest.main()
