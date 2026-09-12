import unittest

import torch

from rte.run_hard_case_experiments import make_case
from rte.spectral_sparse_reference import SparseSpectralFactor, spectral_matrix


class SpectralSparseTests(unittest.TestCase):
    def test_matrix_matches_pseudospectral_action_even_and_odd(self):
        for size in (4, 5):
            rte, _ = make_case("mixed", size, 5, torch.float64, .1)
            matrix = spectral_matrix(rte)
            r = torch.randn_like(rte.source)
            actual = torch.from_numpy(matrix @ r.flatten().numpy()).reshape_as(r)
            torch.testing.assert_close(actual, rte.A_op(r), atol=1e-12, rtol=1e-12)

    def test_exact_factor_and_coefficient_guard(self):
        rte, _ = make_case("mixed", 4, 5, torch.float64, .1)
        factor = SparseSpectralFactor(rte, exact=True)
        r = torch.randn(2, *rte.source.shape[1:], dtype=rte.source.dtype)
        torch.testing.assert_close(rte.A_op(factor(r)), r, atol=1e-12, rtol=1e-12)
        rte.sigma_t.add_(.1)
        with self.assertRaises(RuntimeError):
            factor(r)


if __name__ == "__main__":
    unittest.main()
