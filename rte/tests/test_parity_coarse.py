import unittest

import torch

from rte.parity_coarse import CharacteristicSchurCorrection, GalerkinSchurCorrection, SchurMultilevelControl
from rte.parity_transport import ParitySchur
from rte.run_hard_case_experiments import make_case


class ParityCoarseTests(unittest.TestCase):
    def test_coarse_and_additive_are_hermitian_positive(self):
        torch.set_num_threads(1)
        p = ParitySchur(make_case('mixed', 8, 6, torch.float64, 0.)[0])
        coarse = GalerkinSchurCorrection(p, 4)
        self.assertLess(coarse.symmetry_defect, 1e-12)
        model = SchurMultilevelControl(p, coarse, True)
        x, y = [torch.randn(1, 3, 8, 8, dtype=torch.complex128) for _ in range(2)]
        torch.testing.assert_close((x.conj()*model(y)).sum(), (model(x).conj()*y).sum(), atol=1e-10, rtol=1e-11)
        self.assertGreater(float((x.conj()*model(x)).sum().real), 0.)

    def test_compressed_coarse_equals_fine_galerkin(self):
        torch.set_num_threads(1)
        for size in (8, 16):
            p = ParitySchur(make_case('mixed', size, 6, torch.float64, 0.)[0])
            coarse = GalerkinSchurCorrection(p, 4)
            q = torch.randn(2, 3, 4, 4, dtype=torch.complex128)
            torch.testing.assert_close(coarse.action(q), coarse.direct_action(q), atol=1e-10, rtol=1e-11)

    def test_characteristic_dictionary_equals_direct_projection(self):
        torch.set_num_threads(1)
        p = ParitySchur(make_case('mixed', 16, 6, torch.float64, 0.)[0])
        coarse = CharacteristicSchurCorrection(p, 9)
        q = torch.randn(2, 27, dtype=torch.complex128)
        torch.testing.assert_close(coarse.action(q), coarse.direct_action(q), atol=1e-10, rtol=1e-11)
        torch.testing.assert_close(coarse.restrict(coarse.prolong(q)), q, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
