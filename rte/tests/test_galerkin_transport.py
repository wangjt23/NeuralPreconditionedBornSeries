import unittest

import torch

from rte.galerkin_transport import GalerkinTransport
from rte.multilevel_neural_transport import restrict_field
from rte.run_hard_case_experiments import make_case


class GalerkinTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dealiased_equals_direct_rap_and_is_accretive(self):
        for n in (16, 32):
            fine = make_case('mixed', n, 6, torch.float64, .1)[0]
            coarse = GalerkinTransport(fine, 8)
            value = torch.randn(1, 6, 8, 8, dtype=torch.complex128)
            torch.testing.assert_close(coarse(value), coarse.direct(value), atol=1e-11, rtol=1e-11)
            band = restrict_field(value, 8)
            energy = (band.conj()*coarse(band)).sum().real
            self.assertGreater(float(energy), 0.)

    def test_constant_and_linearity(self):
        fine = make_case('mixed', 32, 6, torch.float64, .1)[0]
        coarse = GalerkinTransport(fine, 8)
        r, s = [torch.randn(1, 6, 8, 8, dtype=torch.complex128) for _ in range(2)]
        torch.testing.assert_close(coarse((.3+.5j)*r+s), (.3+.5j)*coarse(r)+coarse(s))


if __name__ == '__main__':
    unittest.main()
