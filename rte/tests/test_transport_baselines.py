import unittest

import torch

from rte.run_hard_case_experiments import make_case
from rte.transport_baselines import ClassicalTransportStack


class TransportBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def operator(self, case="mixed"):
        return make_case(case, 8, 8, torch.float64, .1)[0]

    def test_shifted_stack_matches_existing_tsa_and_dsa(self):
        from rte.npbs_unet_ablation import inner_base_apply
        rte = self.operator()
        r = torch.randn_like(rte.source)
        for kind in ("tsa", "dsa", "transport"):
            model = ClassicalTransportStack(rte, kind, depth=3, eta_scale=.1, relaxation=.7)
            base_kind = "transport" if kind == "transport" else "transport_" + kind
            torch.testing.assert_close(model(r), inner_base_apply(rte, r, depth=3, kind=base_kind, relaxation=.7))

    def test_zero_shift_homogeneous_reference_is_inverse(self):
        rte = self.operator("anisotropic")
        model = ClassicalTransportStack(rte, "reference", eta_scale=0.)
        r = torch.randn_like(rte.source)
        torch.testing.assert_close(rte.A_op(model(r)), r, atol=1e-12, rtol=1e-12)
        self.assertEqual(float(model.reference.eta.max()), 0.)

    def test_physical_operator_is_unchanged_and_calls_visible(self):
        rte = self.operator()
        r = torch.randn_like(rte.source)
        actual = rte.A_op(r)
        original, count = rte.A_op, []
        rte.A_op = lambda x: (count.append(1), original(x))[1]
        model = ClassicalTransportStack(rte, "tsa", depth=3, eta_scale=.3)
        model(r)
        self.assertEqual(len(count), 5)
        torch.testing.assert_close(rte.A_op(r), actual)

    def test_coefficient_refresh_and_discrete_transport_diffusion(self):
        rte = self.operator("anisotropic")
        model = ClassicalTransportStack(rte, "dsa_transport")
        p1 = torch.fft.fft(rte.phase_matrix[0] * rte.weights)[1].real
        expected = 1. / (2 * (rte.sigma_t - p1 * rte.sigma_s).mean((-2, -1), keepdim=True))
        torch.testing.assert_close(model.reference.diffusion_coeff, expected)
        rte.setup(rte.sigma_t + 1., rte.sigma_s, rte.source)
        model.prepare()
        torch.testing.assert_close(model.reference.sigma_t, rte.sigma_t)
        rte.dx *= 2
        with self.assertRaises(RuntimeError):
            model.prepare()


if __name__ == "__main__":
    unittest.main()
