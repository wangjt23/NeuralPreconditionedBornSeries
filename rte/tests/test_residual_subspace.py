import unittest
from types import SimpleNamespace

import torch

from rte.models import PeriodicRTE2D
from rte.probe_residual_amplification import amplification_probe
from rte.residual_subspace import (
    residual_minimizing_correction, physical_residual_loss, projected_training_loss,
    ResidualMinimizedPreconditioner,
)


class ResidualSubspaceTests(unittest.TestCase):
    def test_matches_independent_least_squares(self):
        torch.manual_seed(123)
        a = torch.randn(9, 9, dtype=torch.complex128) + 4 * torch.eye(9)
        op = lambda x: x @ a.T
        r = torch.randn(3, 9, dtype=torch.complex128)
        baseline = torch.randn_like(r) * 0.01
        directions = [torch.randn_like(r) for _ in range(3)]
        result = residual_minimizing_correction(op, r, baseline, directions, ridge=1e-10)
        for i in range(3):
            w = torch.stack([op(q)[i] for q in directions], dim=1)
            e = (r - op(baseline))[i]
            real_w = torch.cat((w.real, w.imag))
            real_e = torch.cat((e.real, e.imag))
            c = torch.linalg.lstsq(real_w, real_e).solution
            expected = baseline[i] + sum(c[j] * q[i] for j, q in enumerate(directions))
            torch.testing.assert_close(result[i], expected, atol=1e-8, rtol=1e-8)

    def test_safeguards_zero_dependent_and_bad_directions(self):
        torch.manual_seed(3)
        r = torch.randn(4, 10, dtype=torch.complex128)
        op = lambda x: x * torch.arange(1, 11)
        baseline = 10 * r
        directions = [torch.zeros_like(r), r, r, -r]
        got = residual_minimizing_correction(op, r, baseline, directions)
        score = (r - op(got)).abs().square().sum(1)
        for candidate in [torch.zeros_like(r), baseline] + [baseline + q for q in directions]:
            self.assertTrue(torch.all(score <= (r - op(candidate)).abs().square().sum(1) + 1e-10))
        zero = residual_minimizing_correction(op, r * 0, baseline * 0, [r * 0])
        torch.testing.assert_close(zero, r * 0)

    def test_physical_loss_is_mean_square_not_square_of_mean(self):
        r = torch.ones(2, 4, dtype=torch.complex128)
        z = r * torch.tensor([0.0, 0.5]).unsqueeze(1)
        self.assertAlmostEqual(float(physical_residual_loss(lambda x: x, r, z)), 0.625)

    def test_amplification_probe_against_known_singular_value(self):
        template = torch.ones(1, 4, 2, 2, dtype=torch.complex128)
        diagonal = torch.linspace(0.1, 0.9, 16, dtype=torch.float64).reshape_as(template.real)
        ratio, _, _ = amplification_probe(lambda x: x, lambda r: (1 - diagonal) * r,
                                          template, seed=12, steps=100)
        self.assertAlmostEqual(ratio, .9, places=7)

    def test_real_autograd(self):
        torch.manual_seed(7)
        r = torch.randn(1, 6, dtype=torch.float64)
        b = r * 0.1
        q = torch.randn_like(r, requires_grad=True)
        op = lambda x: x * torch.arange(1, 7)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: residual_minimizing_correction(op, r, b, [x]), (q,),
            atol=1e-4, rtol=1e-3))

    def test_zero_output_can_train_and_actual_wrapper_is_safeguarded(self):
        class Model(torch.nn.Module):
            base_calls_per_apply = 2

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
                self.rte = SimpleNamespace(A_op=self.A_op)

            def A_op(self, x):
                return x * torch.arange(1, 5, dtype=x.real.dtype)

            def correction_parts(self, r):
                return 0.1 * r, self.weight * r

        model = Model()
        r = torch.ones(2, 4, dtype=torch.float64)
        loss, _ = projected_training_loss(model, r)
        loss.backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all())
        self.assertGreater(float(model.weight.grad.norm()), 0)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([100.0, -10.0, 3.0, 0.4]))
            projected = ResidualMinimizedPreconditioner(model)(r)
            baseline, delta = model.correction_parts(r)
            projected_loss = physical_residual_loss(model.A_op, r, projected)
            for candidate in (baseline, baseline + delta, r * 0):
                self.assertLessEqual(float(projected_loss),
                                     float(physical_residual_loss(model.A_op, r, candidate)) + 1e-12)


class ReferenceInverseTests(unittest.TestCase):
    def test_lazy_and_batch_independent(self):
        torch.manual_seed(11)
        t = 2 + torch.rand(2, 1, 4, 4, dtype=torch.float64)
        s = t * 0.8
        q = torch.randn(2, 4, 4, 4, dtype=torch.complex128)
        def create(t, s, q):
            return PeriodicRTE2D(shape=(4, 4), dx=0.25, n_angles=4,
                                 fdtype=torch.float64).setup(t, s, q)
        rte = create(t, s, q)
        self.assertIsNone(rte._reference_transport_inverse)
        dsa = rte.transport_dsa_preconditioner(q)
        self.assertIsNone(rte._reference_transport_inverse)
        tsa = rte.reference_transport_solve(q)
        self.assertIsNotNone(rte._reference_transport_inverse)
        for i in range(2):
            solo = create(t[i:i+1], s[i:i+1], q[i:i+1])
            torch.testing.assert_close(dsa[i:i+1], solo.transport_dsa_preconditioner(q[i:i+1]))
            torch.testing.assert_close(tsa[i:i+1], solo.reference_transport_solve(q[i:i+1]))
        matrix = rte._reference_transport_matrix()
        inv = rte.reference_transport_inverse
        torch.testing.assert_close(matrix @ inv, torch.eye(4, dtype=inv.dtype).expand_as(inv),
                                   atol=1e-12, rtol=1e-12)
        rte.setup(t + 1, s, q)
        self.assertIsNone(rte._reference_transport_inverse)


if __name__ == "__main__":
    unittest.main()
