import unittest

import torch

from rte.augmented_fgmres import augmented_fgmres, projected_direction_loss
from rte.source_initialization import solve_from_initial


class AugmentedFGMRESTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(81)
        self.matrix = torch.diag(torch.linspace(.1, 3., 24)).to(torch.complex128)
        self.matrix += .05*torch.randn(24, 24, dtype=torch.complex128)
        self.rhs = torch.randn(1, 2, 3, 4, dtype=torch.complex128)
        self.action = lambda x: (self.matrix@x.flatten()).reshape_as(x)
        self.exact = torch.linalg.solve(self.matrix, self.rhs.flatten()).reshape_as(self.rhs)

    def test_matches_legacy_and_retains_original_rhs_tolerance(self):
        initial = .37*self.exact
        old, status = solve_from_initial(self.action, self.rhs, lambda r:r,
            initial, tolerance=1e-8, maxiter=100)
        new, result = augmented_fgmres(self.action, self.rhs, lambda r:r,
            initial, tolerance=1e-8, maxiter=100)
        torch.testing.assert_close(new, old, atol=1e-9, rtol=1e-8)
        self.assertEqual(status['iterations'], result['iterations'])
        self.assertEqual(result['failed'], 0)

    def test_exact_tail_happy_breakdown_finishes_in_one_extra_direction(self):
        captures = []
        result, status = augmented_fgmres(self.action, self.rhs, lambda r:r,
            inject=lambda x,r: self.exact-x, inject_after=4, tolerance=1e-10,
            capture_prefix=captures.append)
        self.assertEqual(status['classical_calls'], 4)
        self.assertEqual(status['iterations'], 5)
        self.assertEqual(status['injection_accepted'], 1)
        self.assertEqual(status['failed'], 0)
        torch.testing.assert_close(result, self.exact)
        basis = captures[0]['image_basis'][:, 0].flatten(1)
        torch.testing.assert_close((basis.conj()@basis.T).real, torch.eye(4, dtype=torch.float64))
        residual = captures[0]['residual'].flatten()
        torch.testing.assert_close((basis.conj()@residual).real, torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_zero_and_redundant_directions_fall_back_without_restarting(self):
        baseline, plain = augmented_fgmres(self.action, self.rhs, lambda r:r, tolerance=1e-9)
        for proposal in [lambda x,r: torch.zeros_like(r), lambda x,r: self.rhs]:
            result, status = augmented_fgmres(self.action, self.rhs, lambda r:r,
                inject=proposal, tolerance=1e-9)
            self.assertEqual(status['injection_rejected'], 1)
            self.assertEqual(status['classical_calls'], plain['classical_calls'])
            torch.testing.assert_close(result, baseline)

    def test_easy_case_bypasses_injection(self):
        def forbidden(x, r):
            self.fail('already-converged case invoked neural proposal')
        _, status = augmented_fgmres(lambda x:x, self.rhs, lambda x:x, inject=forbidden)
        self.assertEqual(status['injection_calls'], 0)
        self.assertEqual(status['failed'], 0)

    def test_projected_loss_matches_explicit_augmentation_and_backpropagates(self):
        captures = []
        _, _ = augmented_fgmres(self.action, self.rhs, lambda r:r,
            maxiter=4, capture_prefix=captures.append)
        context = captures[0]
        z = (self.exact-context['solution']+.05*torch.randn_like(self.rhs)).requires_grad_()
        basis = context['image_basis'].transpose(0, 1)
        loss = projected_direction_loss(self.action, z, context['residual'], basis)
        q = basis[0].flatten(1).T
        columns = torch.cat((q, self.action(z).flatten()[:, None]), 1)
        real = torch.cat((columns.real, columns.imag))
        r = context['residual'].flatten()
        rreal = torch.cat((r.real, r.imag))
        coefficient = torch.linalg.lstsq(real, rreal).solution
        expected = (rreal-real@coefficient).square().sum()/rreal.square().sum()
        torch.testing.assert_close(loss[0], expected)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())


if __name__ == '__main__':
    unittest.main()
