"""Small CPU counterexamples and independent real least-squares checks."""

import math
import unittest

import torch

from continuation_acceptance import corrected_continuation


def field(values):
    return torch.tensor(values, dtype=torch.complex128).reshape(1, 1, 1, -1)


def apply_matrix(matrix, value):
    return torch.einsum("ij,bj->bi", matrix, value.flatten(1)).reshape_as(value)


def fixture():
    matrix = torch.diag(torch.tensor([1., 2., 3., 4.], dtype=torch.complex128))
    action = lambda value: apply_matrix(matrix, value)
    residual = field([0, 1, 1, 0])
    basis = field([1, 0, 0, 0])[:, None]
    # Actual second Arnoldi vector from source (1,1,1,0) and first image e1.
    frontier = field([2, -1, -1, 0]) / math.sqrt(6)
    return matrix, action, residual, basis, frontier


def independent_remaining(images, residual):
    """Solve the expanded real LS directly; no sequential-projector reuse."""
    columns = torch.stack([value.flatten() for value in images], 1)
    real_columns = torch.cat((columns.real, columns.imag), 0)
    rhs = torch.cat((residual.flatten().real, residual.flatten().imag))
    coefficients = torch.linalg.lstsq(real_columns, rhs, driver="gelsd").solution
    return residual - (columns @ coefficients.to(columns.dtype)).reshape_as(residual)


class AcceptanceContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def assert_explicit_ls(self, action, proposal, residual, basis, result):
        for index in range(len(residual)):
            images = [row for row in basis[index]]
            if bool(result["proposal_accepted"][index]):
                images.append(action(proposal[index:index + 1])[0])
            for step in range(result["ratios"].shape[1]):
                if step and bool(result["classical_step_active"][index, step - 1]):
                    direction = result["classical_directions"][index:index + 1, step - 1]
                    images.append(action(direction)[0])
                expected = independent_remaining(images, residual[index])
                torch.testing.assert_close(result["residuals"][index, step], expected,
                                           atol=2e-12, rtol=2e-10)
                expected_ratio = expected.norm().square() / residual[index].norm().square()
                torch.testing.assert_close(result["ratios"][index, step], expected_ratio,
                                           atol=2e-12, rtol=2e-10)

    def test_original_zero_mask_is_flat_but_ordered_fallback_progresses(self):
        from rte.tail_lookahead import continuation_loss as original_masked_loss

        _, action, residual, basis, frontier = fixture()
        zero = torch.zeros_like(residual)
        masked = original_masked_loss(action, lambda x: x, zero, residual, basis, 2)
        torch.testing.assert_close(masked, torch.ones_like(masked))
        inputs = []
        def baseline(value):
            inputs.append(value.clone())
            return value
        result = corrected_continuation(action, baseline, zero, residual, basis, frontier, 2)
        self.assertFalse(bool(result["proposal_accepted"][0]))
        self.assertEqual(int(result["proposal_rejection_code"][0]), 1)
        torch.testing.assert_close(inputs[0], frontier)
        self.assertAlmostEqual(float(result["ratios"][0, 0]), 1.)
        self.assertAlmostEqual(float(result["ratios"][0, 1]), 1 / 26, places=12)
        self.assertLess(float(result["ratios"][0, -1]), 1e-24)
        self.assert_explicit_ls(action, zero, residual, basis, result)

    def test_finite_dependent_image_preserves_fallback(self):
        matrix, action, residual, basis, frontier = fixture()
        dependent = torch.linalg.solve(matrix, basis[:, 0].flatten()).reshape_as(residual)
        expected = corrected_continuation(action, lambda x: x, torch.zeros_like(residual),
                                          residual, basis, frontier, 2)
        actual = corrected_continuation(action, lambda x: x, dependent, residual, basis, frontier, 2)
        self.assertEqual(int(actual["proposal_rejection_code"][0]), 3)
        torch.testing.assert_close(actual["ratios"], expected["ratios"])
        self.assert_explicit_ls(action, dependent, residual, basis, actual)

    def test_counterexample_matches_actual_deployed_rejection_trace(self):
        from rte.augmented_fgmres import augmented_fgmres

        _, action, residual, basis, frontier = fixture()
        rhs = residual + basis[:, 0]
        used = []
        def first_image_then_identity(value):
            used.append(value.clone())
            return basis[:, 0] if len(used) == 1 else value
        _, deployed = augmented_fgmres(action, rhs, first_image_then_identity,
                                      inject=lambda x, r: torch.zeros_like(r),
                                      inject_after=1, maxiter=3, tolerance=1e-14)
        corrected = corrected_continuation(action, lambda x: x, torch.zeros_like(residual),
                                            residual, basis, frontier, 2)
        self.assertEqual(deployed["injection_rejected"], 1)
        self.assertFalse(deployed["failed"])
        torch.testing.assert_close(used[1], frontier, atol=2e-12, rtol=2e-10)
        prefix_ratio = deployed["trace"][1]
        expected = torch.tensor([(ratio / prefix_ratio)**2 for ratio in deployed["trace"][1:]],
                                dtype=corrected["ratios"].dtype)
        torch.testing.assert_close(corrected["ratios"][0], expected, atol=2e-12, rtol=2e-10)

    def test_near_dependent_image_rejected_relative_to_full_action(self):
        matrix, action, residual, basis, frontier = fixture()
        image = basis[:, 0] + 1e-14 * residual
        proposal = torch.linalg.solve(matrix, image.flatten()).reshape_as(residual)
        result = corrected_continuation(action, lambda x: x, proposal, residual, basis, frontier, 1)
        self.assertEqual(int(result["proposal_rejection_code"][0]), 3)
        self.assertLess(float(result["ratios"][0, -1]), .05)

    def test_teacher_inside_arnoldi_but_outside_images_is_happy_breakdown(self):
        matrix, action, residual, basis, frontier = fixture()
        teacher = torch.linalg.solve(matrix, residual.flatten()).reshape_as(residual)
        result = corrected_continuation(action, lambda x: x, teacher, residual, basis, frontier, 3)
        self.assertTrue(bool(result["proposal_accepted"][0]))
        self.assertTrue(bool(result["happy_breakdown"][0]))
        self.assertTrue(bool(result["accepted_frontier_breakdown"][0]))
        self.assertEqual(int(result["classical_calls"][0]), 0)
        self.assertLess(float(result["ratios"].max()), 1e-24)
        self.assert_explicit_ls(action, teacher, residual, basis, result)

    def test_per_batch_rejection_does_not_freeze_other_members(self):
        matrix, action, residual, basis, frontier = fixture()
        teacher = torch.linalg.solve(matrix, residual.flatten()).reshape_as(residual)
        dependent = basis[:, 0]
        nonfinite = torch.full_like(residual, complex(float("nan"), 0))
        proposals = torch.cat((torch.zeros_like(residual), teacher, dependent, nonfinite))
        r, q, f = residual.repeat(4, 1, 1, 1), basis.repeat(4, 1, 1, 1, 1), frontier.repeat(4, 1, 1, 1)
        result = corrected_continuation(action, lambda x: x, proposals, r, q, f, 2)
        self.assertEqual(result["proposal_rejection_code"].tolist(), [1, 0, 3, 2])
        self.assertTrue(bool(torch.isfinite(result["ratios"]).all()))
        self.assertEqual(result["classical_calls"].tolist(), [2, 0, 2, 2])
        for index in range(4):
            single = corrected_continuation(action, lambda x: x, proposals[index:index + 1],
                                            r[index:index + 1], q[index:index + 1], f[index:index + 1], 2)
            torch.testing.assert_close(result["ratios"][index], single["ratios"][0])

    def test_positive_and_negative_real_proposal_scaling_invariance(self):
        matrix, action, residual, basis, frontier = fixture()
        proposal = field([.2, 1, .3 + .1j, .4j])
        baseline_matrix = torch.eye(4, dtype=matrix.dtype) + .1 * matrix.inverse()
        baseline = lambda value: apply_matrix(baseline_matrix, value)
        expected = corrected_continuation(action, baseline, proposal, residual, basis, frontier, 2)
        self.assertTrue(bool(expected["proposal_accepted"][0]))
        for scale in (.2, 5., -.2, -3.):
            actual = corrected_continuation(action, baseline, scale * proposal,
                                            residual, basis, frontier, 2)
            torch.testing.assert_close(actual["ratios"], expected["ratios"], atol=2e-12, rtol=2e-10)
            torch.testing.assert_close(actual["final_residual"], expected["final_residual"], atol=2e-12, rtol=2e-10)

    def test_rotating_basis_frontier_and_data_preserves_phase(self):
        matrix, action, residual, basis, frontier = fixture()
        matrix = matrix + .15j * torch.eye(4, dtype=matrix.dtype)
        action = lambda value: apply_matrix(matrix, value)
        baseline_matrix = torch.eye(4, dtype=matrix.dtype) + .1j * matrix.inverse()
        baseline = lambda value: apply_matrix(baseline_matrix, value)
        for proposal in (torch.zeros_like(residual), field([.2, 1, .3 + .1j, .4j])):
            expected = corrected_continuation(action, baseline, proposal, residual, basis, frontier, 2)
            for angle in (.37, math.pi / 2, math.pi):
                phase = torch.tensor(complex(math.cos(angle), math.sin(angle)), dtype=matrix.dtype)
                actual = corrected_continuation(action, baseline, phase * proposal, phase * residual,
                                                phase * basis, phase * frontier, 2)
                torch.testing.assert_close(actual["ratios"], expected["ratios"], atol=2e-12, rtol=2e-10)
                torch.testing.assert_close(actual["residuals"], phase * expected["residuals"], atol=2e-12, rtol=2e-10)

    def test_expanded_image_space_is_monotone_and_matches_direct_ls(self):
        _, action, residual, basis, frontier = fixture()
        proposal = field([.2, 1, .3 + .1j, .4j])
        result = corrected_continuation(action, lambda x: x, proposal, residual, basis, frontier, 3)
        self.assertTrue(bool(result["proposal_accepted"][0]))
        sequence = torch.cat((torch.ones(1, 1, dtype=result["ratios"].dtype), result["ratios"]), 1)
        self.assertTrue(bool((sequence[:, 1:] <= sequence[:, :-1] + 2e-12).all()))
        self.assert_explicit_ls(action, proposal, residual, basis, result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
