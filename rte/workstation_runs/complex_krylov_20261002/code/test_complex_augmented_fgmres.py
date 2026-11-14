"""Small CPU oracle/protocol tests for complex augmented flexible GMRES.

The fields are deterministic dense unit examples, not project data. The local
snapshot's exact converter functions are AST-loaded solely to import the
unchanged real-stacked solver without its unavailable training dependencies.
"""

import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
snapshot = ROOT / "remote_reference" / "rte"
if snapshot.is_dir():
    sys.path.insert(0, str(snapshot.parent))
    conversion_path = snapshot / "npbs_unet_ablation.py"
    conversion_names = {"complex_to_real_channels", "real_channels_to_complex"}
    functions = [node for node in ast.parse(conversion_path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in conversion_names]
    if {node.name for node in functions} != conversion_names:
        raise RuntimeError("the archived physical/real-stacked conversion functions are missing")
    conversion_module = types.ModuleType("rte.npbs_unet_ablation")
    conversion_module.__file__ = str(conversion_path)
    conversion_module.__dict__["torch"] = torch
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(conversion_path), "exec"),
         conversion_module.__dict__)
    sys.modules[conversion_module.__name__] = conversion_module

from rte.augmented_fgmres import augmented_fgmres
from complex_augmented_fgmres import _small_lstsq, complex_augmented_fgmres


class ComplexAugmentedFGMRESTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator(device="cpu").manual_seed(318)
        self.matrix = torch.diag(torch.linspace(.6, 3., 8, dtype=torch.float64)).to(torch.complex128)
        self.matrix += .07 * torch.randn(8, 8, generator=generator, dtype=torch.complex128)
        self.rhs = torch.randn(1, 1, 1, 8, generator=generator, dtype=torch.complex128)
        self.action = lambda value: (self.matrix @ value.flatten()).reshape_as(value)
        self.exact = torch.linalg.solve(self.matrix, self.rhs.flatten()).reshape_as(self.rhs)

    def assert_close(self, actual, expected, atol=3e-12, rtol=3e-11):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def assert_original_status_keys(self, status):
        expected = {"iterations", "classical_calls", "injection_calls", "injection_accepted",
                    "injection_rejected", "true_residual", "failed", "trace"}
        self.assertTrue(expected.issubset(status))
        self.assertEqual(len(status["trace"]), status["iterations"] + 1)

    def assert_physical_status(self, result, status, tolerance):
        self.assert_original_status_keys(status)
        residual = float((self.rhs - self.action(result)).norm() / self.rhs.norm())
        self.assertAlmostEqual(status["true_residual"], residual, delta=2e-14)
        self.assertEqual(status["failed"], int(residual > tolerance))
        self.assertEqual(result.shape, self.rhs.shape)
        self.assertEqual(result.dtype, self.rhs.dtype)
        self.assertEqual(result.device.type, "cpu")

    def test_scalar_imaginary_operator_converges_in_one_complex_step(self):
        action = lambda value: 1j * value
        exact = -1j * self.rhs
        result, status = complex_augmented_fgmres(action, self.rhs, lambda value: value,
                                                tolerance=1e-12, maxiter=4, restart=4)
        self.assert_close(result, exact)
        self.assertEqual(status["iterations"], 1)
        self.assertEqual(status["classical_calls"], 1)
        self.assertEqual(status["failed"], 0)
        original, real_status = augmented_fgmres(action, self.rhs, lambda value: value,
                                                tolerance=1e-12, maxiter=4, restart=4)
        self.assert_close(original, exact)
        self.assertEqual(real_status["iterations"], 2)

    def test_dense_nonhermitian_complex_solve_matches_direct_and_real_stacked_oracles(self):
        result, status = complex_augmented_fgmres(self.action, self.rhs, lambda value: value,
                                                tolerance=1e-12, maxiter=8, restart=8)
        self.assert_close(result, self.exact)
        self.assertLessEqual(status["iterations"], 8)
        self.assertEqual(status["failed"], 0)
        self.assert_physical_status(result, status, 1e-12)
        real_result, real_status = augmented_fgmres(self.action, self.rhs, lambda value: value,
                                                  tolerance=1e-12, maxiter=16, restart=16)
        self.assert_close(result, real_result)
        self.assertEqual(real_status["failed"], 0)

    def test_real_matrix_rhs_agree_with_real_stacked_trace_and_direct_solution(self):
        matrix = self.matrix.real.to(torch.complex128)
        rhs = self.rhs.real.to(torch.complex128)
        action = lambda value: (matrix @ value.flatten()).reshape_as(value)
        direct = torch.linalg.solve(matrix, rhs.flatten()).reshape_as(rhs)
        result, status = complex_augmented_fgmres(action, rhs, lambda value: value,
                                                tolerance=1e-12, maxiter=8, restart=8)
        original, old = augmented_fgmres(action, rhs, lambda value: value,
                                        tolerance=1e-12, maxiter=8, restart=8)
        self.assert_close(result, direct)
        self.assert_close(result, original)
        for key in ("iterations", "classical_calls", "injection_calls", "failed"):
            self.assertEqual(status[key], old[key])
        self.assert_close(torch.tensor(status["trace"]), torch.tensor(old["trace"]), atol=1e-10, rtol=1e-8)

    def assert_prefix_geometry(self, prefix, returned_directions=None):
        directions, images = prefix["directions"], prefix["images"]
        vectors, hessenberg = prefix["arnoldi_basis"], prefix["hessenberg"]
        k = len(directions)
        self.assertEqual(directions.shape, (k,) + self.rhs.shape)
        self.assertEqual(images.shape, directions.shape)
        self.assertEqual(vectors.shape, (k + 1,) + self.rhs.shape)
        self.assertEqual(hessenberg.shape, (k + 1, k))
        self.assert_close(prefix["frontier"], vectors[-1], atol=0, rtol=0)
        flattened = vectors.flatten(1)
        self.assert_close(flattened.conj() @ flattened.T,
                          torch.eye(k + 1, dtype=self.rhs.dtype))
        for index, (direction, image) in enumerate(zip(directions, images)):
            self.assert_close(image, self.action(direction))
            reconstructed = (vectors * hessenberg[:, index].reshape((-1, 1, 1, 1, 1))).sum(0)
            self.assert_close(image, reconstructed)
            if returned_directions is not None:
                self.assert_close(direction, returned_directions[index])
        basis = prefix["image_basis"].flatten(1)
        self.assertEqual(len(basis), k)
        self.assert_close(basis.conj() @ basis.T, torch.eye(k, dtype=self.rhs.dtype))
        self.assert_close(basis.conj() @ prefix["residual"].flatten(),
                          torch.zeros(k, dtype=self.rhs.dtype))
        for image in images.flatten(1):
            self.assert_close((basis.conj() @ image) @ basis, image)
        self.assert_close(prefix["residual"], self.rhs - self.action(prefix["solution"]))
        # Independent least squares uses raw accepted A Z, not a preconditioner
        # linearity assumption or a relation Z=M(V) applied after combination.
        coeff = torch.linalg.lstsq(images.flatten(1).T,
                                   prefix["cycle_start_residual"].flatten()).solution
        expected = prefix["cycle_start_solution"] + (
            directions * coeff.reshape((-1, 1, 1, 1, 1))
        ).sum(0)
        self.assert_close(prefix["solution"], expected)

    def test_prefix_keeps_actual_z_raw_images_complex_orthogonality_and_ordered_frontier(self):
        captures = []
        result, status = complex_augmented_fgmres(
            self.action, self.rhs, lambda value: value, inject_after=4,
            tolerance=1e-14, maxiter=4, restart=12, capture_prefix=captures.append,
        )
        self.assertEqual(len(captures), 1)
        prefix = captures[0]
        self.assertEqual(prefix["iterations"], 4)
        self.assertEqual(prefix["classical_calls"], 4)
        self.assert_prefix_geometry(prefix)
        self.assert_close(prefix["solution"], result)
        self.assert_close(prefix["arnoldi_basis"][0], self.rhs / self.rhs.norm())
        self.assert_physical_status(result, status, 1e-14)

    def test_genuinely_nonlinear_stateful_preconditioner_is_valid_through_accepted_z(self):
        directions, captures = [], []

        def nonlinear(value):
            # Componentwise magnitude dependence is nonlinear even on unit
            # Arnoldi vectors; call-dependent complex scaling is flexible.
            z = value / (.7 + value.abs()) * complex(1 + .03 * len(directions), .04)
            directions.append(z.clone())
            return z

        nonlinear_result, nonlinear_status = complex_augmented_fgmres(
            self.action, self.rhs, nonlinear, inject_after=4,
            tolerance=1e-12, maxiter=8, restart=8, capture_prefix=captures.append,
        )
        self.assertEqual(len(captures), 1)
        self.assert_prefix_geometry(captures[0], directions[:4])
        self.assert_close(nonlinear_result, self.exact)
        self.assertEqual(nonlinear_status["failed"], 0)
        self.assertEqual(len(directions), nonlinear_status["classical_calls"])
        self.assert_physical_status(nonlinear_result, nonlinear_status, 1e-12)

    def test_teacher_inside_arnoldi_span_but_outside_az_is_accepted(self):
        prefixes, injections = [], []
        result, status = complex_augmented_fgmres(
            self.action, self.rhs, lambda value: value,
            inject=lambda solution, residual: self.exact - solution,
            inject_after=4, tolerance=1e-12, maxiter=8, restart=8,
            capture_prefix=prefixes.append, capture_injection=injections.append,
        )
        self.assertEqual(status["injection_calls"], 1)
        self.assertEqual(status["injection_accepted"], 1)
        self.assertEqual(status["injection_rejected"], 0)
        self.assertEqual(status["classical_calls"], 4)
        self.assertEqual(status["iterations"], 5)
        self.assertEqual(status["failed"], 0)
        self.assert_close(result, self.exact)
        self.assertEqual(len(injections), 1)
        stage, prefix = injections[0], prefixes[0]
        self.assertEqual(stage["reason"], "accepted")
        self.assertTrue(stage["accepted"])
        self.assertEqual(status["injection_stages"][0]["reason"], "accepted")
        self.assert_close(stage["frontier_before"], prefix["frontier"])
        self.assert_close(stage["z"], (self.exact - prefix["solution"]) / prefix["residual"].norm())
        self.assert_close(stage["image"], self.action(stage["z"]))
        self.assertLess(float(stage["orthogonal_image"].norm()), 1e-11)
        image_basis = prefix["image_basis"].flatten(1)
        image = stage["image"].flatten()
        outside = image - (image_basis.conj() @ image) @ image_basis
        self.assertGreater(float(outside.norm()), .5)
        self.assertAlmostEqual(stage["relative_residual_before"], status["trace"][4], delta=2e-13)
        self.assertAlmostEqual(stage["relative_residual_after"], status["trace"][5], delta=2e-13)

    def assert_rejection_preserves_frontier(self, proposal, reason):
        plain_vectors, rejected_vectors, prefixes, injections = [], [], [], []

        def recorder(history):
            def apply(value):
                history.append(value.clone())
                return value
            return apply

        kwargs = dict(tolerance=1e-13, maxiter=7, restart=7, inject_after=4)
        plain, plain_status = complex_augmented_fgmres(self.action, self.rhs, recorder(plain_vectors), **kwargs)
        result, status = complex_augmented_fgmres(
            self.action, self.rhs, recorder(rejected_vectors), inject=proposal,
            capture_prefix=prefixes.append, capture_injection=injections.append, **kwargs,
        )
        self.assert_close(result, plain)
        self.assertEqual(len(rejected_vectors), len(plain_vectors))
        for actual, expected in zip(rejected_vectors, plain_vectors):
            self.assert_close(actual, expected)
        self.assert_close(torch.tensor(status["trace"]), torch.tensor(plain_status["trace"]))
        for key in ("iterations", "classical_calls", "true_residual", "failed"):
            if isinstance(plain_status[key], float):
                self.assertAlmostEqual(status[key], plain_status[key], delta=3e-13)
            else:
                self.assertEqual(status[key], plain_status[key])
        self.assertEqual(status["injection_calls"], 1)
        self.assertEqual(status["injection_rejected"], 1)
        self.assertEqual(status["injection_accepted"], 0)
        self.assertEqual(len(injections), 1)
        stage = injections[0]
        self.assertEqual(stage["reason"], reason)
        self.assertFalse(stage["accepted"])
        self.assert_close(stage["frontier_before"], prefixes[0]["frontier"])
        self.assert_close(stage["frontier_before"], rejected_vectors[4])
        self.assertAlmostEqual(stage["relative_residual_after"], stage["relative_residual_before"], delta=3e-13)
        return stage

    def test_zero_and_nonfinite_injections_reject_without_displacing_frontier(self):
        for proposal in (lambda solution, residual: torch.zeros_like(residual),
                         lambda solution, residual: torch.full_like(residual, float("nan"))):
            with self.subTest(proposal=proposal):
                self.assert_rejection_preserves_frontier(proposal, "invalid_proposal")

    def test_complex_multiple_of_existing_image_rejects_without_displacing_frontier(self):
        stage = self.assert_rejection_preserves_frontier(
            lambda solution, residual: 1j * self.rhs, "dependent_image"
        )
        self.assert_close(stage["image"], self.action(stage["z"]))
        self.assertLess(float(stage["dependence_column"].norm()), 1e-11)

    def test_nonzero_initial_and_source_normalization_are_original_rhs_relative(self):
        initial = .99 * self.exact
        expected_ratio = float((self.rhs - self.action(initial)).norm() / self.rhs.norm())
        result, status = complex_augmented_fgmres(self.action, self.rhs, lambda value: value,
                                                initial=initial, tolerance=.02, maxiter=8, restart=8)
        self.assert_close(result, initial, atol=0, rtol=0)
        self.assertEqual(status["classical_calls"], 0)
        self.assertEqual(status["iterations"], 0)
        self.assertAlmostEqual(status["trace"][0], expected_ratio, delta=2e-14)
        self.assertAlmostEqual(status["true_residual"], expected_ratio, delta=2e-14)
        self.assertEqual(status["failed"], 0)
        result, status = complex_augmented_fgmres(self.action, self.rhs, lambda value: value,
                                                initial=.37 * self.exact, tolerance=1e-12, maxiter=8, restart=8)
        self.assert_close(result, self.exact)
        self.assert_physical_status(result, status, 1e-12)

    def test_zero_rhs_and_exact_initial_avoid_preconditioner_and_injection(self):
        forbidden = lambda *args: (_ for _ in ()).throw(AssertionError("unexpected callback"))
        zero = torch.zeros_like(self.rhs)
        result, status = complex_augmented_fgmres(forbidden, zero, forbidden,
                                                initial=torch.ones_like(zero), inject=forbidden)
        self.assertTrue(torch.equal(result, zero))
        self.assertEqual(status["iterations"], 0)
        self.assertEqual(status["trace"], [0.])
        self.assertEqual(status["failed"], 0)
        initial = self.rhs.clone()
        rhs = self.action(initial)
        result, status = complex_augmented_fgmres(self.action, rhs, forbidden,
                                                initial=initial, inject=forbidden, tolerance=1e-13)
        self.assert_close(result, initial, atol=0, rtol=0)
        self.assertEqual(status["iterations"], 0)
        self.assertEqual(status["classical_calls"], 0)
        self.assertEqual(status["injection_calls"], 0)
        self.assertEqual(status["failed"], 0)

    def test_restart_and_total_budget_are_honored_with_physical_failure_status(self):
        result, status = complex_augmented_fgmres(self.action, self.rhs, lambda value: value,
                                                tolerance=1e-30, maxiter=5, restart=2)
        self.assertEqual(status["iterations"], 5)
        self.assertEqual(status["classical_calls"], 5)
        self.assertEqual(status["cycles"], 3)
        self.assertEqual(status["failed"], 1)
        self.assert_physical_status(result, status, 1e-30)
        for key in ("maxiter", "restart", "inject_after"):
            with self.subTest(budget=key), self.assertRaises(ValueError):
                complex_augmented_fgmres(self.action, self.rhs, lambda value: value, **{key: 0})

    def test_prefix_after_restart_contains_only_current_cycle_z_images_and_start(self):
        captures = []
        result, status = complex_augmented_fgmres(
            self.action, self.rhs, lambda value: value, inject_after=4,
            tolerance=1e-30, maxiter=5, restart=2, capture_prefix=captures.append,
        )
        self.assertEqual(len(captures), 1)
        prefix = captures[0]
        self.assertEqual(prefix["iterations"], 4)
        self.assertEqual(prefix["classical_calls"], 4)
        self.assertEqual(len(prefix["directions"]), 2)
        self.assertEqual(len(prefix["images"]), 2)
        self.assertGreater(float(prefix["cycle_start_solution"].norm()), 0.)
        self.assert_close(prefix["cycle_start_residual"], self.rhs - self.action(prefix["cycle_start_solution"]))
        self.assert_close(prefix["arnoldi_basis"][0],
                          prefix["cycle_start_residual"] / prefix["cycle_start_residual"].norm())
        self.assert_prefix_geometry(prefix)
        self.assert_physical_status(result, status, 1e-30)

    def test_rank_deficient_complex_small_ls_matches_independent_gelsd_images_and_residual(self):
        w = torch.tensor([1 + 2j, -.3 + .2j, .4 - .8j, 2 - .1j, -1 + .7j], dtype=torch.complex128)
        u = torch.tensor([-.2 + .3j, 1 - .4j, .7 + .1j, -.6 + .9j, .3 - .2j], dtype=torch.complex128)
        u = u - w * (w.conj() @ u) / (w.conj() @ w)
        matrix = torch.stack((w, 1j * w, 1e-18 * u), dim=1)
        right = (2 + 3j) * w + (.7 - .2j) * u
        rcond = max(matrix.shape) * torch.finfo(torch.float64).eps
        expected = torch.linalg.lstsq(matrix, right, rcond=rcond, driver="gelsd")
        coefficients, rank = _small_lstsq(matrix, right)
        self.assertEqual(rank, int(expected.rank))
        self.assertEqual(rank, 1)
        image, oracle_image = matrix @ coefficients, matrix @ expected.solution
        self.assert_close(image, oracle_image)
        self.assert_close(right - image, right - oracle_image)
        self.assertGreater(float((right - image).norm()), .1)
        # The complex optimum preserves the nonreal image coefficient; exact
        # y values are intentionally not compared for dependent columns w, iw.
        projected_coefficient = (w.conj() @ image) / (w.conj() @ w)
        self.assert_close(projected_coefficient, torch.tensor(2 + 3j, dtype=torch.complex128))
        self.assertGreater(float(coefficients.imag.norm()), .1)

    def test_unsuccessful_zero_image_breakdown_is_explicit_and_finite(self):
        result, status = complex_augmented_fgmres(self.action, self.rhs,
                                                lambda value: torch.zeros_like(value),
                                                tolerance=1e-12, maxiter=8, restart=8)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(status["failed"], 1)
        self.assertEqual(status["classical_calls"], 1)
        self.assertGreaterEqual(status["breakdowns"], 1)
        self.assertTrue(status["breakdown_reason"])
        self.assert_physical_status(result, status, 1e-12)

    def test_relative_small_h_rank_loss_stops_even_with_nonzero_arnoldi_remainder(self):
        rhs = torch.zeros(1, 1, 1, 3, dtype=torch.complex128)
        rhs[..., 1] = 1
        first, second = torch.zeros_like(rhs), torch.zeros_like(rhs)
        first[..., 0] = 1e20
        second[..., 2] = 1
        calls, captures = [], []

        def stateful(value):
            calls.append(value.clone())
            if len(calls) > 2:
                raise AssertionError("rank-deficient Arnoldi continuation was attempted")
            return first.clone() if len(calls) == 1 else second.clone()

        def forbidden_injection(*args):
            raise AssertionError("injection was attempted after numerical rank loss")

        # The second image has an exact nonzero remainder e3, but the raw
        # image matrix [1e20 e1, e3] has relative numerical rank one. There
        # is no admissible full-rank QR basis or continued ordered frontier.
        with patch.object(torch.linalg, "qr", side_effect=AssertionError("rank-deficient QR attempted")) as qr:
            result, status = complex_augmented_fgmres(
                lambda value: value, rhs, stateful, inject=forbidden_injection,
                inject_after=2, tolerance=1e-12, maxiter=8, restart=8,
                capture_prefix=captures.append,
            )
            qr.assert_not_called()
        self.assertEqual(len(calls), 2)
        self.assertEqual(status["classical_calls"], 2)
        self.assertEqual(status["iterations"], 2)
        self.assertEqual(status["injection_calls"], 0)
        self.assertEqual(status["rank_deficient_steps"], 1)
        self.assertEqual(status["breakdowns"], 1)
        self.assertEqual(status["breakdown_reason"], "rank_deficient_arnoldi")
        self.assertEqual(status["failed"], 1)
        self.assertAlmostEqual(status["true_residual"], 1.)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(len(captures), 1)
        prefix = captures[0]
        self.assertIsNone(prefix["image_basis"])
        self.assertIsNone(prefix["frontier"])
        self.assertEqual(len(prefix["arnoldi_basis"]), 2)
        self.assert_close(prefix["directions"][0], first, atol=0, rtol=0)
        self.assert_close(prefix["images"][1], second, atol=0, rtol=0)
        self.assertGreater(float(prefix["hessenberg"][-1, -1].abs()), .5)

    def test_wrong_injection_shape_raises_physical_field_error(self):
        with self.assertRaises(ValueError):
            complex_augmented_fgmres(
                self.action, self.rhs, lambda value: value,
                inject=lambda solution, residual: residual[..., :3],
                inject_after=4, tolerance=1e-13, maxiter=8, restart=8,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
