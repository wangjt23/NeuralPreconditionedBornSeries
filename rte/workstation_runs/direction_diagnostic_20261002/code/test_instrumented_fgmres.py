"""CPU protocol tests for the diagnostic clone of the archived RTE solver.

Run from any directory with ``python3 /absolute/path/to/this/file.py``.
The examples are small physical complex fields; all Krylov inner products are
real-stacked, as in the original solver. No model, data, or GPU is needed.
"""

import ast
from pathlib import Path
import math
import sys
import types
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

snapshot = ROOT / "remote_reference" / "rte"
if snapshot.is_dir():
    sys.path.insert(0, str(snapshot.parent))
    # The local source snapshot lacks gmres, reached by npbs_unet_ablation.
    # Isolate its exact conversion functions while importing both solvers
    # unchanged. On FWI no shadow is installed: PYTHONPATH=. supplies the real
    # repository and its normally imported conversion functions/dependencies.
    conversion_path = snapshot / "npbs_unet_ablation.py"
    conversion_names = {"complex_to_real_channels", "real_channels_to_complex"}
    conversion_functions = [
        node for node in ast.parse(conversion_path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in conversion_names
    ]
    if {node.name for node in conversion_functions} != conversion_names:
        raise RuntimeError("The archived physical/real-stacked conversion functions are missing")
    conversion_module = types.ModuleType("rte.npbs_unet_ablation")
    conversion_module.__file__ = str(conversion_path)
    conversion_module.__dict__["torch"] = torch
    exec(
        compile(ast.Module(body=conversion_functions, type_ignores=[]), str(conversion_path), "exec"),
        conversion_module.__dict__,
    )
    sys.modules[conversion_module.__name__] = conversion_module

from rte.augmented_fgmres import augmented_fgmres
from instrumented_fgmres import instrumented_fgmres


class InstrumentedFGMRESProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator(device="cpu").manual_seed(81)
        self.matrix = torch.diag(torch.linspace(.1, 3., 24)).to(torch.complex128)
        self.matrix += .05 * torch.randn(
            24, 24, generator=generator, dtype=torch.complex128
        )
        self.rhs = torch.randn(
            1, 1, 1, 24, generator=generator, dtype=torch.complex128
        )
        self.action = lambda x: (self.matrix @ x.flatten()).reshape_as(x)
        self.precondition = lambda x: x
        self.exact = torch.linalg.solve(self.matrix, self.rhs.flatten()).reshape_as(self.rhs)

    def assert_reference_status(self, actual, expected):
        """Check every original status field, including the full residual trace."""
        self.assertTrue(set(expected).issubset(actual))
        for key, value in expected.items():
            with self.subTest(status_key=key):
                if key == "trace":
                    self.assertEqual(len(actual[key]), len(value))
                    torch.testing.assert_close(
                        torch.tensor(actual[key], dtype=torch.float64),
                        torch.tensor(value, dtype=torch.float64),
                        atol=2e-12, rtol=5e-10,
                    )
                elif isinstance(value, float):
                    self.assertTrue(math.isclose(actual[key], value, abs_tol=2e-12, rel_tol=5e-10))
                else:
                    self.assertEqual(actual[key], value)

    def assert_same_solution(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
        self.assertEqual(actual.dtype, self.rhs.dtype)
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.shape, self.rhs.shape)

    def paired_run(self, diagnostic=None, **kwargs):
        reference, reference_status = augmented_fgmres(
            self.action, self.rhs, self.precondition, **kwargs
        )
        actual, status = instrumented_fgmres(
            self.action, self.rhs, self.precondition,
            **kwargs, **(diagnostic or {}),
        )
        self.assert_same_solution(actual, reference)
        self.assert_reference_status(status, reference_status)
        return actual, status

    def test_no_injection_matches_original_across_restarts_and_nonzero_initial(self):
        _, status = self.paired_run(
            initial=.37 * self.exact, tolerance=1e-12, maxiter=18, restart=7
        )
        self.assertEqual(status["iterations"], 18)
        self.assertEqual(status["injection_calls"], 0)
        self.assertEqual(status["injection_stages"], [])

    def test_exact_teacher_injection_is_accepted_and_capture_matches_actual_step(self):
        prefixes, injections = [], []
        result, status = self.paired_run(
            diagnostic={
                "capture_prefix": prefixes.append,
                "capture_injection": injections.append,
            },
            inject=lambda x, residual: self.exact - x,
            inject_after=4, tolerance=1e-10, maxiter=12, restart=12,
        )
        self.assertEqual(status["classical_calls"], 4)
        self.assertEqual(status["iterations"], 5)
        self.assertEqual(status["injection_accepted"], 1)
        self.assertEqual(status["injection_rejected"], 0)
        self.assertEqual(status["failed"], 0)
        self.assert_same_solution(result, self.exact)
        self.assertEqual(len(prefixes), 1)
        self.assertEqual(len(injections), 1)
        self.assertEqual(len(status["injection_stages"]), 1)
        prefix, stage = prefixes[0], injections[0]
        self.assertEqual(stage["reason"], "accepted")
        self.assertTrue(stage["accepted"])
        self.assertEqual(status["injection_stages"][0]["reason"], "accepted")
        torch.testing.assert_close(stage["frontier_before"], prefix["frontier"], atol=2e-12, rtol=5e-10)
        torch.testing.assert_close(
            stage["z"], (self.exact - prefix["solution"]) / prefix["residual"].norm(),
            atol=2e-12, rtol=5e-10,
        )
        torch.testing.assert_close(stage["image"], self.action(stage["z"]), atol=2e-12, rtol=5e-10)
        self.assertLess(float(stage["orthogonal_image"].norm()), 1e-10)
        self.assertAlmostEqual(stage["relative_residual_before"], status["trace"][4], delta=2e-12)
        self.assertAlmostEqual(stage["relative_residual_after"], status["trace"][5], delta=2e-12)

    def assert_rejected_proposal(self, proposal, reason):
        captures = []
        result, status = self.paired_run(
            diagnostic={"capture_injection": captures.append},
            inject=proposal, inject_after=4,
            tolerance=1e-12, maxiter=18, restart=7,
        )
        plain, plain_status = augmented_fgmres(
            self.action, self.rhs, self.precondition,
            tolerance=1e-12, maxiter=18, restart=7,
        )
        self.assert_same_solution(result, plain)
        self.assertEqual(status["classical_calls"], plain_status["classical_calls"])
        self.assertEqual(status["iterations"], plain_status["iterations"])
        self.assertEqual(status["injection_calls"], 1)
        self.assertEqual(status["injection_rejected"], 1)
        self.assertEqual(status["injection_accepted"], 0)
        self.assertEqual(len(captures), 1)
        self.assertEqual(len(status["injection_stages"]), 1)
        stage = captures[0]
        self.assertEqual(stage["reason"], reason)
        self.assertFalse(stage["accepted"])
        self.assertEqual(status["injection_stages"][0]["reason"], reason)
        self.assertAlmostEqual(stage["relative_residual_before"], status["trace"][4], delta=2e-12)
        self.assertAlmostEqual(stage["relative_residual_after"], stage["relative_residual_before"], delta=2e-12)
        self.assertEqual(stage["frontier_before"].shape, self.rhs.shape)
        return stage

    def test_zero_injection_is_rejected_without_displacing_classical_work(self):
        stage = self.assert_rejected_proposal(
            lambda x, residual: torch.zeros_like(residual), "invalid_proposal"
        )
        # The callback has the same schema even when no operator image exists.
        for field in ("z", "image", "orthogonal_image"):
            self.assertIn(field, stage)

    def test_dependent_image_injection_is_rejected_without_restarting(self):
        # With identity M, rhs is proportional to the first search direction;
        # A(rhs) is therefore already in the span of accepted search images.
        stage = self.assert_rejected_proposal(lambda x, residual: self.rhs, "dependent_image")
        torch.testing.assert_close(stage["image"], self.action(stage["z"]), atol=2e-12, rtol=5e-10)
        self.assertLess(float(stage["orthogonal_image"].norm()), 1e-10)

    def test_four_step_prefix_retains_ordered_frontier_and_search_image_basis(self):
        prefixes, reference_prefixes = [], []
        expected, expected_status = augmented_fgmres(
            self.action, self.rhs, self.precondition,
            inject_after=4, tolerance=1e-12, maxiter=4, restart=25,
            capture_prefix=reference_prefixes.append,
        )
        result, status = instrumented_fgmres(
            self.action, self.rhs, self.precondition,
            inject_after=4, tolerance=1e-12, maxiter=4, restart=25,
            capture_prefix=prefixes.append,
        )
        self.assert_same_solution(result, expected)
        self.assert_reference_status(status, expected_status)
        self.assertEqual(len(prefixes), 1)
        prefix, original = prefixes[0], reference_prefixes[0]
        self.assertEqual(prefix["iterations"], 4)
        self.assertEqual(prefix["classical_calls"], 4)
        self.assertAlmostEqual(prefix["relative_residual"], original["relative_residual"], delta=2e-12)
        for field in ("solution", "residual", "image_basis"):
            torch.testing.assert_close(prefix[field], original[field], atol=2e-11, rtol=2e-10)
        vectors, hessenberg = prefix["arnoldi_basis"], prefix["hessenberg"]
        self.assertEqual(vectors.shape, (5,) + self.rhs.shape)
        self.assertEqual(prefix["frontier"].shape, self.rhs.shape)
        self.assertEqual(prefix["image_basis"].shape, (4,) + self.rhs.shape)
        self.assertEqual(hessenberg.shape, (5, 4))
        torch.testing.assert_close(prefix["frontier"], vectors[-1], atol=0, rtol=0)
        torch.testing.assert_close(vectors[0], self.rhs / self.rhs.norm(), atol=2e-12, rtol=5e-10)
        flattened = vectors.flatten(1)
        torch.testing.assert_close(
            (flattened.conj() @ flattened.T).real,
            torch.eye(5, dtype=torch.float64), atol=2e-12, rtol=5e-10,
        )
        # Ordered columns are essential: a generic orthonormal basis would
        # satisfy the Gram test but would not satisfy the Arnoldi recurrence.
        for j in range(4):
            reconstructed = (vectors * hessenberg[:, j].reshape(5, 1, 1, 1, 1)).sum(0)
            torch.testing.assert_close(self.action(vectors[j]), reconstructed, atol=2e-12, rtol=5e-10)
        images = prefix["image_basis"].flatten(1)
        torch.testing.assert_close(
            (images.conj() @ images.T).real,
            torch.eye(4, dtype=torch.float64), atol=2e-12, rtol=5e-10,
        )
        torch.testing.assert_close(
            (images.conj() @ prefix["residual"].flatten()).real,
            torch.zeros(4, dtype=torch.float64), atol=2e-12, rtol=0,
        )
        # Each A(v_j) lies in AZ and has no component outside image_basis.
        for vector in vectors[:4]:
            image = self.action(vector).flatten()
            projection = ((images.conj() @ image).real[:, None] * images).sum(0)
            torch.testing.assert_close(image, projection, atol=2e-12, rtol=5e-10)

    def test_profile_reports_finite_stage_costs_and_preserves_solution_and_status(self):
        kwargs = dict(
            inject=lambda x, residual: self.exact - x,
            inject_after=4, tolerance=1e-10, maxiter=12, restart=12,
        )
        expected, expected_status = augmented_fgmres(self.action, self.rhs, self.precondition, **kwargs)
        plain, plain_status = instrumented_fgmres(self.action, self.rhs, self.precondition, **kwargs)
        calls = {"operator": 0, "preconditioner": 0, "proposal": 0}

        def counted_action(x):
            calls["operator"] += 1
            return self.action(x)

        def counted_preconditioner(x):
            calls["preconditioner"] += 1
            return x

        def counted_proposal(x, residual):
            calls["proposal"] += 1
            return self.exact - x

        result, status = instrumented_fgmres(
            counted_action, self.rhs, counted_preconditioner,
            **{**kwargs, "inject": counted_proposal}, profile=True,
        )
        self.assert_same_solution(result, expected)
        self.assert_same_solution(result, plain)
        self.assert_reference_status(status, expected_status)
        self.assert_reference_status(status, {key: plain_status[key] for key in expected_status})
        profile = status["profile"]
        seconds, counts = profile["stage_seconds"], profile["stage_counts"]
        self.assertTrue(seconds)
        self.assertTrue(counts)
        self.assertEqual(set(seconds), set(counts))
        for stage_name, elapsed in seconds.items():
            with self.subTest(profile_stage=stage_name):
                self.assertTrue(math.isfinite(elapsed))
                self.assertGreaterEqual(elapsed, 0.)
                self.assertIsInstance(counts[stage_name], int)
                self.assertGreaterEqual(counts[stage_name], 0)
                if counts[stage_name] > 0:
                    self.assertGreater(elapsed, 0.)
        self.assertTrue(math.isfinite(profile["profiled_wall_seconds"]))
        self.assertGreater(profile["profiled_wall_seconds"], 0.)
        self.assertIsInstance(profile["note"], str)
        self.assertTrue(profile["note"])
        # The profile must contain actual callback costs, not just summaries.
        self.assertEqual(set(counts), {"precondition", "action", "neural_callback", "projection_LS", "true_checks"})
        self.assertEqual(counts["precondition"], calls["preconditioner"])
        self.assertEqual(counts["neural_callback"], calls["proposal"])
        self.assertEqual(counts["action"] + counts["true_checks"], calls["operator"])
        self.assertEqual(counts["action"], status["classical_calls"] + status["injection_accepted"])
        self.assertEqual(counts["true_checks"], 2 + 1 + status["iterations"])
        self.assertEqual(counts["projection_LS"], 2 * status["iterations"])
        self.assertEqual(calls["preconditioner"], status["classical_calls"])
        self.assertEqual(calls["proposal"], status["injection_calls"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
