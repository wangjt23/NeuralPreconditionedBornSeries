"""Deterministic CPU tests of finite-probe complex-linearity admissibility.

These small dense examples use no RTE media, source families, held-out seeds,
GPU work, training, or timing measurements. Passing a numerical cap is an
engineering check; it does not establish exact floating-point linearity.
"""

import json
import math
from pathlib import Path
import sys
import unittest

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
from admissibility import check_action


class AdmissibilityTests(unittest.TestCase):
    alpha = .73 + .29j
    beta = -.41 + .37j
    tolerance = 1e-12
    summary_keys = {
        "passed", "finite", "exact_zero", "max_normalized_defect",
        "i_homogeneity", "phase_homogeneity", "real_scaling",
        "complex_combination", "checks",
    }

    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator(device="cpu").manual_seed(319)
        self.matrix = torch.randn(8, 8, generator=generator, dtype=torch.complex128)
        self.matrix += torch.diag(torch.linspace(.7, 2.1, 8, dtype=torch.float64))
        self.x = torch.randn(1, 1, 2, 4, generator=generator, dtype=torch.complex128)
        self.y = torch.randn(1, 1, 2, 4, generator=generator, dtype=torch.complex128)
        self.action = lambda value: (self.matrix @ value.flatten()).reshape_as(value)

    def check(self, action, x=None, y=None, tolerance=None, expected_output_dtype=None):
        options = ({} if expected_output_dtype is None
                   else {"expected_output_dtype": expected_output_dtype})
        metadata = check_action(
            action, self.x if x is None else x, self.y if y is None else y,
            alpha=self.alpha, beta=self.beta,
            tolerance=self.tolerance if tolerance is None else tolerance,
            **options,
        )
        self.assertIsInstance(metadata, dict)
        self.assertTrue(self.summary_keys.issubset(metadata))
        self.assertIsInstance(metadata["checks"], dict)
        self.assert_serializable_scalars(metadata)
        # Python's JSON encoding also catches unsupported leaf types.
        json.dumps(metadata)
        return metadata

    def assert_serializable_scalars(self, value):
        if isinstance(value, dict):
            for key, child in value.items():
                self.assertIsInstance(key, str)
                self.assert_serializable_scalars(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                self.assert_serializable_scalars(child)
        else:
            self.assertIsInstance(value, (str, int, float, bool, type(None)))
            self.assertNotIsInstance(value, torch.Tensor)

    @staticmethod
    def norm(value):
        return float(torch.linalg.vector_norm(value.to(torch.complex128)))

    def independent_ratios(self, action, x=None, y=None):
        x, y = self.x if x is None else x, self.y if y is None else y
        ax, ay = action(x).clone(), action(y).clone()
        expected = {}
        for name, scalar in (
            ("i_homogeneity", 1j),
            ("phase_homogeneity", complex(math.cos(.37), math.sin(.37))),
            ("real_scaling", -1.7),
        ):
            values = []
            for probe, image in ((x, ax), (y, ay)):
                lhs, rhs = action(scalar * probe).clone(), scalar * image
                denominator = abs(scalar) * self.norm(image)
                self.assertGreater(denominator, 0.)
                values.append(self.norm(lhs - rhs) / denominator)
            expected[name] = max(values)
        lhs = action(self.alpha * x + self.beta * y).clone()
        rhs = self.alpha * ax + self.beta * ay
        denominator = abs(self.alpha) * self.norm(ax) + abs(self.beta) * self.norm(ay)
        self.assertGreater(denominator, 0.)
        expected["complex_combination"] = self.norm(lhs - rhs) / denominator
        return expected

    def assert_summary_matches_independent_ratios(self, metadata, expected):
        for key, ratio in expected.items():
            with self.subTest(identity=key):
                self.assertAlmostEqual(float(metadata[key]), ratio, delta=2e-13)
        self.assertAlmostEqual(float(metadata["max_normalized_defect"]),
                               max(expected.values()), delta=2e-13)

    def test_detailed_records_use_reference_and_symmetric_denominators(self):
        action = lambda value: value.abs().square().to(torch.complex128)
        metadata = self.check(action)
        ax, ay = action(self.x), action(self.y)
        records = {}
        for name, scalar in (("i", 1j),
                             ("phase", complex(math.cos(.37), math.sin(.37))),
                             ("real", -1.7)):
            for probe_name, probe, image in (("x", self.x, ax), ("y", self.y, ay)):
                lhs, predicted = action(scalar * probe), scalar * image
                records[name + "_" + probe_name] = (
                    lhs, predicted, abs(scalar) * self.norm(image))
        records["complex_combination"] = (
            action(self.alpha * self.x + self.beta * self.y),
            self.alpha * ax + self.beta * ay,
            abs(self.alpha) * self.norm(ax) + abs(self.beta) * self.norm(ay),
        )
        self.assertEqual(set(metadata["checks"]), set(records))
        required = {"numerator", "denominator", "ratio", "symmetric_denominator",
                    "symmetric_ratio", "eligible", "passed", "lhs_norm", "predicted_norm"}
        for name, (lhs, predicted, denominator) in records.items():
            with self.subTest(identity=name):
                details = metadata["checks"][name]
                self.assertTrue(required.issubset(details))
                numerator, lhs_norm = self.norm(lhs - predicted), self.norm(lhs)
                symmetric_denominator = denominator + lhs_norm
                expected = {
                    "numerator": numerator, "denominator": denominator,
                    "ratio": numerator / denominator,
                    "symmetric_denominator": symmetric_denominator,
                    "symmetric_ratio": numerator / symmetric_denominator,
                    "lhs_norm": lhs_norm, "predicted_norm": self.norm(predicted),
                }
                for key, value in expected.items():
                    self.assertAlmostEqual(float(details[key]), value,
                                           delta=5e-13 * max(1., abs(value)))
                self.assertTrue(details["eligible"])
                self.assertEqual(details["passed"], expected["ratio"] <= self.tolerance)

    def test_complex_linear_map_passes_with_scalar_only_metadata(self):
        metadata = self.check(self.action)
        self.assertTrue(metadata["passed"])
        self.assertTrue(metadata["finite"])
        self.assertTrue(metadata["exact_zero"])
        self.assertLess(float(metadata["max_normalized_defect"]), self.tolerance)
        self.assert_summary_matches_independent_ratios(
            metadata, self.independent_ratios(self.action))
        # The action may return its argument; owned output copies must still
        # make the independently evaluated identity map admissible.
        self.assertTrue(self.check(lambda value: value)["passed"])

    def test_native_complex64_linear_action_uses_its_declared_loose_cap(self):
        low_matrix = self.matrix.to(torch.complex64)
        action = lambda value: (low_matrix @ value.flatten()).reshape_as(value)
        metadata = self.check(action, self.x.to(torch.complex64), self.y.to(torch.complex64),
                              tolerance=1e-5)
        self.assertTrue(metadata["passed"])
        self.assertTrue(metadata["finite"])
        self.assertTrue(metadata["exact_zero"])
        self.assertLess(float(metadata["max_normalized_defect"]), 1e-5)

    def test_wrong_complex_output_dtype_fails_even_with_loose_numerical_cap(self):
        metadata = self.check(lambda value: value.to(torch.complex64), tolerance=1e-4)
        self.assertFalse(metadata["passed"])
        self.assertFalse(metadata["output_contract_passed"])
        self.assertTrue(metadata["finite"])
        self.assertTrue(metadata["exact_zero"])
        self.assertLess(float(metadata["max_normalized_defect"]), 1e-4)

    def test_dtype_contract_checks_every_call_including_zero_and_combination(self):
        for wrong_index in range(10):
            with self.subTest(wrong_call=wrong_index):
                calls = []

                def inconsistent(value):
                    calls.append(value.dtype)
                    return (value.to(torch.complex64) if len(calls) - 1 == wrong_index
                            else value.clone())

                metadata = self.check(inconsistent, tolerance=1e-4)
                self.assertFalse(metadata["passed"])
                self.assertFalse(metadata["output_contract_passed"])
                self.assertFalse(metadata["output_dtype_consistent"])
                self.assertEqual(metadata["expected_output_dtype"], str(torch.complex128))
                self.assertEqual(metadata["action_calls"], 10)
                self.assertEqual(len(calls), 10)

    def test_explicit_complex64_to128_conversion_requires_declared_output_dtype(self):
        x, y = self.x.to(torch.complex64), self.y.to(torch.complex64)
        action = lambda value: value.to(torch.complex128)
        undeclared = self.check(action, x, y, tolerance=1e-5)
        self.assertFalse(undeclared["passed"])
        self.assertFalse(undeclared["output_contract_passed"])
        declared = self.check(action, x, y, tolerance=1e-5,
                              expected_output_dtype=torch.complex128)
        self.assertTrue(declared["passed"])
        self.assertTrue(declared["output_contract_passed"])
        self.assertTrue(declared["output_dtype_consistent"])
        self.assertEqual(declared["expected_output_dtype"], str(torch.complex128))
        self.assertTrue(declared["finite"])
        self.assertTrue(declared["exact_zero"])

    def test_conjugation_and_real_extraction_fail_complex_homogeneity(self):
        for action in (
            lambda value: value.conj(),
            lambda value: value.real.to(torch.complex128),
        ):
            with self.subTest(action=action):
                metadata = self.check(action)
                self.assertFalse(metadata["passed"])
                self.assertTrue(metadata["finite"])
                self.assertTrue(metadata["exact_zero"])
                self.assertGreater(float(metadata["i_homogeneity"]), .1)
                self.assert_summary_matches_independent_ratios(
                    metadata, self.independent_ratios(action))

    def test_nonlinear_magnitude_maps_fail_even_when_phase_equivariant(self):
        for action in (
            lambda value: value.abs().square().to(torch.complex128),
            lambda value: value * value.abs().square(),
        ):
            with self.subTest(action=action):
                metadata = self.check(action)
                self.assertFalse(metadata["passed"])
                self.assertTrue(metadata["finite"])
                self.assertTrue(metadata["exact_zero"])
                self.assertGreater(float(metadata["real_scaling"]), .1)
                self.assert_summary_matches_independent_ratios(
                    metadata, self.independent_ratios(action))

    def test_cast32_then_promote128_passes_loose_cap_and_fails_exact_cap(self):
        low_matrix = self.matrix.to(torch.complex64)

        def rounded(value):
            return (low_matrix @ value.flatten().to(torch.complex64)).reshape_as(value).to(torch.complex128)

        loose = self.check(rounded, tolerance=1e-5)
        strict = self.check(rounded, tolerance=1e-14)
        self.assertTrue(loose["passed"])
        self.assertTrue(loose["finite"])
        self.assertTrue(loose["exact_zero"])
        self.assertGreater(float(loose["max_normalized_defect"]), 1e-14)
        self.assertFalse(strict["passed"])
        self.assert_summary_matches_independent_ratios(loose, self.independent_ratios(rounded))

    def test_affine_bias_and_nonfinite_actions_fail(self):
        biased = self.check(lambda value: self.action(value) + (.03 + .02j))
        self.assertFalse(biased["passed"])
        self.assertFalse(biased["exact_zero"])
        self.assertTrue(biased["finite"])
        tiny_bias = self.check(lambda value: self.action(value) + (1e-30 + 1e-30j))
        self.assertFalse(tiny_bias["passed"])
        self.assertFalse(tiny_bias["exact_zero"])
        self.assertLess(float(tiny_bias["max_normalized_defect"]), self.tolerance)
        for invalid in (float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                metadata = self.check(lambda value: torch.full_like(value, invalid))
                self.assertFalse(metadata["passed"])
                self.assertFalse(metadata["finite"])

    def test_mutating_action_fails_despite_output_ownership(self):
        def mutation(value):
            # A mathematically complex-linear in-place operation must still
            # fail the separate input-ownership/mutation prerequisite.
            value.mul_(2. + .3j)
            return value

        metadata = self.check(mutation)
        self.assertFalse(metadata["passed"])

    def test_shared_output_storage_does_not_falsify_linear_or_nonlinear_checks(self):
        buffer = torch.empty_like(self.x)

        def shared_linear(value):
            buffer.copy_(self.action(value))
            return buffer

        def shared_nonlinear(value):
            buffer.copy_(value.abs().square())
            return buffer

        self.assertTrue(self.check(shared_linear)["passed"])
        metadata = self.check(shared_nonlinear)
        self.assertFalse(metadata["passed"])
        self.assertGreater(float(metadata["max_normalized_defect"]), .1)

    def test_zero_action_and_nonzero_null_probe_are_ineligible_without_clamping(self):
        metadata = self.check(lambda value: torch.zeros_like(value))
        self.assertFalse(metadata["passed"])
        self.assertTrue(metadata["finite"])
        self.assertTrue(metadata["exact_zero"])
        for details in metadata["checks"].values():
            self.assertFalse(details["eligible"])
            self.assertFalse(details["passed"])
            self.assertIsNone(details["ratio"])

        x, y = torch.zeros_like(self.x), torch.zeros_like(self.y)
        x.flatten()[0], y.flatten()[1] = 1 + .2j, 1 - .3j

        def projection(value):
            result = torch.zeros_like(value)
            result.flatten()[1] = value.flatten()[1]
            return result

        metadata = self.check(projection, x=x, y=y)
        self.assertFalse(metadata["passed"])
        self.assertTrue(metadata["finite"])
        self.assertTrue(metadata["exact_zero"])

    def test_non_cancelling_reference_denominator_is_used_for_combination(self):
        # The combined input nearly vanishes, while both reference images are
        # nonzero. Normalizing by the tiny combined image would be misleading.
        y = -(self.alpha / self.beta) * self.x
        metadata = self.check(self.action, y=y)
        self.assertTrue(metadata["passed"])
        self.assert_summary_matches_independent_ratios(
            metadata, self.independent_ratios(self.action, y=y))

    def test_second_probe_is_checked_for_each_homogeneity_identity(self):
        # A probe-dependent branch behaves linearly on x and zero, including
        # the declared scalings of x; y activates the nonlinear branch.
        x = torch.ones_like(self.x)
        y = torch.full_like(self.y, 1 + 2j)

        def conditional(value):
            return (value.abs().square().to(value.dtype)
                    if self.norm(value) > 5. else value.clone())

        metadata = self.check(conditional, x=x, y=y)
        self.assertFalse(metadata["passed"])
        expected = self.independent_ratios(conditional, x=x, y=y)
        self.assertGreater(expected["i_homogeneity"], 1.)
        self.assertGreater(expected["phase_homogeneity"], .1)
        self.assertGreater(expected["real_scaling"], 1.)
        self.assert_summary_matches_independent_ratios(metadata, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
