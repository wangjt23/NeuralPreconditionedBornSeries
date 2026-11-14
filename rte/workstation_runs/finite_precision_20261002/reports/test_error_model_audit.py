"""Independent scalar adversarial tests; no RTE fields/actions/solves or GPU.

Run: python3 deliverables/finite_precision_20261002/test_error_model_audit.py
These test conditional mathematics and scalar validation, not backend contracts.
"""
from fractions import Fraction as F
import math
import unittest

from error_model import (
    acceptance_upper, conditional_action_bound, prospective_budget, stored_scales,
)


def binary(v):
    return F.from_float(float(v))


def scales():
    # Exact stored binary coefficients: a positive nominal margin and a large
    # differentiation norm. No array/action or solver is constructed.
    return stored_scales([[1., 0.]], [-1000., 1000.], [0.],
                         [1.017], [1.], [[1.]], [1.])


class IndependentErrorModelAudit(unittest.TestCase):
    def test_reserved_budget_closes_adversarial_scalar_bound(self):
        # The maximum residual can align with every action/subtraction error.
        # Independently evaluate its extremal rational expression; passing a
        # phase-commutation diagnostic has no role here.
        for n in (1, 12*128*128, 12*256*256):
            with self.subTest(complex_count=n):
                s = scales(); p = prospective_budget(s, field_complex_count=n)
                m = binary(s['coercivity_lower']); L = binary(s['operator_norm_upper'])
                mu = F(1, 2**53); k = 4*n + 16
                nu = k*mu/(1-k*mu)
                rho = binary(p['hypothetical_measured_residual_ceiling'])*(1+nu)/(1-nu)
                eps = binary(p['allowed_outer_absolute_error_per_field_norm'])
                a = mu; b = eps + mu*(L+eps)
                self.assertLess(b, m)
                worst = a+rho+b*(1+a+rho)/(m-b)
                self.assertLessEqual(worst, F(1, 100000))
                self.assertFalse(p['global_backend_contract_established'])
                self.assertFalse(p['empirical_reference_checks_can_certify_this_contract'])

    def test_budget_gets_stricter_when_norm_contract_grows(self):
        s = scales()
        small = prospective_budget(s, field_complex_count=1)
        large = prospective_budget(s, field_complex_count=12*256*256)
        self.assertLess(large['allowed_outer_absolute_error_per_field_norm'],
                        small['allowed_outer_absolute_error_per_field_norm'])

    def test_no_slack_at_observed_threshold(self):
        # 1D example: nominal residual is the observed threshold plus a
        # positive check uncertainty. Even a small error then prevents proof.
        tau = 1e-5
        r = acceptance_upper(tau, 1., 1., 1., 1., 1e-12, 0.)
        self.assertGreater(r['relative_residual_upper'], tau)

    def test_solution_error_criterion_is_independent(self):
        # Source residual can pass while coercivity and field norm make the
        # relative solution-error target fail; denominator remains positive.
        r = acceptance_upper(9e-6, 1., 1., 1., .001, 0., 0.)
        self.assertLess(r['relative_residual_upper'], 1e-5)
        self.assertGreater(r['relative_solution_error_upper'], 1e-3)

    def test_zero_or_negative_solution_norm_lower_bound_is_not_accepted(self):
        self.assertIsNone(acceptance_upper(.1, 1., 1., 1., .1, 0., 0.)
                          ['relative_solution_error_upper'])
        self.assertIsNone(acceptance_upper(0., 1., 1., 0., 1., 0., 0.)
                          ['relative_solution_error_upper'])

    def test_acceptance_output_encloses_exact_scalar_expression(self):
        vals = (.0000091, 1.17, 1.03, 1.1, .019, 2.3e-12, 1.7e-13)
        r = acceptance_upper(*vals)
        rm, xhi, qlo, xlo, m, eps, sub = map(binary, vals)
        residual = rm + eps*xhi + sub
        delta = residual/m
        self.assertGreaterEqual(binary(r['residual_upper']), residual)
        self.assertGreaterEqual(binary(r['relative_residual_upper']), residual/qlo)
        self.assertGreaterEqual(binary(r['relative_solution_error_upper']),
                                delta/(xlo-delta))

    def test_nonfinite_and_invalid_acceptance_inputs_rejected(self):
        valid = [1e-6, 1., 1., .5, .01, 0., 0.]
        for index, value in ((0, math.nan), (5, math.inf), (6, -1.),
                             (2, 0.), (4, 0.), (3, 2.)):
            with self.subTest(index=index, value=value):
                args = valid.copy(); args[index] = value
                with self.assertRaises((AssertionError, ValueError)):
                    acceptance_upper(*args)

    def test_signed_nonnormal_angular_coefficients_remain_conservative(self):
        # K=[[0,-100],[.01,0]], with exact singular values100 and.01.
        # Absolute row/column estimate must cover the large nonnormal norm.
        s = stored_scales([[0., 0.], [0., 0.]], [0.], [0.], [101.], [1.],
                          [[0., -100.], [.01, 0.]], [1., 1.])
        self.assertGreaterEqual(s['scattering_norm_upper'], 100.)
        self.assertLessEqual(s['coercivity_lower'], 1.)
        self.assertGreater(s['coercivity_lower'], 0.)

    def test_negative_scattering_domain_rejected(self):
        with self.assertRaises((AssertionError, ValueError)):
            stored_scales([[1., 0.]], [0.], [0.], [1.], [-1.], [[1.]], [1.])

    def test_unknown_precision_rejected(self):
        for precision in (0, 16, 128):
            with self.subTest(precision=precision):
                with self.assertRaises((AssertionError, ValueError)):
                    conditional_action_bound(scales(), precision, 0., 0.)

    def test_negative_error_contract_and_bad_dot_count_rejected(self):
        for f, i, k in ((-1e-3, 0., None), (0., -1e-3, None),
                        (0., 0., -1), (0., 0., .5)):
            with self.subTest(forward=f, inverse=i, steps=k):
                with self.assertRaises((AssertionError, ValueError)):
                    conditional_action_bound(scales(), 64, f, i, k)

    def test_bad_budget_domain_rejected(self):
        for kwargs in ({'field_complex_count': 0}, {'field_complex_count': -1},
                       {'field_complex_count': 1.5}, {'tolerance': 0.},
                       {'tolerance': -1e-5}, {'allocation': 0.},
                       {'allocation': 1.}, {'allocation': 1.1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((AssertionError, ValueError)):
                    prospective_budget(scales(), **kwargs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
