"""Independent adversarial CPU-only checks; no RTE imports or field data.

These finite witnesses supplement the source proof. They do not prove a global
CPU/GPU backend error bound or certify any saved research candidate.
"""
from fractions import Fraction as F
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import interval_fft as m
from certified_twiddles import twiddle_bounds


def frac(x):
    return F.from_float(float(x))


class IndependentIntervalAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++') or shutil.which('g++')
        if compiler is None:
            raise RuntimeError('Required existing C++ compiler unavailable')
        cls.tmp = tempfile.TemporaryDirectory(prefix='rte_interval_audit_')
        suffix = '.dylib' if sys.platform == 'darwin' else '.so'
        lib = Path(cls.tmp.name)/('audit_primitives'+suffix)
        cls.command = [compiler, '-O2', '-std=c++17', '-fno-fast-math',
                       '-ffp-contract=off', '-shared', '-fPIC',
                       str(Path(__file__).with_name('cpu_interval_primitives.cpp')),
                       '-o', str(lib)]
        subprocess.run(cls.command, check=True, capture_output=True, text=True)
        cls.guard = m.load_native_cpu_backend(lib)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_native_ties_gradual_underflow_and_signed_cancellation(self):
        # Exact results, including underflow halfway between zero/min-subnormal.
        tiny = float(np.nextafter(0., 1.))
        pairs = [(1., 2.**-53), (-1., -2.**-53),
                 (tiny, .5), (-tiny, .5), (2.**-1022, .5)]
        for a, b in pairs:
            for operation, exact in [(m.radd, frac(a)+frac(b)),
                                     (m.rmul, frac(a)*frac(b))]:
                got = operation(m.real_point(np.array(a)), m.real_point(np.array(b)))
                self.assertLessEqual(frac(got.lo), exact)
                self.assertGreaterEqual(frac(got.hi), exact)
        got = m.rsub(m.real_point(np.array(-0.)), m.real_point(np.array(0.)))
        self.assertLessEqual(frac(got.lo), 0)
        self.assertGreaterEqual(frac(got.hi), 0)
        self.assertEqual(m.native_cpu_guard()['guard_flags'], 127)

    def test_odd_length_extreme_scale_norm_exact_squared_bracket(self):
        # An odd reduction length, tiny squares, and widely separated magnitudes.
        x = np.array([2.**100+.0j, 2.**-200+1j*2.**-250, -.75+.125j])
        exact_squared = sum(frac(z.real)**2+frac(z.imag)**2 for z in x)
        lo, hi = m.norm_bounds(x)
        self.assertLessEqual(frac(lo)**2, exact_squared)
        self.assertGreaterEqual(frac(hi)**2, exact_squared)
        with self.assertRaises(FloatingPointError):
            m.norm_bounds(np.array([complex(np.finfo(np.float64).max, 0.)]))

    def test_two_point_fft_nonfinal_axis_and_noncontiguous_input(self):
        original = np.array([[1.+.25j, -.5+.125j], [.75-.5j, .25+.5j],
                             [-.125+.125j, .375-.25j]], dtype=np.complex128)
        x = original.T  # axis zero has length two, noncontiguous storage.
        saved = original.copy()
        for inverse in [False, True]:
            got = m.fft_box(x, axis=0, inverse=inverse)
            for j in range(3):
                for k, sign in [(0, 1), (1, -1)]:
                    for component, bounds in [('real', got.re), ('imag', got.im)]:
                        a = frac(getattr(x[0, j], component))
                        b = frac(getattr(x[1, j], component))
                        exact = (a+sign*b)/(2 if inverse else 1)
                        self.assertLessEqual(frac(bounds.lo[k, j]), exact)
                        self.assertGreaterEqual(frac(bounds.hi[k, j]), exact)
            self.assertFalse(np.shares_memory(got.re.lo, x))
        self.assertTrue(np.array_equal(original, saved))

    def test_eighth_roots_enclose_algebraic_sqrt_half(self):
        real_lo, real_hi, imag_lo, imag_hi = twiddle_bounds(8)
        self.assertLessEqual(frac(real_lo[1])**2, F(1, 2))
        self.assertGreaterEqual(frac(real_hi[1])**2, F(1, 2))
        self.assertLessEqual(frac(-imag_hi[1])**2, F(1, 2))
        self.assertGreaterEqual(frac(-imag_lo[1])**2, F(1, 2))
        inverse = twiddle_bounds(8, inverse=True)
        self.assertTrue(np.array_equal(inverse[0], real_lo))
        self.assertTrue(np.array_equal(inverse[1], real_hi))
        self.assertTrue(np.array_equal(inverse[2], -imag_hi))
        self.assertTrue(np.array_equal(inverse[3], -imag_lo))

    def test_rectangular_constant_field_signed_nonnormal_collision(self):
        # Independent DC-mode oracle on 2x4: streaming vanishes exactly.
        x = np.empty((1, 2, 2, 4), dtype=np.complex128)
        x[:, 0] = .5+.25j; x[:, 1] = -.25+.125j
        d = np.zeros_like(x)
        d[0, 0, :, 1:] = 1j*np.array([1., -2., -.5])
        d[0, 1, :, 1:] = 1j*np.array([-.25, .5, -1.])
        t = np.full((1, 1, 2, 4), 2., dtype=np.float64)
        s = np.full_like(t, .25)
        phase = np.array([[1., -.5], [.25, .75]])
        weights = np.array([.5, .25])
        q = np.empty_like(x)
        for angle in range(2):
            for component in ['real', 'imag']:
                value = 2*frac(getattr(x[0, angle, 0, 0], component))
                value -= F(1, 4)*sum(frac(phase[angle, j])*frac(weights[j])*
                                     frac(getattr(x[0, j, 0, 0], component)) for j in range(2))
                target = float(value)
                self.assertEqual(frac(target), value)
                getattr(q, component)[:, angle] = target
        receipt = m.certify_residual(q, x, d, t, s, phase, weights)
        self.assertTrue(receipt['accepted'])
        self.assertFalse(receipt['global_gpu_contract_established'])
        self.assertGreater(receipt['coercivity_lower'], 0.)
        self.assertEqual(receipt['policy_tolerance_rational'], '1/100000')
        self.assertEqual(receipt['reserve_fraction_rational'], '1/100')

    def test_reserve_rejects_large_vector_radius_despite_small_norm_width(self):
        # Policy-isolation mock, not a claimed physical-action certificate.
        # ||c||=.99tau, E=10^-6, while norm width <10^-7 and ||box||<tau.
        shape = (1, 1, 2, 2)
        q = np.zeros(shape, dtype=np.complex128); q.flat[0] = 1.
        rlo = np.zeros(shape); rhi = np.zeros(shape)
        rlo.flat[0] = rhi.flat[0] = 9.9e-6
        ilo = np.zeros(shape); ihi = np.zeros(shape)
        ilo.flat[0] = -1e-6; ihi.flat[0] = 1e-6
        residual = m.ComplexBox(m.RealBox(rlo, rhi), m.RealBox(ilo, ihi))
        rlower, rupper = m.norm_bounds(residual)
        self.assertLess(rupper-rlower, 1e-7)
        self.assertLess(rupper, 1e-5)
        fake_action = m.csub(m.point(q), residual)
        with patch.object(m, 'physical_action_box', return_value=fake_action):
            receipt = m.certify_residual(q, q, np.zeros_like(q),
                np.ones((1, 1, 2, 2)), np.zeros((1, 1, 2, 2)),
                np.ones((1, 1)), np.ones(1))
        self.assertTrue(receipt['residual_accepted'])
        self.assertFalse(receipt['reserve_radius_accepted'])
        self.assertFalse(receipt['fixed_one_percent_reserve_accepted'])
        self.assertFalse(receipt['accepted'])
        self.assertGreater(receipt['relative_radius_norm_upper'], 1e-7)

    def test_target_and_coercivity_checks_fail_closed(self):
        shape = (1, 1, 2, 2)
        x = np.ones(shape, dtype=np.complex128)
        t = np.ones((1, 1, 2, 2)); s = np.zeros_like(t)
        d = np.zeros_like(x); phase = np.ones((1, 1)); weights = np.ones(1)
        with self.assertRaises(ValueError):
            m.certify_residual(x, x, d, t, s, phase, weights, margin_lower=F(1001, 1000))
        with self.assertRaises(ValueError):
            m.physical_action_box(x, d+1., t, s, phase, weights)
        with self.assertRaises(ValueError):
            m.coefficient_margin(t, t*2., phase, weights)
        with self.assertRaises(TypeError):
            m.real_point(np.array([1], dtype=np.int64))


if __name__ == '__main__':
    unittest.main(verbosity=2)
