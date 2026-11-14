"""Tiny native/exact-Fraction checks for the isolated upper-only specialization."""
from fractions import Fraction as F
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import interval_fft as p
import stagewise_fft as old
import optimized_stagewise as opt
from test_interval_fft import fixture, exact_action


def f(value):
    return F.from_float(float(value))


def upper_squared_exact(box):
    return sum(max(f(a)**2, f(b)**2) for real in (box.re, box.im)
               for a, b in zip(real.lo.reshape(-1), real.hi.reshape(-1)))


class UpperOnlyStagewiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++') or shutil.which('g++')
        if compiler is None:
            raise RuntimeError('Existing system compiler required')
        cls.temporary = tempfile.TemporaryDirectory(prefix='rte-upper-only-tiny-')
        library = Path(cls.temporary.name)/('native.dylib' if sys.platform == 'darwin' else 'native.so')
        dependency = Path(p.__file__).with_name('cpu_interval_primitives.cpp')
        subprocess.run([compiler, '-O2', '-std=c++17', '-fno-fast-math',
                        '-ffp-contract=off', '-shared', '-fPIC', str(dependency),
                        '-o', str(library)], check=True, capture_output=True)
        cls.guard = opt.load_native_cpu_backend(library)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def assertUpperIdenticalAndValid(self, box):
        actual = opt.local_norm_upper(box)
        expected = old._norm(box)[1]
        self.assertEqual(struct.pack('d', actual), struct.pack('d', expected))
        self.assertGreaterEqual(f(actual)**2, upper_squared_exact(box))

    def test_independent_rectangle_corners_and_odd_reduction_trees(self):
        for length in (1, 3, 5, 17):
            lows = np.array([(-3+i)/8 for i in range(length)], dtype=np.float64)
            highs = np.array([(7+i)/8 for i in range(length)], dtype=np.float64)
            box = p.ComplexBox(p.RealBox(lows, highs),
                               p.RealBox(-highs, -lows))
            self.assertUpperIdenticalAndValid(box)

    def test_subnormal_underflow_cancellation_and_cardinal_components(self):
        tiny = np.nextafter(0., 1.)
        for value in (np.array([tiny+0j, -tiny+1j*tiny], np.complex128),
                      np.array([1e-200+1e-200j, -1e-200j], np.complex128),
                      np.array([1.+0j, 0.+1j, -1.-1j], np.complex128),
                      np.array([2.**400+1j*2.**-400], np.complex128)):
            self.assertUpperIdenticalAndValid(p.point(value))

    def test_exact_zero_local_norm_and_no_global_norm_replacement(self):
        routine = p.norm_bounds
        self.assertEqual(opt.local_norm_upper(np.zeros((2, 3, 4), np.complex128)), 0.)
        self.assertIs(p.norm_bounds, routine)
        self.assertUpperIdenticalAndValid(p.point(np.zeros(3, np.complex128)))

    def test_forward_inverse_balls_keep_bitwise_centers_and_radii(self):
        for n in (4, 8, 16):
            x = np.zeros((1, 2, n, n), np.complex128)
            x[0, 0, 1, 2] = .5-.125j
            x[0, 1] = (1+2.**-20)*(.25+.125j)
            for inverse in (False, True):
                expected = old.fft2_ball(old.StagewiseBall(x, F(1, 3)), inverse)
                actual = opt.fft2_ball(opt.StagewiseBall(x, F(1, 3)), inverse)
                self.assertEqual(expected.center.tobytes(), actual.center.tobytes())
                self.assertEqual(expected.radius_upper, actual.radius_upper)
                for a, b in zip(expected.stages, actual.stages):
                    for key in ('incoming_radius_upper', 'local_radius_upper', 'radius_upper'):
                        self.assertEqual(a[key], b[key])

    def test_physical_error_contains_independent_exact_dense_fourpoint_oracle(self):
        args = fixture(); exact = exact_action(*args)
        expected, actual = old.physical_action_ball(*args), opt.physical_action_ball(*args)
        self.assertEqual(expected.center.tobytes(), actual.center.tobytes())
        self.assertEqual(expected.radius_upper, actual.radius_upper)
        error = F(0)
        for a, i, j in np.ndindex((3, 4, 4)):
            re, im = exact[a][i][j]; z = actual.center[0, a, i, j]
            error += (re-f(z.real))**2+(im-f(z.imag))**2
        self.assertLessEqual(error, f(actual.radius_upper)**2)

    def test_certificate_exact_fields_and_gate_decisions_are_unchanged(self):
        args = fixture(); exact = exact_action(*args); q = np.empty_like(args[0])
        for a, i, j in np.ndindex((3, 4, 4)):
            re, im = exact[a][i][j]; q[0, a, i, j] = complex(float(re), float(im))
        expected = old.certify_residual(q, *args, baseline=args[0])
        actual = opt.certify_residual(q, *args, baseline=args[0])
        for key in expected:
            if key in ('costs', 'elapsed_seconds', 'stage_ledger',
                       'native_backend_guard_before', 'native_backend_guard_after',
                       'paired_field_guard_before', 'paired_field_guard_after'):
                continue
            self.assertEqual(expected[key], actual[key], key)
        self.assertTrue(actual['accepted'])
        zero = opt.certify_residual(np.zeros_like(q), np.zeros_like(q),
                                   *args[1:], baseline=np.zeros_like(q))
        self.assertFalse(zero['accepted'])
        self.assertIsNone(zero['relative_solution_error_upper'])

    def test_native_guard_failure_remains_closed(self):
        args = fixture()
        with patch.object(p, '_NATIVE', None):
            with self.assertRaises(RuntimeError):
                opt.certify_residual(args[0], *args)
            with self.assertRaises(RuntimeError):
                opt.paired_field_check(args[0], args[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
