"""Independent tiny exact-Fraction/native-CPU interval fixtures; no GPU/RTE imports."""
import ctypes
from fractions import Fraction as F
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import interval_fft as m


def pair(z):
    return F.from_float(float(z.real)), F.from_float(float(z.imag))


def plus(a, b):
    return a[0]+b[0], a[1]+b[1]


def times(a, b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def dense4(field, inverse=False):
    """Independent dense4x4 DFT using the exact roots1,+/-i,-1."""
    roots = ((F(1), F(0)), (F(0), F(-1)),
             (F(-1), F(0)), (F(0), F(1)))
    out = []
    for plane in field:
        result = []
        for k in range(4):
            row = []
            for l in range(4):
                total = (F(0), F(0))
                for i in range(4):
                    for j in range(4):
                        root = roots[(i*k+j*l) % 4]
                        if inverse:
                            root = (root[0], -root[1])
                        total = plus(total, times(plane[i][j], root))
                row.append((total[0]/16, total[1]/16) if inverse else total)
            result.append(row)
        out.append(result)
    return out


def fixture():
    l = 3
    x = np.array([[[[(a+2*i-j)/8+1j*(2*a-i+3*j)/16
                      for j in range(4)] for i in range(4)] for a in range(l)]],
                 dtype=np.complex128)
    k = np.array([0., 1., -2., -1.])
    dx, dy = np.array([1., .125, -.5]), np.array([0., .5, .25])
    symbol = 1j*(dx[:, None, None]*k[None, :, None]+
                 dy[:, None, None]*k[None, None, :])[None]
    t = np.array([[[[2.+(i+j)/16 for j in range(4)] for i in range(4)]]])
    s = np.full_like(t, .25)
    phase = np.array([[1., .125, -.25], [.5, .75, .0625], [-.125, .25, 1.]])
    weights = np.array([.25, .5, .25])
    return x, symbol, t, s, phase, weights


def exact_action(x, symbol, t, s, phase, weights):
    data = [[[pair(v) for v in row] for row in plane] for plane in x[0]]
    spectral = dense4(data)
    products = [[[times(pair(symbol[0, a, i, j]), spectral[a][i][j])
                  for j in range(4)] for i in range(4)] for a in range(len(data))]
    stream = dense4(products, inverse=True)
    out = []
    for a in range(len(data)):
        plane = []
        for i in range(4):
            row = []
            for j in range(4):
                scatter = (F(0), F(0))
                for b in range(len(data)):
                    coefficient = F.from_float(float(phase[a, b]))*F.from_float(float(weights[b]))
                    scatter = plus(scatter, times((coefficient, F(0)), data[b][i][j]))
                value = plus(stream[a][i][j], times(pair(t[0, 0, i, j]+0j), data[a][i][j]))
                subtraction = times(pair(s[0, 0, i, j]+0j), scatter)
                row.append((value[0]-subtraction[0], value[1]-subtraction[1]))
            plane.append(row)
        out.append(plane)
    return out


def assert_contains(test, box, target):
    for index in np.ndindex(box.shape):
        value = target
        for i in index:
            value = value[i]
        for component, bounds in zip(value, (box.re, box.im)):
            test.assertLessEqual(F.from_float(float(bounds.lo[index])), component)
            test.assertGreaterEqual(F.from_float(float(bounds.hi[index])), component)


class IntervalFFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("Existing system C++ compiler required")
        cls.temp = tempfile.TemporaryDirectory(prefix="rte_interval_cpu_")
        library = Path(cls.temp.name)/("primitives.dylib" if sys.platform == "darwin" else "primitives.so")
        cls.command = [compiler, "-O2", "-std=c++17", "-fno-fast-math",
                       "-ffp-contract=off", "-shared", "-fPIC",
                       str(Path(__file__).with_name("cpu_interval_primitives.cpp")),
                       "-o", str(library)]
        subprocess.run(cls.command, check=True, capture_output=True, text=True)
        cls.guard = m.load_native_cpu_backend(library)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_native_guard_and_no_mode_change(self):
        before = m.native_cpu_guard()
        self.assertEqual(before["guard_flags"], 127)
        self.assertFalse(before["floating_control_modes_modified"])
        x = np.array([1., np.nextafter(0., 1.), 1e-200])
        m.radd(m.real_point(x), m.real_point(x))
        after = m.native_cpu_guard()
        mask = 0xFFC0 if before['architecture'] == 'x86-MXCSR' else ((1 << 24) | (3 << 22))
        self.assertEqual(before['control_register'] & mask, after['control_register'] & mask)

    def test_real_primitives_contain_exact_fraction_corners(self):
        intervals = [(-2., -.5), (-1., 3.), (0., 0.),
                     (np.nextafter(0., 1.), np.nextafter(0., 1.)),
                     (1., np.nextafter(1., 2.))]
        for al, ah in intervals:
            for bl, bh in intervals:
                a = m.RealBox(np.array(al), np.array(ah)); b = m.RealBox(np.array(bl), np.array(bh))
                for operation, exact_op in ((m.radd, lambda x, y: x+y), (m.rmul, lambda x, y: x*y)):
                    out = operation(a, b)
                    for x in (al, ah):
                        for y in (bl, bh):
                            value = exact_op(F.from_float(float(x)), F.from_float(float(y)))
                            self.assertLessEqual(F.from_float(float(out.lo)), value)
                            self.assertGreaterEqual(F.from_float(float(out.hi)), value)

    def test_dense_exact_fourpoint_fft_and_inverse(self):
        x = fixture()[0][0]
        exact_x = [[[pair(z) for z in row] for row in plane] for plane in x]
        before = x.copy()
        for inverse in (False, True):
            box = m.fft2_box(x, inverse=inverse)
            assert_contains(self, box, dense4(exact_x, inverse))
        self.assertTrue(np.array_equal(x, before))

    def test_poweroftwo_roundtrip_spatial_axes_and_output_ownership(self):
        for n in (8, 16):
            x = np.zeros((2, 3, n, n), dtype=np.complex128)
            x[0, 1, 2, 3] = .25-.125j
            x[1, 2] = 1.
            box = m.fft2_box(m.fft2_box(x), inverse=True)
            self.assertTrue(np.all(box.re.lo <= x.real)); self.assertTrue(np.all(box.re.hi >= x.real))
            self.assertTrue(np.all(box.im.lo <= x.imag)); self.assertTrue(np.all(box.im.hi >= x.imag))
            snapshot = box.re.lo.copy(); m.fft2_box(x*2)
            self.assertTrue(np.array_equal(snapshot, box.re.lo)); self.assertFalse(np.shares_memory(x, box.re.lo))

    def test_stored_heterogeneous_action_matches_independent_exact_dense(self):
        args = fixture(); box = m.physical_action_box(*args)
        assert_contains(self, box, [exact_action(*args)])

    def test_rigorous_norm_brackets_and_crossingzero_square(self):
        real = m.RealBox(np.array([-1., 3.]), np.array([2., 3.]))
        zero = m.real_point(np.zeros(2))
        box = m.ComplexBox(real, zero)
        lo, hi = m.norm_bounds(box)
        self.assertLessEqual(F.from_float(lo)**2, F(9))
        self.assertGreaterEqual(F.from_float(hi)**2, F(13))
        tiny = np.array([complex(np.nextafter(0., 1.), 0.)])
        lo, hi = m.norm_bounds(tiny)
        self.assertLessEqual(lo, abs(tiny[0])); self.assertGreaterEqual(hi, abs(tiny[0]))

    def test_exact_solution_fresh_certificate_and_fixed_reserve(self):
        args = fixture(); truth = exact_action(*args)
        q = np.array([[[complex(float(z[0]), float(z[1])) for z in row]
                       for row in plane] for plane in truth], dtype=np.complex128)[None]
        for a, i, j in np.ndindex((3, 4, 4)):
            self.assertEqual(pair(q[0, a, i, j]), truth[a][i][j])
        receipt = m.certify_residual(q, *args)
        self.assertTrue(receipt['accepted']); self.assertTrue(receipt['fixed_one_percent_reserve_accepted'])
        self.assertEqual(receipt['policy_tolerance_rational'], '1/100000')
        self.assertEqual(receipt['reserve_fraction_rational'], '1/100')
        self.assertFalse(receipt['global_gpu_contract_established'])
        margin = F(receipt['coercivity_lower_rational'])
        self.assertEqual(margin, F(receipt['coercivity_lower_numerator'],
                                   receipt['coercivity_lower_denominator']))
        self.assertLessEqual(F.from_float(receipt['coercivity_lower']), margin)
        rupper = F.from_float(receipt['residual_norm_upper'])
        xlower = F.from_float(receipt['candidate_norm_lower'])
        exact_relative_bound = rupper/(margin*xlower-rupper)
        self.assertGreaterEqual(F.from_float(receipt['relative_solution_error_upper']),
                                exact_relative_bound)
        self.assertGreater(receipt['elapsed_seconds'], 0.)
        self.assertGreaterEqual(receipt['residual_norm_upper'], 0.)

    def test_zero_candidate_no_solution_bound_and_zero_source_failclosed(self):
        args = fixture(); q = np.ones_like(args[0])
        receipt = m.certify_residual(q, np.zeros_like(args[0]), *args[1:])
        self.assertFalse(receipt['accepted']); self.assertIsNone(receipt['relative_solution_error_upper'])
        receipt = m.certify_residual(np.zeros_like(q), np.zeros_like(q), *args[1:])
        self.assertFalse(receipt['source_relative_norm_defined']); self.assertFalse(receipt['accepted'])
        self.assertIsNone(receipt['relative_residual_upper'])

    def test_box_radius_is_not_norm_width(self):
        re = m.RealBox(np.array([10., -1.]), np.array([10., 1.]))
        box = m.ComplexBox(re, m.real_point(np.zeros(2)))
        lo, hi = m.norm_bounds(box)
        center = m.point(np.array([10.+0j, 0.+0j]))
        radius = m.norm_bounds(m.csub(box, center))[1]
        self.assertGreater(radius, .99); self.assertLess(hi-lo, .06)

    def test_bad_guard_and_invalid_fields_failclosed(self):
        with patch.object(m, '_NATIVE', None):
            with self.assertRaises(RuntimeError):
                m.certify_residual(np.ones((1, 3, 4, 4), dtype=np.complex128), *fixture())
        for shape in ((3, 5), (3, 1)):
            with self.assertRaises(ValueError): m.fft2_box(np.zeros(shape, dtype=np.complex128))
        with self.assertRaises(FloatingPointError): m.point(np.array([complex(math.nan, 0)]))
        with self.assertRaises(FloatingPointError):
            m.rmul(m.real_point(np.array([np.finfo(np.float64).max])), m.real_point(np.array([2.])))
        args = fixture(); bad = args[1].copy(); bad.real[...] = .01
        with self.assertRaises(ValueError): m.physical_action_box(args[0], bad, *args[2:])
        with self.assertRaises(ValueError): m.certify_residual(np.ones_like(args[0]), *args, margin_lower=100.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
