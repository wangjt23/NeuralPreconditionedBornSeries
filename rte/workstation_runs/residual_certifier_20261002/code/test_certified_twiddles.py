"""Exact algebra and endpoint checks; libm comparisons are secondary only."""

from fractions import Fraction as F
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certified_twiddles as tw


class CertifiedTwiddleTests(unittest.TestCase):
    def test_machin_pi_encloses_independent_fifty_digit_interval(self):
        lo, hi = tw.pi_bounds()
        known_lo = F("3.14159265358979323846264338327950288419716939937510")
        known_hi = known_lo + F(1, 10**50)
        self.assertLessEqual(lo, known_lo)
        self.assertGreaterEqual(hi, known_hi)
        self.assertLess(hi-lo, F(1, 10**40))
        self.assertTrue(F(3) < lo < hi < F(4))
        # Independent rational tangent identity establishes the Machin angle,
        # with 0 < 4 atan(1/5)-atan(1/239) < pi/2 selecting the pi/4 branch.
        a = F(1, 5)
        two = 2*a/(1-a*a)
        four = 2*two/(1-two*two)
        b = F(1, 239)
        self.assertEqual((four-b)/(1+four*b), F(1))

    def test_exact_axis_roots_for_two_and_four(self):
        z, one = F(0), F(1)
        self.assertEqual(tw.twiddle_fraction_bounds(2), ((one,), (one,), (z,), (z,)))
        self.assertEqual(tw.twiddle_fraction_bounds(4),
                         ((one,z), (one,z), (z,-one), (z,-one)))
        self.assertEqual(tw.twiddle_fraction_bounds(4, inverse=True),
                         ((one,z), (one,z), (z,one), (z,one)))
        for n in (8, 128, 256):
            rr = tw.twiddle_bounds(n)
            self.assertEqual(tuple(float(v[0]) for v in rr), (1.,1.,0.,0.))
            self.assertEqual(tuple(float(v[n//4]) for v in rr), (0.,0.,-1.,-1.))

    def test_eighth_roots_enclose_exact_square_root_identity(self):
        rl, rh, il, ih = tw.twiddle_fraction_bounds(8)
        half = F(1, 2)
        for k, real_sign in ((1, 1), (3, -1)):
            rlo,rhi = (rl[k],rh[k]) if real_sign>0 else (-rh[k],-rl[k])
            ilo,ihi = -ih[k],-il[k]
            self.assertTrue(F(0)<rlo<=rhi<F(1))
            self.assertLessEqual(rlo*rlo, half)
            self.assertGreaterEqual(rhi*rhi, half)
            self.assertLessEqual(ilo*ilo, half)
            self.assertGreaterEqual(ihi*ihi, half)

    def test_sixteenth_root_exact_polynomial_and_correct_branch(self):
        lo, hi, _, _ = tw.twiddle_fraction_bounds(16)
        # cos(pi/8) is the unique root above 3/4 of 8*x^4-8*x^2+1;
        # this polynomial is increasing on that branch.
        polynomial = lambda x: 8*x**4 - 8*x*x + 1
        self.assertGreater(lo[1], F(3,4))
        self.assertLessEqual(polynomial(lo[1]), 0)
        self.assertGreaterEqual(polynomial(hi[1]), 0)

    def test_rational_widths_and_outward_float_enclosure_all_radix_stages(self):
        for n in (2,4,8,16,32,64,128,256):
            for inverse in (False,True):
                with self.subTest(n=n,inverse=inverse):
                    exact = tw.twiddle_fraction_bounds(n,inverse)
                    rounded = tw.twiddle_bounds(n,inverse)
                    for component in range(4):
                        self.assertEqual(rounded[component].shape, (n//2,))
                        self.assertEqual(rounded[component].dtype, np.dtype(np.float64))
                        self.assertTrue(np.isfinite(rounded[component]).all())
                        for rational, value in zip(exact[component], rounded[component]):
                            actual = F(*float(value).as_integer_ratio())
                            if component in (1,3):
                                self.assertGreaterEqual(actual, rational)
                            else:
                                self.assertLessEqual(actual, rational)
                    for k in range(n//2):
                        self.assertTrue(0 <= exact[1][k]-exact[0][k] < F(1,10**30))
                        self.assertTrue(0 <= exact[3][k]-exact[2][k] < F(1,10**30))
                        self.assertLessEqual(rounded[0][k],rounded[1][k])
                        self.assertLessEqual(rounded[2][k],rounded[3][k])

    def test_conjugate_inverse_symmetry_exact_and_binary64(self):
        for n in (2,4,8,128,256):
            with self.subTest(n=n):
                forward = tw.twiddle_fraction_bounds(n)
                inverse = tw.twiddle_fraction_bounds(n,True)
                self.assertEqual(inverse[:2],forward[:2])
                self.assertEqual(inverse[2],tuple(-x for x in forward[3]))
                self.assertEqual(inverse[3],tuple(-x for x in forward[2]))
                f,i = tw.twiddle_bounds(n),tw.twiddle_bounds(n,True)
                np.testing.assert_array_equal(i[0],f[0])
                np.testing.assert_array_equal(i[1],f[1])
                np.testing.assert_array_equal(i[2],-f[3])
                np.testing.assert_array_equal(i[3],-f[2])

    def test_unit_circle_covered_by_coordinate_intervals(self):
        exact = tw.twiddle_fraction_bounds(256)
        def magnitude_bounds(lo,hi):
            return (F(0) if lo<=0<=hi else min(abs(lo),abs(hi)),
                    max(abs(lo),abs(hi)))
        for k in range(128):
            rlo,rhi = magnitude_bounds(exact[0][k],exact[1][k])
            ilo,ihi = magnitude_bounds(exact[2][k],exact[3][k])
            self.assertLessEqual(rlo*rlo+ilo*ilo,F(1))
            self.assertGreaterEqual(rhi*rhi+ihi*ihi,F(1))

    def test_cache_is_immutable_even_if_writeable_flag_is_requested(self):
        first,second = tw.twiddle_bounds(128),tw.twiddle_bounds(128)
        self.assertIs(first,second)
        for array in first:
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array[0]=7
            with self.assertRaises(ValueError):
                array.flags.writeable=True

    def test_no_libm_or_numpy_trigonometry_in_fresh_construction(self):
        tw.twiddle_bounds.cache_clear()
        tw.twiddle_fraction_bounds.cache_clear()
        tw.pi_bounds.cache_clear()
        tw._point_sin_cos_bounds.cache_clear()
        blocked = AssertionError("trigonometric backend must not be used")
        with patch.object(np,"sin",side_effect=blocked), \
             patch.object(np,"cos",side_effect=blocked), \
             patch.object(np,"tan",side_effect=blocked), \
             patch.object(np,"arctan",side_effect=blocked), \
             patch.object(np,"exp",side_effect=blocked), \
             patch.object(math,"sin",side_effect=blocked), \
             patch.object(math,"cos",side_effect=blocked), \
             patch.object(math,"atan",side_effect=blocked):
            bounds = tw.twiddle_bounds(8)
        self.assertEqual(bounds[0].shape,(4,))

    def test_secondary_libm_smoke_check_not_a_certificate(self):
        # sqrt identities and rational endpoint comparisons above supply the
        # independent checks. A rounded libm evaluation at a rounded pi angle
        # need not lie in a tight exact-angle interval, especially at axis roots.
        for n in (8,128,256):
            rlo,rhi,ilo,ihi = tw.twiddle_bounds(n)
            for k in range(n//2):
                angle = 2*math.pi*k/n
                empirical = (math.cos(angle),-math.sin(angle))
                allowance = 8*math.ulp(1.0)
                self.assertLessEqual(rlo[k]-allowance,empirical[0])
                self.assertGreaterEqual(rhi[k]+allowance,empirical[0])
                self.assertLessEqual(ilo[k]-allowance,empirical[1])
                self.assertGreaterEqual(ihi[k]+allowance,empirical[1])

    def test_invalid_sizes_and_inverse_types_do_not_alias_valid_cache(self):
        tw.twiddle_bounds(8,True)
        for n in (0,1,3,6,-8,True,8.0,np.int64(8)):
            with self.subTest(n=n),self.assertRaises(ValueError):
                tw.twiddle_bounds(n)
        for inverse in (0,1,"forward",None,np.bool_(True)):
            with self.subTest(inverse=inverse),self.assertRaises(TypeError):
                tw.twiddle_bounds(8,inverse)
        with self.assertRaises(ValueError):
            tw.twiddle_fraction_bounds(7)

    def test_proof_metadata_reports_constant_only_scope(self):
        metadata = tw.twiddle_certificate(256)
        self.assertEqual(metadata['roots'],128)
        self.assertTrue(metadata['rational_width_less_than_1e_minus_30'])
        self.assertLess(F(metadata['rational_width_max']),F(1,10**30))
        self.assertLess(F(metadata['pi_width']),F(1,10**40))
        self.assertIn("constants only",metadata['certifies'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
