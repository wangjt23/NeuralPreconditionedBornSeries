"""Independent, CPU-only standard-library checks of the tiny exact fixture."""

import copy
from fractions import Fraction as F
import math
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_rational_rte import apply_rte, dft2, exact_complex, norm_bounds, norm_squared


Z = (F(0), F(0))


def plus(a, b):
    return (a[0] + b[0], a[1] + b[1])


def times(a, b):
    return (a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0])


def neg(a):
    return (-a[0], -a[1])


def scale_field(field, factor):
    factor = exact_complex(factor)
    return [[[times(exact_complex(z), factor) for z in row] for row in plane]
            for plane in field]


def add_fields(a, b):
    return [[[plus(exact_complex(a[k][x][y]), exact_complex(b[k][x][y]))
              for y in range(4)] for x in range(4)] for k in range(len(a))]


def grid(value):
    return [[value for _ in range(4)] for _ in range(4)]


def field(value=0, angles=1):
    return [grid(value) for _ in range(angles)]


def dense_field(angles=2):
    return [[[(F((a+1)*(x+2) - 3*y, 8), F(2*x + (a+2)*y - 5, 16))
              for y in range(4)] for x in range(4)] for a in range(angles)]


def axis_dft(a):
    """Independent length-four butterfly formula, rather than a root sum."""
    a, b, c, d = [exact_complex(z) for z in a]
    ac = plus(a, neg(c))
    bd = plus(b, neg(d))
    return [plus(plus(a, b), plus(c, d)),
            plus(ac, times((F(0), F(-1)), bd)),
            plus(plus(a, neg(b)), plus(c, neg(d))),
            plus(ac, times((F(0), F(1)), bd))]


def separable_dft(values, inverse=False):
    out = []
    for plane in values:
        converted = [[exact_complex(z) for z in row] for row in plane]
        if inverse:
            converted = [[(z[0], -z[1]) for z in row] for row in converted]
        rows = [axis_dft(row) for row in converted]
        columns = [axis_dft([rows[x][y] for x in range(4)]) for y in range(4)]
        answer = [[columns[y][x] for y in range(4)] for x in range(4)]
        if inverse:
            answer = [[(z[0]/16, -z[1]/16) for z in row] for row in answer]
        out.append(answer)
    return out


def fixture(angles=2):
    directions = [(F(1, 2), F(-1, 4)), (F(-3, 4), F(1, 8))][:angles]
    frequencies = (0, 1, -2, -1)
    kx = [[frequencies[x] for _ in range(4)] for x in range(4)]
    ky = [[frequencies[y] for y in range(4)] for _ in range(4)]
    t = [[F(17+x+2*y, 8) for y in range(4)] for x in range(4)]
    s = [[F(3+x+y, 16) for y in range(4)] for x in range(4)]
    phase = [[F(3, 2), F(1, 4)], [F(-1, 8), F(5, 4)]]
    weights = [F(1, 4), F(3, 4)]
    return dict(directions=directions, kx=kx, ky=ky, t=t, s=s,
                phase=[row[:angles] for row in phase[:angles]], weights=weights[:angles])


class ExactRationalRTETests(unittest.TestCase):
    def assert_norm_enclosure(self, values):
        square = norm_squared(values)
        lo, hi = norm_bounds(values)
        self.assertGreaterEqual(lo, 0.0)
        self.assertFalse(math.isnan(lo) or math.isnan(hi))
        self.assertLessEqual(lo, hi)
        self.assertTrue(math.isfinite(lo))
        self.assertLessEqual(F(*lo.as_integer_ratio()) ** 2, square)
        if math.isfinite(hi):
            self.assertGreaterEqual(F(*hi.as_integer_ratio()) ** 2, square)
        return lo, hi

    def test_exact_stored_float_and_complex_pairs(self):
        self.assertEqual(exact_complex(0.1), (F(*0.1.as_integer_ratio()), F(0)))
        self.assertNotEqual(exact_complex(0.1)[0], F(1, 10))
        self.assertEqual(exact_complex(-0.5+0.25j), (F(-1, 2), F(1, 4)))
        self.assertEqual(exact_complex([F(1, 3), -0.125]), (F(1, 3), F(-1, 8)))
        self.assertEqual(exact_complex(2), (F(2), F(0)))

    def test_reject_nonfinite_and_unsupported_coefficients(self):
        for value in (math.nan, math.inf, -math.inf, complex(1, math.inf)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                exact_complex(value)
        with self.assertRaises(TypeError):
            exact_complex("0.1")
        with self.assertRaises(ValueError):
            exact_complex((1, 2, 3))

    def test_dft_impulse_constant_and_fourier_basis(self):
        impulse = field()
        impulse[0][1][2] = 1
        roots = (1, -1j, -1, 1j)
        expected = [[[exact_complex(roots[(k+2*ell) % 4]) for ell in range(4)]
                     for k in range(4)]]
        self.assertEqual(dft2(impulse), expected)
        constant_expected = field()
        constant_expected[0][0][0] = exact_complex(16)
        self.assertEqual(dft2(field(1)), [[list(map(exact_complex, row))
                                         for row in constant_expected[0]]])
        mode = [[[(1j)**(x+2*y) for y in range(4)] for x in range(4)]]
        mode_expected = field(Z)
        mode_expected[0][1][2] = exact_complex(16)
        self.assertEqual(dft2(mode), mode_expected)

    def test_dense_dft_matches_independent_butterfly_and_inverse(self):
        values = dense_field()
        saved = copy.deepcopy(values)
        expected = separable_dft(values)
        self.assertEqual(dft2(values), expected)
        self.assertEqual(dft2(expected, inverse=True), values)
        self.assertEqual(dft2(values, inverse=True), separable_dft(values, inverse=True))
        self.assertEqual(values, saved)

    def test_nyquist_streaming_preserves_imaginary_response(self):
        values = [[[(-1)**x for _ in range(4)] for x in range(4)]]
        arguments = fixture(1)
        arguments.update(directions=[(1, 0)], t=grid(0), s=grid(0))
        expected = scale_field(values, -2j)
        self.assertEqual(apply_rte(values, **arguments), expected)
        self.assertEqual(apply_rte(field(1), **arguments), field(Z))

    def test_constant_collision_uses_actual_nonsymmetric_weighted_phase(self):
        values = [grid((F(1, 2), F(1, 4))), grid((F(-1, 8), F(3, 4)))]
        arguments = fixture()
        arguments.update(t=grid(F(7, 4)), s=grid(F(3, 8)))
        v0, v1 = values[0][0][0], values[1][0][0]
        # Explicit entries of phase @ diag(weights), not a row-weighted phase.
        scatter0 = plus(times((F(3, 8), F(0)), v0), times((F(3, 16), F(0)), v1))
        scatter1 = plus(times((F(-1, 32), F(0)), v0), times((F(15, 16), F(0)), v1))
        expected = [grid(plus(times((F(7, 4), F(0)), v0),
                              neg(times((F(3, 8), F(0)), scatter0)))),
                    grid(plus(times((F(7, 4), F(0)), v1),
                              neg(times((F(3, 8), F(0)), scatter1))))]
        self.assertEqual(apply_rte(values, **arguments), expected)

    def test_heterogeneous_dense_operator_against_separable_oracle(self):
        values = dense_field()
        arguments = fixture()
        saved = copy.deepcopy((values, arguments))
        spectrum = separable_dft(values)
        symbols = [[[(F(0), F(arguments["kx"][x][y], 2) - F(arguments["ky"][x][y], 4))
                     for y in range(4)] for x in range(4)],
                   [[(F(0), -F(3*arguments["kx"][x][y], 4) + F(arguments["ky"][x][y], 8))
                     for y in range(4)] for x in range(4)]]
        spectral_stream = [[[times(spectrum[a][x][y], symbols[a][x][y])
                             for y in range(4)] for x in range(4)] for a in range(2)]
        expected = separable_dft(spectral_stream, inverse=True)
        for x in range(4):
            for y in range(4):
                v0, v1 = values[0][x][y], values[1][x][y]
                scatters = (plus(times((F(3, 8), F(0)), v0), times((F(3, 16), F(0)), v1)),
                            plus(times((F(-1, 32), F(0)), v0), times((F(15, 16), F(0)), v1)))
                for a in range(2):
                    expected[a][x][y] = plus(expected[a][x][y],
                        plus(times((arguments["t"][x][y], F(0)), values[a][x][y]),
                             neg(times((arguments["s"][x][y], F(0)), scatters[a]))))
        self.assertEqual(apply_rte(values, **arguments), expected)
        self.assertEqual((values, arguments), saved)

    def test_complex_linearity_zero_and_phase(self):
        a = dense_field()
        b = [[[(F(2*aidx-x+3*y, 16), F(x-y-aidx, 8)) for y in range(4)]
              for x in range(4)] for aidx in range(2)]
        alpha, beta = (F(1, 2), F(1, 4)), (F(-3, 4), F(1, 8))
        arguments = fixture()
        combination = add_fields(scale_field(a, alpha), scale_field(b, beta))
        expected = add_fields(scale_field(apply_rte(a, **arguments), alpha),
                              scale_field(apply_rte(b, **arguments), beta))
        self.assertEqual(apply_rte(combination, **arguments), expected)
        self.assertEqual(apply_rte(scale_field(a, 1j), **arguments),
                         scale_field(apply_rte(a, **arguments), 1j))
        self.assertEqual(apply_rte(field(0, 2), **arguments), field(Z, 2))

    def test_rounded_assembled_symbol_is_a_distinct_exact_operator(self):
        def f32(value):
            return struct.unpack("f", struct.pack("f", value))[0]
        ox, oy = f32(0.1), f32(0.3)
        frequencies_x = (0.0, f32(0.3), f32(-0.6), f32(-0.3))
        frequencies_y = (0.0, f32(0.7), f32(-1.4), f32(-0.7))
        kx = [[frequencies_x[x] for _ in range(4)] for x in range(4)]
        ky = [[frequencies_y[y] for y in range(4)] for _ in range(4)]
        arguments = dict(directions=[(ox, oy)], kx=kx, ky=ky, t=grid(0),
                         s=grid(0), phase=[[1]], weights=[1])
        values = [[[(1j)**(x+y) for y in range(4)] for x in range(4)]]
        assembled = [[[(0, f32(f32(ox*kx[x][y]) + f32(oy*ky[x][y])))
                       for y in range(4)] for x in range(4)]]
        exact_coefficient = F(*ox.as_integer_ratio()) * F(*kx[1][1].as_integer_ratio()) \
                            + F(*oy.as_integer_ratio()) * F(*ky[1][1].as_integer_ratio())
        rounded_coefficient = F(*assembled[0][1][1][1].as_integer_ratio())
        self.assertNotEqual(exact_coefficient, rounded_coefficient)
        exact_response = apply_rte(values, **arguments)
        rounded_response = apply_rte(values, stream_symbol=assembled, **arguments)
        self.assertEqual(exact_response, scale_field(values, (0, exact_coefficient)))
        self.assertEqual(rounded_response, scale_field(values, (0, rounded_coefficient)))
        delta = add_fields(exact_response, scale_field(rounded_response, -1))
        self.assertEqual(norm_squared(delta), 16*(exact_coefficient-rounded_coefficient)**2)
        self.assertGreater(norm_squared(delta), 0)

    def test_exact_squared_norm_and_outward_sqrt(self):
        values = field()
        values[0][0][0] = 3+4j
        self.assertEqual(norm_squared(values), F(25))
        lo, hi = self.assert_norm_enclosure(values)
        self.assertLess(lo, 5.0)
        self.assertGreater(hi, 5.0)
        values[0][0][0] = 1+1j
        self.assertEqual(norm_squared(values), F(2))
        lo, hi = self.assert_norm_enclosure(values)
        self.assertLessEqual(hi-lo, 4*math.ulp(math.sqrt(2)))
        self.assertEqual(norm_bounds(field()), (0.0, 0.0))

    def test_norm_bounds_extreme_dyadics_and_stored_float(self):
        cases = [F(1, 1 << 2000), F(1, 1 << 1074), F(1 << 1024),
                 F((1 << 80)+1, 1 << 40), F(*0.1.as_integer_ratio())]
        for value in cases:
            with self.subTest(value=value):
                values = field()
                values[0][0][0] = value
                self.assertEqual(norm_squared(values), value*value)
                self.assert_norm_enclosure(values)
        tiny = field()
        tiny[0][0][0] = F(1, 1 << 2000)
        self.assertEqual(norm_bounds(tiny), (0.0, math.nextafter(0.0, math.inf)))
        huge = field()
        huge[0][0][0] = F(1 << 1024)
        self.assertEqual(norm_bounds(huge), (math.nextafter(math.inf, -math.inf), math.inf))

    def test_strict_shapes_and_no_implicit_frequency_broadcast(self):
        for malformed in ([], grid(0), [[list(range(3)) for _ in range(4)]],
                          [[[0]*4 for _ in range(3)]]):
            with self.subTest(shape=malformed), self.assertRaises((ValueError, TypeError)):
                dft2(malformed)
        arguments = fixture(1)
        for key, replacement in (("kx", [0, 1, -2, -1]), ("ky", [[0]*3]*4),
                                 ("directions", [(1, 0), (0, 1)]),
                                 ("weights", []), ("phase", [[1, 0]])):
            changed = dict(arguments)
            changed[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                apply_rte(field(1), **changed)
        with self.assertRaises(ValueError):
            apply_rte(field(1), stream_symbol=field(0, 2), **arguments)


if __name__ == "__main__":
    unittest.main(verbosity=2)
