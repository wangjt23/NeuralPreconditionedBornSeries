"""Exact, standard-library-only 4 x 4 periodic transport fixture.

Fields have shape L x 4 x 4, with no batch dimension. Complex values are
represented by (real, imaginary) Fraction pairs. Directions have shape L x 2,
phase has shape L x L, weights have length L, and kx, ky, t, s are real 4 x 4
grids. Float inputs denote their exact stored binary values, not decimal ideals.

The optional stream_symbol has shape L x 4 x 4 and contains the *complex*
assembled symbol, including its factor of i. Supplying it changes only the
streaming symbol; omitting it forms i * (Omega_x * kx + Omega_y * ky) exactly
from the supplied stored coefficients. This tiny fixture does not bound a
general-size FFT implementation or reproduce floating-point FFT rounding.
"""

from fractions import Fraction
import math


_ZERO = (Fraction(0), Fraction(0))
_ONE = (Fraction(1), Fraction(0))


def _exact_real(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("exact arithmetic requires finite coefficients")
        return Fraction(*value.as_integer_ratio())
    raise TypeError("real coefficients must be int, float, or Fraction")


def exact_complex(value):
    """Return the exact Fraction pair for a real, complex, or real-pair value."""
    if isinstance(value, complex):
        return (_exact_real(value.real), _exact_real(value.imag))
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("a complex pair must have length two")
        return (_exact_real(value[0]), _exact_real(value[1]))
    return (_exact_real(value), Fraction(0))


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _mul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def _scale(a, value):
    return (a[0] * value, a[1] * value)


def _sequence(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError("{} must have length {}".format(name, size))
    return value


def _field(value, name="field", angles=None):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("{} must be a nonempty L x 4 x 4 field".format(name))
    if angles is not None and len(value) != angles:
        raise ValueError("{} has the wrong angular count".format(name))
    out = []
    for a, plane in enumerate(value):
        _sequence(plane, 4, "{}[{}]".format(name, a))
        converted = []
        for x, row in enumerate(plane):
            _sequence(row, 4, "{}[{}][{}]".format(name, a, x))
            converted.append([exact_complex(cell) for cell in row])
        out.append(converted)
    return out


def _real_grid(value, name):
    _sequence(value, 4, name)
    result = []
    for x, row in enumerate(value):
        _sequence(row, 4, "{}[{}]".format(name, x))
        result.append([_exact_real(cell) for cell in row])
    return result


def dft2(field, inverse=False):
    """Exact spatial DFT on each angle, with backward normalization.

    Forward transforms are unnormalized. Inverse transforms divide by 16.
    The roots are exactly 1, -i, -1, i (conjugated for the inverse).
    """
    values = _field(field)
    roots = (_ONE, (Fraction(0), Fraction(-1)),
             (Fraction(-1), Fraction(0)), (Fraction(0), Fraction(1)))
    if inverse:
        roots = tuple((z[0], -z[1]) for z in roots)
    result = []
    for plane in values:
        transformed = []
        for k in range(4):
            row = []
            for ell in range(4):
                total = _ZERO
                for x in range(4):
                    for y in range(4):
                        total = _add(total, _mul(plane[x][y], roots[(k*x + ell*y) % 4]))
                row.append(_scale(total, Fraction(1, 16)) if inverse else total)
            transformed.append(row)
        result.append(transformed)
    return result


def apply_rte(field, directions, kx, ky, t, s, phase, weights,
              stream_symbol=None):
    """Apply exact streaming + total attenuation - weighted scattering.

    Scattering uses phase @ (weights * field), followed by multiplication by s.
    The result contains exact Fraction pairs and owns all its nested lists.
    """
    values = _field(field)
    angles = len(values)
    _sequence(directions, angles, "directions")
    omega = []
    for a, direction in enumerate(directions):
        _sequence(direction, 2, "directions[{}]".format(a))
        omega.append([_exact_real(component) for component in direction])
    kx = _real_grid(kx, "kx")
    ky = _real_grid(ky, "ky")
    t = _real_grid(t, "t")
    s = _real_grid(s, "s")
    _sequence(phase, angles, "phase")
    phase_exact = []
    for a, row in enumerate(phase):
        _sequence(row, angles, "phase[{}]".format(a))
        phase_exact.append([_exact_real(value) for value in row])
    _sequence(weights, angles, "weights")
    weights_exact = [_exact_real(value) for value in weights]

    if stream_symbol is None:
        symbol = [[[(Fraction(0), omega[a][0] * kx[x][y] + omega[a][1] * ky[x][y])
                    for y in range(4)] for x in range(4)] for a in range(angles)]
    else:
        symbol = _field(stream_symbol, "stream_symbol", angles)
    spectrum = dft2(values)
    streamed_spectrum = [[[_mul(symbol[a][x][y], spectrum[a][x][y])
                          for y in range(4)] for x in range(4)] for a in range(angles)]
    streaming = dft2(streamed_spectrum, inverse=True)
    weighted = [[[_scale(values[a][x][y], weights_exact[a])
                 for y in range(4)] for x in range(4)] for a in range(angles)]
    result = []
    for a in range(angles):
        plane = []
        for x in range(4):
            row = []
            for y in range(4):
                scatter = _ZERO
                for b in range(angles):
                    scatter = _add(scatter, _scale(weighted[b][x][y], phase_exact[a][b]))
                row.append(_sub(_add(streaming[a][x][y], _scale(values[a][x][y], t[x][y])),
                                _scale(scatter, s[x][y])))
            plane.append(row)
        result.append(plane)
    return result


def norm_squared(field):
    """Return the exact, unweighted Euclidean squared norm as a Fraction."""
    values = _field(field)
    return sum((z[0]*z[0] + z[1]*z[1]
                for plane in values for row in plane for z in row), Fraction(0))


def _sqrt_rational_bracket(value):
    """Enclose sqrt(value) in dyadic rationals, using integer arithmetic only."""
    if not value:
        return (Fraction(0), Fraction(0))
    numerator, denominator = value.numerator, value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << (-exponent) < denominator:
        exponent -= 1
    # About 80 significant binary digits, even for subnormal-sized inputs.
    power = 80 - exponent // 2
    if power >= 0:
        scaled_numerator, scaled_denominator = numerator << (2*power), denominator
        unit = Fraction(1, 1 << power)
    else:
        scaled_numerator, scaled_denominator = numerator, denominator << (-2*power)
        unit = Fraction(1 << (-power))
    root = math.isqrt(scaled_numerator // scaled_denominator)
    lower = root * unit
    if root*root * scaled_denominator == scaled_numerator:
        return (lower, lower)
    return (lower, (root + 1) * unit)


def _outward_float(value, upper):
    if not value:
        return 0.0
    try:
        rounded = float(value)
    except OverflowError:
        return math.inf if upper else math.nextafter(math.inf, -math.inf)
    if math.isinf(rounded):
        return math.inf if upper else math.nextafter(math.inf, -math.inf)
    if upper:
        return math.nextafter(rounded, math.inf)
    return max(0.0, math.nextafter(rounded, -math.inf))


def norm_bounds(field):
    """Return conservative outward binary64 (lower, upper) norm bounds.

    Exact squared norms are first bracketed by rational square roots using
    integer isqrt. Each endpoint is then rounded outward with nextafter.
    Underflow may yield a zero lower bound; overflow may yield infinity above.
    """
    lower, upper = _sqrt_rational_bracket(norm_squared(field))
    return (_outward_float(lower, False), _outward_float(upper, True))
