"""Rationally enclosed radix-two roots, rounded outward to binary64.

No trigonometric backend is used. Machin's identity encloses pi with alternating
arctangent sums. Each reduced angle lies in [-pi/4, pi/4]; alternating rational
sin/cos sums enclose its midpoint, and the angle radius is added using their
global Lipschitz constant one. Axis roots are exact. This certifies the constants
only; a complete FFT enclosure still needs certified arithmetic at each stage.
"""

from fractions import Fraction
from functools import lru_cache
import math

import numpy as np


_SERIES_TAIL = Fraction(1, 10**45)
_RATIONAL_WIDTH_REQUIREMENT = Fraction(1, 10**30)


def _atan_reciprocal_bounds(denominator):
    """Alternating arctan(1/q) sum plus its first omitted term."""
    square = denominator*denominator
    power = denominator
    total = Fraction(0)
    index = 0
    sign = 1
    while True:
        total += sign*Fraction(1, (2*index+1)*power)
        power *= square
        index += 1
        sign = -sign
        next_term = sign*Fraction(1, (2*index+1)*power)
        if abs(next_term) <= _SERIES_TAIL:
            return (min(total, total+next_term), max(total, total+next_term))


@lru_cache(maxsize=1)
def pi_bounds():
    """Exact rational enclosure from pi = 16 atan(1/5) - 4 atan(1/239)."""
    five_lo, five_hi = _atan_reciprocal_bounds(5)
    large_lo, large_hi = _atan_reciprocal_bounds(239)
    return (16*five_lo - 4*large_hi, 16*five_hi - 4*large_lo)


@lru_cache(maxsize=None)
def _point_sin_cos_bounds(value):
    """Enclose sin/cos of a rational point of absolute value at most one."""
    if abs(value) > 1:
        raise ValueError("Taylor midpoint must lie in [-1, 1]")
    if not value:
        return ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(1)))
    x = abs(value)
    square = x*x
    sin_sum, sin_term, index = Fraction(0), x, 0
    while True:
        sin_sum += sin_term
        following = -sin_term*square/((2*index+2)*(2*index+3))
        if abs(following) <= _SERIES_TAIL:
            sine = (min(sin_sum, sin_sum+following), max(sin_sum, sin_sum+following))
            break
        sin_term, index = following, index+1
    if value < 0:
        sine = (-sine[1], -sine[0])
    cos_sum, cos_term, index = Fraction(0), Fraction(1), 0
    while True:
        cos_sum += cos_term
        following = -cos_term*square/((2*index+1)*(2*index+2))
        if abs(following) <= _SERIES_TAIL:
            cosine = (min(cos_sum, cos_sum+following), max(cos_sum, cos_sum+following))
            break
        cos_term, index = following, index+1
    return sine, cosine


def _negate(interval):
    return (-interval[1], -interval[0])


def _validate(n, inverse):
    if isinstance(n, bool) or not isinstance(n, int) or n < 2 or n & (n-1):
        raise ValueError("n must be an integer power of two at least two")
    if not isinstance(inverse, bool):
        raise TypeError("inverse must be bool")


def _root_fraction_bounds(n, k, inverse):
    # Nearest integer quadrant, with ties toward the higher quadrant. The
    # exact residual coefficient r satisfies |r| <= 1/4 and theta = m*pi/2+r*pi.
    quadrant = (8*k+n)//(2*n)
    residual = Fraction(4*k-quadrant*n, 2*n)
    if not residual:
        sine, cosine = (Fraction(0), Fraction(0)), (Fraction(1), Fraction(1))
    else:
        p_lo, p_hi = pi_bounds()
        angle_lo, angle_hi = residual*p_lo, residual*p_hi
        if residual < 0:
            angle_lo, angle_hi = angle_hi, angle_lo
        midpoint = (angle_lo+angle_hi)/2
        radius = (angle_hi-angle_lo)/2
        sine, cosine = _point_sin_cos_bounds(midpoint)
        sine = (sine[0]-radius, sine[1]+radius)
        cosine = (cosine[0]-radius, cosine[1]+radius)
    orientation = quadrant % 4
    if orientation == 0:
        real, imag = cosine, sine
    elif orientation == 1:
        real, imag = _negate(sine), cosine
    elif orientation == 2:
        real, imag = _negate(cosine), _negate(sine)
    else:
        real, imag = sine, _negate(cosine)
    if not inverse:
        imag = _negate(imag)
    if real[1]-real[0] >= _RATIONAL_WIDTH_REQUIREMENT or imag[1]-imag[0] >= _RATIONAL_WIDTH_REQUIREMENT:
        raise ArithmeticError("rational twiddle enclosure is not narrow enough")
    return real[0], real[1], imag[0], imag[1]


@lru_cache(maxsize=None, typed=True)
def twiddle_fraction_bounds(n, inverse=False):
    """Four immutable Fraction tuples enclosing exp((+/-)2*pi*i*k/n)."""
    _validate(n, inverse)
    values = [_root_fraction_bounds(n, k, inverse) for k in range(n//2)]
    return tuple(tuple(value[component] for value in values) for component in range(4))


def _outward_float(value, upper):
    rounded = float(value)
    actual = Fraction(*rounded.as_integer_ratio())
    if (actual < value) if upper else (actual > value):
        rounded = math.nextafter(rounded, math.inf if upper else -math.inf)
    actual = Fraction(*rounded.as_integer_ratio())
    if (actual < value) if upper else (actual > value):
        raise ArithmeticError("binary64 conversion did not round outward")
    return rounded


@lru_cache(maxsize=None, typed=True)
def twiddle_bounds(n, inverse=False):
    """Read-only float64 arrays (re_lo, re_hi, im_lo, im_hi), length n//2.

    Forward roots have negative imaginary sign. Inverse roots are conjugate.
    Each float endpoint is verified against its exact Fraction endpoint.
    """
    exact = twiddle_fraction_bounds(n, inverse)
    arrays = []
    for component, values in enumerate(exact):
        upper = component in (1, 3)
        array = np.frombuffer(np.asarray([_outward_float(value, upper) for value in values],
                                        dtype=np.float64).tobytes(), dtype=np.float64)
        arrays.append(array)
    return tuple(arrays)


def twiddle_certificate(n, inverse=False):
    """Small proof metadata; the rational endpoints are available separately."""
    exact = twiddle_fraction_bounds(n, inverse)
    width = max((exact[1][k]-exact[0][k] for k in range(n//2)), default=Fraction(0))
    width = max(width, max((exact[3][k]-exact[2][k] for k in range(n//2)), default=Fraction(0)))
    p_lo, p_hi = pi_bounds()
    return dict(n=n, inverse=inverse, roots=n//2,
                pi_lower=str(p_lo), pi_upper=str(p_hi),
                pi_width=str(p_hi-p_lo), rational_width_max=str(width),
                rational_width_less_than_1e_minus_30=width < _RATIONAL_WIDTH_REQUIREMENT,
                construction='Machin alternating arctan; quadrant reduction; alternating sin/cos at rational midpoint plus Lipschitz radius',
                float_conversion='Fraction comparison and outward binary64 nextafter',
                certifies='twiddle constants only; no FFT/backend arithmetic certificate')
