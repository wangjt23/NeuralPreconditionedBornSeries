"""Input-specific interval certificate for the stored discrete RTE operator.

Certification MANDATORILY uses authored native binary64 add/multiply kernels
compiled without fast math/contraction, under guarded IEEE scalar modes
without FTZ/DAZ, overflow or hidden lower precision. NumPy nextafter returns
adjacent binary64 values. NumPy arithmetic fallback supports conditional
fixtures only: certify_residual fails closed until the native guard succeeds.
Guard flags32/64 repeat read-only mode-register checks, not independent
arithmetic proofs. Finite test witnesses do not prove all CPU operations;
the recorded compiler/IEEE primitive contract remains an explicit trust base.
No NumPy FFT, BLAS, trigonometric function or reduction certifies a result.
The transform uses rationally certified roots from certified_twiddles.

This encloses the exact nominal map with the *actually assembled* original
D64 and exact stored collision bits. It is not a GPU kernel error contract.
"""

from dataclasses import dataclass
from fractions import Fraction
import ctypes
import hashlib
import math
from pathlib import Path
import time

import numpy as np

from certified_twiddles import twiddle_bounds


CPU_CONTRACT = (
    "Authored native add/multiply kernels compiled -fno-fast-math "
    "-ffp-contract=off; IEC559 binary64, FE_TONEAREST and hardware FTZ/DAZ "
    "guards checked; separate correctly rounded CPU operations; "
    "no FTZ/DAZ, overflow, hidden reduced precision or cross-ufunc fusion; "
    "correct adjacent-value nextafter. Rationally certified twiddles; "
    "no NumPy FFT/BLAS/trig/reduction certificate."
)

_NATIVE = None
_NATIVE_METADATA = None


def load_native_cpu_backend(path):
    """Load the authored shared library; no compiler or mode change is hidden.

    The consuming executor compiles the supplied source using the documented
    flags and records its compiler command/version/source/library hashes.
    This loader cannot infer how an arbitrary library was compiled.
    """
    global _NATIVE, _NATIVE_METADATA
    path = Path(path).resolve()
    library = ctypes.CDLL(str(path))
    pointer = ctypes.POINTER(ctypes.c_double)
    for name in ("rte_interval_add", "rte_interval_mul"):
        function = getattr(library, name)
        function.argtypes = (ctypes.c_size_t, pointer, pointer, pointer)
        function.restype = ctypes.c_int
    library.rte_interval_guard.restype = ctypes.c_uint64
    library.rte_interval_control_register.restype = ctypes.c_uint64
    library.rte_interval_architecture.restype = ctypes.c_char_p
    _NATIVE = library
    _NATIVE_METADATA = dict(
        library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        architecture=library.rte_interval_architecture().decode("ascii"),
        compilation_contract="-O2 -std=c++17 -fno-fast-math -ffp-contract=off",
        compilation_provenance_must_be_recorded_by_executor=True)
    return native_cpu_guard()


def native_cpu_guard():
    if _NATIVE is None:
        raise RuntimeError("Load authored native CPU backend before certification")
    flags = int(_NATIVE.rte_interval_guard())
    if flags != 127:
        raise RuntimeError("CPU arithmetic guard failed closed: flags=%d" % flags)
    return dict(_NATIVE_METADATA, guard_flags=flags,
                control_register=int(_NATIVE.rte_interval_control_register()),
                floating_control_modes_modified=False,
                sticky_exception_flags_may_change_in_arithmetic=True)


def _native_operation(name, a, b):
    # Broadcast/indexing/copy are exact bit operations. Native kernels perform
    # every endpoint add/multiply; no NumPy arithmetic backend is assumed here.
    a, b = np.broadcast_arrays(a, b)
    shape = a.shape
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    output = np.empty(shape, dtype=np.float64)
    pointer = ctypes.POINTER(ctypes.c_double)
    status = getattr(_NATIVE, name)(
        output.size, a.ctypes.data_as(pointer), b.ctypes.data_as(pointer),
        output.ctypes.data_as(pointer))
    if status:
        raise FloatingPointError("Native endpoint arithmetic overflow/nonfinite")
    return output


def _finite(a):
    if not np.isfinite(a).all():
        raise FloatingPointError("Nonfinite interval endpoint: fail closed")
    return a


def _down(a):
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        return _finite(np.nextafter(_finite(a), -np.inf))


def _up(a):
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        return _finite(np.nextafter(_finite(a), np.inf))


def _add(a, b):
    if _NATIVE is not None:
        return _native_operation("rte_interval_add", a, b)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        return _finite(np.add(a, b))


def _mul(a, b):
    if _NATIVE is not None:
        return _native_operation("rte_interval_mul", a, b)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        return _finite(np.multiply(a, b))


@dataclass(frozen=True)
class RealBox:
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self):
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        if lo.dtype != np.float64 or hi.dtype != np.float64:
            raise TypeError("Endpoints must be NumPy binary64")
        if lo.shape != hi.shape or not np.all(lo <= hi):
            raise ValueError("Ordered endpoints of equal shape required")
        _finite(lo); _finite(hi)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def shape(self):
        return self.lo.shape


@dataclass(frozen=True)
class ComplexBox:
    re: RealBox
    im: RealBox

    def __post_init__(self):
        if self.re.shape != self.im.shape:
            raise ValueError("Real/imaginary shapes differ")

    @property
    def shape(self):
        return self.re.shape


def real_point(value):
    value = np.asarray(value)
    if value.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("Stored real coefficients must be binary32/binary64")
    x = _finite(np.array(value, dtype=np.float64, copy=True))
    return RealBox(x, x)


def point(value):
    value = np.asarray(value)
    if value.dtype not in (np.dtype("complex64"), np.dtype("complex128"),
                           np.dtype("float32"), np.dtype("float64")):
        raise TypeError("Exact input bits require binary32/64 real/complex arrays")
    re = _finite(np.array(value.real, dtype=np.float64, copy=True))
    im = _finite(np.array(value.imag, dtype=np.float64, copy=True))
    return ComplexBox(RealBox(re, re), RealBox(im, im))


def radd(a, b):
    return RealBox(_down(_add(a.lo, b.lo)), _up(_add(a.hi, b.hi)))


def rneg(a):
    # Binary sign reversal is exact, including signed zero.
    return RealBox(np.negative(a.hi), np.negative(a.lo))


def rsub(a, b):
    return radd(a, rneg(b))


def rmul(a, b):
    corners = (_mul(a.lo, b.lo), _mul(a.lo, b.hi),
               _mul(a.hi, b.lo), _mul(a.hi, b.hi))
    lo = np.minimum(np.minimum(corners[0], corners[1]),
                    np.minimum(corners[2], corners[3]))
    hi = np.maximum(np.maximum(corners[0], corners[1]),
                    np.maximum(corners[2], corners[3]))
    return RealBox(_down(lo), _up(hi))


def cadd(a, b):
    return ComplexBox(radd(a.re, b.re), radd(a.im, b.im))


def csub(a, b):
    return ComplexBox(rsub(a.re, b.re), rsub(a.im, b.im))


def cmul(a, b):
    return ComplexBox(rsub(rmul(a.re, b.re), rmul(a.im, b.im)),
                      radd(rmul(a.re, b.im), rmul(a.im, b.re)))


def scale(a, coefficient):
    coefficient = real_point(coefficient)
    return ComplexBox(rmul(a.re, coefficient), rmul(a.im, coefficient))


def _map(a, function):
    return ComplexBox(RealBox(function(a.re.lo), function(a.re.hi)),
                      RealBox(function(a.im.lo), function(a.im.hi)))


def _slice(a, index):
    return _map(a, lambda x: x[index])


def _concatenate(a, b, axis):
    return ComplexBox(
        RealBox(np.concatenate((a.re.lo, b.re.lo), axis),
                np.concatenate((a.re.hi, b.re.hi), axis)),
        RealBox(np.concatenate((a.im.lo, b.im.lo), axis),
                np.concatenate((a.im.hi, b.im.hi), axis)))


def fft_box(value, axis=-1, inverse=False):
    """Radix2 DIT DFT; inverse divides by the exact power-of-two length.

    Butterfly roots enclose exp(-/+2*pi*i*k/m). Every arithmetic result is
    rounded outwards. Indexing, permutation, reshaping and sign changes are
    exact. `value` is never mutated, and output owns independent storage.
    """
    if not isinstance(value, ComplexBox):
        value = point(value)
    rank = len(value.shape)
    if not -rank <= axis < rank:
        raise ValueError("FFT axis out of range")
    axis %= rank
    value = _map(value, lambda x: np.moveaxis(x, axis, -1))
    n = value.shape[-1]
    if n < 2 or n & (n-1):
        raise ValueError("FFT length must be a power of two >=2")
    bits = n.bit_length()-1
    order = np.array([int(format(i, "0%db" % bits)[::-1], 2)
                      for i in range(n)], dtype=np.int64)
    work = _map(value, lambda x: np.array(x[..., order], copy=True))
    original_shape = work.shape
    for m in (1 << j for j in range(1, bits+1)):
        shaped = _map(work, lambda x: x.reshape(original_shape[:-1]+(n//m, m)))
        left = _slice(shaped, (..., slice(0, m//2)))
        right = _slice(shaped, (..., slice(m//2, m)))
        rl, rh, il, ih = twiddle_bounds(m, inverse=inverse)
        roots = ComplexBox(RealBox(rl, rh), RealBox(il, ih))
        product = cmul(right, roots)
        work = _map(_concatenate(cadd(left, product), csub(left, product), -1),
                    lambda x: x.reshape(original_shape))
    if inverse:
        # 1/n is exactly binary64 for the supported power-of-two lengths.
        work = scale(work, np.array(1./n, dtype=np.float64))
    return _map(work, lambda x: np.moveaxis(x, -1, axis))


def fft2_box(value, inverse=False):
    """Two spatial transforms, axes(-2,-1), backward normalization."""
    if not isinstance(value, ComplexBox):
        value = point(value)
    if len(value.shape) < 2:
        raise ValueError("Two spatial dimensions required")
    return fft_box(fft_box(value, -1, inverse), -2, inverse)


def physical_action_box(candidate, stream_symbol, sigma_t, sigma_s, phase, weights):
    """Enclose F^-1 D64 F x + t*x - s*phase@(weights*x), no BLAS.

    Candidate shape is B,L,Nx,Ny; coefficients have the original broadcasting
    layout. D64 must be the captured original complex128, pure-imaginary symbol.
    The stored real phase/weights/t/s bits are interpreted exactly.
    """
    candidate = np.asarray(candidate)
    stream_symbol = np.asarray(stream_symbol)
    sigma_t, sigma_s = np.asarray(sigma_t), np.asarray(sigma_s)
    phase, weights = np.asarray(phase), np.asarray(weights)
    if candidate.ndim != 4 or stream_symbol.dtype != np.complex128:
        raise ValueError("B,L,Nx,Ny candidate and actual complex128 D64 required")
    b, l, nx, ny = candidate.shape
    if phase.shape != (l, l) or weights.shape != (l,):
        raise ValueError("Stored angular layout mismatch")
    if sigma_t.shape != (b, 1, nx, ny) or sigma_s.shape != sigma_t.shape:
        raise ValueError("Stored material layout must be B,1,Nx,Ny")
    if stream_symbol.shape not in ((b, l, nx, ny), (1, l, nx, ny)):
        raise ValueError("Captured stream-symbol layout mismatch")
    if not np.all(stream_symbol.real == 0):
        raise ValueError("Original skew-Hermitian stream must be pure imaginary")
    if not np.all(sigma_t > 0) or not np.all(sigma_s >= 0):
        raise ValueError("Positive t and nonnegative s required")
    x = point(candidate)
    streaming = fft2_box(cmul(point(stream_symbol), fft2_box(x)), inverse=True)
    weighted = scale(x, weights[None, :, None, None])
    scatter = point(np.zeros(candidate.shape, dtype=np.complex128))
    for j in range(l):
        term = scale(_slice(weighted, (slice(None), slice(j, j+1),
                                      slice(None), slice(None))),
                     phase[None, :, j, None, None])
        scatter = cadd(scatter, term)
    return csub(cadd(streaming, scale(x, sigma_t)), scale(scatter, sigma_s))


def _sum_box(a):
    """Explicit outward pairwise summation; no uncertified reduction."""
    lo, hi = a.lo.reshape(-1), a.hi.reshape(-1)
    if not len(lo):
        raise ValueError("Empty norm")
    while len(lo) > 1:
        pairs = len(lo)//2
        nxt_lo = _down(_add(lo[:2*pairs:2], lo[1:2*pairs:2]))
        nxt_hi = _up(_add(hi[:2*pairs:2], hi[1:2*pairs:2]))
        if len(lo) % 2:
            nxt_lo = np.concatenate((nxt_lo, lo[-1:]))
            nxt_hi = np.concatenate((nxt_hi, hi[-1:]))
        lo, hi = nxt_lo, nxt_hi
    return float(lo[0]), float(hi[0])


def _square(a):
    left, right = _mul(a.lo, a.lo), _mul(a.hi, a.hi)
    lower = _down(np.minimum(left, right))
    lower = np.where((a.lo <= 0) & (a.hi >= 0), 0., lower)
    return RealBox(np.maximum(0., lower), _up(np.maximum(left, right)))


def _sqrt_fraction_bounds(value, bits=100):
    value = Fraction(value)
    if value < 0:
        raise ValueError("Negative square-root bound")
    if value == 0:
        return Fraction(0), Fraction(0)
    # Adaptive dyadic scale retains tiny normal/subnormal inputs without a
    # relative-width guess or a floating square-root/trigonometry contract.
    exponent = value.numerator.bit_length()-value.denominator.bit_length()
    scale_bits = max(bits, bits-exponent//2)
    scale = 1 << (2*scale_bits)
    k = math.isqrt(value.numerator*scale//value.denominator)
    lower = Fraction(k, 1 << scale_bits)
    upper = lower if lower*lower == value else Fraction(k+1, 1 << scale_bits)
    return lower, upper


def _outward_float(value, upper):
    value = Fraction(value)
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError("Scalar certificate overflow")
    stored = Fraction.from_float(result)
    if (stored < value if upper else stored > value):
        result = math.nextafter(result, math.inf if upper else -math.inf)
    if not math.isfinite(result):
        raise FloatingPointError("Scalar certificate overflow")
    return result


def norm_bounds(value):
    """Euclidean enclosure using outward squares/sums and rational sqrt."""
    if not isinstance(value, ComplexBox):
        value = point(value)
    lower, upper = _sum_box(radd(_square(value.re), _square(value.im)))
    lower = max(0., lower)
    lo = _sqrt_fraction_bounds(Fraction.from_float(lower))[0]
    hi = _sqrt_fraction_bounds(Fraction.from_float(upper))[1]
    return _outward_float(lo, False), _outward_float(hi, True)


def coefficient_margin(sigma_t, sigma_s, phase, weights):
    """Exact rational lower bound min(t-s*sqrt(||P||1||P||inf))."""
    t, s, phase, weights = map(np.asarray, (sigma_t, sigma_s, phase, weights))
    l = len(weights)
    if t.shape != s.shape or phase.shape != (l, l) or not np.all(t > 0) or not np.all(s >= 0):
        raise ValueError("Invalid coercivity operands")
    for operand in (t, s, phase, weights):
        real_point(operand)  # validate exact promotable dtype and finite range
    f = lambda x: Fraction.from_float(float(x))
    p = [[abs(f(phase[i, j])*f(weights[j])) for j in range(l)] for i in range(l)]
    row = max(sum(v) for v in p)
    col = max(sum(p[i][j] for i in range(l)) for j in range(l))
    norm_upper = _sqrt_fraction_bounds(row*col)[1]
    margin = min(f(a)-f(b)*norm_upper for a, b in zip(t.reshape(-1), s.reshape(-1)))
    if margin <= 0:
        raise ValueError("No positive stored-coefficient coercivity bound")
    return margin


def certify_residual(source, candidate, stream_symbol, sigma_t, sigma_s,
                     phase, weights, margin_lower=None,
                     residual_tolerance=Fraction(1, 100000),
                     solution_tolerance=Fraction(1, 1000)):
    """Fresh nominal residual and relative solution-error upper bound.

    Source/candidate exact stored bits are never replaced or rounded to64.
    A supplied margin must be <= the independently derived coefficient bound.
    Returned acceptance is conditional solely on the stated CPU primitive
    contract; no GPU transform error hypothesis is used.
    """
    start = time.perf_counter()
    costs = {}
    began = time.perf_counter()
    guard_before = native_cpu_guard()
    costs["initial_guard_seconds"] = time.perf_counter()-began
    tau = (residual_tolerance if isinstance(residual_tolerance, Fraction)
           else Fraction(str(residual_tolerance)))
    solution_tau = (solution_tolerance if isinstance(solution_tolerance, Fraction)
                    else Fraction(str(solution_tolerance)))
    alpha = Fraction(1, 100)
    if tau <= 0 or solution_tau <= 0:
        raise ValueError("Positive exact policy tolerances required")
    source, candidate = np.asarray(source), np.asarray(candidate)
    if source.shape != candidate.shape:
        raise ValueError("Source/candidate shape mismatch")
    began = time.perf_counter()
    margin = coefficient_margin(sigma_t, sigma_s, phase, weights)
    derived_margin = margin
    if margin_lower is not None:
        supplied = (margin_lower if isinstance(margin_lower, Fraction)
                    else Fraction.from_float(float(margin_lower)))
        if supplied <= 0 or supplied > margin:
            raise ValueError("Supplied margin is not within verified bound")
        margin = supplied
    costs["coefficient_margin_seconds"] = time.perf_counter()-began
    began = time.perf_counter()
    qlo, qhi = norm_bounds(source)
    xlo, xhi = norm_bounds(candidate)
    costs["source_candidate_norm_seconds"] = time.perf_counter()-began
    began = time.perf_counter()
    action = physical_action_box(candidate, stream_symbol, sigma_t, sigma_s,
                                 phase, weights)
    costs["action_seconds"] = time.perf_counter()-began
    began = time.perf_counter()
    residual = csub(point(source), action)
    rlo, rhi = norm_bounds(residual)
    costs["residual_norm_seconds"] = time.perf_counter()-began
    began = time.perf_counter()
    # Define a represented center; exact clipping keeps every component inside
    # its interval even when halving a subnormal endpoint rounds to zero.
    def center_component(box):
        half = np.array(.5, dtype=np.float64)
        proposed = _add(_mul(box.lo, half), _mul(box.hi, half))
        center = np.minimum(box.hi, np.maximum(box.lo, proposed))
        if not np.all((box.lo <= center) & (center <= box.hi)):
            raise RuntimeError("Residual center left its enclosure")
        return center
    center = np.empty(residual.shape, dtype=np.complex128)
    center.real[...] = center_component(residual.re)
    center.imag[...] = center_component(residual.im)
    clo, chi = norm_bounds(center)
    elo, ehi = norm_bounds(csub(residual, point(center)))
    costs["reserve_center_radius_seconds"] = time.perf_counter()-began
    began = time.perf_counter()
    upper = Fraction.from_float(rhi)
    delta = upper/margin
    denominator = Fraction.from_float(xlo)-delta
    source_lower = Fraction.from_float(qlo)
    relative_residual = (_outward_float(upper/source_lower, True)
                         if source_lower > 0 else None)
    relative_solution = (_outward_float(delta/denominator, True)
                         if denominator > 0 else None)
    center_upper = Fraction.from_float(chi)
    radius_upper = Fraction.from_float(ehi)
    residual_passed = source_lower > 0 and upper <= tau*source_lower
    center_passed = source_lower > 0 and center_upper <= (1-alpha)*tau*source_lower
    radius_passed = source_lower > 0 and radius_upper <= alpha*tau*source_lower
    solution_passed = denominator > 0 and delta <= solution_tau*denominator
    costs["exact_acceptance_seconds"] = time.perf_counter()-began
    began = time.perf_counter()
    guard_after = native_cpu_guard()
    # Sticky exception-status bits may change. Arithmetic-control bits may not.
    control_mask = (0xFFC0 if guard_before["architecture"] == "x86-MXCSR"
                    else ((1 << 24) | (3 << 22)))
    if ((guard_before["control_register"] ^ guard_after["control_register"])
            & control_mask):
        raise RuntimeError("CPU arithmetic control state changed during certificate")
    costs["final_guard_seconds"] = time.perf_counter()-began
    elapsed = time.perf_counter()-start
    return dict(
        residual_norm_lower=rlo, residual_norm_upper=rhi,
        source_norm_lower=qlo, source_norm_upper=qhi,
        candidate_norm_lower=xlo, candidate_norm_upper=xhi,
        coercivity_lower=_outward_float(margin, False),
        coercivity_lower_rational=str(margin),
        coercivity_lower_numerator=margin.numerator,
        coercivity_lower_denominator=margin.denominator,
        derived_coercivity_lower_rational=str(derived_margin),
        policy_tolerance_rational=str(tau),
        policy_solution_tolerance_rational=str(solution_tau),
        reserve_fraction_rational=str(alpha),
        source_relative_norm_defined=source_lower > 0,
        residual_center_inside_enclosure=True,
        residual_center_norm_lower=clo, residual_center_norm_upper=chi,
        residual_center_norm_width=_outward_float(Fraction.from_float(chi)-Fraction.from_float(clo), True),
        source_norm_width=_outward_float(Fraction.from_float(qhi)-source_lower, True),
        residual_radius_norm_lower=elo, residual_radius_norm_upper=ehi,
        center_plus_radius_residual_norm_upper=_outward_float(center_upper+radius_upper, True),
        relative_center_norm_upper=(_outward_float(center_upper/source_lower, True)
                                    if source_lower > 0 else None),
        relative_radius_norm_upper=(_outward_float(radius_upper/source_lower, True)
                                    if source_lower > 0 else None),
        relative_residual_upper=relative_residual,
        absolute_solution_error_upper=_outward_float(delta, True),
        relative_solution_error_upper=relative_solution,
        residual_accepted=residual_passed,
        reserve_center_accepted=center_passed,
        reserve_radius_accepted=radius_passed,
        fixed_one_percent_reserve_accepted=(center_passed and radius_passed and
                                           residual_passed),
        solution_error_accepted=solution_passed,
        accepted=(residual_passed and center_passed and radius_passed and
                  solution_passed),
        certified_under_cpu_contract=True, cpu_contract=CPU_CONTRACT,
        native_backend_guard_before=guard_before,
        native_backend_guard_after=guard_after,
        global_gpu_contract_established=False,
        target="Exact nominal original stored D64 and collision/source bits",
        costs=costs, elapsed_seconds=elapsed,
    )
