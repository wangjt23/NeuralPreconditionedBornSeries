"""Fresh stored-operator residual certificate with stagewise Euclidean balls.

Every radix2 stage encloses its exact action on the represented binary64
center with the copied native endpoint primitives and certified roots. It
recenters only that local box, then propagates prior uncertainty by the exact
butterfly norm sqrt(2). No uncertain rectangle is carried between stages.
Certification requires the same explicit IEEE/compiler/nextafter contract
as the prior checker. This is no GPU or universal FFT error contract.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math
import time

import numpy as np

import interval_fft as p
from certified_twiddles import twiddle_bounds


load_native_cpu_backend = p.load_native_cpu_backend
native_cpu_guard = p.native_cpu_guard
CPU_CONTRACT = p.CPU_CONTRACT


def _f(value):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite scalar bound')
    return F.from_float(value)


def _upper(value):
    return p._outward_float(value, True)


SQRT2_UPPER = _upper(p._sqrt_fraction_bounds(F(2))[1])


@dataclass(frozen=True)
class StagewiseBall:
    center: np.ndarray
    radius_upper: float = 0.
    stages: tuple = ()

    def __post_init__(self):
        center = np.asarray(self.center)
        if center.dtype != np.complex128:
            raise TypeError('Represented center must be complex128')
        p._finite(center.real); p._finite(center.imag)
        supplied = self.radius_upper
        if isinstance(supplied, bool) or not isinstance(
                supplied, (F, int, np.integer, float, np.float32, np.float64)):
            raise TypeError('Radius must be an exact rational/integer or represented binary float')
        exact = (supplied if isinstance(supplied, F) else F(int(supplied))
                 if isinstance(supplied, (int, np.integer)) else _f(supplied))
        if exact < 0:
            raise ValueError('Nonnegative finite Euclidean radius required')
        object.__setattr__(self, 'center', np.array(center, copy=True))
        object.__setattr__(self, 'radius_upper', _upper(exact))
        object.__setattr__(self, 'stages', tuple(self.stages))


def _as_ball(value):
    if isinstance(value, StagewiseBall):
        return value
    box = p.point(value)
    center = np.empty(box.shape, np.complex128)
    center.real[...] = box.re.lo; center.imag[...] = box.im.lo
    return StagewiseBall(center)


def _norm(box):
    if not isinstance(box, p.ComplexBox):
        box = p.point(box)
    if (np.all(box.re.lo == 0) and np.all(box.re.hi == 0) and
            np.all(box.im.lo == 0) and np.all(box.im.hi == 0)):
        return 0., 0.
    return p.norm_bounds(box)


def _radd(a, b):
    """Outward sum, with exact zero identity/cancellation for point inputs."""
    out = p.radd(a, b)
    al, ah, bl, bh = np.broadcast_arrays(a.lo, a.hi, b.lo, b.hi)
    az, bz = (al == 0) & (ah == 0), (bl == 0) & (bh == 0)
    cancel = (al == ah) & (bl == bh) & (al == np.negative(bl))
    lo = np.where(az, bl, np.where(bz, al, out.lo))
    hi = np.where(az, bh, np.where(bz, ah, out.hi))
    return p.RealBox(np.where(cancel, 0., lo), np.where(cancel, 0., hi))


def _cadd(a, b):
    return p.ComplexBox(_radd(a.re, b.re), _radd(a.im, b.im))


def _csub(a, b):
    return _cadd(a, p.ComplexBox(p.rneg(b.re), p.rneg(b.im)))


def _rmul(a, b):
    out = p.rmul(a, b)
    al, ah, bl, bh = np.broadcast_arrays(a.lo, a.hi, b.lo, b.hi)
    zero = ((al == 0) & (ah == 0)) | ((bl == 0) & (bh == 0))
    a1, b1 = (al == 1) & (ah == 1), (bl == 1) & (bh == 1)
    an, bn = (al == -1) & (ah == -1), (bl == -1) & (bh == -1)
    lo = np.where(a1, bl, np.where(b1, al,
                  np.where(an, np.negative(bh), np.where(bn, np.negative(ah), out.lo))))
    hi = np.where(a1, bh, np.where(b1, ah,
                  np.where(an, np.negative(bl), np.where(bn, np.negative(al), out.hi))))
    return p.RealBox(np.where(zero, 0., lo), np.where(zero, 0., hi))


def _scale(box, coefficient):
    coefficient = p.real_point(coefficient)
    return p.ComplexBox(_rmul(box.re, coefficient), _rmul(box.im, coefficient))


def _center(box):
    def component(real):
        half = np.array(.5, dtype=np.float64)
        proposed = p._add(p._mul(real.lo, half), p._mul(real.hi, half))
        result = np.minimum(real.hi, np.maximum(real.lo, proposed))
        if not np.all((real.lo <= result) & (result <= real.hi)):
            raise RuntimeError('Represented center left local enclosure')
        return result
    center = np.empty(box.shape, dtype=np.complex128)
    center.real[...] = component(box.re); center.imag[...] = component(box.im)
    return center


def _recenter(box):
    center = _center(box)
    # A local rectangle minus its represented point center encloses the exact
    # stage error. This is a vector radius, never a scalar norm-width surrogate.
    return center, _norm(_csub(box, p.point(center)))[1]


def _root_product(right, m, inverse):
    rl, rh, il, ih = twiddle_bounds(m, inverse=inverse)
    roots = p.ComplexBox(p.RealBox(rl, rh), p.RealBox(il, ih))
    shape = right.shape
    arrays = [np.empty(shape, dtype=np.float64) for _ in range(4)]
    cardinal = ((rl == rh) & (il == ih) &
                (((np.abs(rl) == 1) & (il == 0)) |
                 ((rl == 0) & (np.abs(il) == 1))))
    general = ~cardinal
    if np.any(general):
        partial = p.cmul(p._slice(right, (..., general)), p._slice(roots, general))
        for output, endpoint in zip(arrays, (partial.re.lo, partial.re.hi,
                                            partial.im.lo, partial.im.hi)):
            output[..., general] = endpoint
    for re, im in ((1., 0.), (-1., 0.), (0., 1.), (0., -1.)):
        mask = cardinal & (rl == re) & (il == im)
        if not np.any(mask):
            continue
        if re == 1:
            endpoints = (right.re.lo, right.re.hi, right.im.lo, right.im.hi)
        elif re == -1:
            endpoints = (-right.re.hi, -right.re.lo, -right.im.hi, -right.im.lo)
        elif im == 1:
            endpoints = (-right.im.hi, -right.im.lo, right.re.lo, right.re.hi)
        else:
            endpoints = (right.im.lo, right.im.hi, -right.re.hi, -right.re.lo)
        for output, endpoint in zip(arrays, endpoints):
            output[..., mask] = endpoint[..., mask]
    return p.ComplexBox(p.RealBox(arrays[0], arrays[1]),
                        p.RealBox(arrays[2], arrays[3]))


def fft_ball(value, axis=-1, inverse=False):
    """One unnormalized forward or 1/N-normalized inverse radix2 DFT ball."""
    ball = _as_ball(value)
    rank = ball.center.ndim
    if not -rank <= axis < rank:
        raise ValueError('FFT axis out of range')
    axis %= rank
    center = np.moveaxis(ball.center, axis, -1)
    n = center.shape[-1]
    if n < 2 or n & (n-1):
        raise ValueError('FFT length must be a power of two >=2')
    bits = n.bit_length()-1
    order = np.array([int(format(i, '0%db' % bits)[::-1], 2)
                      for i in range(n)], dtype=np.int64)
    work = np.array(center[..., order], copy=True)
    original_shape = work.shape
    radius = ball.radius_upper
    ledger = list(ball.stages)
    for m in (1 << j for j in range(1, bits+1)):
        began = time.perf_counter()
        shaped = p.point(work.reshape(original_shape[:-1]+(n//m, m)))
        left = p._slice(shaped, (..., slice(0, m//2)))
        right = p._slice(shaped, (..., slice(m//2, m)))
        product = _root_product(right, m, inverse)
        local = p._map(p._concatenate(_cadd(left, product), _csub(left, product), -1),
                       lambda x: x.reshape(original_shape))
        point_seconds = time.perf_counter()-began
        began = time.perf_counter()
        work, local_error = _recenter(local)
        local_seconds = time.perf_counter()-began
        before = radius
        radius = _upper(_f(SQRT2_UPPER)*_f(radius)+_f(local_error))
        ledger.append(dict(kind='inverse_fft_stage' if inverse else 'forward_fft_stage',
                           axis=axis, length=n, butterfly_width=m,
                           incoming_radius_upper=before, local_radius_upper=local_error,
                           radius_upper=radius, stage_operator_norm_upper=SQRT2_UPPER,
                           point_interval_seconds=point_seconds,
                           recenter_and_local_norm_seconds=local_seconds))
    if inverse:
        began = time.perf_counter()
        # Exactly represented power-of-two reciprocal; possible underflow is
        # still enclosed locally by native multiplication plus nextafter.
        reciprocal = np.array(1./n, dtype=np.float64)
        work, local_error = _recenter(_scale(p.point(work), reciprocal))
        before = radius
        radius = _upper(_f(radius)*F(1, n)+_f(local_error))
        ledger.append(dict(kind='inverse_normalization', axis=axis, length=n,
                           incoming_radius_upper=before, local_radius_upper=local_error,
                           radius_upper=radius, normalization_rational=str(F(1, n)),
                           elapsed_seconds=time.perf_counter()-began))
    return StagewiseBall(np.moveaxis(work, -1, axis), radius, tuple(ledger))


def fft2_ball(value, inverse=False):
    ball = _as_ball(value)
    if ball.center.ndim < 2:
        raise ValueError('Two spatial axes required')
    return fft_ball(fft_ball(ball, -1, inverse), -2, inverse)


def _validate(candidate, stream_symbol, sigma_t, sigma_s, phase, weights):
    candidate, stream_symbol, sigma_t, sigma_s, phase, weights = map(
        np.asarray, (candidate, stream_symbol, sigma_t, sigma_s, phase, weights))
    if candidate.ndim != 4 or stream_symbol.dtype != np.complex128:
        raise ValueError('B,L,Nx,Ny candidate and captured original D64 required')
    b, l, nx, ny = candidate.shape
    if min(b, l) <= 0 or phase.shape != (l, l) or weights.shape != (l,):
        raise ValueError('Invalid original angular layout')
    if sigma_t.shape != (b, 1, nx, ny) or sigma_s.shape != sigma_t.shape:
        raise ValueError('Original material layout must be B,1,Nx,Ny')
    if stream_symbol.shape not in ((b, l, nx, ny), (1, l, nx, ny)):
        raise ValueError('Original stream-symbol layout mismatch')
    if not np.all(stream_symbol.real == 0):
        raise ValueError('Original skew-Hermitian symbol must be pure imaginary')
    if not np.all(sigma_t > 0) or not np.all(sigma_s >= 0):
        raise ValueError('Positive t and nonnegative s required')
    p.point(candidate); p.point(stream_symbol)
    for array in (sigma_t, sigma_s, phase, weights):
        p.real_point(array)
    return candidate, stream_symbol, sigma_t, sigma_s, phase, weights


def physical_action_ball(candidate, stream_symbol, sigma_t, sigma_s, phase, weights):
    """Exact original D64/collision map enclosed around a represented center."""
    candidate, symbol, t, s, phase, weights = _validate(
        candidate, stream_symbol, sigma_t, sigma_s, phase, weights)
    streaming = fft2_ball(candidate)
    began = time.perf_counter()
    # Pure-imaginary D64 allows exact swaps/signs, then real multiplication.
    imaginary = p.real_point(symbol.imag)
    center_point = p.point(streaming.center)
    product = p.ComplexBox(p.rneg(_rmul(center_point.im, imaginary)),
                           _rmul(center_point.re, imaginary))
    center, local_error = _recenter(product)
    multiplier = float(np.max(np.abs(symbol.imag)))  # exact abs/comparison only
    radius = _upper(_f(multiplier)*_f(streaming.radius_upper)+_f(local_error))
    multiplication = dict(kind='stored_D64_multiplication',
                          operator_norm_upper=multiplier,
                          incoming_radius_upper=streaming.radius_upper,
                          local_radius_upper=local_error, radius_upper=radius,
                          elapsed_seconds=time.perf_counter()-began)
    streaming = fft2_ball(StagewiseBall(center, radius, streaming.stages+(multiplication,)),
                          inverse=True)
    began = time.perf_counter()
    x = p.point(candidate)
    weighted = _scale(x, weights[None, :, None, None])
    scatter = p.point(np.zeros(candidate.shape, dtype=np.complex128))
    for j in range(len(weights)):
        term = _scale(p._slice(weighted, (slice(None), slice(j, j+1),
                                        slice(None), slice(None))),
                      phase[None, :, j, None, None])
        scatter = _cadd(scatter, term)
    collision = _csub(_scale(x, t), _scale(scatter, s))
    collision_center, collision_error = _recenter(collision)
    collision_stage = dict(kind='collision_point_enclosure',
                           local_radius_upper=collision_error,
                           elapsed_seconds=time.perf_counter()-began)
    began = time.perf_counter()
    action_center, addition_error = _recenter(
        _cadd(p.point(streaming.center), p.point(collision_center)))
    action_radius = _upper(_f(streaming.radius_upper)+_f(collision_error)+_f(addition_error))
    addition_stage = dict(kind='stream_collision_addition',
                          incoming_radius_upper=streaming.radius_upper,
                          collision_radius_upper=collision_error,
                          local_radius_upper=addition_error, radius_upper=action_radius,
                          elapsed_seconds=time.perf_counter()-began)
    return StagewiseBall(action_center, action_radius,
                         streaming.stages+(collision_stage, addition_stage))


def ball_norm_bounds(ball):
    ball = _as_ball(ball)
    lower, upper = _norm(ball.center)
    lower = max(F(0), _f(lower)-_f(ball.radius_upper))
    upper = _f(upper)+_f(ball.radius_upper)
    return p._outward_float(lower, False), _upper(upper)


def paired_field_check(candidate, baseline, tolerance=F(1, 10000)):
    """Direct represented-field gate, independent of residual/coercivity gates."""
    guard_before = p.native_cpu_guard()
    tolerance = tolerance if isinstance(tolerance, F) else F(str(tolerance))
    if tolerance != F(1, 10000):
        raise ValueError('Fixed paired-field tolerance is 1/10000')
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    if candidate.shape != baseline.shape:
        raise ValueError('Paired-field shapes differ')
    difference = _csub(p.point(candidate), p.point(baseline))
    dlo, dhi = _norm(difference)
    blo, bhi = _norm(baseline)
    accepted = _f(blo) > 0 and _f(dhi) <= tolerance*_f(blo)
    rejected = _f(bhi) > 0 and _f(dlo) > tolerance*_f(bhi)
    guard_after = p.native_cpu_guard()
    mask = 0xFFC0 if guard_before['architecture'] == 'x86-MXCSR' else ((1 << 24) | (3 << 22))
    if (guard_before['control_register'] ^ guard_after['control_register']) & mask:
        raise RuntimeError('CPU arithmetic control state changed during paired-field check')
    return dict(paired_field_difference_norm_lower=dlo,
                paired_field_difference_norm_upper=dhi,
                paired_field_baseline_norm_lower=blo,
                paired_field_baseline_norm_upper=bhi,
                paired_field_relative_upper=(_upper(_f(dhi)/_f(blo)) if blo > 0 else None),
                paired_field_tolerance_rational=str(tolerance), paired_field_accepted=accepted,
                paired_field_rigorous_rejection=rejected,
                paired_field_guard_before=guard_before, paired_field_guard_after=guard_after,
                paired_field_representation='Exact captured binary32/binary64 input bits; subtraction and norms outward')


def certify_residual(source, candidate, stream_symbol, sigma_t, sigma_s, phase,
                     weights, margin_lower=None, residual_tolerance=F(1, 100000),
                     solution_tolerance=F(1, 1000), baseline=None):
    """Fixed-policy fresh residual, solution-error and optional paired-field gates.

    The direct residual upper bound is C_U+E_U, with stagewise certified vector
    radius E_U. All policy cross-products use exact Fractions. A baseline is
    required to claim the independent paired-field gate; absent baseline does
    not establish that criterion. No arrays are returned by this scalar API.
    """
    started = time.perf_counter(); costs = {}
    began = time.perf_counter(); guard_before = p.native_cpu_guard()
    costs['initial_guard_seconds'] = time.perf_counter()-began
    tau = residual_tolerance if isinstance(residual_tolerance, F) else F(str(residual_tolerance))
    solution_tau = solution_tolerance if isinstance(solution_tolerance, F) else F(str(solution_tolerance))
    alpha = F(1, 100)
    if tau != F(1, 100000) or solution_tau != F(1, 1000):
        raise ValueError('Frozen residual/solution tolerances must remain 1e-5/1e-3')
    source, candidate = np.asarray(source), np.asarray(candidate)
    if source.shape != candidate.shape:
        raise ValueError('Source/candidate shapes differ')
    began = time.perf_counter()
    derived = p.coefficient_margin(sigma_t, sigma_s, phase, weights); margin = derived
    if margin_lower is not None:
        supplied = margin_lower if isinstance(margin_lower, F) else _f(margin_lower)
        if supplied <= 0 or supplied > margin:
            raise ValueError('Supplied margin exceeds derived bound or is nonpositive')
        margin = supplied
    costs['coefficient_margin_seconds'] = time.perf_counter()-began
    began = time.perf_counter(); qlo, qhi = _norm(source); xlo, xhi = _norm(candidate)
    costs['source_candidate_norm_seconds'] = time.perf_counter()-began
    began = time.perf_counter()
    action = physical_action_ball(candidate, stream_symbol, sigma_t, sigma_s, phase, weights)
    costs['action_seconds'] = time.perf_counter()-began
    began = time.perf_counter()
    residual_center, subtraction_error = _recenter(_csub(p.point(source), p.point(action.center)))
    radius = _upper(_f(action.radius_upper)+_f(subtraction_error))
    residual_stage = dict(kind='source_action_subtraction', incoming_radius_upper=action.radius_upper,
                          local_radius_upper=subtraction_error, radius_upper=radius)
    costs['residual_center_and_radius_seconds'] = time.perf_counter()-began
    began = time.perf_counter(); clo, chi = _norm(residual_center)
    rlo = p._outward_float(max(F(0), _f(clo)-_f(radius)), False)
    rhi = _upper(_f(chi)+_f(radius))
    costs['residual_norm_seconds'] = time.perf_counter()-began
    began = time.perf_counter()
    pair_result = paired_field_check(candidate, baseline) if baseline is not None else {
        'paired_field_accepted': None, 'paired_field_relative_upper': None,
        'paired_field_tolerance_rational': str(F(1, 10000))}
    costs['paired_field_seconds'] = time.perf_counter()-began
    began = time.perf_counter()
    qlower, upper = _f(qlo), _f(rhi)
    denominator = margin*_f(xlo)-upper
    residual_ok = qlower > 0 and upper <= tau*qlower
    center_ok = qlower > 0 and _f(chi) <= (1-alpha)*tau*qlower
    radius_ok = qlower > 0 and _f(radius) <= alpha*tau*qlower
    solution_ok = denominator > 0 and upper <= solution_tau*denominator
    accuracy_ok = residual_ok and center_ok and radius_ok and solution_ok
    accepted = accuracy_ok and (baseline is None or pair_result['paired_field_accepted'])
    costs['exact_acceptance_seconds'] = time.perf_counter()-began
    began = time.perf_counter(); guard_after = p.native_cpu_guard()
    mask = 0xFFC0 if guard_before['architecture'] == 'x86-MXCSR' else ((1 << 24) | (3 << 22))
    if (guard_before['control_register'] ^ guard_after['control_register']) & mask:
        raise RuntimeError('CPU arithmetic control state changed during certificate')
    costs['final_guard_seconds'] = time.perf_counter()-began
    return dict(residual_norm_lower=rlo, residual_norm_upper=rhi,
        residual_center_norm_lower=clo, residual_center_norm_upper=chi,
        residual_center_norm_width=_upper(_f(chi)-_f(clo)),
        residual_radius_norm_lower=0., residual_radius_norm_upper=radius,
        source_norm_lower=qlo, source_norm_upper=qhi, source_norm_width=_upper(_f(qhi)-_f(qlo)),
        candidate_norm_lower=xlo, candidate_norm_upper=xhi,
        coercivity_lower=p._outward_float(margin, False),
        coercivity_lower_rational=str(margin), coercivity_lower_numerator=margin.numerator,
        coercivity_lower_denominator=margin.denominator, derived_coercivity_lower_rational=str(derived),
        policy_tolerance_rational=str(tau), policy_solution_tolerance_rational=str(solution_tau),
        reserve_fraction_rational=str(alpha), source_relative_norm_defined=qlo > 0,
        residual_center_representation='Represented complex128 center with certified Euclidean error radius',
        center_plus_radius_residual_norm_upper=rhi,
        relative_center_norm_upper=(_upper(_f(chi)/qlower) if qlo > 0 else None),
        relative_radius_norm_upper=(_upper(_f(radius)/qlower) if qlo > 0 else None),
        relative_residual_upper=(_upper(upper/qlower) if qlo > 0 else None),
        absolute_solution_error_upper=_upper(upper/margin),
        relative_solution_error_upper=(_upper(upper/denominator) if denominator > 0 else None),
        residual_accepted=residual_ok, reserve_center_accepted=center_ok,
        reserve_radius_accepted=radius_ok,
        fixed_one_percent_reserve_accepted=residual_ok and center_ok and radius_ok,
        solution_error_accepted=solution_ok, accuracy_accepted_without_paired_field=accuracy_ok,
        paired_field_gate_required=baseline is not None,
        paired_field_gate_evaluated=baseline is not None,
        paired_field_gate_established=baseline is not None and pair_result['paired_field_accepted'],
        accepted=accepted, **pair_result,
        certified_under_cpu_contract=True, cpu_contract=CPU_CONTRACT,
        native_backend_guard_before=guard_before, native_backend_guard_after=guard_after,
        global_gpu_contract_established=False,
        target='Exact nominal original captured D64 and stored collision/source/candidate bits',
        radius_propagation='Exact stage norm sqrt2; local point intervals and outward scalar norm bounds',
        stage_ledger=list(action.stages)+[residual_stage],
        costs=costs, elapsed_seconds=time.perf_counter()-started)
