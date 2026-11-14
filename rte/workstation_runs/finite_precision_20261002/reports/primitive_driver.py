"""Frozen finite-probe transport arithmetic diagnostic; no full-solver calls.

Stage beside error_model, exact_rational_rte and the unchanged direct-segmented
dependencies. Run from the original repository. Only scalar/hash JSON leaves
this process. No numerical reference below supplies a global backend contract.
"""
import time
_PROCESS_START = time.perf_counter()
import hashlib
import json
import math
import signal
import sys
from pathlib import Path

import numpy as np
import torch

from direct_segmented_tsa import DirectSegmentedTSA
from error_model import (conditional_action_bound, exact, outward_float,
                         prospective_budget, stored_scales)
from exact_rational_rte import apply_rte, exact_complex, norm_bounds
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
from rte.train_source_predictor import build_base, material_operator, source_family


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def host_array(value):
    return value.detach().cpu().numpy().copy()


def operator_arrays(low, high):
    return dict(
        directions=host_array(low.directions).astype(np.float64),
        weights=host_array(low.weights).astype(np.float64),
        phase=host_array(low.phase_matrix).astype(np.float64),
        kx=host_array(low.kx).reshape(-1).astype(np.float64),
        ky=host_array(low.ky).reshape(-1).astype(np.float64),
        t=host_array(low.sigma_t).astype(np.float64),
        s=host_array(low.sigma_s).astype(np.float64),
        D32=host_array(low._stream_symbol(1, torch.complex64,
                                        low.source.device)).astype(np.complex128),
        D64=host_array(high._stream_symbol(1, torch.complex128,
                                          high.source.device)).astype(np.complex128))


def scales_for(a):
    return stored_scales(a['directions'].tolist(), a['kx'].tolist(),
                         a['ky'].tolist(), a['t'].reshape(-1).tolist(),
                         a['s'].reshape(-1).tolist(), a['phase'].tolist(),
                         a['weights'].tolist())


def numpy_action(x, symbol, a):
    """Independent, separately rounded NumPy128 fixed-symbol implementation."""
    x = np.asarray(x, dtype=np.complex128)
    spectrum = np.fft.fftn(x, axes=(-2, -1), norm='backward')
    streaming = np.fft.ifftn(symbol * spectrum, axes=(-2, -1), norm='backward')
    weighted = a['weights'][None, :, None, None] * x
    scatter = np.einsum('lm,bmhw->blhw', a['phase'], weighted, optimize=False)
    return streaming + a['t'] * x - a['s'] * scatter


def diagnostics(got, reference, xnorm, operator_upper):
    assert got.shape == reference.shape, 'Action/reference output shape mismatch'
    finite = bool(np.isfinite(got).all() and np.isfinite(reference).all())
    error = float(np.linalg.norm((got-reference).reshape(-1))) if finite else None
    image = float(np.linalg.norm(reference.reshape(-1))) if finite else None
    valid = finite and math.isfinite(xnorm) and xnorm > 0
    return dict(finite=finite, absolute_error=error, reference_image_norm=image,
                absolute_error_per_input_norm=error/xnorm if valid else None,
                error_over_Lambda_input_norm=error/(operator_upper*xnorm)
                if valid and operator_upper > 0 else None,
                image_relative_error=error/image if valid and image > 0 else None,
                image_cancellation_condition_indicator=operator_upper*xnorm/image
                if valid and image > 0 else None,
                zero_denominator=(not valid or image == 0),
                norm_arithmetic='Ordinary NumPy binary64 diagnostic; no certificate')


def exact_difference(a, b):
    assert len(a) == len(b) and len(a) > 0, 'Exact-field angular count mismatch'
    assert all(len(p) == 4 and all(len(r) == 4 for r in p)
               for p in list(a)+list(b)), 'Exact fields must have shape Lx4x4'
    return [[[(exact_complex(x)[0]-exact_complex(y)[0],
               exact_complex(x)[1]-exact_complex(y)[1])
              for x, y in zip(ar, br)]
             for ar, br in zip(ap, bp)] for ap, bp in zip(a, b)]


def toy_diagnostics(got, reference, field, operator_upper):
    # The stored GPU bits go directly into Fraction subtraction; the exact
    # oracle is NEVER converted to a floating-point field before comparison.
    assert got.shape == (len(field), 4, 4), 'Exact-fixture output shape mismatch'
    assert len(reference) == len(field), 'Exact-fixture reference/input angular mismatch'
    assert bool(np.isfinite(got).all()), 'Nonfinite exact-fixture comparison'
    lo, hi = norm_bounds(exact_difference(got.tolist(), reference))
    xlo, xhi = norm_bounds(field)
    return dict(finite=bool(np.isfinite(got).all()), exact_error_norm_lower=lo,
                exact_error_norm_upper=hi, exact_input_norm_lower=xlo,
                exact_input_norm_upper=xhi,
                absolute_error_per_input_norm_upper=
                outward_float(exact(hi)/exact(xlo), True) if xlo > 0 else None,
                error_over_Lambda_input_norm_upper=
                outward_float(exact(hi)/(exact(operator_upper)*exact(xlo)), True)
                if xlo > 0 and operator_upper > 0 else None,
                exact_target_scope='This fixed 4x4 stored-symbol fixture only')


def toy_fields(angles):
    dense = np.empty((1, angles, 4, 4), dtype=np.complex128)
    for a in range(angles):
        for x in range(4):
            for y in range(4):
                dense[0, a, x, y] = (((a+1)*(x+2)-3*y)/8
                                      + 1j*(2*x+(a+2)*y-5)/16)
    nyquist = np.zeros_like(dense)
    nyquist[0, 0] = np.array([1., -1., 1., -1.])[:, None]
    sparse = np.zeros_like(dense)
    sparse[0, 0, 1, 2] = .25-.125j
    sparse[0, 1, 3, 0] = -.5+.25j
    sparse[0, angles-1, 0, 3] = .125+.0625j
    return [('dense_dyadic', dense), ('x_nyquist_angle0', nyquist),
            ('sparse_dyadic', sparse)]


@torch.no_grad()
def run():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent.parent
    policy = json.loads((root/'policy.json').read_text())
    assert policy['phase'] == 'finite_precision_primitives_v1'
    assert policy['grids'] == [128, 256] and policy['development_ids'] == [0, 3]
    assert policy['development_seed'] == 4626 and policy['max_full_solves'] == 0
    assert policy['full_grid_probe_names'] == [
        'source', 'mixed', 'x_nyquist_angle0', 'unit_complex_random',
        'M_source', 'M_random']
    for name, digest in policy['code_sha256'].items():
        assert sha(root/name) == digest, name
    out = root/'results'
    out.mkdir(exist_ok=False)
    rows, toys, contexts, checks = [], [], [], []
    counters = dict(full_grid_gpu_H_actions=0, fixture_gpu_H_actions=0,
                    preconditioner_actions=0, numpy_fp64_references=0,
                    exact_fixture_cases=0, exact_fixture_apply_calls=0,
                    full_solves=0)
    costs = dict(parent_load_seconds=0., physical_setup_seconds=0.,
                 base_setup_seconds=0., capture_setup_seconds=0.,
                 source_api_setup_seconds=0., probe_setup_seconds=0.,
                 explicit_gpu_action_seconds=0., transfers_seconds=0.,
                 numpy_reference_seconds=0., exact_reference_seconds=0.,
                 exact_scalar_accounting_seconds=0., scalar_model_seconds=0.,
                 fixture_setup_seconds=0., fixture_model_seconds=0.,
                 coefficient_transfer_seconds=0., json_save_seconds=0.)
    peak_allocated = peak_reserved = 0
    result = dict(complete=False, diagnostic_only=True, certified=False,
                  global_backend_contract_established=False,
                  old_entry_failure_preserved=True, full_solves=0,
                  training=False, test9626_touched=False,
                  original_source_exported=False)
    deadline = _PROCESS_START + policy['soft_seconds']

    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        return value

    def track():
        nonlocal peak_allocated, peak_reserved
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())

    def limit():
        track()
        if time.perf_counter() > deadline:
            raise TimeoutError('Frozen primitive60s soft budget')
        if peak_reserved > policy['max_reserved_bytes']:
            raise MemoryError('Frozen primitive4GiB reserved-memory budget')

    def save():
        began = time.perf_counter()
        result.update(counters=counters, costs=costs,
                      owned_process_seconds=time.perf_counter()-_PROCESS_START,
                      peak_allocated_bytes=peak_allocated,
                      peak_reserved_bytes=peak_reserved)
        for name, data in [('actions', rows), ('exact_fixture', toys),
                           ('contexts', contexts), ('checks', checks),
                           ('settings', result)]:
            temp = out/(name+'.json.partial')
            temp.write_text(json.dumps(safe(data), indent=2, allow_nan=False)+'\n')
            temp.replace(out/(name+'.json'))
        costs['json_save_seconds'] += time.perf_counter()-began

    def timed_gpu(action, x, counter):
        limit(); torch.cuda.synchronize(); began = time.perf_counter()
        y = action(x)
        counters[counter] += 1
        torch.cuda.synchronize()
        seconds = time.perf_counter()-began
        costs['explicit_gpu_action_seconds'] += seconds
        limit()
        return y, seconds

    def transfer(x):
        began = time.perf_counter()
        y = host_array(x)
        costs['transfers_seconds'] += time.perf_counter()-began
        limit()
        return y

    def alarm(_signum, _frame):
        raise TimeoutError('Frozen primitive75s owned-process hard budget')

    previous_alarm = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL,
                     max(.001, policy['hard_seconds']-(time.perf_counter()-_PROCESS_START)))
    try:
        assert torch.cuda.is_available()
        result.update(torch_version=torch.__version__, numpy_version=np.__version__,
                      cuda_version=torch.version.cuda,
                      device_name=torch.cuda.get_device_name(),
                      source_api_file_sha256=sha('rte/train_source_predictor.py'),
                      source_api_import_scope='Existing source_family/material_operator/build_base APIs; transitive solver imports permitted; zero full-solver calls',
                      import_and_policy_seconds=time.perf_counter()-_PROCESS_START)
        cuda_matmul = torch.backends.cuda.matmul
        result['backend_flags_read_only'] = dict(
            cuda_matmul_allow_tf32=cuda_matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            **{name: getattr(cuda_matmul, name) for name in (
                'allow_fp16_reduced_precision_reduction',
                'allow_bf16_reduced_precision_reduction')
               if hasattr(cuda_matmul, name)})
        result['conditional_normal_range_assumptions'] = (
            'Standard arithmetic models require normal-range, no overflow/underflow/FTZ, '
            'appropriate componentwise dot/product/addition contracts and uniform FFT '
            'normwise contracts. Finite outputs or these finite probes do not verify '
            'those global assumptions; no flags are changed.')
        for n in policy['grids']:
            limit(); began = time.perf_counter()
            parent = Path(f'rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{n}.pt')
            assert sha(parent) == policy['development_parent_sha256'][str(n)]
            data = torch.load(parent, map_location='cpu', weights_only=True)
            required = ('sigma_t', 'sigma_s', 'centers', 'phases', 'amplitudes',
                        'angular_codes', 'geometry_base64')
            assert all(k in data for k in required)
            assert data['sigma_t'].shape == data['sigma_s'].shape
            assert data['sigma_t'].shape[0] == 16 and data['sigma_t'].shape[-2:] == (n, n)
            cfg, regenerated = source_family(n, 16, 4626)
            for key in required:
                assert torch.equal(data[key], regenerated[key]), key
            costs['parent_load_seconds'] += time.perf_counter()-began
            for index in policy['development_ids']:
                limit(); torch.cuda.synchronize(); began = time.perf_counter()
                ids = torch.tensor([index])
                low, high = material_operator(cfg, data, ids, .1, 'cuda')
                assert low.float_dtype == torch.float32 and high.float_dtype == torch.float64
                torch.cuda.synchronize(); physical = time.perf_counter()-began
                costs['physical_setup_seconds'] += physical
                depth = 8 if n == 128 else 16
                assert policy['strong_factors'][str(n)] == f'tsa{depth}'
                began = time.perf_counter()
                base = build_base(low, f'tsa{depth}')
                torch.cuda.synchronize(); base_seconds = time.perf_counter()-began
                costs['base_setup_seconds'] += base_seconds
                began = time.perf_counter()
                wrapper = DirectSegmentedTSA(base, low.source, chunk=1,
                                             capture_first=False)
                torch.cuda.synchronize(); capture = time.perf_counter()-began
                costs['capture_setup_seconds'] += capture
                # Only these two production M evaluations acquire the two
                # additional diagnostic fields. Capture constructor's original
                # stage warmups are setup and not hidden in this action count.
                began = time.perf_counter()
                gen = torch.Generator(device='cuda').manual_seed(91731+n+index)
                random = torch.complex(torch.randn(low.source.shape, device='cuda', generator=gen),
                                       torch.randn(low.source.shape, device='cuda', generator=gen))
                random = random/random.norm()
                mixed = low.source.roll(1, dims=-2)*(1.13+.21j)
                nyquist = torch.zeros_like(low.source)
                nyquist[:, 0] = ((-1.)**torch.arange(n, device='cuda'))[None, :, None]
                torch.cuda.synchronize()
                costs['probe_setup_seconds'] += time.perf_counter()-began
                msource, ms_seconds = timed_gpu(wrapper, low.source, 'preconditioner_actions')
                saved_source = msource.clone()
                mrandom, mr_seconds = timed_gpu(wrapper, random, 'preconditioner_actions')
                ownership = bool(torch.equal(msource, saved_source))
                source_parity = bool(torch.equal(high.source, low.source.to(torch.complex128)))
                checks.append(dict(size=n, index=index,
                                   M_source_output_owned=ownership,
                                   physical_source_exact_promotion=source_parity,
                                   native_angular_count=low.n_angles,
                                   passed=ownership and source_parity))
                assert ownership and source_parity
                began = time.perf_counter(); a = operator_arrays(low, high)
                coefficient_transfer = time.perf_counter()-began
                costs['coefficient_transfer_seconds'] += coefficient_transfer
                began = time.perf_counter(); scales = scales_for(a)
                budget = prospective_budget(scales, low.source.numel(), 1e-5, .01)
                conditional_floor32 = conditional_action_bound(scales, 32, 0., 0.)
                conditional_floor64 = conditional_action_bound(scales, 64, 0., 0.)
                costs['scalar_model_seconds'] += time.perf_counter()-began
                symbol_delta = float(np.max(np.abs(a['D32']-a['D64'])))
                context = dict(size=n, index=index, development_seed=4626,
                    parent_sha256=policy['development_parent_sha256'][str(n)],
                    angular_count=low.n_angles, physical_seconds=physical,
                    base_seconds=base_seconds, capture_seconds=capture,
                    coefficient_transfer_seconds=coefficient_transfer,
                    M_source_seconds=ms_seconds, M_random_seconds=mr_seconds,
                    source_sha256=array_sha(a_source := transfer(low.source)),
                    assembled_D32_sha256=array_sha(a['D32']),
                    assembled_D64_sha256=array_sha(a['D64']),
                    stored_D32_D64_max_absolute_difference=symbol_delta,
                    strong_name=f'tsa{depth}', strong_relaxation=1.,
                    capture_fft_counts=wrapper.capture_fft_counts,
                    constructor_internal_warm_stages_counted_as_setup=True,
                    scales=scales, prospective_outer_error_budget=budget,
                    hypothetical_zero_fft_error_model32=conditional_floor32,
                    hypothetical_zero_fft_error_model64=conditional_floor64,
                    hypothetical_models_are_not_backend_bounds=True)
                contexts.append(context); save()
                probes = [('source', low.source), ('mixed', mixed),
                          ('x_nyquist_angle0', nyquist),
                          ('unit_complex_random', random),
                          ('M_source', msource), ('M_random', mrandom)]
                for label, field in probes:
                    limit()
                    original = field.to(torch.complex128)
                    x64 = original.to(torch.complex64)
                    xshadow = x64.to(torch.complex128)
                    xo, xc = transfer(original), transfer(xshadow)
                    xnorm, castnorm = (float(np.linalg.norm(v.reshape(-1)))
                                       for v in (xo, xo-xc))
                    assert xnorm > 0 and math.isfinite(xnorm)
                    cast_relative = castnorm/xnorm
                    began = time.perf_counter()
                    ref32 = numpy_action(xc, a['D32'], a)
                    ref64 = numpy_action(xo, a['D64'], a)
                    costs['numpy_reference_seconds'] += time.perf_counter()-began
                    counters['numpy_fp64_references'] += 2
                    images = []
                    for action_name, op, inp, ref in (
                        ('original_low_complex64', low.A_op, x64, ref32),
                        ('original_low_native_complex128_cast_shadow', low.A_op, xshadow, ref32),
                        ('physical_high_complex128', high.A_op, original, ref64)):
                        got, seconds = timed_gpu(op, inp, 'full_grid_gpu_H_actions')
                        image = transfer(got).astype(np.complex128)
                        images.append(image)
                        row = dict(size=n, index=index, probe=label,
                            action=action_name, seconds=seconds,
                            original_input_dtype=str(field.dtype),
                            actual_input_dtype=str(inp.dtype),
                            actual_input_sha256=array_sha(transfer(inp)),
                            original_wide_input_sha256=array_sha(xo),
                            actual_cast_shadow_sha256=array_sha(xc),
                            input_norm=xnorm, input_cast_norm=castnorm,
                            input_cast_relative=cast_relative,
                            physical_input_cast_allowance_per_original_norm=
                            scales['operator_norm_upper']*cast_relative,
                            action_rounding_vs_matched_nominal=diagnostics(
                                image, ref, float(np.linalg.norm((xo if action_name == 'physical_high_complex128' else xc).reshape(-1))),
                                scales['operator_norm_upper']),
                            overall_vs_original_physical_target=diagnostics(
                                image, ref64, xnorm, scales['operator_norm_upper']),
                            D32_D64_symbol_allowance_per_original_norm=
                            symbol_delta*float(np.linalg.norm(xc.reshape(-1)))/xnorm,
                            native_low128_reference_input='actual cast64 promoted exactly to128; shared field with low64'
                            if 'native' in action_name else None,
                            reference_kind='Separately rounded NumPy128 fixed actual assembled symbol',
                            certified=False)
                        rows.append(row)
                        assert row['action_rounding_vs_matched_nominal']['finite']
                        assert row['overall_vs_original_physical_target']['finite']
                    row = rows[-3]
                    row['low32_minus_native128_same_input'] = diagnostics(
                        images[0], images[1], float(np.linalg.norm(xc.reshape(-1))),
                        scales['operator_norm_upper'])
                    row['nominal_D_and_cast_combined_observed'] = diagnostics(
                        ref32, ref64, xnorm, scales['operator_norm_upper'])
                    save()
                # Separate 4x4 fixture with unit-square signed modes, copied
                # actual angular buffers and quarter-grid sampled materials.
                limit(); began = time.perf_counter()
                toy_low = PeriodicRTE2D(shape=(4, 4), dx=.25,
                    n_angles=low.n_angles, anisotropy_g=low.anisotropy_g,
                    eta_scale=low.eta_scale, fdtype=torch.float32, device='cuda')
                for name in ('directions', 'weights', 'phase_matrix'):
                    setattr(toy_low, name, getattr(low, name).clone())
                sample = torch.tensor([0, n//4, n//2, 3*n//4], device='cuda')
                toy_t = low.sigma_t.index_select(-2, sample).index_select(-1, sample)
                toy_s = low.sigma_s.index_select(-2, sample).index_select(-1, sample)
                toy_q = torch.zeros((1, low.n_angles, 4, 4), dtype=torch.complex64, device='cuda')
                toy_low.setup(toy_t, toy_s, toy_q)
                toy_high = high_precision_copy(toy_low)
                torch.cuda.synchronize()
                costs['fixture_setup_seconds'] += time.perf_counter()-began
                began = time.perf_counter(); ta = operator_arrays(toy_low, toy_high)
                toy_coefficient_transfer = time.perf_counter()-began
                costs['coefficient_transfer_seconds'] += toy_coefficient_transfer
                began = time.perf_counter()
                ts = scales_for(ta)
                fixture_model = time.perf_counter()-began
                costs['fixture_model_seconds'] += fixture_model
                kwargs = dict(directions=ta['directions'].tolist(),
                    kx=np.broadcast_to(ta['kx'][:, None], (4, 4)).tolist(),
                    ky=np.broadcast_to(ta['ky'][None, :], (4, 4)).tolist(),
                    t=ta['t'][0, 0].tolist(), s=ta['s'][0, 0].tolist(),
                    phase=ta['phase'].tolist(), weights=ta['weights'].tolist())
                for label, field in toy_fields(low.n_angles):
                    limit()
                    original = torch.as_tensor(field, device='cuda', dtype=torch.complex128)
                    cast = original.to(torch.complex64)
                    cast_field = transfer(cast).astype(np.complex128)
                    # Frozen dyadic probes are deliberately exactly C64
                    # representable. This fixture does not test cast rounding.
                    assert np.array_equal(cast_field, field)
                    exact_field = field[0].tolist()
                    references = {}
                    began = time.perf_counter()
                    for symbol_name in ('D32', 'D64'):
                        limit()
                        references[symbol_name] = apply_rte(
                            exact_field, stream_symbol=ta[symbol_name][0].tolist(), **kwargs)
                        counters['exact_fixture_apply_calls'] += 1
                    costs['exact_reference_seconds'] += time.perf_counter()-began
                    rec = dict(size=n, index=index, probe=label,
                        fixture_shape=[4, 4], fixture_dx=.25,
                        fixture_signed_modes=[0, 1, -2, -1],
                        original_material_sample_indices=[0, n//4, n//2, 3*n//4],
                        angular_count=low.n_angles,
                        coefficient_transfer_seconds=toy_coefficient_transfer,
                        fixture_model_seconds=fixture_model,
                        input_sha256=array_sha(field), input_cast_norm_exact_zero=True,
                        D32_sha256=array_sha(ta['D32']), D64_sha256=array_sha(ta['D64']),
                        exact_nominal_symbol_difference=toy_diagnostics(
                            np.zeros_like(field[0]),
                            exact_difference(references['D32'], references['D64']),
                            exact_field, ts['operator_norm_upper']),
                        scopes='Exact fixed stored-symbol fixture; no full-grid or uniform-backend inference',
                        actions=[])
                    for name, op, inp, target in (
                        ('original_low_complex64', toy_low.A_op, cast, 'D32'),
                        ('physical_high_complex128', toy_high.A_op, original, 'D64')):
                        got, seconds = timed_gpu(op, inp, 'fixture_gpu_H_actions')
                        image = transfer(got)[0]
                        assert bool(np.isfinite(image).all()), 'Nonfinite exact-fixture GPU output'
                        began = time.perf_counter()
                        matched = toy_diagnostics(image, references[target], exact_field,
                                                  ts['operator_norm_upper'])
                        physical_target = toy_diagnostics(image, references['D64'], exact_field,
                                                          ts['operator_norm_upper'])
                        costs['exact_scalar_accounting_seconds'] += time.perf_counter()-began
                        rec['actions'].append(dict(action=name, actual_input_dtype=str(inp.dtype),
                            seconds=seconds, matched_exact_nominal_error=matched,
                            overall_exact_physical_target_error=physical_target))
                    counters['exact_fixture_cases'] += 1
                    toys.append(rec); save()
                # Every next context is constructed independently; no allocator,
                # CUDA setting, or original source/cache is modified or cleaned.
                del wrapper, base, low, high, toy_low, toy_high, probes, a, ta
        expected = {k: policy[k] for k in (
            'full_grid_gpu_H_actions', 'fixture_gpu_H_actions',
            'preconditioner_actions', 'numpy_fp64_references', 'exact_fixture_cases')}
        assert all(counters[k] == v for k, v in expected.items()), counters
        assert counters['exact_fixture_apply_calls'] == policy['exact_reference_actions']
        assert counters['full_solves'] == 0
        limit()
        result.update(complete=True, scope='Four sealed DEV contexts; six full-grid fields and three separate dyadic4x4 fixtures each; no full solves',
            exact_case_count_convention='12 paired fixture cases; 24 exact apply_rte evaluations, one per actual D32/D64 target',
            native_low128_input_convention='Actual cast64 field promoted exactly to128, isolating kernel precision without input-cast confounding',
            implicit_constructor_H_work_separately_billed_as_capture_setup=True,
            result_gate='No result-dependent admission or threshold; certified new admission remains NO-GO without global contracts')
        save()
        print(json.dumps(safe(result), allow_nan=False), flush=True)
        return 0
    except BaseException as error:
        result.update(complete=False, error_type=type(error).__name__,
                      error_message=str(error)[:1000])
        save()
        print(json.dumps(safe(result), allow_nan=False), flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_alarm)


if __name__ == '__main__':
    sys.exit(run())
