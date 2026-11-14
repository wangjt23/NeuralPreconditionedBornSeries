"""Independent scalar/hash receipt audit; stdlib only, never runs an RTE action."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import re

EXPECTED_POLICY = '426fe29b012c7e275a1587e315f0f2f23efe2989f63cf00b13a2a02869f67d08'
EXPECTED_DRIVER = 'd7149407412392cbdbdf548af907ee4348d242e709c6f882b39036f2462027c7'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def binary(v):
    return F.from_float(float(v))


def gamma(k, u):
    return k*u/(1-k*u)


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, list):
        return all(finite_tree(x) for x in value)
    return True


def outward_upper(recorded, exact_value):
    assert binary(recorded) >= exact_value
    assert binary(math.nextafter(recorded, -math.inf)) < exact_value


def conditional_components(s, precision, eta=0.):
    # Independently propagate multiplicative errors as a rational product.
    u = F(1, 2**(24 if precision == 32 else 53))
    d, a, tm = map(binary, (s['streaming_norm_upper'],
                           s['streaming_absolute_terms_upper'], s['sigma_t_max']))
    sb = binary(s['sigma_s_max'])*binary(s['scattering_norm_upper'])
    delta = gamma(2, u)*a
    if precision == 32:
        delta += gamma(2, F(1, 2**53))*a
    e = binary(eta)
    streaming = delta+(d+delta)*((1+e)*(1+u)*(1+e)-1)
    collision = u*tm
    scatter = gamma(8*s['angular_count']+4, u)*sb
    addition = gamma(2, u)*(d+streaming+tm+collision+sb+scatter)
    return dict(total=streaming+collision+scatter+addition, streaming=streaming,
                symbol_formation=delta, collision=collision,
                scattering=scatter, final_additions=addition)


def summary(rows, metric, branch):
    result = {}
    for action in sorted({r['action'] for r in rows}):
        subset = [r for r in rows if r['action'] == action]
        best = max(subset, key=lambda r: r[branch][metric])
        result[action] = dict(count=len(subset), minimum=min(r[branch][metric] for r in subset),
                              maximum=best[branch][metric],
                              witness=dict(size=best['size'], index=best['index'], probe=best['probe']))
    return result


def audit(root):
    load = lambda name: json.loads((root/name).read_text())
    policy = load('policy.json'); stage = load('staging_receipt.json')
    execution = load('execution_receipt.json'); settings = load('results/settings.json')
    contexts = load('results/contexts.json'); rows = load('results/actions.json')
    toys = load('results/exact_fixture.json'); checks = load('results/checks.json')
    assert digest(root/'policy.json') == EXPECTED_POLICY
    assert digest(root/'code/primitive_driver.py') == EXPECTED_DRIVER
    assert execution['policy_sha256'] == EXPECTED_POLICY
    assert execution['independent_final_review_before_execution']['primitive_driver.py'] == EXPECTED_DRIVER
    sealed = policy['code_sha256']
    for name, h in sealed.items():
        assert digest(root/name) == h, name
    for name, h in stage['code_sha256'].items():
        assert digest(root/name) == h, name
    provenance_anomalies = []
    local_review = root.parent.parent/'deliverables/finite_precision_20261002/PRIMITIVE_IMPLEMENTATION_REVIEW.md'
    pin_path = root.parent/'finite_final_review_before_execution.json'
    final_local_review_verified = False
    for name, h in execution['independent_final_review_before_execution'].items():
        p = root/name if name == 'policy.json' else root/('code' if name.endswith('.py') else 'notes')/name
        actual = digest(p)
        if actual != h:
            # Code/policy mismatches are fatal. The final implementation review
            # was pinned locally after staging; preserve the older staged note.
            assert name == 'PRIMITIVE_IMPLEMENTATION_REVIEW.md'
            assert stage['code_sha256']['notes/'+name] == actual
            assert digest(local_review) == h
            assert json.loads(pin_path.read_text()) == execution['independent_final_review_before_execution']
            final_local_review_verified = True
            provenance_anomalies.append(dict(name=name,execution_header_sha256=h,
                actual_and_staging_sha256=actual,
                final_local_review_path=str(local_review), final_local_review_sha256=digest(local_review),
                before_execution_pin_path=str(pin_path), before_execution_pin_sha256=digest(pin_path),
                scope='Final local d714 implementation review pinned in execution header; older d830 staged note retained. Distinct location/version, no code/policy mismatch.'))
    assert policy['phase'] == 'finite_precision_primitives_v1'
    assert policy['grids'] == [128, 256] and policy['development_ids'] == [0, 3]
    assert policy['development_seed'] == 4626
    assert policy['original_acceptance'] == dict(relative_residual=1e-5,
        relative_solution_error_bound=1e-3, paired_field_difference=1e-4)
    assert stage['workspace'] == '/data/wjt/NeuralPreconditionedBornSeries'
    assert stage['preserved_files_verified'] == len(stage['existing_sha256']) == 458
    assert stage['cpu_tests_exit_codes'] == [0, 0, 0]
    assert execution['exit_code'] == 0 and not execution['owned_process_timed_out']
    assert settings['complete'] and settings['diagnostic_only']
    assert not settings['certified'] and not settings['global_backend_contract_established']
    assert settings['old_entry_failure_preserved']
    for key in ('full_solves', 'training', 'test9626_touched', 'original_source_exported'):
        assert not execution[key] and not settings[key]
    assert all(finite_tree(load(p.relative_to(root))) for p in root.rglob('*.json'))
    assert execution['process_wall_seconds'] < policy['soft_seconds'] < policy['hard_seconds']
    assert settings['owned_process_seconds'] < execution['process_wall_seconds']
    assert settings['peak_allocated_bytes'] <= settings['peak_reserved_bytes'] < policy['max_reserved_bytes']

    expected_contexts = {(n, i) for n in (128, 256) for i in (0, 3)}
    assert {(c['size'], c['index']) for c in contexts} == expected_contexts and len(contexts) == 4
    cmap = {(c['size'], c['index']): c for c in contexts}
    assert len(checks) == 4 and all(c['passed'] and c['M_source_output_owned']
        and c['physical_source_exact_promotion'] and c['native_angular_count'] == 12 for c in checks)
    actions = {'original_low_complex64', 'original_low_native_complex128_cast_shadow', 'physical_high_complex128'}
    assert len(rows) == 72
    assert {(r['size'], r['index'], r['probe'], r['action']) for r in rows} == {
        (n, i, probe, action) for n, i in expected_contexts
        for probe in policy['full_grid_probe_names'] for action in actions}
    for r in rows:
        c = cmap[r['size'], r['index']]; L = c['scales']['operator_norm_upper']; norm = r['input_norm']
        assert norm > 0 and r['seconds'] >= 0 and not r['certified']
        assert r['input_cast_norm'] == r['input_cast_relative'] == r['physical_input_cast_allowance_per_original_norm'] == 0
        assert r['original_wide_input_sha256'] == r['actual_cast_shadow_sha256']
        dtype = 'torch.complex64' if r['action'] == 'original_low_complex64' else 'torch.complex128'
        assert r['actual_input_dtype'] == dtype
        if dtype == 'torch.complex128':
            assert r['actual_input_sha256'] == r['actual_cast_shadow_sha256']
        if r['probe'] == 'source' and dtype == 'torch.complex64':
            assert r['actual_input_sha256'] == c['source_sha256']
        for branch in ('action_rounding_vs_matched_nominal', 'overall_vs_original_physical_target'):
            d = r[branch]; err = d['absolute_error']; image = d['reference_image_norm']
            assert d['finite'] and err >= 0 and image > 0 and not d['zero_denominator']
            assert d['absolute_error_per_input_norm'] == err/norm
            assert d['error_over_Lambda_input_norm'] == err/(L*norm)
            assert d['image_relative_error'] == err/image
            assert d['image_cancellation_condition_indicator'] == L*norm/image
        if r['action'] == 'physical_high_complex128':
            assert r['action_rounding_vs_matched_nominal'] == r['overall_vs_original_physical_target']
        assert r['D32_D64_symbol_allowance_per_original_norm'] == c['stored_D32_D64_max_absolute_difference']*norm/norm

    budget_checks = []
    for c in contexts:
        n, i = c['size'], c['index']; s = c['scales']; b = c['prospective_outer_error_budget']
        assert c['development_seed'] == 4626 and c['parent_sha256'] == policy['development_parent_sha256'][str(n)]
        assert c['strong_name'] == policy['strong_factors'][str(n)] and c['strong_relaxation'] == 1.
        assert c['capture_fft_counts'] == dict(fftn=4, ifftn=4)
        assert c['constructor_internal_warm_stages_counted_as_setup']
        assert s['coercivity_lower'] > 0 and s['angular_count'] == c['angular_count'] == 12
        assert c['hypothetical_models_are_not_backend_bounds']
        assert not b['global_backend_contract_established'] and not b['empirical_reference_checks_can_certify_this_contract']
        assert b['field_complex_count'] == 12*n*n and b['tolerance'] == 1e-5 and b['allocation_fraction'] == .01
        mu = F(1, 2**53); tau = F(1, 100000); rho = F(99, 100)*tau
        nu = gamma(4*b['field_complex_count']+16, mu)
        m, L = binary(s['coercivity_lower']), binary(s['operator_norm_upper'])
        rho_upper = rho*(1+nu)/(1-nu)
        expected_allowed = (m*(tau-rho_upper-mu)/(1+tau)-mu*L)/(1+mu)
        allowed = binary(b['allowed_outer_absolute_error_per_field_norm'])
        assert 0 < allowed <= expected_allowed
        assert binary(math.nextafter(b['allowed_outer_absolute_error_per_field_norm'], math.inf)) > expected_allowed
        outward_upper(b['assumed_norm_relative_contract'], nu)
        assert binary(b['hypothetical_measured_residual_ceiling']) <= rho
        for precision in (32, 64):
            record = c[f'hypothetical_zero_fft_error_model{precision}']
            assert record['conditional'] and not record['certified_backend_contract']
            for key, value in conditional_components(s, precision).items():
                outward_upper(record[key], value)
        eta = b['equal_forward_inverse_fft64_contract_required']
        assert eta > 0 and conditional_components(s, 64, eta)['total'] <= allowed
        eps = allowed; a = mu; beta = eps+mu*(L+eps)
        worst = a+rho_upper+beta*(1+a+rho_upper)/(m-beta)
        assert worst <= tau
        high = [r for r in rows if (r['size'], r['index'], r['action']) == (n, i, 'physical_high_complex128')]
        gap = max(r['action_rounding_vs_matched_nominal']['absolute_error_per_input_norm'] for r in high)
        budget_checks.append(dict(size=n, index=i,coercivity_lower=s['coercivity_lower'],
            operator_norm_upper=s['operator_norm_upper'], allowed_outer_absolute_error_per_field_norm=float(allowed),
            required_equal_fft_contract=eta, empirical_high_numpy_gap_per_input_norm_max=gap,
            empirical_gap_over_conditional_budget=gap/float(allowed),
            exact_scalar_budget_reproduced=True, certified_backend_contract=False))

    assert len(toys) == 12
    assert {(t['size'],t['index'],t['probe']) for t in toys} == {
        (n,i,p) for n,i in expected_contexts for p in ('dense_dyadic','x_nyquist_angle0','sparse_dyadic')}
    toy_actions = []
    for t in toys:
        assert t['fixture_shape'] == [4,4] and t['fixture_dx'] == .25
        assert t['fixture_signed_modes'] == [0,1,-2,-1] and t['angular_count'] == 12
        n = t['size']; assert t['original_material_sample_indices'] == [0,n//4,n//2,3*n//4]
        assert t['input_cast_norm_exact_zero'] and len(t['actions']) == 2
        assert {a['action'] for a in t['actions']} == {'original_low_complex64','physical_high_complex128'}
        for a in t['actions']:
            assert a['seconds'] >= 0
            for branch in ('matched_exact_nominal_error','overall_exact_physical_target_error'):
                d = a[branch]
                assert d['finite'] and 0 <= d['exact_error_norm_lower'] <= d['exact_error_norm_upper']
                assert 0 < d['exact_input_norm_lower'] <= d['exact_input_norm_upper']
                outward_upper(d['absolute_error_per_input_norm_upper'],
                    binary(d['exact_error_norm_upper'])/binary(d['exact_input_norm_lower']))
            toy_actions.append(dict(size=t['size'],index=t['index'],probe=t['probe'],**a))

    reconstructed = dict(full_grid_gpu_H_actions=len(rows),fixture_gpu_H_actions=len(toy_actions),
        preconditioner_actions=2*len(contexts),numpy_fp64_references=2*len(contexts)*6,
        exact_fixture_cases=len(toys),exact_fixture_apply_calls=2*len(toys),full_solves=0)
    assert settings['counters'] == reconstructed
    costs = settings['costs']
    for key, field in [('physical_setup_seconds','physical_seconds'),('base_setup_seconds','base_seconds'),
                       ('capture_setup_seconds','capture_seconds')]:
        assert math.isclose(costs[key],sum(c[field] for c in contexts), rel_tol=1e-14)
    direct_seconds = sum(r['seconds'] for r in rows)+sum(r['seconds'] for r in toy_actions)
    factor_seconds = sum(c['M_source_seconds']+c['M_random_seconds'] for c in contexts)
    assert math.isclose(costs['explicit_gpu_action_seconds'],direct_seconds+factor_seconds,rel_tol=1e-14)
    cpu_counts = [int(x) for x in re.findall(r'Ran (\d+) tests?', stage['cpu_tests_output'])]
    assert cpu_counts and all(x>0 for x in cpu_counts)
    maxima = {metric:summary(rows,metric,'action_rounding_vs_matched_nominal') for metric in (
        'absolute_error_per_input_norm','error_over_Lambda_input_norm','image_relative_error','image_cancellation_condition_indicator')}
    return dict(date='2026-10-02', audit_scope='Independent stdlib scalar/hash/count recomputation; no RTE action, field or solve rerun',
        protocol_verified=True,receipt_root=str(root),policy_sha256=digest(root/'policy.json'),
        driver_sha256=digest(root/'code/primitive_driver.py'),sealed_code_hashes_verified=len(sealed),
        all_staging_payload_hashes_verified=len(stage['code_sha256']),before_execution_preservation_receipt_count=458,
        execution_review_header_all_hashes_match=True,
        execution_review_header_all_staged_note_hashes_match=not provenance_anomalies,
        final_local_implementation_review_and_pin_verified=final_local_review_verified,
        provenance_metadata_anomalies=provenance_anomalies,
        remote_preservation_independently_reexecuted=False,cpu_test_counts=cpu_counts,cpu_test_total=sum(cpu_counts),
        counters=reconstructed,all_four_ownership_and_source_checks=True,all_scalars_finite=True,
        all_input_cast_ledgers_zero=True,full_grid_matched_nominal=maxima,
        full_grid_overall_physical=summary(rows,'absolute_error_per_input_norm','overall_vs_original_physical_target'),
        exact_toy_matched=summary(toy_actions,'absolute_error_per_input_norm_upper','matched_exact_nominal_error'),
        exact_toy_overall=summary(toy_actions,'absolute_error_per_input_norm_upper','overall_exact_physical_target_error'),
        budgets=budget_checks,budget_scalar_arithmetic_reproduced=True,
        cost=dict(process_wall_seconds=execution['process_wall_seconds'],settings_clock_seconds=settings['owned_process_seconds'],
            component_seconds=costs,component_timer_sum_seconds=sum(costs.values()),
            process_minus_component_sum_seconds=execution['process_wall_seconds']-sum(costs.values()),
            direct_H_seconds=direct_seconds,factor_application_seconds=factor_seconds,
            peak_allocated_bytes=settings['peak_allocated_bytes'],peak_reserved_bytes=settings['peak_reserved_bytes'],
            json_save_fraction_of_process=costs['json_save_seconds']/execution['process_wall_seconds'],
            timer_scope='Component scopes are partial; context/toy duplicate timing fields must not be summed again. Settings clock precedes final receipt flush.'),
        original_acceptance_unchanged=True,old_entry_failure_preserved=True,diagnostic_only=True,
        certified_backend_contract=False,certified_solver_admission=False,full_solver_accuracy_evaluated=False,
        full_solver_performance_evaluated=False,training=False,test9626_touched=False,
        limits=['NumPy reference uncertainty is not bounded; observed full-grid discrepancies do not certify absolute H error.',
                'Exact-toy bracket arithmetic and code are audited; raw scalar receipts do not contain GPU bits/coefficient fields for rerunning the exact discrepancy.',
                'Toy normalized-Lambda ratios lack the toy-Lambda scalar in raw receipts and were not independently rederived; primary per-input-norm ratios were rederived.',
                'Stored coefficient lower/norm estimates are audited via source/model and sealed hashes, not reexecuted from unexported original coefficients.',
                'Finite probes do not establish uniform FFT/angular/norm contracts or cover unknown future Arnoldi coefficients.',
                'All six full-grid inputs originateC64; zero cast ledgers do not cover genuinely nonrepresentableC128 inputs.',
                '96 direct H probes exclude additional H work inside8M applications and capture construction; these remain costed.'],
        receipt_sha256={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()})


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('receipt_root',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit(args.receipt_root)
    compare_path=args.receipt_root.parent.parent/'deliverables/finite_precision_20261002/RESULT_ANALYSIS.json'
    if compare_path.is_file():
        # Compare only after constructing/verifying our independent raw summary.
        other=json.loads(compare_path.read_text())
        raw_rows=json.loads((args.receipt_root/'results/actions.json').read_text())
        assert other['counts']==result['counters']
        for c in other['contexts']:
            for a in c['actions']:
                subset=[r for r in raw_rows if (r['size'],r['index'],r['action'])==(c['size'],c['index'],a['action'])]
                assert a['records']==len(subset)
                for target,metric in [('max_observed_absolute_error_per_input_norm','absolute_error_per_input_norm'),
                    ('max_observed_image_relative_error','image_relative_error'),
                    ('max_observed_error_over_Lambda_input_norm','error_over_Lambda_input_norm')]:
                    assert a[target]==max(r['action_rounding_vs_matched_nominal'][metric] for r in subset)
                assert a['max_observed_overall_physical_error_per_input_norm']==max(r['overall_vs_original_physical_target']['absolute_error_per_input_norm']for r in subset)
        for a in other['exact_fixture_summary']:
            assert a['max_exact_fixture_error_per_input_norm_upper']==result['exact_toy_matched'][a['action']]['maximum']
            assert a['max_exact_fixture_physical_target_error_per_input_norm_upper']==result['exact_toy_overall'][a['action']]['maximum']
        assert other['process_wall_seconds']==result['cost']['process_wall_seconds']
        assert other['peak_reserved_bytes']==result['cost']['peak_reserved_bytes']
        assert other['costs']==result['cost']['component_seconds']
        result['root_summary_arithmetic_agrees']=True
        result['compared_root_summary_path']=str(compare_path)
        result['compared_root_summary_sha256']=digest(compare_path)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(protocol_verified=result['protocol_verified'],counters=result['counters'],
        policy_sha256=result['policy_sha256'],driver_sha256=result['driver_sha256'],
        process_seconds=result['cost']['process_wall_seconds'],output=str(args.output)),indent=2))
