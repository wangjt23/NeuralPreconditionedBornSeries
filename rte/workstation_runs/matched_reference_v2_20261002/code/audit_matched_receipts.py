"""Independent standard-library audit of the frozen matched-reference screen.

Read preserved local JSON/new-code receipts only. No Torch, CUDA, SSH, original
source export, data generation or training. Timings are setup + median warm
accepted latency, not process-cold latency. Numerical failures are retained and
excluded from the eligible portfolio; malformed protocol receipts are errors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics as st

STRONG = ('eager', 'context_full', 'context_segmented', 'direct_full', 'direct_segmented')
METHODS = STRONG + ('tsa4', 'full_segmented', 'q3_segmented', 'full_raw', 'q3_raw')
WRAPPERS = METHODS[1:]
PAIRS = (('eager', 'context_full'), ('eager', 'context_segmented'),
         ('eager', 'direct_full'), ('eager', 'direct_segmented'),
         ('full_segmented', 'full_raw'), ('q3_segmented', 'q3_raw'))
FROZEN_POLICY = '2e3014b6c5817dc2e619e176c453072397eff7d1aed029a04de814fa915f168c'
FROZEN_DRIVER = 'ce88460d07096c5135eeff4c50bf20f710b260a1990f9546940f83a8f4fdb8c5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(v):
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


def audit(root, expected_policy_sha=FROZEN_POLICY, expected_driver_sha=FROZEN_DRIVER,
          expected_phase='matched_reference_screen'):
    root = Path(root)
    read = lambda n: json.loads((root / 'results' / (n + '.json')).read_text())
    policy = json.loads((root / 'policy.json').read_text())
    rows, checks, setups, apps, prefixes, geometry, settings = [read(n) for n in
        ('solver', 'checks', 'setup', 'applications', 'prefixes', 'geometry', 'settings')]
    errors = []
    def require(value, explanation):
        if not value:
            errors.append(explanation)
    contexts = {(n, i) for n in (128, 256) for i in (0, 3)}
    require(policy['phase'] == expected_phase, 'wrong phase')
    require(policy['grids'] == [128, 256] and policy['development_ids'] == [0, 3], 'wrong development cohort')
    require(policy['timed_repeats'] == 3 and policy['warmup_repeats'] == 1, 'wrong repeat plan')
    require(sha(root / 'policy.json') == expected_policy_sha, 'frozen policy hash mismatch')
    require(policy['code_sha256']['code/matched_screen.py'] == expected_driver_sha, 'frozen driver identity mismatch')
    hashes = {}
    for name, expected in policy['code_sha256'].items():
        hashes[name] = dict(expected=expected, actual=sha(root / name))
        require(hashes[name]['actual'] == expected, 'code hash mismatch: ' + name)
    require(policy['max_full_solves'] == 160 and policy['global_entry_checks'] == 252,
            'frozen count limits mismatch')
    require(settings['complete'] is True, 'incomplete execution')
    for flag in ('training', 'test9626_touched', 'original_source_exported'):
        require(settings.get(flag) is False and policy.get(flag) is False, 'unexpected ' + flag)
    require(len(rows) == 160 and sum(not r['warmup'] for r in rows) == 120, 'full/timed row count mismatch')
    expected_rows = {(n, i, m, rep) for n, i in contexts for m in METHODS for rep in (-1, 0, 1, 2)}
    rowkeys = [(r['size'], r['index'], r['method'], r['repeat']) for r in rows]
    require(len(set(rowkeys)) == len(rows) and set(rowkeys) == expected_rows, 'full identity coverage mismatch')
    rowmap = dict(zip(rowkeys, rows))
    require(len(checks) == 252, 'global entry count mismatch')
    action_keys = [(c['size'], c['index'], c['method'], c.get('probe'), c.get('dtype'))
                   for c in checks if c.get('kind') != 'ownership_cache_view']
    expected_actions = {(n, i, m, p, d) for n, i in contexts for m in WRAPPERS
                        for p in ('source', 'grazing', 'mixed') for d in ('torch.complex64', 'torch.complex128')}
    require(len(action_keys) == 216 and set(action_keys) == expected_actions,
            'action identity coverage mismatch')
    guards = [c for c in checks if c.get('kind') == 'ownership_cache_view']
    require(len(guards) == 36 and {(c['size'], c['index'], c['method']) for c in guards}
            == {(n, i, m) for n, i in contexts for m in WRAPPERS}, 'guard coverage mismatch')
    require(all(c['passed'] is True for c in checks), 'entry failure')
    require(all(c.get('bitwise_equal') is True and c.get('finite') is True
                for c in checks if c.get('kind') != 'ownership_cache_view'), 'action parity/finite failure')
    require(all(c.get('ownership') is True and c.get('guard') is True and c.get('restored') is True
                for c in guards), 'ownership/cache guard failure')
    require(len(apps) == 200, 'measured application count mismatch')
    appkeys = [(a['size'], a['index'], a['method'], a['repeat']) for a in apps]
    require(len(set(appkeys)) == 200 and set(appkeys) ==
            {(n, i, m, r) for n, i in contexts for m in METHODS for r in range(5)}, 'application coverage mismatch')
    require(all(a['finite'] is True and finite(a['seconds']) and a['seconds'] >= 0 for a in apps), 'application finite/time failure')
    setupmap = {(s['size'], s['index']): s for s in setups}
    require(len(setups) == 4 and set(setupmap) == contexts, 'setup coverage mismatch')
    for s in setups:
        depth = 8 if s['size'] == 128 else 16
        require(s['strong_depth'] == depth, 'wrong strong depth')
        expected_capture = {'context_full': dict(fftn=4*depth-1, ifftn=4*depth-1),
            'context_segmented': dict(fftn=4, ifftn=4),
            'direct_full': dict(fftn=4*depth-1, ifftn=4*depth-1),
            'direct_segmented': dict(fftn=4, ifftn=4), 'tsa4': dict(fftn=4, ifftn=4),
            'full_raw': dict(fftn=17, ifftn=17, fft=0, ifft=0),
            'q3_raw': dict(fftn=17, ifftn=17, fft=1, ifft=1)}
        for method, expected in expected_capture.items():
            require(s['capture_fft_counts'][method] == expected, 'capture FFT count mismatch: ' + method)
        for method, info in s['mix_info'].items():
            q3 = method.startswith('q3')
            require(info['application_fft_counts'] == dict(fftn=17, ifftn=17, fft=int(q3), ifft=int(q3)),
                    'compound application FFT declaration mismatch')
            require(info['original_reference_cache_reused'] is True and info['original_dtype_defect_and_addition'] is True,
                    'compound cache/precision declaration mismatch')
            if expected_phase.endswith('_v2'):
                require(info['q3_mask_prepared_before_capture'] == q3 and info['q3_mask_cache_bytes'] == (48 if q3 else 0),
                        'cached-mask preparation declaration mismatch')
    numerical = []
    for r in rows:
        key = (r['size'], r['index'], r['method'], r['repeat'])
        n, i, m, rep = key
        require(r['warmup'] == (rep < 0), 'warmup flag mismatch: ' + str(key))
        numeric = ('seconds', 'acceptance_seconds', 'accepted_seconds', 'setup_seconds',
                   'setup_plus_accepted_seconds', 'independent_true_residual', 'evaluated_margin', 'field_difference')
        require(all(finite(r.get(k)) for k in numeric), 'nonfinite row metric: ' + str(key))
        require(close(r['accepted_seconds'], r['seconds'] + r['acceptance_seconds']), 'accepted-time sum mismatch')
        require(close(r['setup_plus_accepted_seconds'], r['setup_seconds'] + r['accepted_seconds']), 'Q1-time sum mismatch')
        s = setupmap[(n, i)]
        base_name = ('tsa8' if n == 128 else 'tsa16') if m in STRONG else 'tsa4'
        expected_setup = s['physical_seconds'] + s['acceptance_setup_seconds'] + s['base_seconds'][base_name]
        if m != 'eager':
            expected_setup += s['constructor_seconds'][m]
        require(close(r['setup_seconds'], expected_setup), 'per-arm setup reconstruction mismatch: ' + str(key))
        valid = (r['finite'] is True and r['status']['failed'] == 0
                 and finite(r['independent_true_residual']) and 0 <= r['independent_true_residual'] <= 1e-5
                 and finite(r.get('evaluated_error_bound')) and 0 <= r['evaluated_error_bound'] <= 1e-3
                 and finite(r['evaluated_margin']) and r['evaluated_margin'] > 0
                 and finite(r['field_difference']) and 0 <= r['field_difference'] <= 1e-4)
        require(r['accuracy_passed'] == valid, 'accuracy flag differs from metrics: ' + str(key))
        numerical.append(valid)
        if m in STRONG:
            require(r['bitwise_vs_strong'] is True and r['field_difference'] == 0, 'strong field parity failure')
    backend_checks = 0
    for n, i in contexts:
        for rep in (-1, 0, 1, 2):
            for a, b in PAIRS:
                ra, rb = rowmap[(n, i, a, rep)], rowmap[(n, i, b, rep)]
                require(ra['status'] == rb['status'], 'same-recipe status mismatch')
                flags = ra.get('same_recipe_backend_checks', [])
                require(sum(c['other'] == b and c['passed'] is True for c in flags) == 1,
                        'same-recipe stored field-parity receipt missing')
                backend_checks += 1
    require(settings['full_solves'] == len(rows) and settings['measured_full_solves'] == 120,
            'settings full count mismatch')
    require(settings['all_accuracy_passed'] == all(numerical), 'settings accuracy summary mismatch')
    require(settings['global_entry_checks'] == len(checks) and settings['all_entry_passed'] is True
            and settings['all_same_recipe_backend_checks_passed'] is True, 'settings parity summary mismatch')
    cases = []
    for n, i in sorted(contexts):
        for m in METHODS:
            group = [rowmap[(n, i, m, r)] for r in (-1, 0, 1, 2)]
            timed = group[1:]
            record = dict(size=n, index=i, method=m, eligible=all(r['accuracy_passed'] for r in group))
            for k in ('setup_seconds', 'seconds', 'acceptance_seconds', 'accepted_seconds', 'setup_plus_accepted_seconds'):
                record[k] = st.median(r[k] for r in timed)
            record['application_seconds'] = st.median(a['seconds'] for a in apps
                if (a['size'], a['index'], a['method']) == (n, i, m))
            record['iterations'] = st.median(r['status']['iterations'] for r in timed)
            record['classical_calls'] = st.median(r['status']['classical_calls'] for r in timed)
            record['injection_calls'] = st.median(r['status']['injection_calls'] for r in timed)
            cases.append(record)
    casemap = {(c['size'], c['index'], c['method']): c for c in cases}
    grid_cost_summary, conditional_budgets = [], []
    for n in (128, 256):
        original_q1, original_warm, future_q1, future_warm = [], [], [], []
        for i in (0, 3):
            orig = [casemap[(n, i, m)] for m in STRONG if casemap[(n, i, m)]['eligible']]
            portfolio = [casemap[(n, i, m)] for m in METHODS if casemap[(n, i, m)]['eligible']]
            original_q1.append(min(orig, key=lambda c: (c['setup_plus_accepted_seconds'], c['method'])))
            original_warm.append(min(orig, key=lambda c: (c['accepted_seconds'], c['method'])))
            future_q1.append(min(portfolio, key=lambda c: (c['setup_plus_accepted_seconds'], c['method'])))
            future_warm.append(min(portfolio, key=lambda c: (c['accepted_seconds'], c['method'])))
        methods = []
        for m in METHODS:
            cc = [casemap[(n, i, m)] for i in (0, 3)]
            q1 = st.mean(c['setup_plus_accepted_seconds'] for c in cc)
            warm = st.mean(c['accepted_seconds'] for c in cc)
            qsave = 1-q1/st.mean(c['setup_plus_accepted_seconds'] for c in original_q1)
            wsave = 1-warm/st.mean(c['accepted_seconds'] for c in original_warm)
            regress = max(c['setup_plus_accepted_seconds']/b['setup_plus_accepted_seconds']-1
                          for c, b in zip(cc, original_q1))
            methods.append(dict(method=m, mean_setup_ms=1000*st.mean(c['setup_seconds'] for c in cc),
                mean_q1_ms=1000*q1, mean_warm_ms=1000*warm, q1_saving=qsave, warm_saving=wsave,
                worst_individual_q1_regression=regress,
                gate=all(c['eligible'] for c in cc) and qsave >= .1 and wsave >= .1 and regress <= .1))
        grid_cost_summary.append(dict(size=n, methods=methods,
            future_best_q1=[dict(index=i, method=c['method'], ms=1000*c['setup_plus_accepted_seconds'])
                            for i, c in zip((0, 3), future_q1)],
            future_best_warm=[dict(index=i, method=c['method'], ms=1000*c['accepted_seconds'])
                              for i, c in zip((0, 3), future_warm)]))
        available = [m for m in ('full_segmented', 'full_raw')
                     if all(casemap[(n, i, m)]['eligible'] for i in (0, 3))]
        if available:
            chosen = min(available, key=lambda m: (st.mean(casemap[(n, i, m)]['setup_plus_accepted_seconds'] for i in (0, 3)), m))
            cc = [casemap[(n, i, chosen)] for i in (0, 3)]
            setup_mean, warm_mean = (st.mean(c[k] for c in cc) for k in ('setup_seconds', 'accepted_seconds'))
            q1best, warmbest = (st.mean(c[k] for c in values) for k, values in
                               (('setup_plus_accepted_seconds', future_q1), ('accepted_seconds', future_warm)))
            budget = min(.9*q1best-setup_mean, .9*warmbest)
            conditional_budgets.append(dict(size=n, candidate=chosen,
                mean_current_setup_ms=1000*setup_mean, mean_current_warm_ms=1000*warm_mean,
                mean_future_best_q1_ms=1000*q1best, mean_future_best_warm_ms=1000*warmbest,
                max_mean_warm_if_setup_unchanged_and_no_extra_acquisition_ms=1000*budget,
                required_mean_warm_reduction_ms=1000*(warm_mean-budget),
                required_mean_warm_reduction_fraction=1-budget/warm_mean,
                scope='Necessary aggregate10%Q1+warm arithmetic only under unchanged recorded setup, zero added acquisition and frozen all-ten comparator; individual-regression constraint remains. Neither attainable headroom nor a timing lower bound.'))
    gates = {}
    for rule in ('full', 'q3'):
        grids = {}
        for n in (128, 256):
            candidates = [rule + suffix for suffix in ('_segmented', '_raw')
                          if all(casemap[(n, i, rule + suffix)]['eligible'] for i in (0, 3))]
            if not candidates:
                grids[str(n)] = dict(selected=None, passed=False)
                continue
            selected = min(candidates, key=lambda m: (st.mean(casemap[(n, i, m)]['setup_plus_accepted_seconds'] for i in (0, 3)), m))
            detail = []
            for i in (0, 3):
                eligible = [casemap[(n, i, m)] for m in STRONG if casemap[(n, i, m)]['eligible']]
                require(bool(eligible), 'no accurate original comparator')
                cq = min(eligible, key=lambda c: (c['setup_plus_accepted_seconds'], c['method']))
                cw = min(eligible, key=lambda c: (c['accepted_seconds'], c['method']))
                candidate = casemap[(n, i, selected)]
                detail.append(dict(index=i, q1_baseline=cq['method'], warm_baseline=cw['method'],
                    q1_baseline_seconds=cq['setup_plus_accepted_seconds'], warm_baseline_seconds=cw['accepted_seconds'],
                    q1_candidate_seconds=candidate['setup_plus_accepted_seconds'], warm_candidate_seconds=candidate['accepted_seconds'],
                    q1_saving_fraction=1-candidate['setup_plus_accepted_seconds']/cq['setup_plus_accepted_seconds']))
            qs = 1-st.mean(d['q1_candidate_seconds'] for d in detail)/st.mean(d['q1_baseline_seconds'] for d in detail)
            ws = 1-st.mean(d['warm_candidate_seconds'] for d in detail)/st.mean(d['warm_baseline_seconds'] for d in detail)
            grids[str(n)] = dict(selected=selected, mean_q1_saving_fraction=qs, mean_warm_saving_fraction=ws,
                worst_individual_q1_saving_fraction=min(d['q1_saving_fraction'] for d in detail),
                passed=qs >= .1 and ws >= .1 and all(d['q1_saving_fraction'] >= -.1 for d in detail), cases=detail)
        gates[rule] = dict(grids=grids, passed_both_grids=all(g['passed'] for g in grids.values()))
    require(len(prefixes) == 4 and {(p['size'], p['index']) for p in prefixes} == contexts, 'prefix coverage mismatch')
    for p in prefixes:
        require(close(p['total_diagnostic_seconds'], p['seconds'] + p['initialization_seconds'] + p['independent_verification_seconds']),
                'prefix acquisition total mismatch')
        require(p['status']['iterations'] == 4 and p['status']['classical_calls'] == 4, 'prefix is not four classical steps')
        require(finite(p['prefix_gap']) and finite(p['real_AZ_gram_defect']), 'nonfinite prefix diagnostic')
    require(len(geometry) == 8 and {(g['size'], g['index'], g['context']) for g in geometry}
            == {(n, i, c) for n, i in contexts for c in ('source', 'prefix4')}, 'geometry coverage mismatch')
    signals = []
    for n in (128, 256):
        for context in ('source', 'prefix4'):
            group = [g for g in geometry if g['size'] == n and g['context'] == context]
            delta = st.mean(g['composite_q3']['best_real_line_remaining_ratio']**2
                            - g['composite_full']['best_real_line_remaining_ratio']**2 for g in group)
            signals.append(dict(size=n, context=context, mean_full_squared_line_gain=delta, predeclared_signal_passed=delta >= .02))
    for g in geometry:
        require(g['oracle_deployable'] is False and g['range_optimality_claim'] is False, 'oracle qualification missing')
        for label in ('composite_q3', 'composite_full', 'increment_q3', 'increment_full'):
            d = g[label]
            require(all(finite(v) for v in d.values()), 'nonfinite direction metric')
            if d['target_norm'] > 1e-20 and d['image_norm'] > 1e-20:
                predicted = math.sqrt(max(0., 1-d['real_cosine']**2))
                require(abs(predicted-d['best_real_line_remaining_ratio']) <= 1e-7, 'real-line scalar identity mismatch')
        span = g['two_composite_span_oracle']
        require(0 <= span['rank'] <= 2, 'invalid two-span rank')
        require(span['attained_relative_residual'] <= min(g['composite_q3']['best_real_line_remaining_ratio'],
                g['composite_full']['best_real_line_remaining_ratio']) + 1e-7, 'attained span worse than either contained direction')
    diagnostic_cost = sum(p['total_diagnostic_seconds'] for p in prefixes) + sum(g['total_geometry_diagnostic_seconds'] for g in geometry)
    diagnostic_cost += sum(next(g['homogeneous_diagnostic_matrix_seconds'] for g in geometry
                               if (g['size'], g['index']) == c) for c in contexts)
    for key, expected in (('geometry_targets', 8), ('truncated_prefix_solves', 4)):
        require(settings[key] == expected, 'settings ' + key + ' mismatch')
    require(settings['wall_seconds'] <= policy['soft_seconds'] + 1, 'soft runner bound exceeded')
    require(settings['peak_reserved_bytes'] <= policy['max_reserved_bytes'], 'memory bound exceeded')
    execution = json.loads((root / 'execution_receipt.json').read_text()) if (root / 'execution_receipt.json').exists() else None
    if execution is not None:
        require(execution['exit_code'] == 0 and execution['owned_process_timed_out'] is False, 'owned process failed/timed out')
        require(execution['process_wall_seconds'] <= policy['hard_seconds'], 'hard process bound exceeded')
        require(execution['policy_sha256'] == expected_policy_sha, 'execution policy mismatch')
    receipt_hashes = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file() and p.suffix != '.pyc'}
    stage = json.loads((root / 'staging_receipt.json').read_text())
    stage_hashes_ok = all(sha(root / n) == h for n, h in stage['code_sha256'].items())
    require(stage_hashes_ok and stage['cpu_tests_exit'] == 0, 'staging payload/preflight mismatch')
    staging_verification = dict(files_verified=stage['files_verified'], cpu_tests_exit=stage['cpu_tests_exit'],
        cpu_tests_seconds=stage['cpu_tests_seconds'], all_staged_payload_hashes_match=stage_hashes_ok,
        staged_payload_count=len(stage['code_sha256']))
    return dict(protocol_passed=not errors, protocol_errors=errors, source=str(root.resolve()),
        counts=dict(full=len(rows), timed=120, warmup=40, entries=len(checks), actions=len(action_keys),
                    guard_bundles=len(guards), measured_applications=len(apps), same_recipe_full_pairs=backend_checks,
                    prefixes=len(prefixes), geometry=len(geometry)),
        numerical=dict(all_accuracy_passed=all(numerical), accuracy_failures=sum(not v for v in numerical),
            solver_failures=sum(r['status']['failed'] != 0 for r in rows),
            min_margin=min(r['evaluated_margin'] for r in rows), max_true_residual=max(r['independent_true_residual'] for r in rows),
            max_evaluated_error_bound=max(r['evaluated_error_bound'] for r in rows if finite(r.get('evaluated_error_bound'))),
            max_field_difference=max(r['field_difference'] for r in rows)),
        cases=cases, gates=gates, geometry_signals=signals, geometry=geometry, prefixes=prefixes,
        independent_grid_cost_summary=grid_cost_summary, conditional_current_full_action_budget=conditional_budgets,
        staging_verification=staging_verification,
        diagnostic_acquisition_seconds=diagnostic_cost, setup=setups, settings=settings, execution=execution,
        code_hashes=hashes, receipt_sha256=receipt_hashes,
        caveats=['Within-medium median then mean across two media/grid; repeats are not independent media.',
                 'Q1 is recorded algorithm setup plus warm accepted time; source I/O/process startup and QA are excluded.',
                 'Gate warm denominator is the cheapest accurate original backend per metric, independently of Q1 selection.',
                 'Bitwise field equality is independently checked in frozen driver and preserved flags; fields are not exported.',
                 'Geometry is an attained real one/two-proposal diagnostic; it is not a full-range optimum or a learning result.',
                 'Diagnostic acquisition total sums unique prefix/H0 costs and per-target totals, not nested timing fields.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-policy-sha', default=FROZEN_POLICY)
    p.add_argument('--expected-driver-sha', default=FROZEN_DRIVER)
    p.add_argument('--expected-phase', default='matched_reference_screen')
    args = p.parse_args()
    result = audit(args.root, args.expected_policy_sha, args.expected_driver_sha, args.expected_phase)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('protocol_passed', 'protocol_errors', 'counts', 'numerical', 'gates', 'geometry_signals')}, indent=2))
    raise SystemExit(0 if result['protocol_passed'] else 1)
