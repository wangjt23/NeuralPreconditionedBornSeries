"""Independent local JSON/new-code audit of the single frozen beta2 screen.

Standard library only. No Torch, CUDA, SSH, data generation or training.
All timing repeats reduce within medium before grid means. Q1 is recorded
algorithm setup plus warm accepted latency, not process-cold latency.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st

STRONG = ('eager', 'context_full', 'context_segmented', 'direct_full', 'direct_segmented')
PRIOR = STRONG + ('tsa4', 'full_segmented', 'q3_segmented', 'full_raw', 'q3_raw')
METHODS = PRIOR + ('gain_two',)
PAIRS = (('eager', 'context_full'), ('eager', 'context_segmented'), ('eager', 'direct_full'),
         ('eager', 'direct_segmented'), ('full_segmented', 'full_raw'), ('q3_segmented', 'q3_raw'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


def audit(root, expected_policy_sha, expected_driver_sha):
    root = Path(root)
    load = lambda p: json.loads((root / p).read_text())
    policy, execution, stage = (load(p) for p in ('policy.json', 'execution_receipt.json', 'staging_receipt.json'))
    rows, checks, apps, setups, geometry, prefixes, settings = (load('results/' + n + '.json') for n in
        ('solver', 'checks', 'applications', 'setup', 'geometry', 'prefixes', 'settings'))
    errors = []
    def require(condition, explanation):
        if not condition:
            errors.append(explanation)
    contexts = {(n, i) for n in (128, 256) for i in (0, 3)}
    require(policy['phase'] == 'gain_two_screen', 'wrong phase')
    require(type(policy['fixed_gain']) is int and policy['fixed_gain'] == 2 and policy['gain_sweep'] is False,
            'policy does not forbid gain selection/sweep')
    require(policy['candidate'] == 'gain_two' and policy['prior_portfolio'] == list(PRIOR)
            and policy['strong_portfolio'] == list(STRONG), 'candidate/prior comparison family changed')
    require(policy['prior_negative_policy_sha256'] == '67e5cbdfaf734265f9b06ffb7f18cee75a6ab52c22cf8c5a46bce7cee0c30c48',
            'previous full-backend selection provenance changed')
    require(policy['grids'] == [128, 256] and policy['development_ids'] == [0, 3], 'wrong cohort')
    require(policy['candidate_backend_by_grid'] == {'128': 'raw', '256': 'direct_segmented'}, 'backend was not prescribed mapping')
    require(policy['timed_repeats'] == 3 and policy['warmup_repeats'] == 1, 'wrong repeat plan')
    require(policy['max_full_solves'] == 176 and policy['global_entry_checks'] == 280, 'wrong frozen counts')
    require(policy['max_truncated_prefix_solves'] == 4 and policy['max_geometry_targets'] == 8
            and policy['measured_applications'] == 220, 'frozen diagnostic/application envelope changed')
    require(policy['maxiter'] == 100 and policy['restart'] == 25 and policy['tolerance'] == 1e-5
            and policy['relative_error_bound_limit'] == 1e-3 and policy['paired_field_difference_limit'] == 1e-4,
            'numerical acceptance/solver settings changed')
    require(policy['soft_seconds'] == 95 and policy['hard_seconds'] == 120
            and policy['max_reserved_bytes'] == 12*1024**3, 'resource envelope changed')
    require(all(policy[k] is False for k in ('new_package','large_campaign','git_write')), 'unauthorized scope flag')
    require('Require the same Q1/warm/worst-regression gate under this stricter sensitivity for any superiority claim.'
            in policy['research_charge_rule'], 'stricter claim rule missing')
    require(sha(root / 'policy.json') == expected_policy_sha, 'policy hash mismatch')
    require(policy['code_sha256']['code/gain_two_screen.py'] == expected_driver_sha, 'driver identity mismatch')
    hashes = {n: dict(expected=h, actual=sha(root/n)) for n, h in policy['code_sha256'].items()}
    require(all(v['actual'] == v['expected'] for v in hashes.values()), 'new-code hash mismatch')
    require(settings['complete'] is True and settings['fixed_gain'] == 2
            and settings['candidate_backend_by_grid'] == policy['candidate_backend_by_grid'], 'completion/gain/backend mismatch')
    for flag in ('training', 'test9626_touched', 'original_source_exported'):
        require(policy.get(flag) is False and settings.get(flag) is False, 'unexpected ' + flag)
    keys = [(r['size'], r['index'], r['method'], r['repeat']) for r in rows]
    expected = {(n, i, m, rep) for n, i in contexts for m in METHODS for rep in (-1, 0, 1, 2)}
    require(len(rows) == 176 and len(set(keys)) == 176 and set(keys) == expected, 'full identity coverage mismatch')
    rm = dict(zip(keys, rows))
    require(sum(r['warmup'] for r in rows) == 44 and sum(not r['warmup'] for r in rows) == 132, 'warm/timed count mismatch')
    action = [c for c in checks if c.get('kind') != 'ownership_cache_view']
    guard = [c for c in checks if c.get('kind') == 'ownership_cache_view']
    actionkeys = [(c['size'], c['index'], c['method'], c['probe'], c['dtype']) for c in action]
    require(len(checks) == 280 and len(action) == 240 and len(guard) == 40, 'entry/bundle counts mismatch')
    require(len(set(actionkeys)) == 240 and set(actionkeys) ==
            {(n, i, m, p, d) for n, i in contexts for m in METHODS[1:] for p in ('source','grazing','mixed')
             for d in ('torch.complex64','torch.complex128')}, 'action identity mismatch')
    require({(c['size'],c['index'],c['method']) for c in guard} ==
            {(n,i,m) for n,i in contexts for m in METHODS[1:]}, 'guard identity mismatch')
    require(all(c['passed'] is True for c in checks), 'entry failure')
    require(all(c['bitwise_equal'] is True and c['finite'] is True for c in action), 'action parity/finite failure')
    require(all(all(c[k] is True for k in ('ownership','guard','beta_guard','restored')) for c in guard), 'guard/ownership failure')
    appkeys = [(a['size'],a['index'],a['method'],a['repeat']) for a in apps]
    require(len(apps) == 220 and len(set(appkeys)) == 220 and set(appkeys) ==
            {(n,i,m,r) for n,i in contexts for m in METHODS for r in range(5)}, 'application coverage mismatch')
    require(all(a['finite'] is True and finite(a['seconds']) and a['seconds'] >= 0 for a in apps), 'application finite/time failure')
    sm = {(s['size'],s['index']):s for s in setups}
    require(len(setups) == 4 and set(sm) == contexts, 'setup coverage mismatch')
    for s in setups:
        n = s['size']; depth = 8 if n == 128 else 16
        require(s['strong_depth'] == depth and s['fixed_gain'] == 2
                and s['candidate_backend'] == policy['candidate_backend_by_grid'][str(n)], 'setup depth/gain/backend mismatch')
        qa = s['start_source_parity']
        require(all(qa[k] is True for k in ('reference_cache_bitwise_equal','source_bitwise_equal','initialization_bitwise_equal','passed')),
                'same cache/source/start QA failure')
        for name, v in s['base_parameters'].items():
            require(v == dict(kind='tsa', depth=4 if name == 'tsa4' else depth, eta_scale=0., relaxation=1.), 'base recipe mismatch')
        expected_fft = {'context_full':dict(fftn=4*depth-1,ifftn=4*depth-1),
            'direct_full':dict(fftn=4*depth-1,ifftn=4*depth-1),
            'context_segmented':dict(fftn=4,ifftn=4),'direct_segmented':dict(fftn=4,ifftn=4),'tsa4':dict(fftn=4,ifftn=4),
            'full_raw':dict(fftn=17,ifftn=17,fft=0,ifft=0),'q3_raw':dict(fftn=17,ifftn=17,fft=1,ifft=1)}
        if n == 128:
            expected_fft['gain_two'] = dict(fftn=17,ifftn=17,fft=1,ifft=1)
        else:
            require(s['capture_fft_counts']['gain_two'] is None, 'segmented compound unexpectedly owns whole capture')
        require(all(s['capture_fft_counts'][m] == v for m,v in expected_fft.items()), 'capture FFT count mismatch')
        info = s['mix_info']['gain_two']
        require(info['beta'] == 2 and info['gain_tunable'] is False and info['weighting_before_one_reference_solve'] is True,
                'gain semantics declaration mismatch')
        require(info['application_fft_counts'] == dict(fftn=17,ifftn=17,fft=1,ifft=1)
                and info['q3_mask_cache_bytes'] == 48 and info['original_reference_cache_reused'] is True,
                'gain cache/FFT declaration mismatch')
    valid_rows = []
    for r in rows:
        n,i,m,rep = r['size'],r['index'],r['method'],r['repeat']
        require(r['warmup'] == (rep < 0), 'warmup flag mismatch')
        require(all(finite(r[k]) and r[k] >= 0 for k in ('seconds','acceptance_seconds','accepted_seconds','setup_seconds','setup_plus_accepted_seconds')),
                'invalid recorded duration')
        require(close(r['accepted_seconds'],r['seconds']+r['acceptance_seconds'])
                and close(r['setup_plus_accepted_seconds'],r['setup_seconds']+r['accepted_seconds']), 'cost arithmetic mismatch')
        s = sm[n,i]; base = ('tsa8' if n == 128 else 'tsa16') if m in STRONG else 'tsa4'
        expected_setup = s['physical_seconds'] + s['acceptance_setup_seconds'] + s['base_seconds'][base]
        if m != 'eager': expected_setup += s['constructor_seconds'][m]
        require(close(r['setup_seconds'],expected_setup), 'complete per-arm setup reconstruction mismatch')
        valid = (r['finite'] is True and r['status']['failed'] == 0
                 and finite(r['independent_true_residual']) and 0 <= r['independent_true_residual'] <= 1e-5
                 and finite(r['evaluated_error_bound']) and 0 <= r['evaluated_error_bound'] <= 1e-3
                 and finite(r['evaluated_margin']) and r['evaluated_margin'] > 0
                 and finite(r['field_difference']) and 0 <= r['field_difference'] <= 1e-4)
        valid_rows.append(valid); require(r['accuracy_passed'] == valid, 'accuracy flag disagrees with independent metrics')
        if m in STRONG: require(r['bitwise_vs_strong'] is True and r['field_difference'] == 0, 'strong field parity failure')
    pairs = 0
    for n,i in contexts:
        for rep in (-1,0,1,2):
            for a,b in PAIRS:
                ra,rb = rm[n,i,a,rep],rm[n,i,b,rep]
                require(ra['status'] == rb['status'] and sum(v['other'] == b and v['passed'] is True
                    for v in ra.get('same_recipe_backend_checks',[])) == 1, 'same-recipe field/status receipt mismatch')
                pairs += 1
    require(settings['all_accuracy_passed'] == all(valid_rows) and settings['all_entry_passed'] is True
            and settings['all_same_recipe_backend_checks_passed'] is True, 'settings accuracy/parity summary mismatch')
    require(settings['full_solves'] == 176 and settings['measured_full_solves'] == 132
            and settings['global_entry_checks'] == 280, 'settings full/entry counts mismatch')
    require(len(prefixes) == 4 and {(p['size'],p['index']) for p in prefixes} == contexts, 'prefix coverage mismatch')
    pm = {(p['size'],p['index']):p for p in prefixes}
    for p in prefixes:
        require(close(p['total_diagnostic_seconds'],p['seconds']+p['initialization_seconds']+p['independent_verification_seconds']), 'prefix charge mismatch')
        require(p['status']['iterations'] == 4 and p['status']['classical_calls'] == 4, 'prefix is not four classical steps')
    require(len(geometry) == 8 and {(g['size'],g['index'],g['context']) for g in geometry}
        == {(n,i,c) for n,i in contexts for c in ('source','prefix4')}, 'geometry coverage mismatch')
    for g in geometry:
        require(g['fixed_gain'] == 2 and g['actual_literal_bitwise_equal'] is True and g['physical_H_evaluated'] is True
                and g['oracle_used'] is False and g['range_optimality_claim'] is False, 'actual gain geometry/provenance mismatch')
        for k in ('composite_q3','composite_full','composite_gain_two'):
            d = g[k]
            require(all(finite(v) for v in d.values()), 'nonfinite geometry metric')
            if d['target_norm'] > 1e-20 and d['image_norm'] > 1e-20:
                require(abs(d['best_real_line_remaining_ratio']-math.sqrt(max(0.,1-d['real_cosine']**2))) < 1e-7,
                        'real-line scalar geometry identity mismatch')
        q,f,b = (g[k] for k in ('composite_q3','composite_full','composite_gain_two'))
        require(close(g['squared_best_line_reduction_vs_full'],f['best_real_line_remaining_ratio']**2-b['best_real_line_remaining_ratio']**2)
                and close(g['unit_squared_reduction_vs_full'],f['unit_remaining_ratio']**2-b['unit_remaining_ratio']**2), 'geometry gain arithmetic mismatch')
        require(close(g['prefix_acquisition_seconds'],pm[g['size'],g['index']]['total_diagnostic_seconds'] if g['context']=='prefix4' else 0.),
                'geometry prefix cost duplication mismatch')
    acquisition = {c:pm[c]['total_diagnostic_seconds']+sum(g['total_geometry_diagnostic_seconds'] for g in geometry
                    if (g['size'],g['index']) == c) for c in contexts}
    cases=[]
    for n,i in sorted(contexts):
        for m in METHODS:
            group=[rm[n,i,m,r] for r in (-1,0,1,2)];timed=group[1:]
            c=dict(size=n,index=i,method=m,eligible=all(r['accuracy_passed'] for r in group))
            for k in ('setup_seconds','seconds','acceptance_seconds','accepted_seconds','setup_plus_accepted_seconds'):
                c[k]=st.median(r[k] for r in timed)
            for k in ('iterations','classical_calls','injection_calls','injection_rejected'):
                c[k]=st.median(r['status'][k] for r in timed)
            c['application_seconds']=st.median(a['seconds'] for a in apps if (a['size'],a['index'],a['method'])==(n,i,m))
            cases.append(c)
    cm={(c['size'],c['index'],c['method']):c for c in cases};cost=[]
    for n in (128,256):
        details=[]
        for i in (0,3):
            prior=[cm[n,i,m] for m in PRIOR if cm[n,i,m]['eligible']]
            qb=min(prior,key=lambda c:(c['setup_plus_accepted_seconds'],c['method']))
            wb=min(prior,key=lambda c:(c['accepted_seconds'],c['method']));candidate=cm[n,i,'gain_two']
            details.append(dict(index=i,q1_baseline=qb['method'],warm_baseline=wb['method'],q1_baseline_seconds=qb['setup_plus_accepted_seconds'],
                warm_baseline_seconds=wb['accepted_seconds'],candidate=candidate,diagnostic_acquisition_seconds=acquisition[n,i],
                charged_q1_seconds=candidate['setup_plus_accepted_seconds']+acquisition[n,i]))
        qsave=1-st.mean(d['candidate']['setup_plus_accepted_seconds'] for d in details)/st.mean(d['q1_baseline_seconds'] for d in details)
        wsave=1-st.mean(d['candidate']['accepted_seconds'] for d in details)/st.mean(d['warm_baseline_seconds'] for d in details)
        charge=1-st.mean(d['charged_q1_seconds'] for d in details)/st.mean(d['q1_baseline_seconds'] for d in details)
        worst=max(d['candidate']['setup_plus_accepted_seconds']/d['q1_baseline_seconds']-1 for d in details)
        charged_worst=max(d['charged_q1_seconds']/d['q1_baseline_seconds']-1 for d in details)
        eligible=all(d['candidate']['eligible'] for d in details)
        cost.append(dict(size=n,fixed_backend=policy['candidate_backend_by_grid'][str(n)],cases=details,
            primary_q1_saving=qsave,warm_saving=wsave,worst_q1_regression=worst,
            research_charged_q1_saving=charge,research_charged_worst_q1_regression=charged_worst,
            primary_gate=eligible and qsave>=.1 and wsave>=.1 and worst<=.1,
            research_charged_gate=eligible and charge>=.1 and wsave>=.1 and charged_worst<=.1))
    geom=[]
    for n in (128,256):
        for context in ('source','prefix4'):
            group=[g for g in geometry if g['size']==n and g['context']==context]
            geom.append(dict(size=n,context=context,mean_squared_line_gain_vs_full=st.mean(g['squared_best_line_reduction_vs_full'] for g in group),
                mean_unit_squared_gain_vs_full=st.mean(g['unit_squared_reduction_vs_full'] for g in group)))
    require(settings['geometry_targets']==8 and settings['truncated_prefix_solves']==4, 'settings geometry/prefix count mismatch')
    require(execution['exit_code']==0 and execution['owned_process_timed_out'] is False and execution['policy_sha256']==expected_policy_sha,
            'execution completion/policy mismatch')
    require(execution['process_wall_seconds']<=policy['hard_seconds'] and settings['wall_seconds']<=policy['soft_seconds'], 'time limit exceeded')
    require(settings['peak_reserved_bytes']<=policy['max_reserved_bytes'], 'memory envelope exceeded')
    stageok=all(sha(root/n)==h for n,h in stage['code_sha256'].items())
    require(stageok and stage['cpu_tests_exit']==0, 'staging/preflight mismatch')
    return dict(protocol_passed=not errors,protocol_errors=errors,counts=dict(full=176,timed=132,warm=44,entries=280,
            actions=240,guard_bundles=40,applications=220,prefixes=4,geometry=8,same_recipe_full_pairs=pairs),
        all_accuracy_passed=all(valid_rows),accuracy_failures=sum(not v for v in valid_rows),full_solver_failures=sum(r['status']['failed']!=0 for r in rows),
        min_margin=min(r['evaluated_margin'] for r in rows),max_residual=max(r['independent_true_residual'] for r in rows if finite(r['independent_true_residual'])),
        max_evaluated_bound=max(r['evaluated_error_bound'] for r in rows if finite(r['evaluated_error_bound'])),
        max_field_difference=max(r['field_difference'] for r in rows if finite(r['field_difference'])),
        cost=cost,cases=cases,geometry_summary=geom,geometry=geometry,prefixes=prefixes,setup=setups,
        diagnostic_acquisition_seconds=sum(acquisition.values()),primary_both_grid_gate=all(c['primary_gate'] for c in cost),
        charged_both_grid_gate=all(c['research_charged_gate'] for c in cost),settings=settings,execution=execution,code_hashes=hashes,
        staging=dict(files_verified=stage['files_verified'],cpu_tests_exit=stage['cpu_tests_exit'],payload_count=len(stage['code_sha256']),all_payload_hashes_match=stageok),
        receipt_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.suffix!='.pyc'},
        scope=['Within-medium median then two-medium grid means; timing repeats are not independent media.',
               'Q1 and warm use independent successful prior-ten minima; these are comparison envelopes, not deployed selectors.',
               'Research sensitivity adds unique prefix and both target totals per context to candidate Q1 only, without amortization.',
               'Historical matched-v2 acquisition remains provenance; it is not silently made runtime acquisition or summed twice.',
               'Actual computed beta2 is physically evaluated; geometry is not a full-range optimum, coefficient teacher or learning result.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-policy-sha',required=True);p.add_argument('--expected-driver-sha',required=True)
    a=p.parse_args();result=audit(a.root,a.expected_policy_sha,a.expected_driver_sha)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('protocol_passed','protocol_errors','counts','all_accuracy_passed','full_solver_failures',
        'primary_both_grid_gate','charged_both_grid_gate','cost','geometry_summary')},indent=2))
    raise SystemExit(0 if result['protocol_passed'] else 1)
