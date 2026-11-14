"""Independent standard-library analysis of frozen matched-screen receipts."""
import argparse, collections, hashlib, json, math, pathlib, statistics

p = argparse.ArgumentParser()
p.add_argument('root', type=pathlib.Path)
p.add_argument('destination', type=pathlib.Path)
a = p.parse_args()
load = lambda name: json.loads((a.root / name).read_text())
policy = load('policy.json')
execution = load('execution_receipt.json')
settings = load('results/settings.json')
rows = load('results/solver.json')
checks = load('results/checks.json')
geometry = load('results/geometry.json')
prefixes = load('results/prefixes.json')
apps = load('results/applications.json')
setups = load('results/setup.json')
assert execution['exit_code'] == 0 and not execution['owned_process_timed_out']
assert settings['complete'] and len(rows) == 160 and len(checks) == 252
assert len(geometry) == 8 and len(prefixes) == 4 and len(apps) == 200 and len(setups) == 4
assert len({(r['size'], r['index'], r['method'], r['repeat']) for r in rows}) == len(rows)
assert sum(r['warmup'] for r in rows) == 40
assert all(c['passed'] for c in checks)
assert settings['peak_reserved_bytes'] <= policy['max_reserved_bytes']
assert settings['wall_seconds'] <= policy['soft_seconds']
assert execution['policy_sha256'] == hashlib.sha256((a.root/'policy.json').read_bytes()).hexdigest()
hashes = {}
for rel, expected in policy['code_sha256'].items():
    actual = hashlib.sha256((a.root/rel).read_bytes()).hexdigest()
    assert actual == expected, rel
    hashes[rel] = actual
groups = collections.defaultdict(list)
for r in rows:
    assert r['warmup'] == (r['repeat'] == -1)
    assert math.isclose(r['accepted_seconds'], r['seconds'] + r['acceptance_seconds'], rel_tol=1e-12)
    assert math.isclose(r['setup_plus_accepted_seconds'], r['setup_seconds'] + r['accepted_seconds'], rel_tol=1e-12)
    passed = r['finite'] and r['status']['failed'] == 0 and r['independent_true_residual'] <= 1e-5 and r['evaluated_error_bound'] is not None and math.isfinite(r['evaluated_error_bound']) and 0 <= r['evaluated_error_bound'] <= 1e-3 and r['field_difference'] <= 1e-4
    assert passed == r['accuracy_passed']
    groups[r['size'], r['index'], r['method']].append(r)
parity = [c for r in rows for c in r.get('same_recipe_backend_checks', [])]
assert len(parity) == 96 and all(c['passed'] for c in parity)
contexts = collections.defaultdict(dict)
for (n, index, method), rr in groups.items():
    assert len(rr) == 4 and {r['repeat'] for r in rr} == {-1,0,1,2}
    timed = [r for r in rr if not r['warmup']]
    assert len({r['setup_seconds'] for r in rr}) == 1
    context = dict(method=method, eligible=all(r['accuracy_passed'] for r in rr),
        setup_ms=rr[0]['setup_seconds']*1000,
        warm_accepted_ms=statistics.median(r['accepted_seconds'] for r in timed)*1000,
        q1_ms=statistics.median(r['setup_plus_accepted_seconds'] for r in timed)*1000,
        warm_measured_ms=[r['accepted_seconds']*1000 for r in timed],
        timed_status=timed[0]['status'],
        max_residual=max(r['independent_true_residual'] for r in rr),
        max_evaluated_bound=max(r['evaluated_error_bound'] or 0 for r in rr),
        max_paired_field_difference=max(r['field_difference'] for r in rr),
        application_ms=statistics.median(r['seconds'] for r in apps if r['size']==n and r['index']==index and r['method']==method)*1000)
    contexts[n,index][method] = context
cost = []
for n in policy['grids']:
    cc = [contexts[n,i] for i in policy['development_ids']]
    original_q1 = [min((v for k,v in c.items() if k in policy['strong_portfolio'] and v['eligible']), key=lambda v:(v['q1_ms'],v['method'])) for c in cc]
    original_warm = [min((v for k,v in c.items() if k in policy['strong_portfolio'] and v['eligible']), key=lambda v:(v['warm_accepted_ms'],v['method'])) for c in cc]
    selected = {}
    for kind in ('full','q3'):
        available = [m for m in (kind+'_segmented',kind+'_raw') if all(c[m]['eligible'] for c in cc)]
        selected[kind] = min(available, key=lambda m:(statistics.mean(c[m]['q1_ms'] for c in cc),m)) if available else None
    summaries = {}
    for method in policy['strong_portfolio'] + policy['control_portfolio']:
        values = [c[method] for c in cc]
        q1 = statistics.mean(v['q1_ms'] for v in values)
        warm = statistics.mean(v['warm_accepted_ms'] for v in values)
        q1_saving = 1-q1/statistics.mean(v['q1_ms'] for v in original_q1)
        warm_saving = 1-warm/statistics.mean(v['warm_accepted_ms'] for v in original_warm)
        regress = max(v['q1_ms']/b['q1_ms']-1 for v,b in zip(values,original_q1))
        summaries[method] = dict(eligible=all(v['eligible'] for v in values),mean_q1_ms=q1,mean_warm_accepted_ms=warm,
            mean_setup_ms=statistics.mean(v['setup_ms'] for v in values),q1_saving_fraction=q1_saving,
            warm_saving_fraction=warm_saving,worst_case_q1_regression_fraction=regress,
            grid_gate_passed=all(v['eligible'] for v in values) and q1_saving>=.10 and warm_saving>=.10 and regress<=.10)
    matched = {}
    for backend in ('segmented','raw'):
        full, q3 = summaries['full_'+backend], summaries['q3_'+backend]
        matched[backend] = dict(full_vs_q3_q1_saving_fraction=1-full['mean_q1_ms']/q3['mean_q1_ms'],
             full_vs_q3_warm_saving_fraction=1-full['mean_warm_accepted_ms']/q3['mean_warm_accepted_ms'])
    cost.append(dict(size=n,case_ids=policy['development_ids'],original_q1_comparator=[v['method'] for v in original_q1],
        original_warm_comparator=[v['method'] for v in original_warm],
        mean_original_best_q1_ms=statistics.mean(v['q1_ms'] for v in original_q1),
        mean_original_best_warm_ms=statistics.mean(v['warm_accepted_ms'] for v in original_warm),
        selected_backend_by_rule=selected,method_summary=summaries,matched_backend_pair=matched,
        cases=[dict(index=i,methods=contexts[n,i],future_classical_best_q1=min((v for v in contexts[n,i].values() if v['eligible']),key=lambda v:(v['q1_ms'],v['method']))['method']) for i in policy['development_ids']]))
geom = []
for n in policy['grids']:
    for label in ('source','prefix4'):
        gg = [g for g in geometry if g['size']==n and g['context']==label]
        assert len(gg) == 2
        improvement = statistics.mean(g['composite_q3']['best_real_line_remaining_ratio']**2-g['composite_full']['best_real_line_remaining_ratio']**2 for g in gg)
        geom.append(dict(size=n,context=label,mean_squared_best_line_residual_reduction=improvement,predeclared_signal_passed=improvement>=.02,
            q3_mean_best_line_remaining=statistics.mean(g['composite_q3']['best_real_line_remaining_ratio'] for g in gg),
            full_mean_best_line_remaining=statistics.mean(g['composite_full']['best_real_line_remaining_ratio'] for g in gg),
            attained_two_proposal_mean_remaining=statistics.mean(g['two_composite_span_oracle']['attained_relative_residual'] for g in gg),
            mean_discarded_input_angle_fraction=statistics.mean(g['discarded_input_angle_fraction'] for g in gg),
            mean_q3_higher_output_angle_fraction=statistics.mean(g['q3_higher_output_angle_fraction'] for g in gg)))
diagnostic_seconds = sum(p['total_diagnostic_seconds'] for p in prefixes) + sum(s for (n,i),s in {(g['size'],g['index']):g['homogeneous_diagnostic_matrix_seconds'] for g in geometry}.items()) + sum(g['total_geometry_diagnostic_seconds'] for g in geometry)
result = dict(complete=True,policy_sha256=execution['policy_sha256'],code_sha256=hashes,
    rows=160,timed_rows=120,warmup_rows=40,checks=252,same_recipe_backend_checks=96,geometry_targets=8,prefixes=4,measured_applications=200,
    all_accuracy_passed=all(r['accuracy_passed'] for r in rows),failed_rows=[{k:r[k] for k in ('size','index','method','repeat','status','accuracy_passed')} for r in rows if not r['accuracy_passed']],
    max_true_residual=max(r['independent_true_residual'] for r in rows),max_evaluated_relative_bound=max(r['evaluated_error_bound'] or 0 for r in rows),max_paired_field_difference=max(r['field_difference'] for r in rows),
    runner_seconds=settings['wall_seconds'],process_seconds=execution['process_wall_seconds'],peak_reserved_bytes=settings['peak_reserved_bytes'],peak_allocated_bytes=settings['peak_allocated_bytes'],
    separate_diagnostic_total_seconds=diagnostic_seconds,max_prefix_recurrence_gap=max(p['prefix_gap'] for p in prefixes),max_real_AZ_gram_defect=max(p['real_AZ_gram_defect'] for p in prefixes),
    max_low_vs_high_defect_gap=max(g['low_vs_high_defect_relative'] for g in geometry),max_homogeneous_full_closure=max(g['full_homogeneous_closure_relative'] for g in geometry),max_correction_additivity=max(g['correction_additivity_relative'] for g in geometry),max_physical_image_additivity=max(g['physical_image_additivity_relative'] for g in geometry),
    cost=cost,geometry=geom,any_single_fixed_arm_passes_both_grids=any(all(next(c for c in cost if c['size']==n)['method_summary'][m]['grid_gate_passed'] for n in policy['grids']) for m in policy['control_portfolio']),
    selected_full_passes_both_grids=all(c['selected_backend_by_rule']['full'] is not None and c['method_summary'][c['selected_backend_by_rule']['full']]['grid_gate_passed'] for c in cost),
    selected_q3_passes_both_grids=all(c['selected_backend_by_rule']['q3'] is not None and c['method_summary'][c['selected_backend_by_rule']['q3']]['grid_gate_passed'] for c in cost),
    cost_scope=policy['cost_scope'],diagnostic_scope='Prefix including initialization and independent verification plus homogeneous matrix once per context plus total geometry per target. Nested acquire/fit timers are not summed again.',
    comparator_rule='Independent per-case eligible original-five Q1 and warm minima. All successful ten arms form the future denominator.',
    learned_claim=False,range_optimum_claim=False,training=False,test9626_touched=False,original_source_exported=False)
a.destination.parent.mkdir(parents=True,exist_ok=True)
a.destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('code_sha256','cost')},indent=2))
