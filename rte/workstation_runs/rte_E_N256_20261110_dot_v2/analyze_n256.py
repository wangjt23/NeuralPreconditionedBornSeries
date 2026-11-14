"""Audit saved N256 receipts only. No torch or solver execution."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
d=json.loads((P/'gate_01/results.json').read_text());watch=json.loads((P/'WATCHDOG.json').read_text())
assert watch['attempt']==2 and watch['prior_failed_attempt_controller_seconds']==375.6764374298509
assert abs(watch['cumulative_controller_seconds']-watch['elapsed_seconds']-375.6764374298509)<1e-9
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert d['source_hashes']==d['source_hashes_after'] and d['E_source_hashes']==d['E_source_hashes_after']
assert d['plan_sha256']==digest(P/'FROZEN_PLAN.md')=='3be997b61d1f65d10ca7c254ff7f2b3d752cfc7c5694513a5e7244f22b75ea3f'
assert all(digest(P/f)==h for f,h in d['own_sources'].items())
assert d['source_ids']==[0,4,7,14] and d['grid']==256 and d['modes']==list(range(-8,9))
assert d['inner_tolerance']==1e-4 and d['inner_maxiter']==8 and d['outer_tolerance']==1e-5 and d['outer_maxiter']==100
assert d['block_target']==1e-10 and d['block_max_corrections']==3
assert not any(d[k] for k in ['heldout9626','training','new_teachers'])
qual=[d['qualification']]+[b['qualification'] for b in d['batches'] if 'qualification' in b]
mode_rows=[r for q in qual for r in q.get('modes',[])]
edge_rows=[r for q in qual for r in q.get('boundary_cases',[])]
profiles=[z for b in d['batches'] for p in b.get('runtime_inverse_profiles',[]) for z in p]
all_A0=[v for r in mode_rows+edge_rows for v in r['original_A0_relative']]
out=dict(result=d.get('result'),error=d.get('error'),stop_reason=d.get('stop_reason'),worker_wall_seconds=d['wall_seconds'],controller=watch,worker_peak_RSS_bytes=d['peak_process_RSS_bytes'],sources_unchanged=True,source_identity=d['source_identity'],material_identity=d.get('material_identity'),background_identity=d.get('background_identity'),stages=[{k:v for k,v in s.items() if k!='admission'} for s in d['stages']],qualification=dict(initial_factor_seconds=d['qualification'].get('factor_seconds'),initial_qualification_seconds=d['qualification'].get('qualification_seconds'),logical_factor_bytes=d['qualification'].get('factor_bytes'),mode_records=len(mode_rows),boundary_records=len(edge_rows),original_A0_columns=len(all_A0),max_original_A0_residual=max(all_A0,default=None),max_qualification_corrections=max((r['repair']['corrections'] for r in mode_rows+edge_rows),default=None),all_zero_columns_exact=all(r['strict_zero_solution'] for r in edge_rows),all_tiny_preserved=all(r['tiny_nonzero_preserved'] is not False for r in edge_rows)),batches=[],strongest_control=d.get('strongest_control'),strongest_control_seconds=d.get('strongest_control_seconds'))
for b in d['batches']:
    r={k:v for k,v in b.items() if k in ['method','status','common','requested_depth','effective_depth','preconditioner_object','factor_seconds','factor_bytes','setup_observed_seconds','first_RHS_observed_seconds','complete_four_RHS_observed_seconds','online_phase_seconds','inner_phase_seconds','calls','inner_calls','A_call_counts','peak_process_RSS_bytes']}
    r['expected_depth']=16 if b['method']=='tsa16_zero' else 64
    r['recorded_constructor_depth']=b.get('common',{}).get('depth')
    r['method_binding_valid']=r['expected_depth']==r['recorded_constructor_depth']==b.get('requested_depth')==b.get('effective_depth')==b.get('preconditioner_object',{}).get('actual_depth')
    if r['method_binding_valid']:
        assert b['preconditioner_object']['type']=='rte.transport_baselines.ClassicalTransportStack' and b['preconditioner_object']['kind']=='tsa'
        assert b['preconditioner_object']['reference_shape']==[256,256] and b['preconditioner_object']['cache_shape']==[1,256,256,12,12]
    if 'qualification' in b:r['qualification_seconds']=b['qualification']['qualification_seconds']
    if 'runtime_inverse_profiles' in b:
        ps=b['runtime_inverse_profiles'];zs=[z for p in ps for z in p]
        r.update(inverse_applications=len(ps),logical_mode_inversions=len(zs),runtime_corrections_histogram={str(k):sum(z['corrections']==k for z in zs) for k in range(4)},max_runtime_original_G_residual=max(max(z['residual_trace'][-1]) for z in zs))
    out['batches'].append(r)
if d['rows']:
    out['accuracy']=dict(rows=len(d['rows']),accepted_rows=sum(r['accepted'] for r in d['rows']),max_original_A_residual=max(r['true_residual'] for r in d['rows']),max_evaluated_relative_bound=max(r['evaluated_relative_bound'] for r in d['rows']),max_field_difference=max(r['relative_field_difference'] for r in d['rows']),max_inner_original_projected_residual=max((x['true_residual'] for r in d['rows'] for x in r['inner']),default=None))
    out['method_accuracy']={m:dict(max_original_A_residual=max(r['true_residual'] for r in d['rows'] if r['method']==m),max_evaluated_bound=max(r['evaluated_relative_bound'] for r in d['rows'] if r['method']==m),max_field_difference=max(r['relative_field_difference'] for r in d['rows'] if r['method']==m)) for m in {r['method'] for r in d['rows']}}
completed={b['method']:b for b in out['batches'] if b['status']=='completed'}
out['all_method_bindings_valid']=all(b['method_binding_valid'] for b in out['batches'])
if not out['all_method_bindings_valid']:
    out['interpretation']='MECHANICAL_METHOD_BINDING_FAILURE: the tsa16_zero-labelled batch used depth64. Do not treat it as TSA16 evidence or promote this incomplete attempt.'
if len(completed)==4:
    assert out['all_method_bindings_valid']
    e=completed['E'];out['E_cost_gains']={m:1-e['complete_four_RHS_observed_seconds']/b['complete_four_RHS_observed_seconds'] for m,b in completed.items() if m!='E'}
    assert all(r['accepted'] for r in d['rows']) and len(d['rows'])==16
    assert all(not x['failed'] and x['true_residual']<=1e-4 for r in d['rows'] for x in r['inner'])
    assert out['accuracy']['max_original_A_residual']<=1e-5 and out['accuracy']['max_evaluated_relative_bound']<=1e-3 and out['accuracy']['max_field_difference']<=1e-4
    assert out['qualification']['max_original_A0_residual']<=1e-10 and out['qualification']['max_qualification_corrections']<=3
    assert e['max_runtime_original_G_residual']<=1e-10 and e['logical_mode_inversions']==17*e['inverse_applications']
    assert d['material_identity']['exact_nested'] and all(r['exact_nested'] and not r['neural_field_used'] for r in d['source_identity'])
    assert e['factor_bytes']==1105707336 and completed['compact4']['factor_bytes']==1359249408
    assert d['wall_seconds']<600 and watch['elapsed_seconds']<600 and all(r['seconds']<180 for r in d['stages'])
    assert d['peak_process_RSS_bytes']<=6*1024**3 and watch['reaped'] and watch['returncode']==0 and watch['stop_reason'] is None
    best=min((b for m,b in completed.items() if m!='E'),key=lambda r:r['complete_four_RHS_observed_seconds'])
    assert d['strongest_control']==best['method']
    if d['result']=='N256_PRELIMINARY_FORWARD_PASS':assert e['complete_four_RHS_observed_seconds']<best['complete_four_RHS_observed_seconds']
    elif d['result']=='NO_GO_N256_COMPLETE_COST':assert e['complete_four_RHS_observed_seconds']>=best['complete_four_RHS_observed_seconds']
    else:raise AssertionError('Unexpected full-completion outcome')
(P/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
