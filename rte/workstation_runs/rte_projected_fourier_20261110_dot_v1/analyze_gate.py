"""Read saved receipts only. No torch import or numerical experiment."""
import json, hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
d=json.loads((P/'gate_01/results.json').read_text())
assert d['production_hashes']==d['production_hashes_after']
assert d['dependency_hashes']==d['dependency_hashes_after']
assert d['frozen_plan_sha256']=='19b95306d355f2fae8f5e46d038d207f49fa49d53f989ac6a8991890cf1ea902'
assert hashlib.sha256((P/'FROZEN_PLAN.md').read_bytes()).hexdigest()==d['frozen_plan_sha256']
assert all(hashlib.sha256((P/f).read_bytes()).hexdigest()==h for f,h in d['own_sources'].items())
assert d['source_ids']==[0,4,7,14] and d['modes']==list(range(-8,9))
assert d['inner_tolerance']==1e-4 and d['inner_maxiter']==8 and d['outer_tolerance']==1e-5 and d['outer_maxiter']==100
assert d['block_target']==1e-10 and d['block_max_corrections']==3
assert not any(d[k] for k in ['heldout9626','neural_training','new_accurate_teachers'])
passed=d.get('result')=='PROJECTED_FOURIER_TWO_ORDERS_PASS'
result=dict(result=d.get('result'),error=d.get('error'),stop_reason=d.get('stop_reason'),wall_seconds=d['wall_seconds'],peak_process_RSS_bytes=d['process_peak_RSS_bytes'],production_and_dependencies_unchanged=True,reference_seconds=d.get('reference_seconds'),stages=[{k:v for k,v in x.items() if k!='admission'} for x in d['stages']],batches=[],pairs=[])
records=[d['qualification']]+[b for b in d['batches'] if 'action_inverse_probes' in b]
probes=[p for r in records for p in r.get('action_inverse_probes',[])]
result['probe_records']=len(probes)
for key in ['action_difference','lifted_action_difference','inverse_field_difference','inverse_original_A_image_difference']:
    vals=[v for p in probes for v in p.get(key,[])]
    result['max_'+key]=max(vals,default=None)
result['zero_probe_records']=sum(p.get('strict_zero_solution',False) for p in probes)
modechecks=[p for r in [d['qualification']]+d['batches'] for p in r.get('mode_qualifications',[])]
result['independent_A0_checks']=len(modechecks)
result['max_original_A0_residual']=max((x['original_A0_residual'] for x in modechecks),default=None)
result['original_high_A_ledger']=dict(independent_mode_checks=len(modechecks),action_inverse_probe_calls=3*len(probes),ordinary_reference_calls_source_derived=sum(2*r['status']['iterations']+2 for r in d['references']),measured_batch_calls=sum(sum(b.get('A_call_counts',{}).get(k,0) for k in ['outer_A','inner_A','full_injection_A','final_A']) for b in d['batches']),scope='Visible high.A_op calls: modes/probes/references source-derived; measured solver boundaries instrumented. Excludes internal low-precision TSA stages and matrix-free G repair actions.')
result['original_high_A_ledger']['total']=sum(result['original_high_A_ledger'][k] for k in ['independent_mode_checks','action_inverse_probe_calls','ordinary_reference_calls_source_derived','measured_batch_calls'])
for b in d['batches']:
    row={k:v for k,v in b.items() if k in ['method','repeat','status','physical_seconds','TSA_seconds','factor_seconds','coefficient_seconds','coefficient_bytes','factor_bytes','mode_qualification_seconds','probe_seconds','setup_observed_seconds','first_RHS_observed_seconds','complete_four_RHS_observed_seconds','online_phase_seconds','inner_phase_seconds','A_call_counts','calls','inner_calls','process_peak_RSS_bytes']}
    ps=b.get('runtime_inverse_profiles',[])
    row['runtime_inverse_applications']=len(ps)
    row['logical_mode_inversions']=sum(len(p) for p in ps)
    row['runtime_corrections_histogram']={str(k):sum(z['corrections']==k for p in ps for z in p) for k in range(4)}
    row['max_runtime_original_G_residual']=max((max(z['residual_trace'][-1]) for p in ps for z in p),default=None)
    result['batches'].append(row)
if d['rows']:
    result.update(candidate_rows=len(d['rows']),accepted_rows=sum(r['accepted'] for r in d['rows']),max_original_A_residual=max(r['true_residual'] for r in d['rows']),max_evaluated_relative_bound=max(r['evaluated_relative_bound'] for r in d['rows']),max_field_difference=max(r['relative_field_difference'] for r in d['rows']),max_inner_residual=max((x['true_residual'] for r in d['rows'] for x in r['inner']),default=None),max_independent_full_projected_inner_residual=max((x['independent_full_projected_residual'] for r in d['rows'] for x in r['inner'] if 'independent_full_projected_residual' in x),default=None))
if d.get('result')=='NO_GO_FORWARD_COMPLETE_COST':
    assert len(d['rows'])==12 and all(r['accepted'] for r in d['rows'])
    assert len(d['batches'])==3 and all(b['status']=='completed' for b in d['batches'])
    assert d['qualification']['passed'] and len(probes)==12 and len(modechecks)==51
    assert all(result['max_'+k]<=1e-10 for k in ['action_difference','lifted_action_difference','inverse_field_difference','inverse_original_A_image_difference'])
    assert result['max_original_A0_residual']<=1e-10 and result['max_original_A_residual']<=1e-5 and result['max_evaluated_relative_bound']<=1e-3 and result['max_field_difference']<=1e-4
    assert result['max_inner_residual']<=1e-4 and result['max_independent_full_projected_inner_residual']<=1e-4
    for b in result['batches']:
        assert b['calls']==d['expected_calls'][b['method']]
        if b['method']!='compact4':assert b['inner_calls']==d['expected_inner'] and b['runtime_inverse_applications']==19 and b['logical_mode_inversions']==323 and b['max_runtime_original_G_residual']<=1e-10
        if b['method']=='full_E':assert b['A_call_counts']['inner_A']==46
        if b['method']=='reduced_E':assert b['A_call_counts']['compressed_A']==46 and b['A_call_counts']['full_injection_A']==4
    assert d['wall_seconds']<600 and all(x['seconds']<180 for x in d['stages'])
for rep in [0,1]:
    bs={b['method']:b for b in result['batches'] if b['repeat']==rep and b['status']=='completed'}
    if len(bs)==3:
        c,e,r=[bs[k] for k in ['compact4','full_E','reduced_E']]
        result['pairs'].append(dict(repeat=rep,reduced_vs_full_E_gain=1-r['complete_four_RHS_observed_seconds']/e['complete_four_RHS_observed_seconds'],reduced_vs_compact_gain=1-r['complete_four_RHS_observed_seconds']/c['complete_four_RHS_observed_seconds'],inner_reduction=1-r['inner_phase_seconds']/e['inner_phase_seconds'],reduced_first=r['first_RHS_observed_seconds'],full_E_first=e['first_RHS_observed_seconds'],compact_first=c['first_RHS_observed_seconds']))
if passed:
    assert len(d['rows'])==24 and all(r['accepted'] for r in d['rows'])
    assert len(result['pairs'])==2 and all(x['reduced_vs_full_E_gain']>0 and x['reduced_vs_compact_gain']>0 for x in result['pairs'])
    assert d['wall_seconds']<600 and all(x['seconds']<180 for x in d['stages'])
    assert all(result['max_'+k]<=1e-10 for k in ['action_difference','lifted_action_difference','inverse_field_difference','inverse_original_A_image_difference'])
    assert result['max_original_A0_residual']<=1e-10 and result['max_original_A_residual']<=1e-5 and result['max_evaluated_relative_bound']<=1e-3 and result['max_field_difference']<=1e-4
    for b in result['batches']:
        assert b['calls']==d['expected_calls'][b['method']]
        if b['method']!='compact4':assert b['inner_calls']==d['expected_inner'] and b['runtime_inverse_applications']==19 and b['logical_mode_inversions']==323 and b['max_runtime_original_G_residual']<=1e-10
        if b['method']=='full_E':assert b['A_call_counts']['inner_A']==46
        if b['method']=='reduced_E':assert b['A_call_counts']['compressed_A']==46 and b['A_call_counts']['full_injection_A']==4
(P/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
