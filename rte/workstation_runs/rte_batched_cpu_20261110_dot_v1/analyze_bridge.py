"""Static verification of the sealed batched CPU bridge receipts."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
q=json.loads((P/'qualification_01/results.json').read_text())
f=json.loads((P/'full_01/results.json').read_text())
for s in [q,f]:
    assert s['complete'] and s['passed'] and not s.get('error')
    assert s['affinity']==[0] and s['device']=='cpu' and s['threads']==1
    assert s['production_hashes']==s['production_hashes_after']
    assert s['wall_seconds']<180 and s['previous_numerical_wall_seconds']+s['wall_seconds']<600
    assert (s['grid'],s['L'],s['g'],s['inclusion_amplitude'])==(128,12,.65,3)
    assert (s['block_target'],s['block_max_corrections'],s['inner_tolerance'],s['inner_maxiter'],s['outer_tolerance'],s['outer_maxiter'])==(1e-10,3,1e-4,8,1e-5,100)
    for name,h in s['own_sources'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
assert q['own_sources']==f['own_sources'] and q['E_sources']==f['E_sources']
assert abs(f['previous_numerical_wall_seconds']-q['wall_seconds'])<1e-9
action=[r for r in q['records'] if r['kind']=='original_action']
cases=[r for r in q['records'] if r['kind']=='inverse_case']
assert [r['k'] for r in action]==list(range(-8,9))
assert [c['name'] for c in cases]==['locked_vectors','all_zero','mixed_columns','mixed_inactive_modes']
def flat(x):
    if isinstance(x,list):return [v for y in x for v in flat(y)]
    return [x]
for r in action:assert max(r['sequential_action_difference'])<=1e-12 and max(r['independent_original_A0_difference'])<=1e-12
for c in cases:
    assert c['batched_status']['success'] and max(flat(c['batched_status']['corrections']))<=3
    assert [r['k'] for r in c['modes']]==list(range(-8,9))
    for r in c['modes']:
        assert r['sequential_status']['success'] and r['strict_zero_solution']
        assert max(r['inverse_difference'])<=1e-10 and max(r['batched_original_G_residual'])<=1e-10 and max(r['independent_original_A0_residual'])<=1e-10
        assert r['tiny_nonzero_preserved'] in [True,None]
    if c['name']=='all_zero':assert max(flat(c['batched_status']['corrections']))==0 and max(flat(c['batched_status']['residual_trace']))==0
qual=f['mode_qualification'];assert qual['repair_status']['success'] and len(qual['records'])==17
assert [r['k'] for r in qual['records']]==list(range(-8,9))
assert max(v for r in qual['records'] for v in r['original_A0_residual'])<=1e-10
rows=f['rows'];assert [r['source_id'] for r in rows]==[0,4,7,14]
assert len(f['reference_checks'])==4 and all(not r['status']['failed'] and r['status']['true_residual']<=1e-5 for r in f['reference_checks'])
for r in rows:
    assert r['accepted'] and not r['status']['failed'] and r['true_residual']<=1e-5
    assert 0<=r['evaluated_relative_bound']<=1e-3 and r['relative_field_difference']<=1e-4
    assert r['status']['classical_calls']==2 and r['status']['injection_calls']==r['status']['injection_accepted']==1 and r['status']['injection_rejected']==0
    assert all(not i['failed'] and i['true_residual']<=1e-4 for i in r['inner'])
    assert abs(r['accepted_seconds']-r['solve_seconds']-r['check_seconds'])<1e-9
assert f['batch']['inner_calls']==[5,5,5,4] and f['batch']['outer_calls']==[2,2,2,2]
assert len(f['repair_profiles'])==f['batch']['batched_inverse_applications']==19
assert sum(r['batched_inverse_applications'] for r in rows)==19
for r in f['repair_profiles']:
    assert r['success'] and max(flat(r['corrections']))<=3 and max(flat(r['residual_trace'][-1]))<=1e-10
assert sum(r['batched_raw_calls'] for r in f['repair_profiles'])==f['batch']['batched_raw_calls']==38
assert sum(r['batched_original_calls'] for r in f['repair_profiles'])==f['batch']['batched_original_G_calls']==38
setup=f['setup'];batch=f['batch']
assert abs(sum(setup[k] for k in ['physical_seconds','TSA_seconds','factor_seconds','original_A0_validation_seconds'])-setup['complete_seconds'])<1e-9
assert abs(sum(r['accepted_seconds'] for r in rows)-batch['online_four_RHS_seconds'])<1e-9
assert abs(setup['complete_seconds']+batch['online_four_RHS_seconds']-batch['complete_four_RHS_seconds'])<1e-9
gap=batch['observed_batch_wall_seconds']-batch['complete_four_RHS_seconds'];assert gap>=0
assert abs(batch['first_RHS_seconds']-setup['complete_seconds']-rows[0]['accepted_seconds'])<1e-9
summary=dict(passed=True,stages_seconds=[q['wall_seconds'],f['wall_seconds']],total_numerical_wall_seconds=q['wall_seconds']+f['wall_seconds'],peak_process_RSS_bytes=[q['process_peak_RSS_bytes'],f['process_peak_RSS_bytes']],max_original_action_vs_sequential=max(max(r['sequential_action_difference']) for r in action),max_independent_original_A0_action_difference=max(max(r['independent_original_A0_difference']) for r in action),max_sequential_inverse_difference=max(max(r['inverse_difference']) for c in cases for r in c['modes']),max_qualified_original_G_inverse_residual=max(max(r['batched_original_G_residual']) for c in cases for r in c['modes']),max_qualified_independent_A0_inverse_residual=max(max(r['independent_original_A0_residual']) for c in cases for r in c['modes']),qualification_cases=[c['name'] for c in cases],inverse_qualification_columns=sum(len(r['inverse_difference']) for c in cases for r in c['modes']),qualified_strict_zero_columns=sum(r['strict_zero_columns'] for c in cases for r in c['modes']),full_batch=batch,full_setup=setup,unallocated_observed_batch_seconds=gap,first_RHS_phase_seconds=batch['first_RHS_seconds'],first_RHS_conservative_upper_seconds=batch['first_RHS_seconds']+gap,max_outer_original_residual=max(r['true_residual'] for r in rows),max_evaluated_relative_bound=max(r['evaluated_relative_bound'] for r in rows),max_field_difference=max(r['relative_field_difference'] for r in rows),max_inner_original_projected_residual=max(i['true_residual'] for r in rows for i in r['inner']),max_runtime_original_G_residual=max(max(flat(r['residual_trace'][-1])) for r in f['repair_profiles']),max_runtime_corrections=max(max(flat(r['corrections'])) for r in f['repair_profiles']),full_factor_logical_bytes=f['factor_logical_bytes'],reference_research_seconds=f['reference_research_seconds'],source_hashes=f['own_sources'],production_hashes_unchanged=True)
summary['source_derived_call_counts']=dict(note='Static counts from frozen solver source and successful no-restart/no-rejection statuses, not independent runtime counters or GPU kernel-launch counts.',candidate_outer_original_A_and_final_checks=sum(2*r['status']['iterations']+4 for r in rows),candidate_inner_projected_original_A=sum(2*i['iterations']+2 for r in rows for i in r['inner']),candidate_independent_setup_original_A0=17,ordinary_reference_original_A=sum(2*r['status']['iterations']+2 for r in f['reference_checks']),ordinary_reference_TSA64=sum(r['status']['iterations'] for r in f['reference_checks']),qualification_stage_original_A0=17+sum(len(c['modes']) for c in cases),candidate_setup_batched_LU=13,candidate_setup_batched_LU_solve=23,candidate_online_batched_LU_solve=13*batch['batched_raw_calls'])
print(json.dumps(summary,indent=2))
