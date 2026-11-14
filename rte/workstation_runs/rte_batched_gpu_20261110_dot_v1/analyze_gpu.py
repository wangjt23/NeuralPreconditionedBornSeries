"""Static audit of the completed first-pair GPU no-go; no CUDA import."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;s=json.loads((P/'gpu_gate_01/results.json').read_text())
assert s['complete'] and s['result']=='NO_GO_FIRST_COMPLETE_PAIR' and not s.get('error')
assert s['production_hashes']==s['production_hashes_after']
assert hashlib.sha256((P/'gpu_gate.py').read_bytes()).hexdigest()==s['source_sha256']
assert s['affinity']==[0] and s['grid']==128 and s['L']==12 and s['g']==.65 and s['amplitude']==3
assert (s['prefix'],s['tangent_cut'],s['inner_tolerance'],s['inner_maxiter'],s['outer_tolerance'],s['outer_maxiter'],s['block_target'],s['block_max_corrections'])==(1,8,1e-4,8,1e-5,100,1e-10,3)
assert s['wall_seconds']<180 and s['wall_seconds']<s['total_budget_limit']
assert len(s['batches'])==2 and len(s['rows'])==8 and len(s['references'])==4
assert all(not r['status']['failed'] and r['status']['true_residual']<=1e-5 for r in s['references'])
c,a=s['batches'];assert (c['method'],a['method'])==('compact4','batched17')
assert c['repeat']==a['repeat']==0 and c['status']==a['status']=='completed'
assert c['base_setup']['coefficient_hashes']==a['base_setup']['coefficient_hashes']
assert a['setup_observed_seconds']<c['complete_four_RHS_observed_seconds']<a['complete_four_RHS_observed_seconds']
assert c['calls']==[4,3,4,3] and a['calls']==[2,2,2,2] and a['inner_calls']==[5,5,5,4]
def flat(x):return [v for y in x for v in flat(y)] if isinstance(x,list) else [x]
for b in [c,a]:
    rr=[r for r in s['rows'] if r['method']==b['method']]
    assert [r['source_id'] for r in rr]==[0,4,7,14]
    for r in rr:
        assert r['accepted'] and not r['status']['failed'] and r['true_residual']<=1e-5
        assert 0<=r['evaluated_relative_bound']<=1e-3 and r['relative_field_difference']<=1e-4
        assert all(not i['failed'] and i['true_residual']<=1e-4 for i in r['inner'])
    online=sum(r['solve_seconds']+r['check_seconds'] for r in rr)
    assert abs(online-b['online_phase_seconds'])<1e-9
    assert b['complete_four_RHS_observed_seconds']>=b['setup_observed_seconds']+online
    assert b['first_RHS_observed_seconds']>=b['setup_observed_seconds']+rr[0]['solve_seconds']+rr[0]['check_seconds']
qual=a['original_mode_qualification'];assert [r['k'] for r in qual['records']]==list(range(-8,9))
assert max(r['original_A0_residual'] for r in qual['records'])<=1e-10
for name in ['repair_status','zero_status','mixed_status']:
    z=qual[name];assert z['success'] and max(flat(z['corrections']))<=3 and max(flat(z['residual_trace'][-1]))<=1e-10
assert max(flat(qual['zero_status']['corrections']))==0 and max(flat(qual['zero_status']['residual_trace']))==0
profiles=a['repair_profiles'];assert len(profiles)==19
for r in profiles:assert r['success'] and max(flat(r['corrections']))<=3 and max(flat(r['residual_trace'][-1]))<=1e-10
summary=dict(result=s['result'],whole_research_wall_seconds=s['wall_seconds'],context_initialization_seconds=s['warmup']['context_seconds'],common_neutral_warmup_seconds=s['warmup']['neutral_library_seconds'],reference_research_seconds=s['reference_research_seconds'],methods={b['method']:{k:b[k] for k in ['setup_observed_seconds','factor_synchronized_seconds','capture_or_qualification_synchronized_seconds','first_RHS_observed_seconds','complete_four_RHS_observed_seconds','online_phase_seconds','calls','inner_calls','logical_factor_bytes','GPU_allocated_at_entry','peak_GPU_allocated_bytes','peak_GPU_reserved_bytes']} for b in [c,a]},candidate_over_compact_complete_ratio=a['complete_four_RHS_observed_seconds']/c['complete_four_RHS_observed_seconds'],candidate_over_compact_first_ratio=a['first_RHS_observed_seconds']/c['first_RHS_observed_seconds'],candidate_online_budget_at_unchanged_setup=c['complete_four_RHS_observed_seconds']-a['setup_observed_seconds'],candidate_required_online_reduction_at_unchanged_setup=1-(c['complete_four_RHS_observed_seconds']-a['setup_observed_seconds'])/a['online_phase_seconds'],max_original_A_residual=max(r['true_residual'] for r in s['rows']),max_evaluated_relative_bound=max(r['evaluated_relative_bound'] for r in s['rows']),max_field_difference=max(r['relative_field_difference'] for r in s['rows']),max_inner_original_projected_residual=max(i['true_residual'] for r in s['rows'] for i in r['inner']),max_independent_original_A0_residual=max(r['original_A0_residual'] for r in qual['records']),max_runtime_original_G_residual=max(max(flat(r['residual_trace'][-1])) for r in profiles),runtime_max_corrections=max(max(flat(r['corrections'])) for r in profiles),batched_inverse_calls=len(profiles),logical_mode_inverse_calls=17*len(profiles),batched_raw_calls=sum(r['batched_raw_calls'] for r in profiles),batched_original_G_calls=sum(r['batched_original_calls'] for r in profiles),per_method_base_H2D_bytes=c['base_setup']['H2D_bytes'],per_method_exact_value_check_D2H_bytes=c['base_setup']['exact_value_check_D2H_bytes'],candidate_additional_qualification_H2D_bytes=qual['vector_H2D_bytes'],source_sha256=s['source_sha256'],batched_source_sha256=s['batched_source_sha256'],production_hashes_unchanged=True,no_reverse_pair_run=True)
print(json.dumps(summary,indent=2))
