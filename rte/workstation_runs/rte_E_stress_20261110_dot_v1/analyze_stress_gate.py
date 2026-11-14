"""Static verification of the frozen E-on-B conditional CPU stress gate."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
s=json.loads((P/'stress_01/results.json').read_text())
b=json.loads((P.parent/'rte_inclusion_stress_20261110_dot_v1/stress_cpu_01/settings.json').read_text())
e=json.loads((P.parent/'rte_angular_full_20261110_dot_v1/full_01/settings.json').read_text())
assert s['complete'] and not s.get('error') and s['result']=='CPU_STRESS_ABBA_PASS'
assert hashlib.sha256((P/'run_stress_gate.py').read_bytes()).hexdigest()==s['source_sha256']
assert s['production_hashes']==s['production_hashes_after']==b['production_hashes']==e['production_hashes']
assert s['dependency_hashes']==s['dependency_hashes_after']==e['dependency_hashes']
assert s['E_source_hashes']==e['source_hashes'] and s['B_source_hashes']==b['source_hashes']
assert s['rhs_sha256']==b['rhs_sha256']==e['rhs_sha256'] and s['source_ids']==[0,4,7,14]
assert (s['grid'],s['L'],s['g'],s['inclusion_amplitude'],s['prefix'],s['tangent_cut'])==(128,12,.65,6.,1,8)
assert (s['inner_tolerance'],s['inner_maxiter'],s['outer_tolerance'],s['outer_maxiter'],s['block_target'],s['block_max_corrections'])==(1e-4,8,1e-5,100,1e-10,3)
assert s['affinity']==[0] and s['threads']==1 and s['device']=='cpu'
assert s['wall_seconds']<600 and all(x['seconds']<180 and x['status']=='completed' for x in s['stages'])
identity=s['material_identity']
for k in ['baseline_total_sha256','baseline_scatter_sha256','stress_total_sha256','stress_scatter_sha256']:assert identity[k]==b['material_identity'][k]
assert identity['amplitude3_byte_equal_original']
assert identity['info']==b['material_identity']['stress_info']
assert len(s['references'])==4 and all(not r['status']['failed'] and r['status']['true_residual']<=1e-5 for r in s['references'])
assert [(x['method'],x['repeat'],x['position']) for x in s['batches']]==[('compact4',0,0),('sequential_E',0,1),('sequential_E',1,0),('compact4',1,1)]
assert len(s['rows'])==16
for r in s['rows']:
    assert r['accepted'] and not r['status']['failed'] and r['true_residual']<=1e-5
    assert 0<=r['evaluated_relative_bound']<=1e-3 and r['relative_field_difference']<=1e-4
    assert r['status']['classical_calls']==s['expected_calls'][r['method']][s['source_ids'].index(r['source_id'])]
    if r['method']=='sequential_E':assert len(r['inner'])==1 and r['inner'][0]['iterations']==6 and not r['inner'][0]['failed'] and r['inner'][0]['true_residual']<=1e-4
    else:assert not r['inner']
profiles=[];qual=[]
for batch in s['batches']:
    assert batch['status']=='completed' and batch['background']==identity['backgrounds']['amplitude6']
    rows=[r for r in s['rows'] if r['method']==batch['method'] and r['repeat']==batch['repeat']]
    assert [r['source_id'] for r in rows]==s['source_ids']
    assert abs(batch['online_phase_seconds']-sum(r['solve_seconds']+r['check_seconds'] for r in rows))<1e-9
    assert batch['complete_four_RHS_observed_seconds']>=batch['setup_observed_seconds']+batch['online_phase_seconds']
    assert batch['first_RHS_observed_seconds']>=batch['setup_observed_seconds']+rows[0]['solve_seconds']+rows[0]['check_seconds']
    if batch['method']=='sequential_E':
        assert batch['inner_calls']==[6,6,6,6] and batch['calls']==[2,3,2,3]
        assert len(batch['mode_qualifications'])==17 and [r['k'] for r in batch['mode_qualifications']]==list(range(-8,9))
        qual.extend(batch['mode_qualifications']);assert len(batch['repair_profiles'])==24
        for call in batch['repair_profiles']:
            assert [r['k'] for r in call]==list(range(-8,9))
            profiles.extend(call)
    else:assert batch['calls']==[4,4,4,4]
assert all(r['original_A0_residual']<=1e-10 and r['repair_status']['success'] for r in qual)
assert all(r['success'] and r['corrections']<=3 and max(r['residual_trace'][-1])<=1e-10 for r in profiles)
pairs=[]
for rep in [0,1]:
    batches={x['method']:x for x in s['batches'] if x['repeat']==rep};c,a=batches['compact4'],batches['sequential_E']
    assert a['complete_four_RHS_observed_seconds']<c['complete_four_RHS_observed_seconds']
    pairs.append(dict(repeat=rep,compact_complete=c['complete_four_RHS_observed_seconds'],E_complete=a['complete_four_RHS_observed_seconds'],gain=1-a['complete_four_RHS_observed_seconds']/c['complete_four_RHS_observed_seconds'],compact_first=c['first_RHS_observed_seconds'],E_first=a['first_RHS_observed_seconds'],compact_setup=c['setup_observed_seconds'],E_setup=a['setup_observed_seconds'],compact_online=c['online_phase_seconds'],E_online=a['online_phase_seconds']))
summary=dict(result=s['result'],pairs=pairs,material_identity=identity,stage_seconds={x['name']:x['seconds'] for x in s['stages']},whole_research_wall_seconds=s['wall_seconds'],reference_research_seconds=s['reference_research_seconds'],process_peak_RSS_bytes=s['process_peak_RSS_bytes'],accepted_candidate_rows=len(s['rows']),independent_A0_checks=len(qual),mode_inverse_calls=len(profiles),max_independent_original_A0_residual=max(r['original_A0_residual'] for r in qual),max_runtime_original_G_residual=max(max(r['residual_trace'][-1]) for r in profiles),max_runtime_corrections=max(r['corrections'] for r in profiles),correction_counts={str(i):sum(r['corrections']==i for r in profiles) for i in range(4)},max_original_A_residual=max(r['true_residual'] for r in s['rows']),max_evaluated_relative_bound=max(r['evaluated_relative_bound'] for r in s['rows']),max_field_difference=max(r['relative_field_difference'] for r in s['rows']),max_inner_original_projected_residual=max(i['true_residual'] for r in s['rows'] for i in r['inner']),source_sha256=s['source_sha256'],E_source_hashes=s['E_source_hashes'],B_source_hashes=s['B_source_hashes'],production_and_dependencies_unchanged=True)
summary['batch_components']=[{k:r[k] for k in ['method','repeat','physical_setup_seconds','TSA_setup_seconds','factor_setup_seconds','mode_qualification_seconds','setup_observed_seconds','logical_factor_bytes']} for r in s['batches']]
oldrows=json.loads((P.parent/'rte_inclusion_stress_20261110_dot_v1/stress_cpu_01/rows.json').read_text())
oldinner={r['source_id']:r['inner_records'][0]['true_residual'] for r in oldrows if r['method']=='exact8_inner'}
summary['max_inner_residual_difference_vs_B']=max(abs(r['inner'][0]['true_residual']-oldinner[r['source_id']]) for r in s['rows'] if r['method']=='sequential_E')
print(json.dumps(summary,indent=2))
