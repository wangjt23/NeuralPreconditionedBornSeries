"""Independent read-only saved-receipt audit; no Torch or solver imports."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
d=json.loads((P/'gate_01/results.json').read_text());w=json.loads((P/'WATCHDOG.json').read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert d['plan_sha256']==digest(P/'FROZEN_PLAN.md')=='4b24e2d47d714a189c4b6e5ec6b2c4dae1b330409f8ae12fe3ee6c26f0b200d2'
assert all(digest(P/f)==h for f,h in d['own_sources'].items())
assert d['source_hashes']==d['source_hashes_after'] and d['E_source_hashes']==d['E_source_hashes_after']
assert d['E_source_hashes']['angular_modes.py']=='d152afea90179efa8342a4f12631979951503785330f3aab08d4b7f1fe32a481'
assert d['source_ids']==[0,4,7,14] and d['grid']==128 and d['modes']==list(range(-8,9))
assert d['inner_tolerance']==1e-4 and d['inner_maxiter']==8 and d['outer_tolerance']==1e-5 and d['outer_maxiter']==100
assert d['block_target']==1e-10 and d['block_max_corrections']==3
assert not any(d[k] for k in ['heldout9626','training','new_teachers'])
out=dict(result=d.get('result'),error=d.get('error'),stop_reason=d.get('stop_reason'),source_hashes_unchanged=True,worker_wall_seconds=d['wall_seconds'],peak_RSS_bytes=d['peak_process_RSS_bytes'],controller=w,source_identity=d['source_identity'],material_identity=d.get('material_identity'),background_identity=d.get('background_identity'),stages=[{k:v for k,v in s.items() if k!='admission'} for s in d['stages']],batches=[])
assert all(r['q_sha256']==r['expected_sha256'] and r['unpermuted'] and not r['neural_field_used'] for r in d['source_identity'])
m=d.get('material_identity',{})
if m.get('passed'):
    assert m['multiplier']==3 and m['inverse']==43
    assert all(all(r.values()) for r in m['identity_and_inverse'].values())
    bg=m['background'];assert bg['original']['axis']==bg['permuted']['axis']==0
    assert bg['total_max_absolute_difference']==bg['scatter_max_absolute_difference']==bg['total_unequal_elements']==bg['scatter_unequal_elements']==0
    assert bg['original']['total_sha256']==bg['permuted']['total_sha256'] and bg['original']['scatter_sha256']==bg['permuted']['scatter_sha256']
    assert all(r['modular_FFT_relative_difference']<=1e-12 and r['spectrum_premise'] and r['permuted_outside_P8_energy_fraction']>r['original_outside_P8_energy_fraction'] for r in m['spectrum'].values())
quals=[d['qualification']]+[b['qualification'] for b in d['batches'] if 'qualification' in b]
mode=[r for q in quals for r in q.get('modes',[])];edge=[r for q in quals for r in q.get('boundary_cases',[])]
a0=[v for r in mode+edge for v in r['original_A0_relative']]
out['qualification']=dict(mode_records=len(mode),boundary_records=len(edge),original_A0_columns=len(a0),max_original_A0_residual=max(a0,default=None),max_corrections=max((r['repair']['corrections'] for r in mode+edge),default=None),all_zero_columns_exact=all(r['strict_zero_solution'] for r in edge),all_tiny_preserved=all(r['tiny_nonzero_preserved'] is not False for r in edge),preliminary_factor_seconds=d['qualification'].get('factor_seconds'),preliminary_checks_seconds=d['qualification'].get('qualification_seconds'))
for b in d['batches']:
    r={k:v for k,v in b.items() if k in ['method','status','requested_depth','effective_depth','preconditioner_object','common','factor_seconds','factor_bytes','setup_observed_seconds','first_RHS_observed_seconds','complete_four_RHS_observed_seconds','online_phase_seconds','inner_phase_seconds','calls','inner_calls','A_call_counts','peak_process_RSS_bytes']}
    expected=16 if b['method']=='tsa16_zero' else 64
    assert b['requested_depth']==b['effective_depth']==b['preconditioner_object']['actual_depth']==b['common']['depth']==expected
    assert b['preconditioner_object']['kind']=='tsa' and b['preconditioner_object']['type']=='rte.transport_baselines.ClassicalTransportStack'
    assert b['preconditioner_object']['reference_shape']==[128,128] and b['preconditioner_object']['cache_shape']==[1,128,128,12,12]
    if 'qualification' in b:r['qualification_seconds']=b['qualification']['qualification_seconds']
    if 'runtime_inverse_profiles' in b:
        profiles=b['runtime_inverse_profiles'];zs=[z for a in profiles for z in a]
        r.update(inverse_applications=len(profiles),logical_mode_inversions=len(zs),max_runtime_original_G_residual=max(max(z['residual_trace'][-1]) for z in zs),runtime_corrections_histogram={str(i):sum(z['corrections']==i for z in zs) for i in range(4)})
    out['batches'].append(r)
if d['rows']:
    out['accuracy']=dict(rows=len(d['rows']),accepted_rows=sum(r['accepted'] for r in d['rows']),max_original_A_residual=max(r['true_residual'] for r in d['rows']),max_evaluated_bound=max(r['evaluated_relative_bound'] for r in d['rows']),max_field_difference=max(r['relative_field_difference'] for r in d['rows']),max_inner_original_projected_residual=max((x['true_residual'] for r in d['rows'] for x in r['inner']),default=None))
    out['method_accuracy']={name:dict(max_original_A_residual=max(r['true_residual'] for r in d['rows'] if r['method']==name),max_evaluated_bound=max(r['evaluated_relative_bound'] for r in d['rows'] if r['method']==name),max_field_difference=max(r['relative_field_difference'] for r in d['rows'] if r['method']==name)) for name in {r['method'] for r in d['rows']}}
completed={b['method']:b for b in out['batches'] if b['status']=='completed'}
if len(completed)==4:
    assert len(d['rows'])==16 and all(r['accepted'] for r in d['rows'])
    assert out['accuracy']['max_original_A_residual']<=1e-5 and out['accuracy']['max_evaluated_bound']<=1e-3 and out['accuracy']['max_field_difference']<=1e-4
    assert all(not x['failed'] and x['true_residual']<=1e-4 for r in d['rows'] for x in r['inner'])
    assert out['qualification']['max_original_A0_residual']<=1e-10 and out['qualification']['max_corrections']<=3
    e=completed['E'];assert e['max_runtime_original_G_residual']<=1e-10 and e['logical_mode_inversions']==17*e['inverse_applications']
    assert e['factor_bytes']==276633928 and completed['compact4']['factor_bytes']==302137344
    assert all(s['seconds']<90 for s in d['stages']) and d['wall_seconds']<240 and w['elapsed_seconds']<240
    assert d['peak_process_RSS_bytes']<=4*1024**3 and w['returncode']==0 and w['reaped'] and w['stop_reason'] is None
    best=min((b for name,b in completed.items() if name!='E'),key=lambda b:b['complete_four_RHS_observed_seconds'])
    assert d['strongest_control']==best['method']
    out.update(strongest_control=best['method'],strongest_control_seconds=best['complete_four_RHS_observed_seconds'],E_cost_gains={name:1-e['complete_four_RHS_observed_seconds']/b['complete_four_RHS_observed_seconds'] for name,b in completed.items() if name!='E'})
    if d['result']=='PERMUTATION_PRELIMINARY_FORWARD_PASS':assert e['complete_four_RHS_observed_seconds']<best['complete_four_RHS_observed_seconds']
    else:assert d['result']=='NO_GO_PERMUTATION_COMPLETE_COST' and e['complete_four_RHS_observed_seconds']>=best['complete_four_RHS_observed_seconds']
(P/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
