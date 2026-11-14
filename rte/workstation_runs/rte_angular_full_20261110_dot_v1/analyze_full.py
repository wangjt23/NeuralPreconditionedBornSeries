"""Read-only receipt validation and reproducible aggregate statistics."""
import json, hashlib, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
read = lambda name: json.loads((HERE / 'full_01' / (name + '.json')).read_text())
s, rows, batches = read('settings'), read('rows'), read('batches')
qual, profiles, refs = read('mode_qualifications'), read('repair_profiles'), read('reference_checks')
assert s['complete'] and not s.get('error')
assert s['production_hashes'] == s['production_hashes_after']
assert s['source_ids'] == [0, 4, 7, 14]
assert s['device'] == 'cpu' and s['affinity'] == [0] and s['threads'] == 1
assert (s['grid'],s['L'],s['g'],s['inclusion_amplitude']) == (128,12,.65,3.)
assert (s['inner_tolerance'],s['inner_maxiter'],s['block_target'],s['block_max_corrections']) == (1e-4,8,1e-10,3)
assert (s['outer_tolerance'],s['outer_maxiter']) == (1e-5,100)
assert not s['heldout9626'] and not s['neural_training'] and not s['new_accurate_teachers']
for f,h in s['source_hashes'].items(): assert hashlib.sha256((HERE/f).read_bytes()).hexdigest() == h
assert len(refs)==4 and all(not r['status']['failed'] for r in refs)
assert all(r['status']['true_residual']<=1e-5 for r in refs)
if s['result']=='SCREEN_AND_THREE_PAIRS_COMPLETE': assert len(batches)==12 and len(rows)==48 and s['screen_passed']
assert len(rows) == 4*len(batches) and all(r['accepted'] for r in rows)
assert all(r['true_residual']<=1e-5 and r['evaluated_relative_bound']<=1e-3 and r['relative_field_difference']<=1e-4 for r in rows)
assert len(qual)==len(batches)
expected=[(stage,rep,method,pos) for stage,orders in [('screen',s['screen_orders']),('confirmation',s['confirmation_orders'] if s['screen_passed'] else [])] for rep,order in enumerate(orders) for pos,method in enumerate(order)]
assert [(b['stage'],b['repeat'],b['method'],b['method_order']) for b in batches]==expected
zero=json.loads((HERE/'zero_columns_01/results.json').read_text())
assert zero['complete'] and zero['source_hashes']['angular_modes.py']==s['source_hashes']['angular_modes.py']
for z in zero['records']:
    assert z['zero_output_exact'] and z['tiny_nonzero_not_discarded']
    assert z['all_zero']['success'] and z['all_zero']['corrections']==0
    assert z['mixed']['success'] and max(z['mixed']['residual_trace'][-1])<=1e-10

def key(x): return (x['stage'],x['repeat'],x['method'])
def near(x,y): assert abs(x-y)<1e-8, (x,y)
batch_rows={}
for b in batches:
    rr=[r for r in rows if key(r)==key(b)]; batch_rows[key(b)]=rr
    assert [r['source_id'] for r in rr]==s['source_ids']
    near(b['setup_seconds'],sum(b[k] for k in ['physical_setup_seconds','TSA_setup_seconds','factor_setup_seconds','independent_validation_seconds']))
    near(b['online_four_rhs_seconds'],sum(r['accepted_seconds'] for r in rr))
    near(b['complete_four_rhs_seconds'],b['setup_seconds']+b['online_four_rhs_seconds'])
    for r in rr:
        near(r['accepted_seconds'],r['solve_seconds']+r['check_seconds'])
        near(r['Q1_seconds'],b['setup_seconds']+r['accepted_seconds'])
        assert r['status']['classical_calls'] == (4 if r['source_id'] in [0,7] else 3) if b['method']=='compact4_inject' else r['status']['classical_calls']==2
        for inner in r['inner_records']:
            assert not inner['failed'] and inner['iterations']<=8 and inner['true_residual']<=1e-4

mode_hashes={}
for q in qual:
    assert not q.get('failed')
    if q['method']=='compact4_inject': assert not q['records']; continue
    assert [r['k'] for r in q['records']]==list(range(-8,9))
    for r in q['records']:
        assert r['original_A0_residual']<=1e-10 and r['repair_status']['success']
        assert r['repair_status']['corrections']<=3
        if r['k'] in mode_hashes: assert mode_hashes[r['k']]==r['RHS_sha256']
        mode_hashes[r['k']]=r['RHS_sha256']
assert len(profiles)==sum(b['method']=='angular8_inner' for b in batches)
all_mode_calls=[]
for p in profiles:
    rr=batch_rows[key(p)]
    assert len(p['applications'])==sum(r['mode_inverse_calls'] for r in rr)
    for application in p['applications']:
        assert [x['k'] for x in application]==list(range(-8,9))
        for x in application:
            assert x['success'] and x['corrections']<=3 and max(x['residual_trace'][-1])<=1e-10
            all_mode_calls.append(x)

def pair(stage,rep):
    bb={b['method']:b for b in batches if b['stage']==stage and b['repeat']==rep}
    c,a=bb['compact4_inject'],bb['angular8_inner']
    item=dict(stage=stage,repeat=rep,compact_seconds=c['complete_four_rhs_seconds'],angular_seconds=a['complete_four_rhs_seconds'],angular_gain_vs_compact=1-a['complete_four_rhs_seconds']/c['complete_four_rhs_seconds'],angular_wall_gain_vs_compact=1-a['observed_batch_wall_seconds']/c['observed_batch_wall_seconds'],compact_first_rhs_seconds=batch_rows[key(c)][0]['Q1_seconds'],angular_first_rhs_seconds=batch_rows[key(a)][0]['Q1_seconds'])
    if 'dense8_inner' in bb:
        d=bb['dense8_inner'];item.update(dense_seconds=d['complete_four_rhs_seconds'],angular_gain_vs_dense=1-a['complete_four_rhs_seconds']/d['complete_four_rhs_seconds'],dense_first_rhs_seconds=batch_rows[key(d)][0]['Q1_seconds'])
    return item

pairs=[pair(stage,rep) for stage in ['screen','confirmation'] for rep in sorted(set(b['repeat'] for b in batches if b['stage']==stage))]
summary={}
for stage in ['screen','confirmation']:
    summary[stage]={}
    for method in ['compact4_inject','dense8_inner','angular8_inner']:
        bb=[b for b in batches if b['stage']==stage and b['method']==method]
        if not bb:continue
        fields=['physical_setup_seconds','TSA_setup_seconds','factor_setup_seconds','independent_validation_seconds','setup_seconds','online_four_rhs_seconds','complete_four_rhs_seconds','observed_batch_wall_seconds','factor_or_logical_stored_bytes']
        summary[stage][method]={f:dict(median=statistics.median(b[f] for b in bb),minimum=min(b[f] for b in bb),maximum=max(b[f] for b in bb)) for f in fields}
        summary[stage][method]['first_rhs_seconds']=[batch_rows[key(b)][0]['Q1_seconds'] for b in bb]
        summary[stage][method]['outer_calls']=[b['calls'] for b in bb]
        summary[stage][method]['inner_calls']=[b['inner_calls'] for b in bb]

result=dict(complete=True,result=s['result'],raw_rows=len(rows),batches=len(batches),ordinary_reference_solves=len(refs),independent_original_A0_checks=sum(len(q['records']) for q in qual),mode_inverse_calls=len(all_mode_calls),max_original_A0_residual=max(r['original_A0_residual'] for q in qual for r in q['records']),max_mode_repair_residual=max(max(x['residual_trace'][-1]) for x in all_mode_calls),max_mode_corrections=max(x['corrections'] for x in all_mode_calls),correction_histogram={str(i):sum(x['corrections']==i for x in all_mode_calls) for i in range(4)},max_original_A_residual=max(r['true_residual'] for r in rows),max_evaluated_relative_error_bound=max(r['evaluated_relative_bound'] for r in rows),max_relative_field_difference=max(r['relative_field_difference'] for r in rows),pairs=pairs,summary=summary,total_research_wall_seconds=s['wall_seconds'],reference_research_seconds=s['reference_research_seconds'],source_hashes=s['source_hashes'],production_hashes_unchanged=True,all_cost_identities_verified=True)
result.update(max_raw_band_original_G_residual=max(max(x['residual_trace'][0]) for x in all_mode_calls),max_inner_original_projected_A_residual=max(i['true_residual'] for r in rows for i in r['inner_records']),sum_observed_batch_wall_seconds=sum(b['observed_batch_wall_seconds'] for b in batches))
result['original_A0_checks_by_method']={m:dict(count=sum(len(q['records']) for q in qual if q['method']==m),max_residual=max(r['original_A0_residual'] for q in qual if q['method']==m for r in q['records'])) for m in ['dense8_inner','angular8_inner']}
print(json.dumps(result,indent=2))
