"""Root analysis of the fixed screen, independent of the receipt auditor."""
import hashlib,json,pathlib,statistics as s
root=pathlib.Path('receipts/gain_two_execution');out=pathlib.Path('deliverables/gain_two')
load=lambda n:json.loads((root/n).read_text())
policy=load('policy.json');rows=load('results/solver.json');setups=load('results/setup.json')
geo=load('results/geometry.json');prefix=load('results/prefixes.json');apps=load('results/applications.json')
checks=load('results/checks.json');settings=load('results/settings.json');execution=load('execution_receipt.json')
assert hashlib.sha256((root/'policy.json').read_bytes()).hexdigest()=='7adf90dd59c439d5b17e3b47513ebdbaaf1cf3f7b3a5f31d75e44c977e912832'
assert len(rows)==176 and len(checks)==280 and len(apps)==220 and len(geo)==8 and len(prefix)==4
assert settings['complete'] and execution['exit_code']==0 and not execution['owned_process_timed_out']
assert len({(r['size'],r['index'],r['method'],r['repeat']) for r in rows})==176
cases={};cost=[];geometry=[]
for n in policy['grids']:
 for i in policy['development_ids']:
  for method in policy['prior_portfolio']+[policy['candidate']]:
   a=[r for r in rows if (r['size'],r['index'],r['method'])==(n,i,method)];timed=[r for r in a if not r['warmup']]
   assert len(a)==4 and len(timed)==3
   valid=all(r['finite'] and r['status']['failed']==0 and 0<=r['independent_true_residual']<=1e-5 and r['evaluated_error_bound'] is not None and 0<=r['evaluated_error_bound']<=1e-3 and r['field_difference']<=1e-4 for r in a)
   assert valid==all(r['accuracy_passed'] for r in a)
   c={k:s.median(r[k] for r in timed) for k in ('setup_seconds','accepted_seconds','setup_plus_accepted_seconds')}
   c.update(size=n,index=i,method=method,eligible=valid,iterations=s.median(r['status']['iterations'] for r in timed),calls=s.median(r['status']['classical_calls'] for r in timed),rejected_injections=s.median(r['status']['injection_rejected'] for r in timed),application_seconds=s.median(r['seconds'] for r in apps if (r['size'],r['index'],r['method'])==(n,i,method)))
   cases[n,i,method]=c
for n in policy['grids']:
 details=[]
 for i in policy['development_ids']:
  a=cases[n,i,'gain_two'];pri=[cases[n,i,m] for m in policy['prior_portfolio'] if cases[n,i,m]['eligible']]
  q=min(pri,key=lambda c:(c['setup_plus_accepted_seconds'],c['method']));w=min(pri,key=lambda c:(c['accepted_seconds'],c['method']))
  acquire=sum(p['total_diagnostic_seconds'] for p in prefix if (p['size'],p['index'])==(n,i))+sum(g['total_geometry_diagnostic_seconds'] for g in geo if (g['size'],g['index'])==(n,i))
  details.append(dict(index=i,candidate=a,q1_baseline=q,warm_baseline=w,unique_acquisition_seconds=acquire,charged_q1=a['setup_plus_accepted_seconds']+acquire))
 mean=lambda f:s.mean(f(d) for d in details)
 qb=mean(lambda d:d['q1_baseline']['setup_plus_accepted_seconds']);wb=mean(lambda d:d['warm_baseline']['accepted_seconds'])
 cq=mean(lambda d:d['candidate']['setup_plus_accepted_seconds']);cw=mean(lambda d:d['candidate']['accepted_seconds']);cs=mean(lambda d:d['candidate']['setup_seconds']);ca=mean(lambda d:d['unique_acquisition_seconds'])
 qgain=1-cq/qb;wgain=1-cw/wb;chargedgain=1-(cq+ca)/qb
 worst=max(d['candidate']['setup_plus_accepted_seconds']/d['q1_baseline']['setup_plus_accepted_seconds']-1 for d in details)
 chargedworst=max(d['charged_q1']/d['q1_baseline']['setup_plus_accepted_seconds']-1 for d in details)
 eligible=all(d['candidate']['eligible'] for d in details)
 fullmethod='full_raw' if n==128 else 'full_segmented';q3method='q3_raw' if n==128 else 'q3_segmented'
 fullwarm=s.mean(cases[n,i,fullmethod]['accepted_seconds'] for i in policy['development_ids']);q3warm=s.mean(cases[n,i,q3method]['accepted_seconds'] for i in policy['development_ids'])
 cost.append(dict(size=n,cases=details,candidate_backend=policy['candidate_backend_by_grid'][str(n)],q1_baseline_seconds=qb,warm_baseline_seconds=wb,candidate_q1_seconds=cq,candidate_warm_seconds=cw,candidate_setup_seconds=cs,diagnostic_acquisition_seconds=ca,
  primary_q1_saving=qgain,warm_saving=wgain,research_charged_q1_saving=chargedgain,worst_q1_regression=worst,research_charged_worst_q1_regression=chargedworst,
  primary_gate=eligible and qgain>=.1 and wgain>=.1 and worst<=.1,research_charged_gate=eligible and chargedgain>=.1 and wgain>=.1 and chargedworst<=.1,
  same_backend_full_warm_saving=1-cw/fullwarm,same_backend_q3_warm_saving=1-cw/q3warm,
  necessary_whole_warm_allowance_no_new_cost=min(.9*wb,.9*qb-cs),necessary_whole_warm_allowance_research_charged=min(.9*wb,.9*qb-cs-ca),
  required_further_warm_reduction=max(0.,1-min(.9*wb,.9*qb-cs)/cw),budget_scope='Necessary mean accounting ceiling with current setup, zero additional learning/acquisition cost; not achievable oracle budget or hardware time floor.'))
 for label in ('source','prefix4'):
  a=[g for g in geo if g['size']==n and g['context']==label]
  geometry.append(dict(size=n,context=label,mean_squared_best_line_gain_vs_full=s.mean(g['squared_best_line_reduction_vs_full'] for g in a),mean_unit_squared_gain_vs_full=s.mean(g['unit_squared_reduction_vs_full'] for g in a),mean_best_line_remaining={m:s.mean(g['composite_'+m]['best_real_line_remaining_ratio'] for g in a) for m in ('q3','full','gain_two')}))
result=dict(cost=cost,geometry_summary=geometry,cases=list(cases.values()),counts=dict(full=len(rows),timed=sum(not r['warmup'] for r in rows),entries=len(checks),applications=len(apps),geometry=len(geo),prefixes=len(prefix)),all_accuracy=all(r['accuracy_passed'] for r in rows),all_entry_parity=all(c['passed'] for c in checks),max_residual=max(r['independent_true_residual'] for r in rows),max_evaluated_bound=max(r['evaluated_error_bound'] for r in rows),max_paired_field_difference=max(r['field_difference'] for r in rows),min_margin=min(r['evaluated_margin'] for r in rows),unique_diagnostic_acquisition_seconds=sum(p['total_diagnostic_seconds'] for p in prefix)+sum(g['total_geometry_diagnostic_seconds'] for g in geo),settings=settings,execution=execution,
 primary_both_grid_gate=all(c['primary_gate'] for c in cost),research_charged_both_grid_gate=all(c['research_charged_gate'] for c in cost))
(out/'RESULT_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ('counts','all_accuracy','all_entry_parity','max_residual','max_evaluated_bound','unique_diagnostic_acquisition_seconds','primary_both_grid_gate','research_charged_both_grid_gate','cost','geometry_summary')},indent=2))
