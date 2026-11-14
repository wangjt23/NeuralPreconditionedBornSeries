"""Regenerate summaries from immutable JSON receipts; no torch required."""
import json,hashlib,statistics
from pathlib import Path
P=Path(__file__).resolve().parent

def read(run,name):return json.loads((P/run/(name+'.json')).read_text())
def check(file,h):
 actual=hashlib.sha256((P/file).read_bytes()).hexdigest()
 if h!=actual:raise RuntimeError(f'source hash mismatch: {file}')
result={}
s=read('cpu_probe_01','settings');check('recycle_probe.py',s['source_hash']);assert s['complete'];assert s['production_hashes']==s['production_hashes_after']
r=read('cpu_probe_01','rows');a=read('cpu_probe_01','audits')
result['recycling']=dict(attempts=len(r),failures=sum(not x['accepted'] for x in r),wall_seconds=s['wall_seconds'],scope='One CPU mechanism pass (recorded repeat=-1), not a repeated timing confirmation.',methods={x['method']:dict(batch_seconds=x['batch_complete_seconds'],calls=[y['status']['iterations'] for y in r if y['method']==x['method']],rank=x['rank'],AU_C_max=x['AU_C_max'],orthogonality_defect=x['orthogonality_defect'],basis_bytes=x['bytes']) for x in a})
for run in ['exact_background_01','nested_background_01']:
 if not (P/run).exists():continue
 s=read(run,'settings');assert s['complete']
 for file,h in s['source_hashes'].items():check(file,h)
 r=read(run,'rows');methods={}
 for m in s['methods']:
  rows=[x for x in r if x['method']==m]
  setup=s['method_shared_setup_seconds'][m];online=sum(x['accepted_seconds'] for x in rows)
  methods[m]=dict(setup_seconds=setup,online_four_rhs_seconds=online,complete_four_rhs_seconds=setup+online,calls=[x['status']['classical_calls'] for x in rows],Q1_seconds=[x['Q1_seconds'] for x in rows],inner_calls=[v['iterations'] for x in rows for v in x.get('inner_records',[])],inner_seconds=sum(v['seconds'] for x in rows for v in x.get('inner_records',[])),inner_failed=sum(v['failed'] for x in rows for v in x.get('inner_records',[])),inner_true_residuals=[v['true_residual'] for x in rows for v in x.get('inner_records',[])])
 base=methods['compact4_inject']
 for m,d in methods.items():
  d['batch_gain_vs_compact4_percent']=100*(1-d['complete_four_rhs_seconds']/base['complete_four_rhs_seconds'])
  # Linear extrapolation is a cost diagnostic only, not an actual new workload.
  savings_per_rhs=(base['online_four_rhs_seconds']-d['online_four_rhs_seconds'])/4
  d['estimated_break_even_rhs_vs_compact4']=max(0.,(d['setup_seconds']-base['setup_seconds'])/savings_per_rhs) if savings_per_rhs>0 else None
 result[run]=dict(attempts=len(r),failures=sum(not x['accepted'] for x in r),wall_seconds=s['wall_seconds'],methods=methods,scope=s['interpretation'],max_residual=max(x['true_residual'] for x in r),max_bound=max(x['evaluated_relative_bound'] for x in r),max_field_difference=max(x['relative_field_difference'] for x in r))
for run in ['mode_probe_01','block_costs_01']:
 if (P/run).exists():
  v=read(run,'results');check('exact_mode_probe.py' if run=='mode_probe_01' else 'exact_block_costs.py',v['source_hash']);result[run]=v
result['accounting']=dict(complete_outer_solves=sum(v['attempts'] for v in result.values() if isinstance(v,dict) and 'attempts' in v),failed_outer_solves=sum(v['failures'] for v in result.values() if isinstance(v,dict) and 'failures' in v),total_recorded_main_wall_seconds=sum(v.get('wall_seconds',0) for v in result.values() if isinstance(v,dict)),GPU_used=False,neural_training=False,new_teachers=False,heldout9626=False)
(P/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
