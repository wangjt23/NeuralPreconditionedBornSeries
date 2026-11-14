"""Deterministic stdlib scalar reduction; no source, field, CUDA or solver."""
import argparse,hashlib,json,pathlib

def analyze(root):
 root=pathlib.Path(root)
 load=lambda name:json.loads((root/name).read_text())
 policy=load('policy.json');settings=load('results/settings.json');receipt=load('execution_receipt.json')
 actions=load('results/actions.json');fixtures=load('results/exact_fixture.json');contexts=load('results/contexts.json');checks=load('results/checks.json')
 assert policy['max_full_solves']==0 and settings['full_solves']==0 and receipt['full_solves']==0
 assert settings['diagnostic_only']and not settings['certified']and not settings['global_backend_contract_established']
 assert hashlib.sha256((root/'policy.json').read_bytes()).hexdigest()==receipt['policy_sha256']
 for name,digest in policy['code_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 complete=settings['complete']and receipt['exit_code']==0
 if complete:
  assert len(actions)==72 and len(fixtures)==12 and len(contexts)==len(checks)==4
  expected=dict(full_grid_gpu_H_actions=72,fixture_gpu_H_actions=24,preconditioner_actions=8,numpy_fp64_references=48,exact_fixture_cases=12,exact_fixture_apply_calls=24,full_solves=0)
  assert settings['counters']==expected
 groups=[]
 for context in contexts:
  n,i=context['size'],context['index'];rows=[r for r in actions if(r['size'],r['index'])==(n,i)]
  group=dict(size=n,index=i,angular_count=context['angular_count'],scales=context['scales'],prospective_outer_error_budget=context['prospective_outer_error_budget'],stored_D32_D64_max_absolute_difference=context['stored_D32_D64_max_absolute_difference'],actions=[])
  for name in sorted({r['action']for r in rows}):
   selected=[r for r in rows if r['action']==name]
   assert all(r['action_rounding_vs_matched_nominal']['finite']and r['overall_vs_original_physical_target']['finite']for r in selected)
   key='absolute_error_per_input_norm';worst=max(selected,key=lambda r:r['action_rounding_vs_matched_nominal'][key])
   allowed=context['prospective_outer_error_budget']['allowed_outer_absolute_error_per_field_norm']
   observed=worst['action_rounding_vs_matched_nominal'][key]
   group['actions'].append(dict(action=name,records=len(selected),max_observed_absolute_error_per_input_norm=observed,worst_probe=worst['probe'],max_observed_image_relative_error=max(r['action_rounding_vs_matched_nominal']['image_relative_error']for r in selected),max_observed_error_over_Lambda_input_norm=max(r['action_rounding_vs_matched_nominal']['error_over_Lambda_input_norm']for r in selected),max_observed_overall_physical_error_per_input_norm=max(r['overall_vs_original_physical_target'][key]for r in selected),observed_over_prospective_outer_budget=observed/allowed if name=='physical_high_complex128'and allowed>0 else None,reference_uncertainty_bound_established=False))
  groups.append(group)
 toy_groups=[]
 for name in sorted({a['action']for f in fixtures for a in f['actions']}):
  entries=[(f,a)for f in fixtures for a in f['actions']if a['action']==name]
  key='absolute_error_per_input_norm_upper';f,a=max(entries,key=lambda item:item[1]['matched_exact_nominal_error'][key])
  assert all(a['matched_exact_nominal_error']['finite']for _,a in entries)
  toy_groups.append(dict(action=name,cases=len(entries),max_exact_fixture_error_per_input_norm_upper=a['matched_exact_nominal_error'][key],worst=dict(size=f['size'],index=f['index'],probe=f['probe']),max_exact_fixture_physical_target_error_per_input_norm_upper=max(a['overall_exact_physical_target_error'][key]for _,a in entries),scope='Verified finite-input exact4x4 fixture only; no full-grid/global error contract'))
 return dict(complete=complete,exit_code=receipt['exit_code'],policy_sha256=receipt['policy_sha256'],counts=settings['counters'],contexts=groups,exact_fixture_summary=toy_groups,all_ownership_source_checks_passed=all(c['passed']for c in checks),all_full_grid_input_cast_norms_zero=all(r['input_cast_norm']==0 for r in actions),input_cast_scope='All predefined fields originateC64; no nonrepresentableC128 casting test',process_wall_seconds=receipt['process_wall_seconds'],driver_reported_seconds=settings['owned_process_seconds'],peak_allocated_bytes=settings['peak_allocated_bytes'],peak_reserved_bytes=settings['peak_reserved_bytes'],costs=settings['costs'],backend_flags=settings['backend_flags_read_only'],certified_global_contract=False,certified_full_solver_admission='NO-GO',precision_repair_selected=False,old_complex_entry_failure_preserved=True,full_solves=0,training=False,test9626_touched=False,original_source_exported=False,interpretation='NumPy comparisons are separately rounded finite engineering evidence; exact4x4 brackets certify only fixture differences; observed ratios do not establish missing global backend assumptions or solver economics')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('output');args=parser.parse_args()
 result=analyze(args.root);pathlib.Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
