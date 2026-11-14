"""Scalar-only exact policy audit; no numerical fields or native execution."""
import argparse,json,math,pathlib,statistics
from fractions import Fraction as F
def f(x):return F.from_float(float(x))
def load(p):return json.loads(p.read_text())
def numerical(value):
 if isinstance(value,dict):return {k:numerical(v)for k,v in value.items()if 'seconds'not in k and k not in ('costs','native_backend_guard_before','native_backend_guard_after')}
 if isinstance(value,list):return[numerical(v)for v in value]
 return value
def certificate(c):
 q,r,e,center,x=map(f,[c['source_norm_lower'],c['residual_norm_upper'],c['residual_radius_norm_upper'],c['residual_center_norm_upper'],c['candidate_norm_lower']])
 m=F(c['coercivity_lower_rational']);tau=F(1,100000);alpha=F(1,100);den=m*x-r
 assert c['policy_tolerance_rational']==str(tau)and c['reserve_fraction_rational']==str(alpha)
 assert c['policy_solution_tolerance_rational']==str(F(1,1000))
 tests=dict(residual_accepted=q>0 and r<=tau*q,reserve_center_accepted=q>0 and center<=(1-alpha)*tau*q,
  reserve_radius_accepted=q>0 and e<=alpha*tau*q,solution_error_accepted=den>0 and r<=F(1,1000)*den)
 accuracy=all(tests.values());tests['accuracy_accepted_without_paired_field']=accuracy
 pair=f(c['paired_field_baseline_norm_lower'])>0 and f(c['paired_field_difference_norm_upper'])<=F(1,10000)*f(c['paired_field_baseline_norm_lower'])
 tests['paired_field_accepted']=pair;tests['accepted']=accuracy and pair
 assert all(c[k]==v for k,v in tests.items()),'Stored predicate mismatch'
 assert r>=center+e
 assert f(c['relative_radius_norm_upper'])*q>=e
 assert f(c['relative_residual_upper'])*q>=r
 assert c['global_gpu_contract_established'] is False
 for g in ('native_backend_guard_before','native_backend_guard_after'):assert c[g]['guard_flags']==127
 for stage in c['stage_ledger']:
  if stage['kind']in('forward_fft_stage','inverse_fft_stage'):
   assert f(stage['stage_operator_norm_upper'])**2>=2
   assert f(stage['radius_upper'])>=f(stage['stage_operator_norm_upper'])*f(stage['incoming_radius_upper'])+f(stage['local_radius_upper'])
  elif stage['kind']=='inverse_normalization':
   assert f(stage['radius_upper'])>=f(stage['incoming_radius_upper'])*F(stage['normalization_rational'])+f(stage['local_radius_upper'])
  elif stage['kind']=='stored_D64_multiplication':
   assert f(stage['radius_upper'])>=f(stage['operator_norm_upper'])*f(stage['incoming_radius_upper'])+f(stage['local_radius_upper'])
 return dict(accepted=c['accepted'],relative_residual_upper=c['relative_residual_upper'],relative_radius_upper=c['relative_radius_norm_upper'],
  reserve_usage_fraction=float(e/(alpha*tau*q)),relative_solution_error_upper=c['relative_solution_error_upper'],paired_relative_upper=c['paired_field_relative_upper'])
def main(root,output):
 batch=load(root/'batch_results.json');assert batch['complete']and batch['all_processes_successful']
 assert len(batch['jobs'])==5 and not any(r['timeout']for r in batch['jobs'])
 pre=load(root/'preflight_receipt.json');assert pre['prior_files_verified']==576 and pre['tests_returncode']==0
 a=load(root/'jobA/results/certificates.json');b=load(root/'jobB/certificates.json');pairs=load(root/'jobA/results/pairs.json')
 assert len(a)==8 and len(b)==4 and len(pairs)==12
 evidence=[]
 for arm,rows in [('A',a),('B',b)]:
  for row in rows:
   evidence.append(dict(arm=arm,size=row['size'],index=row['index'],role=row['role'],call_seconds=row['certificate_call_seconds'],**certificate(row['certificate'])))
 for row in pairs:
  accept=f(row['paired_field_baseline_norm_lower'])>0 and f(row['paired_field_difference_norm_upper'])<=F(1,10000)*f(row['paired_field_baseline_norm_lower'])
  reject=f(row['paired_field_baseline_norm_upper'])>0 and f(row['paired_field_difference_norm_lower'])>F(1,10000)*f(row['paired_field_baseline_norm_upper'])
  assert accept==row['paired_field_accepted']and reject==row['paired_field_rigorous_rejection']
  assert accept if row['role']=='positive'else reject
 indexed={(r['size'],r['index'],r['role']):r for r in a};comparisons=[]
 for row in b:
  original=indexed[(row['size'],row['index'],row['role'])]
  assert row['snapshot_sha256']==original['snapshot_sha256']
  assert numerical(row['certificate'])==numerical(original['certificate']),'A/B numerical output mismatch'
  comparisons.append(dict(size=row['size'],index=row['index'],role=row['role'],identical_numerical_certificate=True,
   A_seconds=original['certificate_call_seconds'],B_seconds=row['certificate_call_seconds'],
   descriptive_call_saving_fraction=1-row['certificate_call_seconds']/original['certificate_call_seconds']))
 verdicts={};physics=[]
 for job,expected in [('C',dict(H=52,M=28,S=16)),('D',dict(H=36,M=4,S=0))]:
  rows=load(root/('job'+job)/'results.json');settings=load(root/('job'+job)/'settings.json');manifest=load(root/('job'+job)/'candidates/manifest.json')
  assert settings['complete'] and settings['full_solves']==settings['training_steps']==0
  assert all(settings['counts'][k]==v for k,v in expected.items())
  assert len(rows)==(24 if job=='C'else 16)
  assert len(manifest['snapshots'])==(60 if job=='C'else 36)
  assert all(v['saved_before_physical_check']for v in manifest['snapshots'])
  assert all(v['constructed_image_linearity_gap']<=1e-10 and v['certification_established']is False for v in rows)
  checks=[]
  for n in (128,256):
   for context in ('source','one_strong_action_defect'):
    groups={name:[v for v in rows if v['size']==n and v['context']==context and v['map']==name]for name in set(v['map']for v in rows)}
    assert all(len(v)==2 for v in groups.values())
    means={name:dict(mean_actual_line_ratio=statistics.mean(v['constructed_field_actual_H_line_ratio']for v in values),
      mean_squared_actual_line_ratio=statistics.mean(v['constructed_field_actual_H_line_ratio']**2 for v in values),
      mean_apply_seconds=statistics.mean(v['median_apply_seconds']for v in values))for name,values in groups.items()}
    if job=='C':
     quality=means['material_local']['mean_squared_actual_line_ratio']<=means['strong']['mean_squared_actual_line_ratio']+.02
     cost=means['material_local']['mean_apply_seconds']<=.25*means['strong']['mean_apply_seconds'];passed=quality and cost
     checks.append(passed);physics.append(dict(job=job,size=n,context=context,means=means,quality_screen=quality,component_cost_screen=cost,screen_passed=passed))
    else:
     gain=means['reference_column']['mean_squared_actual_line_ratio']-means['teacher_column']['mean_squared_actual_line_ratio']
     passed=gain>=.02;checks.append(passed);physics.append(dict(job=job,size=n,context=context,means=means,squared_information_gain=gain,information_screen_passed=passed,acquisition_cost_unknown_not_zero=True))
  verdicts[job]=dict(all_strata_screen_passed=all(checks),passing_strata=sum(checks),strata=len(checks),fullsolver_or_learning_admission=False)
 commutator=load(root/'jobC/commutator.json');assert len(commutator)==8
 assert all(r['identity_discrepancy_relative']<=1e-10 for r in commutator)
 settingsA=load(root/'jobA/results/settings.json');settingsB=load(root/'jobB/settings.json');freeze=load(root/'jobA/snapshot_ledger.json')
 assert settingsA['complete']and settingsB['complete']and freeze['complete']
 assert freeze['counts']==dict(full_solves=0,training_steps=0,new_seed_generation=0,M=2,H=2)
 native=pre['native_sha256'];assert settingsA['native_sha256']==settingsB['native_sha256']==native
 output.mkdir(exist_ok=True)
 report=dict(complete=True,A=dict(all_eight_accepted=all(v['accepted']for v in evidence if v['arm']=='A'),positive_pairs=6,rigorous_negative_pairs=6,
   four_distinct_N256_engineering_fields=True,no_original_source_N256_solver_admission=True),
 B=dict(all_four_numerically_identical=True,comparisons=comparisons,A_matched_N256_helper_seconds=sum(v['A_seconds']for v in comparisons),B_matched_N256_helper_seconds=sum(v['B_seconds']for v in comparisons),
   concurrent_load_and_unequal_total_job_case_counts=True,controlled_cold_throughput_benchmark=False),
 certificate_evidence=evidence,physics_verdicts=verdicts,physics_strata=physics,
 commutator_summary=dict(streaming_penalty_relative_min=min(r['streaming_material_penalty_relative']for r in commutator),streaming_penalty_relative_max=max(r['streaming_material_penalty_relative']for r in commutator),
  reference_closure_relative_max=max(r['reference_closure_relative']for r in commutator),local_collision_closure_relative_max=max(r['local_collision_closure_relative']for r in commutator),identity_discrepancy_relative_max=max(r['identity_discrepancy_relative']for r in commutator)),
 costs=dict(batch_supervisor_wall_seconds=batch['elapsed_seconds'],external_job_processes=batch['jobs'],submission_seconds=load(root/'submission_receipt.json')['submission_seconds'],
  preservation_preflight_seconds=pre['preservation_seconds'],compile_seconds=pre['compile_seconds'],tests_seconds=pre['tests_seconds'],A_internal=settingsA,B_internal=settingsB,fixture_generation=freeze),
 exact_predicates_recomputed=True,stage_radius_propagation_checked=True,original_N256_NO_CERTIFICATE_preserved=True,full_solves=0,training_steps=0)
 (output/'RESULT_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:report[k]for k in ['complete','A','B','physics_verdicts','commutator_summary','full_solves','training_steps']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();main(a.root,a.output)
