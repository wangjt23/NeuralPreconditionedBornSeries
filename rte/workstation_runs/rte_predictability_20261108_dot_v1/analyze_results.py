"""Reproduce scalar summaries from immutable recorded rows. Python standard library."""
import json,statistics,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
sources={"prepare_01":"prepare_predictability.py","predict_01":"fit_and_evaluate.py","matched_01":"matched_rank.py","screen_01":"screen_structured.py","registered_01":"registered_frame.py","registered_cost_01":"registered_cost.py","depth_01":"screen_depth.py","hardgate_01":"hardgate.py","hardgate_born_02":"hardgate_born.py","hardgate_final_03":"hardgate_final.py"}
out={"aggregation":"Median of3 measured repeats within each case, then equal-case mean. Warmup retained but excluded. Failed rows remain visible and never called speedups.","stages":{}}
for stage,source in sources.items():
 s=json.loads((P/stage/"settings.json").read_text());actual=hashlib.sha256((P/source).read_bytes()).hexdigest()
 assert actual==s["source_hash"],(stage,actual,s["source_hash"])
 assert s["complete"],stage
 st={"complete":True,"source_hash":actual,"wall_seconds":s["wall_seconds"],"recorded_attempts":s.get("rows",0),"recorded_failures":s.get("failures",0)}
 if (P/stage/"rows.json").exists():
  allrows=json.loads((P/stage/"rows.json").read_text());groups={}
  for row in allrows:
   if row["repeat"]<0:continue
   group=(row.get("split",""),row.get("condition",""),row["method"])
   groups.setdefault(group,[]).append(row)
  st["methods"]=[]
  for (split,condition,method),rows in groups.items():
   idkey="index" if "index" in rows[0] else "source_id";ids=sorted({r[idkey] for r in rows})
   metric=lambda k:statistics.mean(statistics.median(r[k] for r in rows if r[idkey]==i) for i in ids)
   v=dict(split=split,condition=condition,method=method,cases=len(ids),measured_attempts=len(rows),failures=sum(not r["accepted"] for r in rows),
    mean_case_median_accepted_ms=metric("accepted_seconds")*1000,mean_case_median_Q1_ms=metric("Q1_seconds")*1000,
    classical_calls=statistics.mean(r["status"].get("classical_calls",r["status"]["iterations"]) for r in rows))
   if "shared_setup_seconds" in rows[0]:
    setups={r["shared_setup_seconds"] for r in rows};assert len(setups)==1
    v["shared_setup_ms"]=next(iter(setups))*1000;v["actual_batch4_ms"]=v["shared_setup_ms"]+4*v["mean_case_median_accepted_ms"]
   st["methods"].append(v)
 out["stages"][stage]=st
out["totals"]={"stage_wall_seconds":sum(x["wall_seconds"] for x in out["stages"].values()),
 "recorded_end_to_end_attempts":sum(x["recorded_attempts"] for x in out["stages"].values()),
 "recorded_end_to_end_failures":sum(x["recorded_failures"] for x in out["stages"].values()),
 "new_unique_medium_source_teachers":4,"new_exact_reference_solve_executions":12,"auxiliary_truncated_rhs_solves":32,
 "existing_teacher_cases_reused":144,"neural_training":False,"fitted_model":"One kernel-ridge/POD model on96 existing train IDs, hyperparameter selected on32 internal IDs.",
 "reused_development":16,"reserved_seed9626_touched":False,"N256_tested_this_round":False}
c=json.loads((P/"prepare_01/counterfactuals.json").read_text())
out["input_counterfactuals"]={}
for cut in [8,16]:
 z=[r for r in c if r["cut"]==cut]
 out["input_counterfactuals"][str(cut)]={k:{"mean":statistics.mean(r[k] for r in z),"max":max(r[k] for r in z)} for k in ["retained_output_relative_difference","removed_input_relative_norm","visible_residual_identity_error","auxiliary_explicit_residual"]}
reg=json.loads((P/"registered_01/settings.json").read_text())
out["registration"]={name:{split:{rank:{"mean":statistics.mean(vals),"max":max(vals)} for rank,vals in errors.items()} for split,errors in rec["errors"].items()} for name,rec in reg["results"].items()}
final=out["stages"]["hardgate_final_03"]["methods"]
valid=[r for r in final if r["method"]!="oracle_C8" and r["failures"]==0]
best=min(valid,key=lambda x:x["actual_batch4_ms"]);oracle=next(r for r in final if r["method"]=="oracle_C8");baseline=next(r for r in final if r["method"]=="tsa64")
out["hard_scene_gate"]={"best_classical_batch4":best,"oracle":oracle,"tsa64":baseline,
 "oracle_batch4_change_vs_best_classical_percent":100*(oracle["actual_batch4_ms"]/best["actual_batch4_ms"]-1),
 "oracle_warm_change_vs_best_classical_percent":100*(oracle["mean_case_median_accepted_ms"]/best["mean_case_median_accepted_ms"]-1),
 "extra_per_RHS_ms_for_5percent_batch4_gain":(.95*best["actual_batch4_ms"]-oracle["actual_batch4_ms"])/4,
 "extra_per_RHS_ms_break_even_batch4":(best["actual_batch4_ms"]-oracle["actual_batch4_ms"])/4}
(P/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out["totals"],indent=2));print(json.dumps(out["hard_scene_gate"],indent=2))
