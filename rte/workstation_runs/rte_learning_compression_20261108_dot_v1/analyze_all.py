"""Reproduce paired physical-case reductions without numerical solver work."""
import json,statistics as st,random
from pathlib import Path
P=Path(__file__).resolve().parent
result={}
for folder in ["oracle_01","geometry_01","semicoarse_01"]:
 rows=json.loads((P/folder/"rows.json").read_text());contexts=json.loads((P/folder/"contexts.json").read_text())
 methods=list(dict.fromkeys(r["method"] for r in rows));cases={}
 for method in methods:
  cases[method]=[]
  for i in range(16):
   v=[r for r in rows if r["method"]==method and r["index"]==i and r["repeat"]>=0];assert len(v)==3
   cases[method].append(dict(index=i,warm_ms=1000*st.median(r["accepted_seconds"] for r in v),
    Q1_ms=1000*st.median(r["Q1_seconds"] for r in v),setup_ms=1000*st.median(r["setup_seconds"] for r in v),
    classical_calls=st.median(r["status"]["classical_calls"] for r in v),failures=sum(not r["accepted"] for r in v)))
 base=cases["TSA16_native"];summaries=[]
 for method,v in cases.items():
  a=dict(method=method,per_case=v,failures=sum(x["failures"] for x in v),
   **{k:st.mean(x[k] for x in v) for k in ["warm_ms","Q1_ms","setup_ms","classical_calls"]})
  for metric in ["warm_ms","Q1_ms"]:
   b=st.mean(x[metric] for x in base);a[metric+"_percent_vs_native"]=100*(a[metric]/b-1)
   rng=random.Random(20261109);boot=[]
   for _ in range(2000):
    ids=[rng.randrange(16) for _ in range(16)]
    boot.append(100*(st.mean(v[i][metric] for i in ids)/st.mean(base[i][metric] for i in ids)-1))
   boot.sort();a[metric+"_bootstrap95_percent"]=[boot[50],boot[1949]]
  a["remaining_ms_for_5pct_target"]=.95*st.mean(x["Q1_ms"] for x in base)-a["Q1_ms"]
  if method in contexts[0].get("representations",{}):
   metrics=[c["representations"][method] for c in contexts]
   a["coefficient_complex_count"]=metrics[0]["coefficient_complex_count"]
   a["dof_fraction"]=metrics[0]["dof_fraction"]
   for key in ["field_relative_error","optimal_quotient_residual_ratio","decoder_seconds"]:
    vals=[m[key] for m in metrics]
    a[key]=dict(min=min(vals),max=max(vals),mean=st.mean(vals),median=st.median(vals))
  summaries.append(a)
 result[folder]=dict(settings=json.loads((P/folder/"settings.json").read_text()),summaries=summaries,
  recorded_failures=sum(not r["accepted"] for r in rows),max_original_residual=max(r["true_residual"] for r in rows),
  max_bound=max(r["evaluated_relative_bound"] for r in rows),max_field=max(r["relative_field_difference"] for r in rows))
 if folder=="semicoarse_01":
  result[folder]["implementation_checks"]={k:dict(max=max(c[k] for c in contexts),mean=st.mean(c[k] for c in contexts))
   for k in ["streaming_commutation_error","restriction_prolongation_error","collocation_RAP_relative_defect_on_prefix","semicoarse_setup"]}
result["qualification"]="Each case median of3 repeats, then equal physical-case mean. Same16 reused N128 development media throughout. Bootstrap conditional on these media, not independent confirmation; setup was measured once per case. Oracle coefficient acquisition is free, inference absent. No N256 representation claim."
(P/"combined_analysis.json").write_text(json.dumps(result,indent=2)+"\n")
for name in ["oracle_01","geometry_01","semicoarse_01"]:
 print(name)
 for r in result[name]["summaries"]:
  print(r["method"],"warm %.3f Q1 %.3f calls %.4f change %+.2f%% budget5 %.3f"%(r["warm_ms"],r["Q1_ms"],r["classical_calls"],r["Q1_ms_percent_vs_native"],r["remaining_ms_for_5pct_target"]))
