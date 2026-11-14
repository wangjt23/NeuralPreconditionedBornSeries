"""CPU analysis of no-training compressed-answer oracle receipts."""
import json,statistics as st,random,math
from pathlib import Path
P=Path(__file__).resolve().parent
rows=json.loads((P/"oracle_01/rows.json").read_text());ctx=json.loads((P/"oracle_01/contexts.json").read_text())
methods=list(dict.fromkeys(r["method"] for r in rows));cases={}
for m in methods:
 cases[m]=[]
 for i in range(16):
  z=[r for r in rows if r["method"]==m and r["index"]==i and r["repeat"]>=0]
  assert len(z)==3 and all(r["accepted"] for r in z)
  cases[m].append(dict(index=i,warm_ms=1000*st.median(r["accepted_seconds"] for r in z),
   Q1_ms=1000*st.median(r["Q1_seconds"] for r in z),calls=st.median(r["status"]["classical_calls"] for r in z),
   injections=st.median(r["status"].get("injection_calls",0) for r in z)))
base=cases["TSA16_native"];out=[]
for m,v in cases.items():
 a=dict(method=m,**{k:st.mean(x[k] for x in v) for k in ["warm_ms","Q1_ms","calls","injections"]},per_case=v)
 for metric in ["warm_ms","Q1_ms"]:
  b=st.mean(x[metric] for x in base)
  a[metric+"_percent"]=100*(a[metric]/b-1)
  rng=random.Random(20261109);bs=[]
  for _ in range(2000):
   picks=[rng.randrange(16) for _ in range(16)]
   bs.append(100*(st.mean(v[i][metric] for i in picks)/st.mean(base[i][metric] for i in picks)-1))
  bs.sort();a[metric+"_bootstrap95_percent"]=[bs[50],bs[1949]]
 a["inference_budget_for_5pct_ms"]=.95*st.mean(x["Q1_ms"] for x in base)-a["Q1_ms"]
 if m in ctx[0]["representations"]:
  z=[c["representations"][m] for c in ctx]
  a["coefficient_complex_count"]=z[0]["coefficient_complex_count"]
  a["dof_fraction"]=z[0]["dof_fraction"]
  for key in ["field_relative_error","raw_original_residual","optimal_quotient_residual_ratio","decoder_seconds"]:
   w=[x[key] for x in z];a[key]=dict(min=min(w),max=max(w),mean=st.mean(w),median=st.median(w))
 out.append(a)
result=dict(scope="Full16 reused N128 development. Free exact teacher coefficients, actual decoder and continuation cost. Not an ML result.",rows=len(rows),failures=sum(not r["accepted"] for r in rows),
 max_original_residual=max(r["true_residual"] for r in rows),
 max_bound=max(r["evaluated_relative_bound"] for r in rows),
 max_field=max(r["relative_field_difference"] for r in rows),summaries=out)
(P/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
for r in out:
 print(r["method"],"warm %.3f Q1 %.3f calls %.4f Q1diff %+.2f%% budget5 %.3f"%(r["warm_ms"],r["Q1_ms"],r["calls"],r["Q1_ms_percent"],r["inference_budget_for_5pct_ms"]),r["Q1_ms_bootstrap95_percent"])
 if "field_relative_error" in r:
  print(" field",r["field_relative_error"],"quotient rho",r["optimal_quotient_residual_ratio"],"decoder",r["decoder_seconds"])
