"""Reproduce scalar reductions. No solver, training, or GPU work."""
import json, statistics, random, hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
def reduce(folder, variants=False):
    rows=json.loads((P/folder/"rows.json").read_text())
    measured=[r for r in rows if r["repeat"]>=0]
    groups={}
    for r in measured:
        k=(r["grid"],r.get("variant","all"),r["method"],r["index"])
        groups.setdefault(k,[]).append(r)
    case={}
    for k,v in groups.items():
        case[k]=dict(index=k[3], n=len(v), failures=sum(not r["accepted"] for r in v),
            warm_ms=statistics.median(r["accepted_seconds"] for r in v)*1000,
            Q1_ms=statistics.median(r["Q1_seconds"] for r in v)*1000,
            setup_ms=statistics.median(r["setup_seconds"] for r in v)*1000,
            calls=statistics.median(r["status"].get("classical_calls",r["status"]["iterations"]) for r in v),
            iterations=statistics.median(r["status"]["iterations"] for r in v),
            max_r=max(r["true_residual"] for r in v),
            max_bound=max(float(r["evaluated_relative_bound"]) for r in v),
            max_field=max(r["relative_field_difference"] for r in v))
    cohorts={}
    for k,v in case.items():cohorts.setdefault(k[:3],[]).append(v)
    out=[]
    for k,vs in sorted(cohorts.items()):
        native=[case[(k[0],k[1],"TSA16_native",v["index"])] for v in vs]
        record=dict(grid=k[0],variant=k[1],method=k[2],cases=len(vs),attempts=sum(v["n"] for v in vs),
            failures=sum(v["failures"] for v in vs),
            **{f:statistics.mean(v[f] for v in vs) for f in ["warm_ms","Q1_ms","setup_ms","calls","iterations"]},
            **{f:max(v[f] for v in vs) for f in ["max_r","max_bound","max_field"]},
            per_case=sorted(vs,key=lambda v:v["index"]))
        for f in ["warm_ms","Q1_ms"]:
            d=statistics.mean(v[f] for v in vs)-statistics.mean(v[f] for v in native)
            b=statistics.mean(v[f] for v in native)
            record[f+"_difference_vs_native"]=d
            record[f+"_percent_vs_native"]=100*d/b
            rng=random.Random(20261108+k[0])
            boot=[]
            for _ in range(4000):
                picks=[rng.randrange(len(vs)) for _ in vs]
                a=statistics.mean(vs[i][f] for i in picks)
                b0=statistics.mean(native[i][f] for i in picks)
                boot.append(100*(a/b0-1))
            boot.sort()
            record[f+"_paired_bootstrap95_percent"]=[boot[100],boot[3899]]
        out.append(record)
    return dict(recorded_attempts=len(rows),measured_attempts=len(measured),
                recorded_failures=sum(not r["accepted"] for r in rows),
                measured_failures=sum(not r["accepted"] for r in measured), summaries=out)
a={name:reduce(name) for name in ["complete_01","frontier_01"]}
a["qualification"]="Medians of three repeats per physical material, then equal-weight physical-material mean. Complete_01 is four reused development cases per grid; frontier_01 is all16 reused development cases. Bootstrap resamples paired physical materials, conditional on reused development set, not independent confirmation. Setup measured once per material; timing intervals do not resolve setup variability. Failures are retained, not credited as valid speedups."
def clean(o):
    if isinstance(o,float) and (o==float("inf") or o==-float("inf")):return str(o)
    if isinstance(o,dict):return {k:clean(v) for k,v in o.items()}
    if isinstance(o,list):return [clean(v) for v in o]
    return o
(P/"analysis.json").write_text(json.dumps(clean(a),indent=2)+"\n")
for name in ["complete_01","frontier_01"]:
    print(name, "recorded",a[name]["recorded_attempts"],"failed",a[name]["recorded_failures"])
    for r in a[name]["summaries"]:
        print(r["grid"],r["variant"],r["method"],"warm=%.3f Q1=%.3f calls=%.4f fails=%d warmdiff=%+.2f%% Q1diff=%+.2f%%"%(r["warm_ms"],r["Q1_ms"],r["calls"],r["failures"],r["warm_ms_percent_vs_native"],r["Q1_ms_percent_vs_native"]), "warmCI",*[round(x,2) for x in r["warm_ms_paired_bootstrap95_percent"]])
