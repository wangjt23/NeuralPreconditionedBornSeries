"""Root receipt reduction, independently implemented with the standard library."""
import argparse,hashlib,json,math,statistics
from pathlib import Path

def analyze(root):
    root=Path(root)
    def read(name):return json.loads((root/name).read_text())
    policy=read('policy.json');stage=read('staging_receipt.json');execution=read('execution_receipt.json')
    assert hashlib.sha256((root/'policy.json').read_bytes()).hexdigest()=='acd73134c63839726d3b699c60e795b8da90d14709a9529fb94e905ce59ebd93'
    assert stage['files_verified']==420 and stage['cpu_tests_exit_codes']==[0,0]
    for name,h in stage['code_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
    checks=read('results/checks.json');linear=read('results/admissibility.json');setup=read('results/setup.json');rows=read('results/solver.json');settings=read('results/settings.json')
    assert execution['exit_code']==0 and not execution['owned_process_timed_out'] and execution['process_wall_seconds']<=120
    assert settings['wall_seconds']<=95 and settings['peak_reserved_bytes']<=12*1024**3
    assert len(checks)==112 and all(c['passed'] for c in checks)
    assert len(linear)==108 and sum(c['action_calls'] for c in linear)==1080 and len(setup)==4
    summary=[];failed=[]
    for label,cap in policy['admissibility_tolerances'].items():
        group=[c for c in linear if c['action']==label];assert len(group)==12
        summary.append(dict(action=label,cap=cap,failed=sum(not c['passed'] for c in group),maximum_defect=max(c['max_normalized_defect'] for c in group)))
        for c in group:
            for identity,v in c['checks'].items():
                ratio=v['numerator']/v['denominator'];assert math.isclose(ratio,v['ratio'],rel_tol=1e-14)
                assert v['passed']==(ratio<=cap)
                if not v['passed']:failed.append(dict(size=c['size'],index=c['index'],action=label,probe=c['probe'],identity=identity,numerator=v['numerator'],denominator=v['denominator'],defect=ratio,cap=cap,cap_multiple=ratio/cap))
    result=dict(policy_sha256=execution['policy_sha256'],entry_obstruction=settings['entry_obstruction'],settings=settings,execution=execution,map_summary=summary,failed_identities=failed,source_start_qa=[dict(size=s['size'],index=s['index'],**s['source_start_qa']) for s in setup],mandatory_acquisition_seconds=sum(s['mandatory_admissibility_seconds'] for s in setup),cpu_tests=31)
    if settings['entry_obstruction']:
        assert not rows and settings['full_solves']==0 and failed
        result.update(outcome='Frozen numerical admissibility entry obstruction; no full solves',full_cost_comparison=None,exact_linearity_claim=False)
        return result
    assert settings['complete'] and len(rows)==96 and sum(not r['warmup'] for r in rows)==72
    assert all(r['accuracy_passed'] for r in rows)
    candidates=[]
    for size in (128,256):
        for index in (0,3):
            for method in policy['methods']:
                group=[r for r in rows if (r['size'],r['index'],r['method'])==(size,index,method)];assert len(group)==4
                timed=[r for r in group if not r['warmup']];assert len(timed)==3
                candidates.append(dict(size=size,index=index,method=method,accepted_seconds=statistics.median(r['accepted_seconds'] for r in timed),q1_seconds=statistics.median(r['setup_plus_accepted_seconds'] for r in timed),operational_q1_seconds=statistics.median(r['operational_q1_seconds'] for r in timed),classical_calls=statistics.median(r['status']['classical_calls'] for r in timed),iterations=statistics.median(r['status']['iterations'] for r in timed),eligible=all(r['accuracy_passed'] for r in group)))
    gates={}
    for ledger in ('q1_seconds','operational_q1_seconds'):
        for mode in ('contemporary_five','same_factor'):
            gs=[]
            for n in (128,256):
                detail=[]
                for i in (0,3):
                    ca=next(r for r in candidates if (r['size'],r['index'],r['method'])==(n,i,'complex_direct_segmented'))
                    controls=[r for r in candidates if (r['size'],r['index'])==(n,i) and r['method']!='complex_direct_segmented' and (mode!='same_factor' or r['method']=='direct_segmented')]
                    q=min(controls,key=lambda r:r[ledger]);w=min(controls,key=lambda r:r['accepted_seconds'])
                    detail.append(dict(index=i,candidate=ca,q1_baseline=q,warm_baseline=w))
                baseq=statistics.mean(d['q1_baseline'][ledger] for d in detail);caq=statistics.mean(d['candidate'][ledger] for d in detail)
                basew=statistics.mean(d['warm_baseline']['accepted_seconds'] for d in detail);caw=statistics.mean(d['candidate']['accepted_seconds'] for d in detail)
                qs=1-caq/baseq;ws=1-caw/basew;reg=max(d['candidate'][ledger]/d['q1_baseline'][ledger]-1 for d in detail)
                gs.append(dict(size=n,baseline_q1=baseq,candidate_q1=caq,baseline_warm=basew,candidate_warm=caw,q1_saving=qs,warm_saving=ws,worst_q1_regression=reg,passed=qs>=.1 and ws>=.1 and reg<=.1,cases=detail))
            gates[ledger+'_'+mode]=gs
    realparity=[r for r in rows if r['method'] not in ('direct_segmented','complex_direct_segmented')]
    assert len(realparity)==64 and all(r['same_recipe_field_status_parity'] for r in realparity)
    result.update(outcome='Complete prespecified solver screen',cases=candidates,gates=gates,real_parity_count=64,maximum_true_residual=max(r['independent_true_residual'] for r in rows),maximum_evaluated_bound=max(r['evaluated_error_bound'] for r in rows),maximum_field_difference=max(r['field_difference'] for r in rows),full_cost_comparison=True,exact_linearity_claim=False)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('output');args=parser.parse_args()
    Path(args.output).write_text(json.dumps(analyze(args.root),indent=2,allow_nan=False)+'\n')
