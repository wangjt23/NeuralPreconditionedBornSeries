"""Independent raw-receipt audit; stdlib only, no original imports or compute.

Handles a completed numerical-entry obstruction as an executed negative, and a
complete six-arm solver screen. A launch/timeout/partial receipt is distinct.
Cost gates reduce repeats within a material, then average the two materials.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st

REAL = ('eager', 'context_full', 'context_segmented', 'direct_full', 'direct_segmented')
CANDIDATE = 'complex_direct_segmented'
METHODS = REAL + (CANDIDATE,)
MAPS = ('physical_H128','low_H64','reference_G64','reference_R064','strong_eager64',
        'strong_eager128','deployed_backcast128','segmented_native128','physical_H_after_deployed')
CAPS = dict(zip(MAPS, (1e-12,2e-6,5e-6,5e-6,1e-4,1e-12,1e-4,1e-12,1e-3)))
CONTEXTS = {(n,i) for n in (128,256) for i in (0,3)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def close(a,b):
    return finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)


def reduce_cases(rows, methods):
    cases=[]
    for n,i in sorted(CONTEXTS):
        for method in methods:
            group=[r for r in rows if (r['size'],r['index'],r['method'])==(n,i,method)]
            timed=[r for r in group if not r['warmup']]
            if not timed:
                continue
            cases.append(dict(size=n,index=i,method=method,eligible=all(r.get('accuracy_passed') is True for r in group),
                setup_seconds=st.median(r['setup_seconds'] for r in timed),
                accepted_seconds=st.median(r['accepted_seconds'] for r in timed),
                q1_seconds=st.median(r['setup_plus_accepted_seconds'] for r in timed),
                operational_q1_seconds=st.median(r.get('operational_q1_seconds',r['setup_plus_accepted_seconds']) for r in timed),
                iterations=st.median(r['status']['iterations'] for r in timed),
                classical_calls=st.median(r['status']['classical_calls'] for r in timed),
                injection_rejected=st.median(r['status'].get('injection_rejected',0) for r in timed)))
    return cases


def gate(cases, comparators, candidate_q1='q1_seconds', baseline_q1='q1_seconds'):
    result=[]
    for n in (128,256):
        details=[]
        for i in (0,3):
            c=next(r for r in cases if (r['size'],r['index'],r['method'])==(n,i,CANDIDATE))
            controls=[r for r in comparators if (r['size'],r['index'])==(n,i) and r['eligible']]
            if not controls:
                raise ValueError(f'no eligible comparator at {n}/{i}')
            q=min(controls,key=lambda r:r[baseline_q1]);w=min(controls,key=lambda r:r['accepted_seconds'])
            details.append(dict(index=i,candidate=c,q1_baseline=q,warm_baseline=w,
                q1_ratio=c[candidate_q1]/q[baseline_q1],warm_ratio=c['accepted_seconds']/w['accepted_seconds']))
        B=st.mean(d['q1_baseline'][baseline_q1] for d in details)
        W=st.mean(d['warm_baseline']['accepted_seconds'] for d in details)
        T=st.mean(d['candidate'][candidate_q1] for d in details)
        U=st.mean(d['candidate']['accepted_seconds'] for d in details)
        saving=1-T/B;warm=1-U/W;worst=max(d['q1_ratio']-1 for d in details)
        eligible=all(d['candidate']['eligible'] for d in details)
        result.append(dict(size=n,cases=details,mean_baseline_q1_seconds=B,mean_candidate_q1_seconds=T,
            mean_baseline_warm_seconds=W,mean_candidate_warm_seconds=U,q1_saving=saving,
            warm_saving=warm,worst_q1_regression=worst,all_candidate_accuracy=eligible,
            gate=eligible and saving>=.1 and warm>=.1 and worst<=.1))
    return result


def audit(root, expected_policy_sha, expected_driver_sha, historical_root=None, historical_summary=None):
    root=Path(root);errors=[]
    def require(ok,description):
        if not ok:errors.append(description)
    def load(name,default=None):
        p=root/name
        return json.loads(p.read_text()) if p.exists() else default
    policy=load('policy.json',{});execution=load('execution_receipt.json',{});stage=load('staging_receipt.json',{})
    settings=load('results/settings.json',{});checks=load('results/checks.json',[])
    linear=load('results/admissibility.json',[]);setups=load('results/setup.json',[]);rows=load('results/solver.json',[])
    require(sha(root/'policy.json')==expected_policy_sha,'policy hash mismatch')
    require(policy.get('phase')=='complex_krylov_screen' and policy.get('methods')==list(METHODS),'wrong phase/methods')
    require(policy.get('candidate')==CANDIDATE and policy.get('candidate_backend')=='direct_segmented'
            and policy.get('strong_factors')=={'128':'tsa8/reference_tail','256':'tsa16/zero'},'candidate/factors changed')
    for flag in ('selection','parameter_sweep','new_teacher','optional_span_fit','training','test9626_touched',
                 'original_source_exported','git_write','new_package','large_campaign'):
        require(policy.get(flag) is False,'policy scope flag '+flag)
    require(policy.get('grids')==[128,256] and policy.get('development_ids')==[0,3] and policy.get('development_seed')==4626,'cohort changed')
    for name,value in dict(timed_repeats=3,warmup_repeats=1,max_full_solves=96,global_entry_checks=112,
        admissibility_records=108,soft_seconds=95,hard_seconds=120,max_reserved_bytes=12*1024**3,
        restart=25,maxiter=100,tolerance=1e-5,relative_error_bound_limit=1e-3,paired_field_difference_limit=1e-4).items():
        require(policy.get(name)==value,'policy changed '+name)
    require(policy.get('primary_paired_full_solves')==32 and policy.get('admissibility_action_calls')==1080
            and policy.get('same_real_recipe_full_pairs')==64 and policy.get('inject_after')==4,'paired/acquisition/injection budget changed')
    require(policy.get('admissibility_tolerances')==CAPS,'admissibility caps changed')
    hashes={}
    for name,h in policy.get('code_sha256',{}).items():
        actual=sha(root/name) if (root/name).exists() else None
        hashes[name]=dict(expected=h,actual=actual);require(actual==h,'code hash mismatch '+name)
    require(policy.get('code_sha256',{}).get('code/complex_screen.py')==expected_driver_sha,'driver identity mismatch')
    stage_hashes=stage.get('code_sha256',{});stageok=bool(stage_hashes) and all((root/n).exists() and sha(root/n)==h for n,h in stage_hashes.items())
    require(stageok and stage.get('cpu_tests_exit')==0,'staging hashes/CPU preflight mismatch')
    require(execution.get('policy_sha256')==expected_policy_sha,'execution policy identity mismatch')
    require(execution.get('owned_process_timed_out') is False,'owned execution timeout')
    require(finite(execution.get('process_wall_seconds')) and execution['process_wall_seconds']<=120,'hard time envelope')
    if settings:
        require(finite(settings.get('wall_seconds')) and settings['wall_seconds']<=95,'soft time envelope')
        require(finite(settings.get('peak_reserved_bytes')) and settings['peak_reserved_bytes']<=12*1024**3,'memory envelope')
        for flag in ('training','test9626_touched','original_source_exported'):
            require(settings.get(flag) is False,'scope flag '+flag)
    actions=[c for c in checks if c.get('kind')!='ownership_cache_view'];guards=[c for c in checks if c.get('kind')=='ownership_cache_view']
    expected_action={(n,i,m,p,d) for n,i in CONTEXTS for m in REAL[1:] for p in ('source','grazing','mixed') for d in ('torch.complex64','torch.complex128')}
    actionkeys=[(c['size'],c['index'],c['method'],c['probe'],c['dtype']) for c in actions]
    expected_linear={(n,i,m,p) for n,i in CONTEXTS for m in MAPS for p in ('source_mixed','nyquist_source','unit_complex_random')}
    linearkeys=[(c['size'],c['index'],c['action'],c['probe']) for c in linear]
    completed_entry=settings.get('entry_obstruction') is True or settings.get('complete') is True
    if completed_entry:
        require(len(checks)==112 and len(actions)==96 and len(guards)==16 and len(set(actionkeys))==96 and set(actionkeys)==expected_action,'entry action coverage')
        require({(c['size'],c['index'],c['method']) for c in guards}=={(n,i,m) for n,i in CONTEXTS for m in REAL[1:]},'guard coverage')
        require(len(linear)==108 and len(set(linearkeys))==108 and set(linearkeys)==expected_linear,'admissibility coverage')
    else:
        require(len(checks)<=112 and len(set(actionkeys))==len(actionkeys) and set(actionkeys)<=expected_action,'partial entry bounds/identity')
        require(len(linear)<=108 and len(set(linearkeys))==len(linearkeys) and set(linearkeys)<=expected_linear,'partial admissibility bounds/identity')
    for c in actions:
        require(c['passed']==(c.get('bitwise_equal') is True and c.get('finite') is True),'action passed flag inconsistent')
    for c in guards:
        require(c['passed']==all(c.get(k) is True for k in ('ownership','guard','restored')),'guard passed flag inconsistent')
    admissibility_flags=[];map_summary=[]
    for c in linear:
        require(c['tolerance']==CAPS[c['action']],'record cap changed')
        require(c.get('action_calls')==10,'admissibility action count')
        coeff=c.get('coefficients',{});alpha=complex(.73,.29);beta=complex(-.41,.37);phase=complex(math.cos(.37),math.sin(.37))
        require(coeff==dict(alpha=dict(real=.73,imaginary=.29),beta=dict(real=-.41,imaginary=.37),phase_angle=.37,
                            phase=dict(real=phase.real,imaginary=phase.imag),real_scale=-1.7),'probe coefficients changed')
        qc=c.get('checks',{});valid=[]
        for name,v in qc.items():
            outputs=c.get('outputs',{})
            if 'x' in outputs and 'y' in outputs:
                nx,ny=outputs['x']['norm'],outputs['y']['norm']
                if finite(nx) and finite(ny):
                    expected_denominator=(abs(alpha)*nx+abs(beta)*ny if name=='complex_combination'
                        else (1. if name.startswith('i_') else abs(phase) if name.startswith('phase_') else 1.7)
                        * outputs[name.rsplit('_',1)[1]]['norm'])
                    require(close(v.get('denominator'),expected_denominator),'reference-output normalization mismatch '+name)
            eligible=v.get('finite') is True and finite(v.get('denominator')) and v['denominator']>0
            recomputed=v['numerator']/v['denominator'] if eligible and finite(v.get('numerator')) else None
            require((close(v.get('ratio'),recomputed) if recomputed is not None else v.get('ratio') is None),'identity ratio arithmetic '+name)
            passed=eligible and finite(recomputed) and recomputed<=c['tolerance']
            require(v.get('passed')==passed,'identity passed flag '+name);valid.append(passed)
        passed=(c.get('finite') is True and c.get('exact_zero') is True and c.get('output_contract_passed') is True
                and c.get('input_mutated') is False and not c.get('errors') and len(qc)==7 and all(valid))
        require(c.get('passed')==passed,'record passed flag inconsistency');admissibility_flags.append(passed)
        ratios=[v['ratio'] for v in qc.values() if finite(v.get('ratio'))]
        if ratios:require(close(c.get('max_normalized_defect'),max(ratios)),'max identity defect mismatch')
    for action in MAPS:
        group=[c for c in linear if c['action']==action];ratios=[c['max_normalized_defect'] for c in group if finite(c.get('max_normalized_defect'))]
        map_summary.append(dict(action=action,cap=CAPS[action],records=len(group),failures=sum(c.get('passed') is not True for c in group),max_defect=max(ratios) if ratios else None))
    sm={(s['size'],s['index']):s for s in setups}
    if completed_entry:require(len(setups)==4 and set(sm)==CONTEXTS,'setup coverage')
    qa=[]
    for s in setups:
        n=s['size'];depth=8 if n==128 else 16
        require(s['depth']==depth and s['strong_name']==f'tsa{depth}' and s['recipe']==('reference_tail' if n==128 else 'tsa16_zero'),'physical recipe changed')
        require(s['base_parameters']==dict(kind='tsa',depth=depth,eta_scale=0.,relaxation=1.),'factor parameters changed')
        fft={'context_full':dict(fftn=4*depth-1,ifftn=4*depth-1),'direct_full':dict(fftn=4*depth-1,ifftn=4*depth-1),
             'context_segmented':dict(fftn=4,ifftn=4),'direct_segmented':dict(fftn=4,ifftn=4)}
        require(s['capture_fft_counts']==fft,'FFT work count mismatch')
        q=s['source_start_qa'];qa.append(all(q.get(k) is True for k in ('source_is_exactly_real','source_bitwise_equal','initialization_bitwise_equal','reference_cache_shared')))
        require(s.get('complex_capture_ZW') is False and s.get('complex_current_raw_image_live') is True,'retained-image storage declaration mismatch')
        require(finite(s['evaluated_margin']) and s['evaluated_margin']>0,'nonpositive evaluated margin')
        require(all(finite(s[k]) and s[k]>=0 for k in ('physical_seconds','base_seconds','acceptance_setup_seconds','mandatory_admissibility_seconds')),'nonfinite/negative setup cost')
    entry_ok=all(c.get('passed') is True for c in checks) and all(admissibility_flags) and all(qa)
    outcome=('complete_screen' if settings.get('complete') is True else 'completed_entry_obstruction' if settings.get('entry_obstruction') is True else 'partial_or_execution_failure')
    if outcome=='completed_entry_obstruction':
        require(not entry_ok and not rows and settings.get('full_solves')==0,'entry obstruction/full solve rule broken')
        require(execution.get('exit_code')==0,'entry obstruction did not complete normally')
    elif outcome=='complete_screen':
        require(entry_ok and execution.get('exit_code')==0,'full solves admitted through failed entry')
        expected_rows={(n,i,m,rep) for n,i in CONTEXTS for m in METHODS for rep in (-1,0,1,2)}
        keys=[(r['size'],r['index'],r['method'],r['repeat']) for r in rows]
        require(len(rows)==96 and len(set(keys))==96 and set(keys)==expected_rows,'full solve coverage')
        require(sum(r['warmup'] for r in rows)==24,'warmup count')
    else:
        require(not rows,'unexpected full solves in partial launch/entry outcome')
        require(False,'incomplete screen/entry receipt; preserve as execution blocker')
    valid_rows=[];pairs=0
    rm={(r['size'],r['index'],r['method'],r['repeat']):r for r in rows}
    for r in rows:
        s=sm[(r['size'],r['index'])];kind='complex' if r['method']==CANDIDATE else 'real'
        cap=0 if r['method']=='eager' else s['constructor_seconds']['direct_segmented' if kind=='complex' else r['method']]
        op=sum(s[k] for k in ('physical_seconds','base_seconds','acceptance_setup_seconds'))+cap+s['solver_preparation_seconds'][kind]
        require(close(r['operational_setup_seconds'],op),'own constructor/preparation accounting')
        require(close(r['setup_seconds'],op+s['mandatory_admissibility_seconds']),'common admissibility acquisition charge')
        require(close(r['mandatory_acquisition_seconds'],s['mandatory_admissibility_seconds']),'mandatory charge mismatch')
        require(close(r['accepted_seconds'],r['seconds']+r['acceptance_seconds']),'accepted latency arithmetic')
        require(close(r['operational_q1_seconds'],op+r['accepted_seconds']),'operational Q1 arithmetic')
        require(close(r['setup_plus_accepted_seconds'],r['setup_seconds']+r['accepted_seconds']),'fully charged Q1 arithmetic')
        valid=(r.get('finite') is True and r['status']['failed']==0 and finite(r.get('independent_true_residual')) and 0<=r['independent_true_residual']<=1e-5
               and finite(r.get('evaluated_error_bound')) and 0<=r['evaluated_error_bound']<=1e-3
               and finite(r.get('field_difference')) and 0<=r['field_difference']<=1e-4)
        require(r.get('accuracy_passed')==valid,'accuracy flag differs from reported metrics');valid_rows.append(valid)
        require(r['status']['iterations']<=100 and r['status']['classical_calls']<=100,'iteration/call cap')
        require(r['status'].get('injection_calls',0)<=(1 if r['size']==128 else 0),'injection recipe changed')
        if r['method'] in REAL[:-1]:
            ref=rm.get((r['size'],r['index'],'direct_segmented',r['repeat']))
            require(ref is not None and r.get('same_recipe_field_status_parity') is True and r.get('bitwise_vs_real_direct_segmented') is True and r['status']==ref['status'],'real same-recipe field/status parity');pairs+=1
    cases=reduce_cases(rows,METHODS) if outcome=='complete_screen' else []
    economics={};historical=None
    if cases:
        controls=[c for c in cases if c['method'] in REAL]
        economics['contemporary_five_fully_charged']=gate(cases,controls)
        economics['contemporary_five_operational']=gate(cases,controls,'operational_q1_seconds','operational_q1_seconds')
        economics['same_factor_fully_charged']=gate(cases,[c for c in controls if c['method']=='direct_segmented'])
        if historical_root:
            hr=Path(historical_root);old=json.loads((hr/'results/solver.json').read_text());oldpolicy=json.loads((hr/'policy.json').read_text())
            require(oldpolicy['phase']=='gain_two_screen' and oldpolicy['fixed_gain']==2,'historical prior identity')
            # Frozen policy pins the prior derived summary, not solver.json.
            hs=Path(historical_summary) if historical_summary else Path('deliverables/gain_two/RESULT_AUDIT.json')
            require(hs.exists() and sha(hs)==policy.get('historical_receipt_sha256'),'historical summary receipt hash mismatch')
            historical=reduce_cases(old,tuple(oldpolicy['prior_portfolio'])+('gain_two',))
            for item in policy.get('historical_prior_eleven',[]):
                group=[r for r in historical if (r['size'],r['index'])==(item['size'],item['index']) and r['eligible']]
                q=min(group,key=lambda r:r['q1_seconds']);w=min(group,key=lambda r:r['accepted_seconds'])
                require((q['method'],q['q1_seconds'],w['method'],w['accepted_seconds'])==
                        (item['q1_method'],item['q1_seconds'],item['warm_method'],item['warm_seconds']), 'historical policy envelope mismatch')
            economics['historical_eleven_sensitivity']=gate(cases,historical)
        else:require(False,'historical-eleven sensitivity source absent')
    decisive=(bool(economics) and all(c['gate'] for c in economics['contemporary_five_fully_charged'])
              and 'historical_eleven_sensitivity' in economics and all(c['gate'] for c in economics['historical_eleven_sensitivity']))
    return dict(protocol_passed=not errors,protocol_errors=errors,outcome=outcome,entry_passed=entry_ok,
        cost_gate_evaluated=bool(cases),decisive_cost_gate=decisive if cases else None,
        counts=dict(entries=len(checks),capture_actions=len(actions),guard_bundles=len(guards),admissibility_records=len(linear),
            admissibility_action_calls=sum(c.get('action_calls',0) for c in linear),full=len(rows),timed=sum(not r['warmup'] for r in rows),warm=sum(r['warmup'] for r in rows),same_recipe_full_pairs=pairs),
        admissibility=map_summary,admissibility_failed_records=[c for c in linear if c.get('passed') is not True],
        all_accuracy_passed=all(valid_rows) if rows else None,accuracy_failures=sum(not v for v in valid_rows),
        full_solver_failures=sum(r['status']['failed']!=0 for r in rows),
        max_residual=max((r['independent_true_residual'] for r in rows if finite(r.get('independent_true_residual'))),default=None),
        max_evaluated_bound=max((r['evaluated_error_bound'] for r in rows if finite(r.get('evaluated_error_bound'))),default=None),
        max_field_difference=max((r['field_difference'] for r in rows if finite(r.get('field_difference'))),default=None),
        mandatory_acquisition_seconds=sum(s['mandatory_admissibility_seconds'] for s in setups),cases=cases,economics=economics,
        setup=setups,settings=settings,execution=execution,code_hashes=hashes,staging=dict(all_hashes_match=stageok,cpu_tests_exit=stage.get('cpu_tests_exit')),
        receipt_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.suffix!='.pyc'},
        scope=['No Torch/original source import, fields or GPU execution in this auditor.',
               'Physical accuracy flags recomputed from scalar receipts; H/fields not reevaluated locally.',
               'Current five and stored eleven envelopes are separate; history is not a contemporaneous timing comparison.',
               'Common scientific acquisition charged to all six; operational costs retained separately.',
               'Companion changes complex coefficient freedom and SVD rank safeguards; no pure coefficient-only timing attribution.',
               'Engineering admissibility caps are finite-probe checks, not global linearity/error bounds.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--expected-policy-sha',required=True);p.add_argument('--expected-driver-sha',required=True);p.add_argument('--historical-root',type=Path)
    p.add_argument('--historical-summary',type=Path)
    a=p.parse_args();result=audit(a.root,a.expected_policy_sha,a.expected_driver_sha,a.historical_root,a.historical_summary)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('protocol_passed','protocol_errors','outcome','entry_passed','counts','admissibility','all_accuracy_passed','decisive_cost_gate','economics')},indent=2))
    raise SystemExit(0 if result['protocol_passed'] else 1)
