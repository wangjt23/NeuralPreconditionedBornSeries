import time,json,gc,hashlib
import torch
from bridge_common import Receipt,construct,project8,qualify_all_modes,BatchedAngularModes,HERE,ROOT,E,current_rss_bytes
from rte.source_initialization import solve_from_initial
from rte.augmented_fgmres import augmented_fgmres

@torch.no_grad()
def main():
    assert json.loads((HERE/'qualification_01/results.json').read_text())['passed']
    receipt=Receipt('full_01');error=None
    try:
        old=json.loads((E/'full_01/settings.json').read_text())
        saved=torch.load(ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True)
        rhs=[saved['bank']['q'][i:i+1].to(torch.complex128) for i in old['source_ids']]
        receipt.data['rhs_sha256']={str(i):hashlib.sha256(saved['bank']['q'][i:i+1].numpy().tobytes()).hexdigest() for i in old['source_ids']};assert receipt.data['rhs_sha256']==old['rhs_sha256'];del saved;gc.collect()
        t=time.perf_counter();low,high,base,margin,physical,tsa=construct(rhs[0]);pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128);references={};receipt.data['reference_checks']=[]
        for sid,b in zip(old['source_ids'],rhs):
            ti=time.perf_counter();u,st=solve_from_initial(high.A_op,b,pre,None,tolerance=1e-5,maxiter=100)
            receipt.data['reference_checks'].append(dict(source_id=sid,seconds=time.perf_counter()-ti,status=st));assert not st['failed'] and st['true_residual']<=1e-5
            references[sid]=u;receipt.save()
        receipt.data['reference_research_seconds']=time.perf_counter()-t
        del pre,base,low,high;gc.collect()
        tick=time.perf_counter();low,high,base,margin,physical,tsa=construct(rhs[0]);pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128)
        t=time.perf_counter();factor=BatchedAngularModes(low,high);factor_time=time.perf_counter()-t
        receipt.data['factor_phases']=factor.setup;receipt.data['factor_logical_bytes']=factor.bytes;receipt.data['RSS_after_factor_bytes']=current_rss_bytes()
        qualification={};receipt.data['mode_qualification']=qualification;qualify_all_modes(factor,low,high,qualification)
        setup=physical+tsa+factor_time+qualification['seconds'];receipt.data['setup']=dict(physical_seconds=physical,TSA_seconds=tsa,factor_seconds=factor_time,original_A0_validation_seconds=qualification['seconds'],complete_seconds=setup)
        receipt.data['rows']=[];receipt.data['repair_profiles']=factor.profiles
        for sid,b in zip(old['source_ids'],rhs):
            inner=[];before=len(factor.profiles);bn=float(b.norm());t=time.perf_counter()
            def inject(x,r):
                ti=time.perf_counter();z,st=solve_from_initial(lambda v:project8(high.A_op(v)),project8(r),lambda v:project8(factor.apply(project8(v))),None,tolerance=1e-4,maxiter=8)
                inner.append(dict(**st,seconds=time.perf_counter()-ti));return z
            u,st=augmented_fgmres(high.A_op,b,pre,None,inject=inject,inject_after=1,tolerance=1e-5,maxiter=100)
            solve=time.perf_counter()-t;t=time.perf_counter();res=float((b-high.A_op(u)).norm()/bn);delta=res*bn/margin;bound=delta/(float(u.norm())-delta);diff=float((u-references[sid]).norm()/references[sid].norm());ok=bool(torch.isfinite(u).all()) and not st['failed'] and res<=1e-5 and bound<=1e-3 and diff<=1e-4;check=time.perf_counter()-t;st.pop('trace',None)
            row=dict(source_id=sid,solve_seconds=solve,check_seconds=check,accepted_seconds=solve+check,first_RHS_if_standalone_seconds=setup+solve+check,status=st,inner=inner,accepted=ok,true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff,batched_inverse_applications=len(factor.profiles)-before)
            receipt.data['rows'].append(row)
            assert ok and all(not x['failed'] and x['true_residual']<=1e-4 for x in inner),row
            assert st['classical_calls']==2 and [x['iterations'] for x in inner]==([4] if sid==14 else [5]),row
        observed=time.perf_counter()-tick
        online=sum(x['accepted_seconds'] for x in receipt.data['rows'])
        receipt.data['batch']=dict(complete_four_RHS_seconds=setup+online,online_four_RHS_seconds=online,observed_batch_wall_seconds=observed,first_RHS_seconds=receipt.data['rows'][0]['first_RHS_if_standalone_seconds'],outer_calls=[x['status']['classical_calls'] for x in receipt.data['rows']],inner_calls=[x['iterations'] for r in receipt.data['rows'] for x in r['inner']],batched_inverse_applications=len(factor.profiles),logical_mode_inverse_applications=17*len(factor.profiles),batched_raw_calls=sum(x['batched_raw_calls'] for x in factor.profiles),batched_original_G_calls=sum(x['batched_original_calls'] for x in factor.profiles))
        receipt.data['passed']=True
    except BaseException as exc:error=exc;raise
    finally:receipt.finish(error)
if __name__=='__main__':main()
