import time,gc
import torch
from bridge_common import Receipt,construct,locked_vectors,stable_relative,independent_A0_residual,FullAngularModes,BatchedAngularModes,MODES,current_rss_bytes

@torch.no_grad()
def main():
    receipt=Receipt('qualification_01');error=None
    try:
        low,high,base,margin,physical,tsa=construct(torch.zeros(1,12,128,128,dtype=torch.complex128));del base;gc.collect()
        receipt.data['common_physical_seconds']=physical;receipt.data['unused_TSA_preparation_seconds']=tsa
        t=time.perf_counter();sequential=FullAngularModes(low,high);receipt.data['sequential_factor_seconds']=time.perf_counter()-t
        t=time.perf_counter();batched=BatchedAngularModes(low,high);receipt.data['batched_factor_seconds']=time.perf_counter()-t
        receipt.data.update(batched_factor_phases=batched.setup,batched_logical_factor_bytes=batched.bytes,sequential_logical_factor_bytes=sequential.bytes,RSS_with_both_factors_bytes=current_rss_bytes());receipt.save()
        q=locked_vectors();t=time.perf_counter();action=batched.original(q);receipt.data['batched_original_action_seconds']=time.perf_counter()-t
        t=time.perf_counter();seq_action=torch.stack([sequential.blocks[k].original(q[j]) for j,k in enumerate(MODES)]);receipt.data['sequential_original_actions_seconds']=time.perf_counter()-t
        differences=stable_relative(action-seq_action,seq_action)
        for j,k in enumerate(MODES):
            independent=independent_A0_residual(high,low,k,q[j],action[j])
            row=dict(kind='original_action',k=k,sequential_action_difference=differences[j].tolist(),independent_original_A0_difference=independent.tolist());receipt.data['records'].append(row)
            assert bool((differences[j]<=1e-12).all()) and bool((independent<=1e-12).all()),row
        receipt.save()
        zero=torch.zeros_like(q);mix=torch.cat((zero,q,1e-200*q,zero),dim=2)
        inactive=mix.clone();inactive[[0,8,16]]=0
        cases=[('locked_vectors',q),('all_zero',torch.cat((zero,zero),dim=2)),('mixed_columns',mix),('mixed_inactive_modes',inactive)]
        for name,input in cases:
            t=time.perf_counter();x,st=batched.repaired(input);batch_time=time.perf_counter()-t
            case=dict(kind='inverse_case',name=name,batched_seconds=batch_time,batched_status=st,modes=[]);receipt.data['records'].append(case)
            assert st['success'],case
            active=(input!=0).any(dim=1)
            assert torch.equal(x.masked_select(~active[:,None]),torch.zeros_like(x.masked_select(~active[:,None])))
            assert bool(((x!=0).any(dim=1)[active]).all())
            direct=stable_relative(input-batched.original(x),input)
            for j,k in enumerate(MODES):
                t=time.perf_counter();xs,ss=sequential.blocks[k].repaired(input[j]);seq_time=time.perf_counter()-t
                diff=stable_relative((x[j]-xs)[None],xs[None])[0]
                independent=independent_A0_residual(high,low,k,x[j],input[j])
                row=dict(k=k,sequential_seconds=seq_time,sequential_status=ss,inverse_difference=diff.tolist(),batched_original_G_residual=direct[j].tolist(),independent_original_A0_residual=independent.tolist(),strict_zero_columns=int((~active[j]).sum()),strict_zero_solution=True,tiny_nonzero_preserved=(bool(x[j,:,2].abs().max()>0) if input.shape[2]==4 and active[j,2] else None))
                case['modes'].append(row)
                assert ss['success'] and bool((diff<=1e-10).all()) and bool((direct[j]<=1e-10).all()) and bool((independent<=1e-10).all()),row
            receipt.save()
        receipt.data['passed']=True
    except BaseException as exc:error=exc;raise
    finally:receipt.finish(error)
if __name__=='__main__':main()
