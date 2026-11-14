"""Five retained failures, same media/source; no solver benchmark or training."""
import hashlib
import json
from pathlib import Path
import time
import torch
from algebraic_tsa import AlgebraicTSA
from rte.transport_baselines import ClassicalTransportStack
from rte.train_source_predictor import material_operator

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def norm(x):
    return float(x.norm())

def rel(x, y):
    return norm(x-y)/max(norm(y), 1e-30)

def ff(x): return torch.fft.fftn(x, dim=(-2,-1))
def iff(x): return torch.fft.ifftn(x, dim=(-2,-1))

def spectrum(x):
    n=x.shape[-1]
    k=torch.fft.fftfreq(n,device=x.device)*n
    radius=torch.maximum(k[:,None].abs(),k[None,:].abs())
    power=ff(x).abs().square().sum((0,1))
    total=float(power.sum())
    return dict(high_frequency_power_fraction=float(power[radius>=n/4].sum())/max(total,1e-300),
                nyquist_power_fraction=float(power[radius>=n/2].sum())/max(total,1e-300))

@torch.no_grad()
def run():
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parent.parent
    out=root/'diagnosis_extended';out.mkdir(exist_ok=False)
    prior=Path('rte/workstation_runs/classical_tsa_algebra_20261002/operator_checks')
    settings=json.loads((prior/'settings.json').read_text())
    old=json.loads((prior/'cases.json').read_text())
    failed=[r for r in old if not r['passed']]
    assert len(failed)==5
    setup=json.loads(Path('rte/workstation_runs/resume_audit_20261002/runtime_gate/settings.json').read_text())
    # Configuration recorded by the same existing source-family material builder.
    from rte.train_source_predictor import source_family
    rows=[];start=time.perf_counter()
    for case in failed:
        n,i=case['size'],case['index']
        parent=Path(f'rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{n}.pt')
        assert sha(parent)==settings['development_parent_sha256'][str(n)]
        data=torch.load(parent,map_location='cpu',weights_only=True)
        cfg,_=source_family(n,16,4626)
        low,physical=material_operator(cfg,data,torch.tensor([i]),.1,'cuda')
        base=ClassicalTransportStack(low,'tsa',16,0.)
        fused=AlgebraicTSA(low,depth=16,eta_scale=0.)
        b=low.source
        stream=base.reference._stream_symbol(1,torch.complex64,b.device).to(torch.complex128)
        gamma=base.reference.sigma0.to(torch.complex128)
        t=low.sigma_t.to(torch.complex128)
        def S(x):return iff(stream*ff(x))
        def L(x):return S(x)+gamma*x
        def A(x):return S(x)+t*x-low.scattering_op(x)
        z=torch.zeros_like(b);zf=torch.zeros_like(b)
        stages=[]
        for j in range(16):
            d=b if j==0 else b-low.A_op(z)
            df=b if j==0 else b-low.A_op(zf)
            g=base.reference.G(d)
            ahat=low.A_op(g)
            ho=d-ahat
            hf=(base.reference.sigma0.to(g.dtype)-low.sigma_t.to(g.dtype))*g+low.scattering_op(g)
            co=base.reference.reference_transport_solve(ho)
            cf=base.reference.reference_transport_solve(hf)
            single= g+co
            zs=z+single
            fsingle=fused.single(df)
            zfs=zf+fsingle
            row=dict(stage=j,field_relative=rel(zfs,zs),image_relative=rel(physical.A_op(zfs.to(torch.complex128)),physical.A_op(zs.to(torch.complex128))),
                     accumulated_field_delta_spectrum=spectrum((zfs-zs).to(torch.complex128)),
                     accumulated_image_delta_spectrum=spectrum(physical.A_op((zfs-zs).to(torch.complex128))))
            acc=((zfs.to(torch.complex128)-zs.to(torch.complex128))-
                 (zf.to(torch.complex128)-z.to(torch.complex128)+fsingle.to(torch.complex128)-single.to(torch.complex128)))
            row['outer_accumulation_rounding_difference_image_norm']=norm(physical.A_op(acc))
            row['outer_accumulation_rounding_difference_norm']=norm(acc)
            if j in (0,7,15):
                gd=g.to(torch.complex128);dd=d.to(torch.complex128)
                hlocal=(gamma-t)*gd+low.scattering_op(gd)
                terms=dict(inverse_roundtrip=L(gd)-dd,local_rounding=hf.to(torch.complex128)-hlocal,
                           physical_action_rounding=ahat.to(torch.complex128)-A(gd),
                           subtraction_rounding=-(ho.to(torch.complex128)-(dd-ahat.to(torch.complex128))))
                delta=hf.to(torch.complex128)-ho.to(torch.complex128)
                reconstruction=sum(terms.values())
                contributions={}
                for name,value in terms.items():
                    transported=base.reference.reference_transport_solve(value)
                    contributions[name]=dict(norm=norm(value),relative_to_old_defect=norm(value)/max(norm(ho),1e-30),
                        physical_image_norm_after_reference=norm(physical.A_op(transported)),spectrum=spectrum(value))
                rt=iff(ff(gd))-gd
                g128=base.reference.G(dd)
                rdelta=base.reference.reference_transport_solve(delta)
                actual_rdelta=cf.to(torch.complex128)-co.to(torch.complex128)
                rround=actual_rdelta-rdelta
                sadd=(g+cf).to(torch.complex128)-(g+co).to(torch.complex128)-actual_rdelta
                dhat=ff(d);denom=base.reference._stream_symbol(1,d.dtype,d.device)+base.reference.sigma0.to(d.dtype)
                ghat=dhat/denom
                assert torch.equal(iff(ghat),g)
                lam=denom.to(torch.complex128)
                fftterms=dict(inverse_fft_output=iff(lam*(ff(gd)-ghat.to(torch.complex128))),
                              division=iff(lam*ghat.to(torch.complex128)-dhat.to(torch.complex128)),
                              forward_fft=iff(dhat.to(torch.complex128)-ff(dd)))
                row['common_input_diagnosis']=dict(old_defect_norm=norm(ho),local_defect_norm=norm(hf),delta_norm=norm(delta),
                    decomposition_relative_error=rel(reconstruction,delta),contributions=contributions,
                    inverse_error_recomputed_g128_norm=norm(L(g128)-dd),
                    inverse_error_same_g128_norm=norm(L(gd)-dd),fft_roundtrip_same_g128_norm=norm(rt),
                    original_defect_cancellation_ratio=(norm(d)+norm(ahat))/max(norm(ho),1e-30),
                    local_defect_cancellation_ratio=(norm((gamma-t)*gd)+norm(low.scattering_op(gd)))/max(norm(hlocal),1e-30),
                    double_stream_symbol_promotion_discrepancy_relative=rel(physical.streaming_op(gd),S(gd)),
                    local_action_arithmetic_delta_image_norm=norm(physical.A_op(base.reference.reference_transport_solve(delta))),
                    old_single_norm=norm(single),local_single_difference_image_norm=norm(physical.A_op((fused.single(d)-single).to(torch.complex128))))
                row['common_input_diagnosis']['remaining_rounding']=dict(reference_action_difference_image_norm=norm(physical.A_op(rround)),
                    single_addition_difference_image_norm=norm(physical.A_op(sadd)),
                    decomposition_image_closure_norm=norm(physical.A_op((g+cf).to(torch.complex128)-(g+co).to(torch.complex128)-rdelta-rround-sadd)),
                    reference_difference_image_norm=norm(physical.A_op(actual_rdelta)))
                row['common_input_diagnosis']['G_fft_ledger']=dict(components={k:dict(norm=norm(v),image_after_reference_norm=norm(physical.A_op(base.reference.reference_transport_solve(v)))) for k,v in fftterms.items()},
                    closure_relative_error=rel(sum(fftterms.values()),L(gd)-dd),
                    fp32_roundtrip_ghat_relative=rel(ff(g),ghat))
            stages.append(row);z,zf=zs,zfs
        rows.append(dict(size=n,index=i,prior_case=case,stages=stages,final_field_relative=rel(zf,z),
                         final_image_relative=rel(physical.A_op(zf.to(torch.complex128)),physical.A_op(z.to(torch.complex128)))))
        (out/'cases.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        print(f'diagnosed N{n} ID{i}: image {rows[-1]["final_image_relative"]:.9g}',flush=True)
    torch.cuda.synchronize()
    result=dict(complete=True,cases=len(rows),stages=80,common_input_decompositions=15,wall_seconds=time.perf_counter()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                script_sha256=sha(__file__),prior_case_sha256=sha(prior/'cases.json'),
                unchanged_gate=dict(complex64=1e-4,complex128=1e-9),reserved_seed_touched=False,
                decomposition_note='Double arithmetic evaluates the actual FP32-assembled stream symbol, stored coefficients/phase/weights and the same rounded g. The acceptance image remains the original promoted-buffer physical operator.')
    (out/'settings.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
