"""Prespecified complex companion versus frozen original strong real controls."""
import hashlib,json,math,random,time
from pathlib import Path
import torch
from graph_classical_v2 import GraphClassicalV2
from segmented_tsa import SegmentedTSA
from direct_segmented_tsa import DirectSegmentedTSA
from direct_full_graph import DirectFullGraph
from complex_augmented_fgmres import complex_augmented_fgmres,_small_lstsq
from admissibility import check_action
from rte.benchmark_tail_solver import solve
from rte.global_source_predictor import balance_integrated_source
from rte.train_source_predictor import build_base,material_operator,source_family

REAL=('eager','context_full','context_segmented','direct_full','direct_segmented')
METHODS=REAL+('complex_direct_segmented',)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sync():torch.cuda.synchronize()

@torch.no_grad()
def run():
    torch.set_num_threads(1);root=Path(__file__).resolve().parent.parent;policy=json.loads((root/'policy.json').read_text())
    assert policy['phase']=='complex_krylov_screen' and policy['methods']==list(METHODS)
    for n,h in policy['code_sha256'].items():assert sha(root/n)==h
    out=root/'results';out.mkdir(exist_ok=False);start=time.perf_counter();deadline=start+policy['soft_seconds']
    checks=[];linear=[];setups=[];rows=[];contexts=[];peak_allocated=peak_reserved=0
    def track():
        nonlocal peak_allocated,peak_reserved
        peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated());peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved())
    def limit():
        track()
        if time.perf_counter()>deadline:raise TimeoutError('frozen complex-Krylov soft deadline')
        if peak_reserved>policy['max_reserved_bytes']:raise MemoryError('frozen complex-Krylov memory envelope')
    def safe(v):
        if isinstance(v,float) and not math.isfinite(v):return None
        if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [safe(x) for x in v]
        return v
    def save():
        for n,v in [('checks',checks),('admissibility',linear),('setup',setups),('solver',rows)]:
            (out/(n+'.json')).write_text(json.dumps(safe(v),indent=2,allow_nan=False)+'\n')
    for n in policy['grids']:
        parent=Path(f'rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{n}.pt')
        assert sha(parent)==policy['development_parent_sha256'][str(n)]
        data=torch.load(parent,map_location='cpu',weights_only=True);cfg,reg=source_family(n,16,4626)
        for k in ('sigma_t','sigma_s','centers','phases','amplitudes','angular_codes','geometry_base64'):assert torch.equal(data[k],reg[k])
        for index in policy['development_ids']:
            limit();sync();t=time.perf_counter();low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda');sync();physical=time.perf_counter()-t
            name='tsa8' if n==128 else 'tsa16';recipe='reference_tail' if n==128 else 'tsa16_zero'
            sync();t=time.perf_counter();base=build_base(low,name);sync();base_setup=time.perf_counter()-t
            wrappers={};capture={};order=list(REAL[1:])
            if index%2:order.reverse()
            for method in order:
                sync();t=time.perf_counter()
                w=(GraphClassicalV2(base,low.source) if method=='context_full' else
                   SegmentedTSA(base,low.source,chunk=1,capture_first=False) if method=='context_segmented' else
                   DirectFullGraph(base,low.source) if method=='direct_full' else DirectSegmentedTSA(base,low.source))
                sync();capture[method]=time.perf_counter()-t;wrappers[method]=w
            sync();t=time.perf_counter();P=high.phase_matrix.to(torch.complex128)*high.weights[None,:]
            mu=torch.linalg.eigvalsh((P+P.mH)/2).max().real;margin=float((high.sigma_t-high.sigma_s*mu).min());qnorm=float(high.source.norm());sync();acceptance_setup=time.perf_counter()-t
            assert margin>0 and math.isfinite(margin)
            stripe=(-1.)**torch.arange(n,device='cuda');grazing=torch.zeros_like(low.source);grazing[:,0]=stripe[None,None,:]
            mixed=low.source.roll(1,-2)*(1.13+.21j);probes=[('source',low.source),('grazing',grazing),('mixed',mixed)]
            for dtype in (torch.complex64,torch.complex128):
                for label,p in probes:
                    rhs=p.to(dtype);expected=base(rhs).to(torch.complex128)
                    for method,w in wrappers.items():
                        got=w(rhs);passed=bool(torch.equal(got,expected) and torch.isfinite(got).all())
                        checks.append(dict(size=n,index=index,method=method,probe=label,dtype=str(dtype),bitwise_equal=bool(torch.equal(got,expected)),finite=bool(torch.isfinite(got).all()),passed=passed))
            for method,w in wrappers.items():
                held=w(low.source);copy=held.clone();w(grazing);owned=bool(torch.equal(held,copy));old=base.reference._reference_transport_inverse
                base.reference._reference_transport_inverse=old.transpose(1,2)
                try:
                    try:w(low.source)
                    except RuntimeError:guard=True
                    else:guard=False
                finally:base.reference._reference_transport_inverse=old
                restored=bool(torch.equal(w(low.source),base(low.source).to(torch.complex128)))
                checks.append(dict(size=n,index=index,method=method,kind='ownership_cache_view',ownership=owned,guard=guard,restored=restored,passed=owned and guard and restored))
            save();assert all(c['passed'] for c in checks)
            # Gate is mandatory, frozen and complete before ANY full solve.
            sync();began=time.perf_counter();generator=torch.Generator(device='cuda').manual_seed(91731+n+index)
            r1=torch.complex(torch.randn(low.source.shape,device='cuda',generator=generator),torch.randn(low.source.shape,device='cuda',generator=generator));r1=r1/r1.norm()
            r2=torch.complex(torch.randn(low.source.shape,device='cuda',generator=generator),torch.randn(low.source.shape,device='cuda',generator=generator));r2=r2/r2.norm()
            pairs=[('source_mixed',low.source,mixed),('nyquist_source',grazing,low.source),('unit_complex_random',r1,r2)]
            deployed=lambda r:wrappers['direct_segmented'](r.to(torch.complex64)).to(torch.complex128)
            maps=[('physical_H128',high.A_op,torch.complex128),('low_H64',low.A_op,torch.complex64),
                  ('reference_G64',base.reference.G,torch.complex64),('reference_R064',base.reference.reference_transport_solve,torch.complex64),
                  ('strong_eager64',base,torch.complex64),('strong_eager128',base,torch.complex128),
                  ('deployed_backcast128',deployed,torch.complex128),('segmented_native128',wrappers['direct_segmented'],torch.complex128),
                  ('physical_H_after_deployed',lambda r:high.A_op(deployed(r)),torch.complex128)]
            for label,A,dtype in maps:
                for probe,x,y in pairs:
                    limit();record=check_action(A,x.to(dtype),y.to(dtype),tolerance=policy['admissibility_tolerances'][label])
                    record.update(size=n,index=index,action=label,probe=probe);linear.append(record);save()
            source_equal=bool(torch.equal(high.source,low.source.to(torch.complex128)));source_real=bool(torch.count_nonzero(high.source.imag)==0)
            eager_initial=None if n==256 else balance_integrated_source(high,base.reference.reference_transport_solve(low.source).to(torch.complex128),high.source)
            segmented_initial=None if n==256 else balance_integrated_source(high,wrappers['direct_segmented'].reference.reference_transport_solve(low.source).to(torch.complex128),high.source)
            initial_equal=True if n==256 else bool(torch.equal(eager_initial,segmented_initial))
            source_image=high.A_op(high.source);nyquist_image=high.A_op(grazing.to(torch.complex128));source_factor=deployed(high.source)
            real_diagnostics=dict(source_imaginary_norm=float(high.source.imag.norm()),source_is_exactly_real=source_real,
                H_source_imaginary_relative=float(source_image.imag.norm()/source_image.norm().clamp_min(1e-30)),
                H_nyquist_imaginary_relative=float(nyquist_image.imag.norm()/nyquist_image.norm().clamp_min(1e-30)),
                M_source_imaginary_relative=float(source_factor.imag.norm()/source_factor.norm().clamp_min(1e-30)),
                real_field_constraint_in_original_solver=False,source_bitwise_equal=source_equal,initialization_bitwise_equal=initial_equal,reference_cache_shared=wrappers["direct_segmented"].reference is base.reference)
            sync();admissibility_seconds=time.perf_counter()-began
            # Library/solver preparation is charged, counterbalanced ex ante.
            preparation={};prep_order=['real','complex'] if index%2==0 else ['complex','real']
            for kind in prep_order:
                sync();t=time.perf_counter();dtype=torch.float64 if kind=='real' else torch.complex128
                h=torch.eye(26,25,dtype=dtype,device='cuda');right=torch.arange(26,dtype=torch.float64,device='cuda').to(dtype)
                if kind=='real':torch.linalg.lstsq(h,right).solution
                else:_small_lstsq(h,right)
                sync();preparation[kind]=time.perf_counter()-t
            record=dict(size=n,index=index,strong_name=name,recipe=recipe,depth=base.depth,
                base_parameters=dict(kind=base.kind,depth=base.depth,eta_scale=base.eta_scale,relaxation=base.relaxation),
                physical_seconds=physical,base_seconds=base_setup,constructor_seconds=capture,acceptance_setup_seconds=acceptance_setup,
                mandatory_admissibility_seconds=admissibility_seconds,solver_preparation_seconds=preparation,
                source_start_qa=real_diagnostics,evaluated_margin=margin,capture_fft_counts={m:w.capture_fft_counts for m,w in wrappers.items()},
                actual_vector_real_storage_equal=True,max_real_hessenberg_bytes=26*25*8,max_complex_hessenberg_bytes=26*25*16,
                complex_capture_ZW=False,complex_current_raw_image_live=True,reference_inverse_bytes=base.reference._reference_transport_inverse.numel()*base.reference._reference_transport_inverse.element_size())
            setups.append(record);save()
            current=[r for r in linear if (r['size'],r['index'])==(n,index)]
            gate=all(r['passed'] for r in current) and source_equal and source_real and initial_equal and real_diagnostics['reference_cache_shared']
            contexts.append(dict(n=n,index=index,low=low,high=high,base=base,wrappers=wrappers,name=name,recipe=recipe,margin=margin,qnorm=qnorm,setup=record,entry_passed=gate))
            print(f'complex admissibility N{n} ID{index}: {gate}',flush=True)
    assert len(checks)==policy['global_entry_checks'] and len(linear)==policy['admissibility_records']
    if not all(c['entry_passed'] for c in contexts):
        track();result=dict(complete=False,entry_obstruction=True,full_solves=0,global_entry_checks=len(checks),admissibility_records=len(linear),wall_seconds=time.perf_counter()-start,peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,training=False,test9626_touched=False,original_source_exported=False)
        (out/'settings.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return
    # No stopped gain/reference-composition variants are run.
    for c in contexts:
        n,index,low,high,base,wrappers,name,recipe,margin,qnorm=(c[k] for k in ('n','index','low','high','base','wrappers','name','recipe','margin','qnorm'))
        setup=c['setup'];order=list(METHODS);rng=random.Random(79331+n+index)
        for repeat in range(-1,policy['timed_repeats']):
            rng.shuffle(order);values={};batch=[]
            for method in order:
                limit();track();torch.cuda.reset_peak_memory_stats();before_allocated=torch.cuda.memory_allocated();sync();t=time.perf_counter()
                if method=='complex_direct_segmented':
                    initial=None if n==256 else balance_integrated_source(high,base.reference.reference_transport_solve(low.source).to(torch.complex128),high.source)
                    inject=None if n==256 else lambda x,r:base.reference.reference_transport_solve(r.to(torch.complex64))
                    v,status=complex_augmented_fgmres(high.A_op,high.source,lambda r:wrappers['direct_segmented'](r.to(torch.complex64)),initial,inject=inject,inject_after=4,tolerance=1e-5,maxiter=100,restart=25)
                    status.pop('trace',None)
                else:v,status=solve(recipe,low,high,{name:base if method=='eager' else wrappers[method]}, {})
                sync();seconds=time.perf_counter()-t;values[method]=v
                sync();t=time.perf_counter();res=float((high.source-high.A_op(v)).norm());delta=res/margin;den=float(v.norm())-delta;bound=delta/den if den>0 else None;finite=bool(torch.isfinite(v).all());imaginary_relative=float(v.imag.norm()/v.norm().clamp_min(1e-30));sync();acceptance=time.perf_counter()-t
                kind='complex' if method=='complex_direct_segmented' else 'real';cap=0. if method=='eager' else setup['constructor_seconds']['direct_segmented' if kind=='complex' else method]
                operational_setup=setup['physical_seconds']+setup['base_seconds']+setup['acceptance_setup_seconds']+cap+setup['solver_preparation_seconds'][kind]
                charged_setup=operational_setup+setup['mandatory_admissibility_seconds']
                track();batch.append(dict(size=n,index=index,method=method,repeat=repeat,warmup=repeat<0,seconds=seconds,acceptance_seconds=acceptance,accepted_seconds=seconds+acceptance,
                    operational_setup_seconds=operational_setup,setup_seconds=charged_setup,operational_q1_seconds=operational_setup+seconds+acceptance,setup_plus_accepted_seconds=charged_setup+seconds+acceptance,
                    mandatory_acquisition_seconds=setup['mandatory_admissibility_seconds'],status=status,independent_true_residual=res/qnorm,evaluated_error_bound=bound,evaluated_margin=margin,finite=finite,
                    solve_peak_allocated_bytes=torch.cuda.max_memory_allocated(),solve_extra_peak_allocated_bytes=max(0,torch.cuda.max_memory_allocated()-before_allocated),solve_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    final_imaginary_relative=imaginary_relative,training=False))
            reference=values['direct_segmented']
            for r in batch:
                r['field_difference']=float((values[r['method']]-reference).norm()/reference.norm());r['bitwise_vs_real_direct_segmented']=bool(torch.equal(values[r['method']],reference))
                r['accuracy_passed']=r['finite'] and r['status']['failed']==0 and r['independent_true_residual']<=1e-5 and r['evaluated_error_bound'] is not None and math.isfinite(r['evaluated_error_bound']) and 0<=r['evaluated_error_bound']<=1e-3 and r['field_difference']<=1e-4
            for method in REAL[:-1]:
                a=next(r for r in batch if r['method']==method);b=next(r for r in batch if r['method']=='direct_segmented')
                equal=bool(torch.equal(values[method],reference) and a['status']==b['status']);a['same_recipe_field_status_parity']=equal;assert equal
            rows.extend(batch);save();assert len(rows)<=policy['max_full_solves']
        print(f'complex full solves N{n} ID{index} complete',flush=True)
    track();assert len(rows)==policy['max_full_solves']
    result=dict(complete=True,entry_obstruction=False,wall_seconds=time.perf_counter()-start,full_solves=len(rows),measured_full_solves=sum(not r['warmup'] for r in rows),
        global_entry_checks=len(checks),admissibility_records=len(linear),all_entry_passed=all(r['passed'] for r in checks),all_admissibility_passed=all(r['passed'] for r in linear),
        all_accuracy_passed=all(r['accuracy_passed'] for r in rows),all_same_recipe_parity_passed=True,peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,
        training=False,test9626_touched=False,original_source_exported=False,scope='One complex-coefficient SVD companion, strongest same explicit-segmented factors, four extra original real controls. Entire operational setup and common mandatory entry acquisition charged separately. No range fit or stopped gain variants.')
    (out/'settings.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
