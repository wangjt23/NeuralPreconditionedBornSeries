"""Bounded from-start TSA4 plus existing transport-response range control.

Cached coefficients reproduce one source-specific online CGLS trajectory;
acquisition is excluded only in explicitly optimistic arms, not deployable.
"""
import hashlib,json,math,random,time
from pathlib import Path
import torch
from transport_response_range import TransportResponseRange
from graph_classical_v2 import GraphClassicalV2
from rte.augmented_fgmres import augmented_fgmres
from rte.benchmark_tail_solver import solve
from rte.global_source_predictor import balance_integrated_source
from rte.train_source_predictor import build_base,material_operator,source_family

def sync():torch.cuda.synchronize()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dot(a,b):return (a.conj()*b).real.sum()
def tensor_sha(v):return hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()

@torch.no_grad()
def run():
    torch.set_num_threads(1);root=Path(__file__).resolve().parent.parent
    policy=json.loads((root/'policy.json').read_text())
    for p,h in policy['code_sha256'].items():assert sha(root/p)==h
    out=root/'results';out.mkdir(exist_ok=False);started=time.perf_counter();deadline=started+policy['soft_seconds']
    checks=[];setups=[];applications=[];teacher_records=[];rows=[]
    def check_time():
        if time.perf_counter()>deadline:raise TimeoutError('response bounded process deadline')
    def save():
        for name,value in [('checks',checks),('setup',setups),('applications',applications),('teachers',teacher_records),('solver',rows)]:
            (out/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    for n in (128,256):
        parent=Path(f'rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{n}.pt');assert sha(parent)==policy['development_parent_sha256'][str(n)]
        data=torch.load(parent,map_location='cpu',weights_only=True);cfg,reg=source_family(n,16,4626)
        for key in ('sigma_t','sigma_s','centers','phases','amplitudes','angular_codes','geometry_base64'):assert torch.equal(data[key],reg[key])
        for index in (0,3):
            check_time();sync();began=time.perf_counter();low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda');sync();physical_setup=time.perf_counter()-began
            bases={};base_setup={};graphs={};graph_setup={}
            strong='tsa8' if n==128 else 'tsa16';recipe='reference_tail' if n==128 else 'tsa16_zero'
            for name in ('tsa4',strong):
                sync();began=time.perf_counter();bases[name]=build_base(low,name);sync();base_setup[name]=time.perf_counter()-began
                sync();began=time.perf_counter();graphs[name]=GraphClassicalV2(bases[name],low.source);sync();graph_setup[name]=time.perf_counter()-began
            sync();began=time.perf_counter();response=TransportResponseRange(high);gi=torch.linalg.inv(response.gram);sync();response_setup=time.perf_counter()-began
            sync();began=time.perf_counter()
            P=high.phase_matrix.to(torch.complex128)*high.weights[None,:];mu=torch.linalg.eigvalsh((P+P.mH)/2).max().real
            margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0 and math.isfinite(margin)
            sync();acceptance_setup=time.perf_counter()-began
            stripe=(-1.)**torch.arange(n,device='cuda');grazing=torch.zeros_like(low.source);grazing[:,0]=stripe[None,None,:]
            # Exact TSA4 arithmetic, new depth, plus actual full-response identity.
            for dtype in (torch.complex64,torch.complex128):
                for label,probe in [('source',low.source),('grazing',grazing)]:
                    a=bases['tsa4'](probe.to(dtype)).to(torch.complex128);b=graphs['tsa4'](probe.to(dtype))
                    checks.append(dict(size=n,index=index,kind='TSA4',probe=label,dtype=str(dtype),bitwise_equal=bool(torch.equal(a,b)),passed=bool(torch.equal(a,b) and torch.isfinite(b).all())))
            probe=response.angular_restrict(high.source);z=response.lift(probe)
            H0=response.homogeneous_matrix();spectrum=torch.fft.fftn(z,dim=(-2,-1));image=torch.einsum('bhwij,bjhw->bihw',H0,spectrum);image=torch.fft.ifftn(image,dim=(-2,-1))
            expected=response.angular_lift(probe);parity=float((image-expected).norm()/expected.norm().clamp_min(1e-30))
            other=high.source.roll(1,-2)*(1.13+.21j);lhs=dot(response.action(probe),other);rhs=dot(probe,response.adjoint_action(other));adj=float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()).clamp_min(1))
            checks.append(dict(size=n,index=index,kind='response',homogeneous_identity=parity,physical_adjoint=adj,passed=parity<=1e-9 and adj<=1e-10))
            del H0,spectrum,image,expected,z
            save();assert all(r['passed'] for r in checks)
            setrec=dict(size=n,index=index,physical_seconds=physical_setup,acceptance_setup_seconds=acceptance_setup,base_seconds=base_setup,graph_seconds=graph_setup,response_seconds=response_setup,response_bytes=response.response_symbol.numel()*response.response_symbol.element_size(),evaluated_margin=margin)
            setups.append(setrec)
            def initial(base):
                return None if n==256 else balance_integrated_source(high,base.reference.reference_transport_solve(low.source).to(torch.complex128),high.source)
            def cheap(r,graph=True):return (graphs['tsa4'] if graph else bases['tsa4'])(r.to(torch.complex64)).to(torch.complex128)
            def homogeneous(r,graph=True):
                g=cheap(r,graph);d=r-high.A_op(g);c=torch.einsum('ij,bjhw->bihw',gi,response.angular_restrict(d));return g+response.lift(c)
            # Cheap setup/application floor comes before any adaptive coarse solve.
            for repeat in range(7):
                for method,fn in [('cheap_graph',cheap),('homogeneous_graph',homogeneous),('response_adjoint',response.adjoint_action),('response_lift',response.lift)]:
                    argument=probe if method=='response_lift' else high.source
                    sync();began=time.perf_counter();value=fn(argument);sync();duration=time.perf_counter()-began
                    if repeat>=2:applications.append(dict(size=n,index=index,method=method,repeat=repeat-2,seconds=duration,finite=bool(torch.isfinite(value).all())))
            save()
            if policy['phase']=='cost':
                def control(method):
                    if method.startswith('strong_'):
                        return solve(recipe,low,high,{strong:graphs[strong] if method.endswith('_graph') else bases[strong]},{})
                    use_graph=method.endswith('_graph')
                    prec=(lambda r:cheap(r,use_graph)) if method.startswith('cheap_') else (lambda r:homogeneous(r,use_graph))
                    return augmented_fgmres(high.A_op,high.source,prec,initial(bases['tsa4']),tolerance=1e-5,maxiter=100,restart=25)
                methods=['strong_eager','strong_graph','cheap_eager','cheap_graph','homogeneous_eager','homogeneous_graph']
                rng=random.Random(94103+n+index)
                for repeat in range(-1,policy['timed_repeats']):
                    check_time();rng.shuffle(methods);results={};batch=[]
                    for method in methods:
                        sync();began=time.perf_counter();value,status=control(method);sync();duration=time.perf_counter()-began;results[method]=value
                        sync();began=time.perf_counter();res=float((high.source-high.A_op(value)).norm());ratio=res/float(high.source.norm());delta=res/margin;den=float(value.norm())-delta;bound=delta/den if den>0 else None;finite=bool(torch.isfinite(value).all());sync();acceptance_seconds=time.perf_counter()-began
                        setup=physical_setup+acceptance_setup+base_setup[strong if method.startswith('strong_') else 'tsa4']
                        if method.endswith('_graph'):setup+=graph_setup[strong if method.startswith('strong_') else 'tsa4']
                        if method.startswith('homogeneous_'):setup+=response_setup
                        batch.append(dict(size=n,index=index,method=method,repeat=repeat,warmup=repeat<0,seconds=duration,acceptance_seconds=acceptance_seconds,accepted_seconds=duration+acceptance_seconds,setup_seconds=setup,setup_plus_accepted_seconds=setup+duration+acceptance_seconds,status=status,independent_true_residual=ratio,evaluated_error_bound=bound,evaluated_margin=margin,finite=finite,acquisition_excluded=False,offline_training_seconds=0.))
                    for row in batch:
                        value=results[row['method']];field=float((value-results['strong_eager']).norm()/results['strong_eager'].norm())
                        row.update(field_difference=field,bitwise_vs_strong=bool(torch.equal(value,results['strong_eager'])))
                        row['accuracy_passed']=row['finite'] and row['status']['failed']==0 and row['independent_true_residual']<=1e-5 and row['evaluated_error_bound'] is not None and row['evaluated_error_bound']<=1e-3 and field<=1e-4
                    rows.extend(batch);save()
                print(f'response cost screen N{n} ID{index} complete',flush=True)
                del response,graphs,bases,low,high
                continue
            cache=[];inner=[]
            def online(r):
                check_time();g=cheap(r);d=r-high.A_op(g);c=torch.zeros_like(response.angular_restrict(d));t=d.clone();s=response.adjoint_action(t);p=s.clone();gamma=dot(s,s);normal0=float(s.norm());reached=False;k=0
                for k in range(policy['inner_cap']):
                    check_time();q=response.action(p);den=dot(q,q)
                    if not bool(den>0):break
                    alpha=gamma/den;c=c+alpha*p;t=t-alpha*q;snew=response.adjoint_action(t);newgamma=dot(snew,snew)
                    if float(snew.norm())<=policy['inner_normal_tolerance']*max(normal0,1e-30):reached=True;break
                    p=snew+(newgamma/gamma)*p;gamma=newgamma
                true=d-response.action(c);normal=float(response.adjoint_action(true).norm())/max(normal0,1e-30)
                inner.append(dict(call=len(cache),iterations=k+1,normal_ratio=normal,target_reached=bool(reached and normal<=policy['inner_normal_tolerance']),attained_residual_ratio=float(true.norm()/d.norm().clamp_min(1e-30))))
                cache.append((r.clone(),c.clone()));return g+response.lift(c)
            sync();began=time.perf_counter();online_value,online_status=augmented_fgmres(high.A_op,high.source,online,initial(bases['tsa4']),tolerance=1e-5,maxiter=100,restart=25);sync();online_seconds=time.perf_counter()-began
            teacher_records.append(dict(size=n,index=index,seconds=online_seconds,status=online_status,inner=inner,cached_calls=len(cache),input_sha256=[tensor_sha(r) for r,c in cache],coefficient_sha256=[tensor_sha(c) for r,c in cache],scope='one actual classical adaptive control; RHS-specific cachedcoefficient acquisition; nolearned/fine-teacher-bank claim'))
            save()
            def custom(method,check_inputs=False):
                if method in ('strong_eager','strong_graph'):
                    return solve(recipe,low,high,{strong:graphs[strong] if method=='strong_graph' else bases[strong]},{})
                use_graph=method.endswith('_graph')
                if method.startswith('cheap_'):prec=lambda r:cheap(r,use_graph)
                elif method.startswith('homogeneous_'):prec=lambda r:homogeneous(r,use_graph)
                else:
                    cursor=[0]
                    def prec(r):
                        j=cursor[0];cursor[0]+=1
                        if j>=len(cache):raise RuntimeError('cached replay exceeded acquired trajectory')
                        old,c=cache[j]
                        if check_inputs:assert torch.equal(r,old),'cached trajectory differs'
                        g=cheap(r,use_graph);d=r-high.A_op(g)
                        if method.startswith('extracted_'):response.adjoint_action(d)
                        return g+response.lift(c.clone())
                result,status=augmented_fgmres(high.A_op,high.source,prec,initial(bases['tsa4']),tolerance=1e-5,maxiter=100,restart=25)
                return result,status
            # Independent trajectory check before any cached timing, both eager
            # and replay preserve the same input/corrector sequence exactly.
            for method in ('cached_eager','cached_graph','extracted_graph'):
                v,st=custom(method,True);assert torch.equal(v,online_value) and st['true_residual']==online_status['true_residual']
            methods=['strong_eager','strong_graph','cheap_eager','cheap_graph','homogeneous_eager','homogeneous_graph','cached_eager','cached_graph','extracted_graph']
            rng=random.Random(94103+n+index)
            for repeat in range(-1,3):
                check_time();rng.shuffle(methods);results={};batch=[]
                for method in methods:
                    sync();began=time.perf_counter();value,status=custom(method);sync();duration=time.perf_counter()-began;results[method]=value
                    res=float((high.source-high.A_op(value)).norm());ratio=res/float(high.source.norm());delta=res/margin;den=float(value.norm())-delta;bound=delta/den if den>0 else None
                    setup=physical_setup+base_setup[strong if method.startswith('strong_') else 'tsa4']
                    if method.endswith('_graph'):setup+=graph_setup[strong if method.startswith('strong_') else 'tsa4']
                    if method.startswith(('homogeneous_','cached_','extracted_')):setup+=response_setup
                    batch.append(dict(size=n,index=index,method=method,repeat=repeat,warmup=repeat<0,seconds=duration,setup_seconds=setup,setup_plus_warm_seconds=setup+duration,status=status,independent_true_residual=ratio,evaluated_error_bound=bound,evaluated_margin=margin,finite=bool(torch.isfinite(value).all()),acquisition_excluded=method.startswith(('cached_','extracted_')),offline_training_seconds=0.))
                for row in batch:
                    value=results[row['method']];field=float((value-results['strong_eager']).norm()/results['strong_eager'].norm())
                    row.update(field_difference=field,bitwise_vs_strong=bool(torch.equal(value,results['strong_eager'])))
                    row['accuracy_passed']=row['finite'] and row['status']['failed']==0 and row['independent_true_residual']<=1e-5 and row['evaluated_error_bound'] is not None and row['evaluated_error_bound']<=1e-3 and field<=1e-4
                rows.extend(batch);save()
            print(f'response from-start N{n} ID{index} complete',flush=True)
            del cache,response,graphs,bases,online_value
    result=dict(complete=True,phase=policy['phase'],wall_seconds=time.perf_counter()-started,total_timed_and_warm_solves=len(rows),measured_solves=sum(not r['warmup'] for r in rows),all_accuracy_passed=all(r['accuracy_passed'] for r in rows),all_checks_passed=all(r['passed'] for r in checks),independent_media=2,training=False,test9626_touched=False,workload='primary Q1permedium; current four inputs, offlineacquisition charged separately; conditionalQ16 sensitivity only',peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope='6 classical arms with fully charged extra acceptance checks before any CGLS acquisition' if policy['phase']=='cost' else '9arms, one actualadaptivecontrol and3verifiedcachedsanities perinput; optimisticcached inverse excludedacquisition, nolearnedwin')
    (out/'settings.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
