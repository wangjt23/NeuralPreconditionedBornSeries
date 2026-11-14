"""Bounded, development-only full-spatial q3 Galerkin quality/cost oracle."""
import hashlib, json, math, random, statistics, time
from pathlib import Path
import torch
from angular_galerkin import AngularGalerkin
from graph_classical import GraphClassical
from instrumented_fgmres import instrumented_fgmres
from rte.augmented_fgmres import augmented_fgmres
from rte.benchmark_tail_solver import solve
from rte.global_source_predictor import balance_integrated_source
from rte.train_source_predictor import build_base, material_operator, source_family

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sync(): torch.cuda.synchronize()
def rel(a,b): return float((a-b).norm()/b.norm().clamp_min(1e-30))
def dot(a,b): return (a.conj()*b).real.sum()
def project(v,basis):
    if basis is not None:
        for _ in range(2):
            for q in basis: v=v-q*dot(q,v)
    return v

@torch.no_grad()
def run():
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parent.parent
    policy=json.loads((root/'angular_policy_v2.json').read_text())
    for p,h in policy['code_sha256'].items(): assert sha(root/p)==h
    out=root/'angular_oracle'; out.mkdir(exist_ok=False)
    started=time.perf_counter(); deadline=started+policy['soft_budget_seconds']
    records=[]; parity=[]; timings=[]; setups=[]; ranges=[]
    def check_time():
        if time.perf_counter()>deadline: raise TimeoutError('angular oracle process budget reached')
    def save():
        for name,value in [('coarse_cases',records),('parity',parity),('solver_cases',timings),('setup',setups),('range_cases',ranges)]:
            (out/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    for n in policy['grids']:
        parent=Path(f'rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{n}.pt')
        assert sha(parent)==policy['development_parent_sha256'][str(n)]
        data=torch.load(parent,map_location='cpu',weights_only=True); cfg,reg=source_family(n,16,4626)
        for key in ('sigma_t','sigma_s','centers','phases','amplitudes','angular_codes','geometry_base64'): assert torch.equal(data[key],reg[key])
        for index in policy['development_ids']:
            check_time(); sync(); began=time.perf_counter()
            low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda'); sync(); physical_setup=time.perf_counter()-began
            name='tsa8' if n==128 else 'tsa16'; frozen='reference_tail' if n==128 else 'tsa16_zero'
            sync(); began=time.perf_counter(); base=build_base(low,name);sync();base_setup=time.perf_counter()-began
            graph=GraphClassical(base,low.source); precondition=lambda r:graph(r.to(torch.complex64))
            sync();began=time.perf_counter(); coarse=AngularGalerkin(high);sync();coarse_setup=time.perf_counter()-began
            P=high.phase_matrix.to(torch.complex128)*high.weights[None,:]
            mu=torch.linalg.eigvalsh((P+P.mH)/2).max().real
            margin=float((high.sigma_t-high.sigma_s*mu).min()); gm=float(torch.linalg.eigvalsh((coarse.gram+coarse.gram.mH)/2).min())
            assert margin>0 and math.isfinite(margin) and float(coarse.coercivity_bound().min())>0
            # Deterministic, non-physical probes never become validation cases.
            coordinates=torch.arange(n,device='cuda',dtype=torch.float64)
            probe=coarse.restrict(high.source)+torch.exp(1j*coordinates)[None,None,None,:]*.013
            other=probe.roll(1,-2)*(1.17+.31j)
            direct=coarse.restrict(high.A_op(coarse.lift(probe)))
            action=coarse.action(probe)
            lhs=dot(coarse.action(probe),other);rhs=dot(probe,coarse.adjoint(other))
            fullprobe=coarse.lift(other)+high.source*.17j
            lhsfull=dot(coarse.full_action(probe),fullprobe);rhsfull=dot(probe,coarse.full_adjoint(fullprobe))
            record=dict(size=n,index=index,projection_relative=rel(action,direct),
                adjoint_relative=float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()).clamp_min(1)),
                full_adjoint_relative=float((lhsfull-rhsfull).abs()/torch.maximum(lhsfull.abs(),rhsfull.abs()).clamp_min(1)),
                gram_defect=float((coarse.gram-torch.eye(3,device='cuda')).norm()),evaluated_margin=margin,
                evaluated_coarse_margin=float(coarse.coercivity_bound().min()))
            record['passed']=max(record['projection_relative'],record['adjoint_relative'],record['full_adjoint_relative'])<=1e-10
            parity.append(record);save();assert record['passed']
            initial=None if n==256 else balance_integrated_source(high,base.reference.reference_transport_solve(low.source).to(torch.complex128),high.source)
            captured=[]
            _,status=instrumented_fgmres(high.A_op,high.source,precondition,initial,maxiter=4,capture_prefix=captured.append)
            assert len(captured)==1 and captured[0]['image_basis'] is not None
            context=captured[0];contexts=[('source',high.source,None),('prefix4',context['residual'],context['image_basis'])]
            proposals={};targets={};basis_by={}
            for label,target,basis in contexts:
                check_time(); sync();began=time.perf_counter(); crhs=coarse.restrict(target)
                c,cs=augmented_fgmres(coarse.action,crhs,coarse.reference_solve,tolerance=1e-8,maxiter=100,restart=25)
                z=coarse.lift(c);sync();seconds=time.perf_counter()-began
                true=float((crhs-coarse.action(c)).norm()/crhs.norm().clamp_min(1e-30))
                image=high.A_op(z);useful=project(image,basis)
                alpha=float(dot(target,useful)/dot(useful,useful).clamp_min(1e-30))
                raw_ratio=float((target-image).norm()/target.norm());useful_ratio=float((target-alpha*useful).norm()/target.norm())
                records.append(dict(size=n,index=index,context=label,seconds=seconds,independent_coarse_relative=true,
                    coarse_accepted=bool(cs['failed']==0 and true<=1e-8 and torch.isfinite(z).all()),
                    full_residual_ratio=raw_ratio,useful_line_residual_ratio=useful_ratio,
                    coarse_rhs_norm=float(crhs.norm()),absolute_coarse_residual=float((crhs-coarse.action(c)).norm()),correction_norm=float(z.norm()),useful_image_norm=float(useful.norm()),coarse_calls=cs['classical_calls'],status=cs,original_prefix_status=status if label=='prefix4' else None))
                proposals[label]=z; targets[label]=target; basis_by[label]=basis
                proposals[label+'_reference']=coarse.lift(coarse.reference_solve(crhs));save()
            setups.append(dict(size=n,index=index,physical_seconds=physical_setup,base_seconds=base_setup,
                graph_seconds=graph.setup_seconds,coarse_seconds=coarse_setup,coarse_inverse_bytes=coarse.reference_inverse.numel()*coarse.reference_inverse.element_size()))
            # Acquisition is intentionally excluded from cached arms. They include
            # an owned device copy, original image/projection/checks and initialization.
            def custom(method):
                if method=='graph_baseline': return solve(frozen,low,high,{name:graph},{})
                if method.endswith('_initial'):
                    init=proposals['source_range' if method=='cached_range_initial' else 'source'].clone()
                    inject=(lambda x,r:base.reference.reference_transport_solve(r.to(torch.complex64))) if n==128 else None
                else:
                    init=None if n==256 else balance_integrated_source(high,base.reference.reference_transport_solve(low.source).to(torch.complex128),high.source)
                    key='prefix4_range' if method=='cached_range_tail' else ('prefix4_reference' if method=='cached_qreference_tail' else 'prefix4')
                    inject=lambda x,r:proposals[key].clone()
                return augmented_fgmres(high.A_op,high.source,precondition,init,inject=inject,tolerance=1e-5,maxiter=100,restart=25)
            methods=['graph_baseline','cached_galerkin_tail','cached_qreference_tail','cached_galerkin_initial']
            rng=random.Random(52103+n+index)
            for repeat in range(-1,3):
                check_time(); rng.shuffle(methods);results={};batch=[]
                for method in methods:
                    sync();began=time.perf_counter();value,st=custom(method);sync();duration=time.perf_counter()-began
                    results[method]=value;normres=float((high.source-high.A_op(value)).norm());ratio=normres/float(high.source.norm())
                    delta=normres/margin;den=float(value.norm())-delta;bound=delta/den if den>0 else None
                    batch.append(dict(size=n,index=index,method=method,repeat=repeat,warmup=repeat<0,seconds=duration,
                        independent_true_residual=ratio,evaluated_margin=margin,evaluated_error_bound=bound,
                        finite=bool(torch.isfinite(value).all()),status=st))
                for row in batch:
                    value=results[row['method']];field=rel(value,results['graph_baseline'])
                    row.update(field_difference=field,bitwise_equal_vs_graph=bool(torch.equal(value,results['graph_baseline'])))
                    row['accuracy_passed']=row['finite'] and row['status']['failed']==0 and row['independent_true_residual']<=1e-5 and row['evaluated_error_bound'] is not None and row['evaluated_error_bound']<=1e-3 and field<=1e-4
                timings.extend(batch);save()
            # Right-preconditioned CGLS image-range diagnostic. Always retain
            # attained values; projected range has no coercivity lower bound.
            for label,target,basis in contexts:
                check_time();sync();began=time.perf_counter()
                C=lambda y:project(coarse.full_action(coarse.reference_solve(y)),basis)
                CT=lambda v:coarse.reference_adjoint_solve(coarse.full_adjoint(project(v,basis)))
                y=torch.zeros_like(coarse.restrict(target));t=target.clone();s=CT(t);p=s.clone();gamma=dot(s,s);normal0=float(s.norm())
                reached=False;k=0
                for k in range(100):
                    check_time();q=C(p);den=dot(q,q)
                    if not bool(den>0):break
                    a=gamma/den;y=y+a*p;t=t-a*q;snew=CT(t);gnew=dot(snew,snew)
                    if float(snew.norm())<=1e-8*max(normal0,1e-30):reached=True;s=snew;break
                    p=snew+(gnew/gamma)*p;s=snew;gamma=gnew
                c=coarse.reference_solve(y);z=coarse.lift(c);proposals[label+'_range']=z;physical_image=project(high.A_op(z),basis);t=target-physical_image
                rawgrad=coarse.full_adjoint(project(t,basis));sync();duration=time.perf_counter()-began
                upper=float(t.norm());normal=float(CT(t).norm())/max(normal0,1e-30)
                lower=math.sqrt(max(0.,upper**2-float(rawgrad.norm())**2/(margin**2*gm))) if basis is None else None
                ranges.append(dict(size=n,index=index,context=label,iterations=k+1,seconds=duration,
                    attained_relative_residual=upper/float(target.norm()),independent_preconditioned_normal_relative=normal,
                    target_reached=bool(reached and normal<=1e-8),raw_gradient_norm=float(rawgrad.norm()),
                    raw_range_minimum_relative_lower=lower/float(target.norm()) if lower is not None else None,
                    raw_range_minimum_relative_upper=upper/float(target.norm()),
                    bound_scope='raw HQ coercivity/Gram ordinary-FP64 bracket' if basis is None else 'attained upper bound only; AZ projected range loses coercivity bound',
                    optimality_claim=False,finite=bool(torch.isfinite(z).all() and torch.isfinite(t).all())))
                save()
            # Attained LS proposals get their own complete cached continuation
            # timings even when stationarity is underresolved. A negative result
            # cannot reject the optimal angular range; positive attained saving
            # is a feasible headroom witness, not an inverse/optimality claim.
            methods=['graph_baseline','cached_range_tail','cached_range_initial']
            for repeat in range(-1,3):
                check_time();rng.shuffle(methods);results={};batch=[]
                for method in methods:
                    sync();began=time.perf_counter();value,st=custom(method);sync();duration=time.perf_counter()-began
                    results[method]=value;normres=float((high.source-high.A_op(value)).norm());ratio=normres/float(high.source.norm())
                    delta=normres/margin;den=float(value.norm())-delta;bound=delta/den if den>0 else None
                    batch.append(dict(size=n,index=index,method=method,repeat=repeat,warmup=repeat<0,timing_block='attained_range',seconds=duration,
                        independent_true_residual=ratio,evaluated_margin=margin,evaluated_error_bound=bound,finite=bool(torch.isfinite(value).all()),status=st))
                for row in batch:
                    value=results[row['method']];field=rel(value,results['graph_baseline'])
                    row.update(field_difference=field,bitwise_equal_vs_graph=bool(torch.equal(value,results['graph_baseline'])))
                    row['accuracy_passed']=row['finite'] and row['status']['failed']==0 and row['independent_true_residual']<=1e-5 and row['evaluated_error_bound'] is not None and row['evaluated_error_bound']<=1e-3 and field<=1e-4
                timings.extend(batch);save()
            del coarse,base,graph,low,high,context,proposals,targets,basis_by
            print(f'angular q3 N{n} ID{index} complete',flush=True)
    result=dict(complete=True,wall_seconds=time.perf_counter()-started,independent_media=2,
        parity_passed=all(x['passed'] for x in parity),all_coarse_accepted=all(x['coarse_accepted'] for x in records),
        all_full_solves_accuracy_passed=all(x['accuracy_passed'] for x in timings),range_optimality_claim=False,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        training=False,test_touched=False,scope='q3 cached oracle; acquisition excluded from cached timings, included separately; no ML success claim')
    (out/'settings.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
