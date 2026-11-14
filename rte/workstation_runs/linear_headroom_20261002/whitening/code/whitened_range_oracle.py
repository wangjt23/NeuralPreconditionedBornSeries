"""One fixed full-image-whitened LS follow-up on the same eight dev contexts."""
import hashlib,json,math,random,time
from pathlib import Path
import torch
from angular_galerkin import AngularGalerkin
from image_whitening import ImageWhitening
from graph_classical_v2 import GraphClassicalV2
from instrumented_fgmres import instrumented_fgmres
from rte.augmented_fgmres import augmented_fgmres
from rte.benchmark_tail_solver import solve
from rte.global_source_predictor import balance_integrated_source
from rte.train_source_predictor import build_base,material_operator,source_family

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sync():torch.cuda.synchronize()
def dot(a,b):return (a.conj()*b).real.sum()
def project(v,basis):
    if basis is not None:
        for _ in range(2):
            for q in basis:v=v-q*dot(q,v)
    return v

@torch.no_grad()
def run():
    torch.set_num_threads(1);root=Path(__file__).resolve().parent.parent
    policy=json.loads((root/'policy.json').read_text())
    for p,h in policy['code_sha256'].items():assert sha(root/p)==h
    out=root/'results';out.mkdir(exist_ok=False);start=time.perf_counter();deadline=start+policy['soft_seconds']
    ranges=[];rows=[];checks=[];setups=[]
    def check_time():
        if time.perf_counter()>deadline:raise TimeoutError('whitening oracle soft budget reached')
    def save():
        for n,v in [('range',ranges),('solver',rows),('checks',checks),('setup',setups)]:
            (out/(n+'.json')).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    for n in (128,256):
        parent=Path(f'rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{n}.pt')
        assert sha(parent)==policy['development_parent_sha256'][str(n)]
        data=torch.load(parent,map_location='cpu',weights_only=True);cfg,reg=source_family(n,16,4626)
        for key in ('sigma_t','sigma_s','centers','phases','amplitudes','angular_codes','geometry_base64'):assert torch.equal(data[key],reg[key])
        for index in (0,3):
            check_time();low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda')
            name='tsa8' if n==128 else 'tsa16';recipe='reference_tail' if n==128 else 'tsa16_zero'
            base=build_base(low,name);graph=GraphClassicalV2(base,low.source);prec=lambda r:graph(r.to(torch.complex64))
            coarse=AngularGalerkin(high);sync();began=time.perf_counter();whitening=ImageWhitening(coarse);sync();setup=time.perf_counter()-began
            normal=whitening.full_reference_symbol.mH@whitening.full_reference_symbol
            identity=whitening.matrix.mH@normal@whitening.matrix
            defect=float((identity-torch.eye(3,device='cuda')).norm(dim=(-2,-1)).max())
            probe=coarse.restrict(high.source);other=probe.roll(1,-2)*(1.3+.2j)
            left=dot(whitening.solve(probe),other);right=dot(probe,whitening.adjoint_solve(other))
            adj=float((left-right).abs()/torch.maximum(left.abs(),right.abs()).clamp_min(1))
            checks.append(dict(size=n,index=index,normal_whitening_max_identity_defect=defect,adjoint_relative=adj,
                reference_min_eigenvalue=whitening.minimum_reference_eigenvalue,reference_max_eigenvalue=whitening.maximum_reference_eigenvalue,
                passed=defect<=1e-8 and adj<=1e-10 and whitening.minimum_reference_eigenvalue>0))
            save();assert checks[-1]['passed']
            P=high.phase_matrix.to(torch.complex128)*high.weights[None,:];mu=torch.linalg.eigvalsh((P+P.mH)/2).max().real
            margin=float((high.sigma_t-high.sigma_s*mu).min());gram=float(torch.linalg.eigvalsh((coarse.gram+coarse.gram.mH)/2).min())
            assert margin>0 and math.isfinite(margin)
            init=None if n==256 else balance_integrated_source(high,base.reference.reference_transport_solve(low.source).to(torch.complex128),high.source)
            caps=[];_,ps=instrumented_fgmres(high.A_op,high.source,prec,init,maxiter=4,capture_prefix=caps.append)
            assert len(caps)==1 and caps[0]['image_basis'] is not None
            cap=caps[0];proposals={}
            for label,target,basis in [('source',high.source,None),('prefix4',cap['residual'],cap['image_basis'])]:
                check_time();sync();began=time.perf_counter()
                C=lambda y:project(coarse.full_action(whitening.solve(y)),basis)
                CT=lambda v:whitening.adjoint_solve(coarse.full_adjoint(project(v,basis)))
                y=torch.zeros_like(coarse.restrict(target));t=target.clone();s=CT(t);p=s.clone();gamma=dot(s,s);normal0=float(s.norm());reached=False;k=0
                for k in range(100):
                    check_time();q=C(p);den=dot(q,q)
                    if not bool(den>0):break
                    alpha=gamma/den;y=y+alpha*p;t=t-alpha*q;snew=CT(t);newgamma=dot(snew,snew)
                    if float(snew.norm())<=1e-8*max(normal0,1e-30):reached=True;break
                    p=snew+(newgamma/gamma)*p;gamma=newgamma
                c=whitening.solve(y);z=coarse.lift(c);t=target-project(high.A_op(z),basis)
                rawgrad=coarse.full_adjoint(project(t,basis));normalratio=float(CT(t).norm())/max(normal0,1e-30)
                upper=float(t.norm());lower=math.sqrt(max(0.,upper**2-float(rawgrad.norm())**2/(margin**2*gram))) if basis is None else None
                sync();duration=time.perf_counter()-began;proposals[label]=z
                ranges.append(dict(size=n,index=index,context=label,iterations=k+1,seconds=duration,target_reached=bool(reached and normalratio<=1e-8),
                    independent_preconditioned_normal_relative=normalratio,attained_relative_residual=upper/float(target.norm()),raw_gradient_norm=float(rawgrad.norm()),
                    raw_range_minimum_relative_lower=lower/float(target.norm()) if lower is not None else None,
                    raw_range_minimum_relative_upper=upper/float(target.norm()),evaluated_margin=margin,
                    finite=bool(torch.isfinite(z).all() and torch.isfinite(t).all()),optimality_claim=False,
                    bound_scope='rawHQ physicalmGram ordinaryFP64 conditionalbracket' if basis is None else 'AZ projected attainable upper only'))
                save()
            setups.append(dict(size=n,index=index,whitening_seconds=setup,graph_seconds=graph.setup_seconds,
                whitening_cache_bytes=whitening.matrix.numel()*whitening.matrix.element_size(),full_reference_symbol_bytes=whitening.full_reference_symbol.numel()*whitening.full_reference_symbol.element_size()))
            methods=['graph_baseline','cached_whitened_tail','cached_whitened_initial'];rng=random.Random(73103+n+index)
            for repeat in range(-1,3):
                check_time();rng.shuffle(methods);results={};batch=[]
                for method in methods:
                    sync();began=time.perf_counter()
                    if method=='graph_baseline':value,status=solve(recipe,low,high,{name:graph},{})
                    else:
                        if method.endswith('initial'):
                            initial=proposals['source'].clone();inject=(lambda x,r:base.reference.reference_transport_solve(r.to(torch.complex64))) if n==128 else None
                        else:
                            initial=None if n==256 else balance_integrated_source(high,base.reference.reference_transport_solve(low.source).to(torch.complex128),high.source)
                            inject=lambda x,r:proposals['prefix4'].clone()
                        value,status=augmented_fgmres(high.A_op,high.source,prec,initial,inject=inject,tolerance=1e-5,maxiter=100,restart=25)
                    sync();duration=time.perf_counter()-began;results[method]=value
                    residual=float((high.source-high.A_op(value)).norm());ratio=residual/float(high.source.norm());delta=residual/margin;den=float(value.norm())-delta;bound=delta/den if den>0 else None
                    batch.append(dict(size=n,index=index,method=method,repeat=repeat,warmup=repeat<0,seconds=duration,status=status,
                        independent_true_residual=ratio,evaluated_error_bound=bound,evaluated_margin=margin,finite=bool(torch.isfinite(value).all())))
                for row in batch:
                    value=results[row['method']];diff=float((value-results['graph_baseline']).norm()/results['graph_baseline'].norm())
                    row.update(field_difference=diff,bitwise_equal_vs_graph=bool(torch.equal(value,results['graph_baseline'])))
                    row['accuracy_passed']=row['finite'] and row['status']['failed']==0 and row['independent_true_residual']<=1e-5 and row['evaluated_error_bound'] is not None and row['evaluated_error_bound']<=1e-3 and diff<=1e-4
                rows.extend(batch);save()
            print(f'whitened q3 N{n} ID{index} complete',flush=True)
    result=dict(complete=True,wall_seconds=time.perf_counter()-start,all_checks_passed=all(r['passed'] for r in checks),all_full_accuracy=all(r['accuracy_passed'] for r in rows),
        independent_media=2,total_full_solves=len(rows),measured_full_solves=sum(not r['warmup'] for r in rows),all_range_normal_targets_reached=all(r['target_reached'] for r in ranges),
        optimality_claim=False,training=False,test9626_touched=False,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (out/'settings.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
