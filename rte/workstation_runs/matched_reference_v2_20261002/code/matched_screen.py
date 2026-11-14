"""Matched original-dtype reference correction; no training or range-optimum claim."""
import hashlib,json,math,random,time
from pathlib import Path
import torch
from graph_classical_v2 import GraphClassicalV2
from segmented_tsa import SegmentedTSA
from direct_segmented_tsa import DirectSegmentedTSA
from direct_full_graph import DirectFullGraph
from matched_reference import ReferenceDefectAction,RawReferenceDefectGraph
from rte.augmented_fgmres import augmented_fgmres
from rte.benchmark_tail_solver import solve
from rte.global_source_predictor import balance_integrated_source
from rte.train_source_predictor import build_base,material_operator,source_family

STRONG=('eager','context_full','context_segmented','direct_full','direct_segmented')
METHODS=STRONG+('tsa4','full_segmented','q3_segmented','full_raw','q3_raw')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sync():torch.cuda.synchronize()
def dot(a,b):return (a.conj()*b).real.sum()
def project(v,basis):
    if basis is not None:
        for _ in range(2):
            for q in basis:v=v-q*dot(q,v)
    return v
def direction_metrics(target,image,basis):
    b=project(target,basis);a=project(image,basis);bn=float(b.norm());an=float(a.norm());den=dot(a,a)
    alpha=float(dot(b,a)/den) if float(den)>0 else 0.
    return dict(target_norm=bn,image_norm=an,real_alpha=alpha,unit_remaining_ratio=float((b-a).norm())/max(bn,1e-30),
        best_real_line_remaining_ratio=float((b-alpha*a).norm())/max(bn,1e-30),
        real_cosine=float(dot(b,a))/max(bn*an,1e-30),projection_removed_image_fraction=float((image-a).norm())/max(float(image.norm()),1e-30))

def span_metrics(target,images,basis):
    began=time.perf_counter();b=project(target,basis);cols=[project(v,basis) for v in images];norms=[v.norm() for v in cols]
    unit=[v/n.clamp_min(1e-30) for v,n in zip(cols,norms)];gram=torch.stack([torch.stack([dot(a,c) for c in unit]) for a in unit]);rhs=torch.stack([dot(a,b) for a in unit])
    ev,V=torch.linalg.eigh(gram);cut=1e-12*ev.max().clamp_min(1.);inv=torch.where(ev>cut,ev.clamp_min(1e-30).reciprocal(),torch.zeros_like(ev));coef=V@(inv*(V.T@rhs))
    remaining=b-sum((c*a for c,a in zip(coef,unit)),torch.zeros_like(b));sync()
    return dict(attained_relative_residual=float(remaining.norm()/b.norm().clamp_min(1e-30)),rank=int((ev>cut).sum()),gram_eigenvalues=[float(x) for x in ev],
        coefficients=[float(c/n.clamp_min(1e-30)) for c,n in zip(coef,norms)],small_fit_and_project_seconds=time.perf_counter()-began,scope='attained two-proposal real span only; no range optimum; actual H image acquisition recorded separately')

@torch.no_grad()
def run():
    torch.set_num_threads(1);root=Path(__file__).resolve().parent.parent;policy=json.loads((root/'policy.json').read_text());assert policy['phase']=='matched_reference_screen_v2'
    for n,h in policy['code_sha256'].items():assert sha(root/n)==h
    out=root/'results';out.mkdir(exist_ok=False);start=time.perf_counter();deadline=start+policy['soft_seconds']
    checks=[];setups=[];apps=[];geometry=[];prefixes=[];rows=[];contexts=[]
    def check_time():
        if time.perf_counter()>deadline:raise TimeoutError('frozen matched-reference soft deadline')
        if torch.cuda.max_memory_reserved()>policy['max_reserved_bytes']:raise MemoryError('frozen matched-reference memory envelope exceeded')
    def save():
        for n,v in [('checks',checks),('setup',setups),('applications',apps),('geometry',geometry),('prefixes',prefixes),('solver',rows)]:
            (out/(n+'.json')).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    for n in policy['grids']:
        parent=Path(f'rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{n}.pt');assert sha(parent)==policy['development_parent_sha256'][str(n)]
        data=torch.load(parent,map_location='cpu',weights_only=True);cfg,reg=source_family(n,16,4626)
        for k in ('sigma_t','sigma_s','centers','phases','amplitudes','angular_codes','geometry_base64'):assert torch.equal(data[k],reg[k])
        for index in policy['development_ids']:
            check_time();sync();t=time.perf_counter();low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda');sync();physical=time.perf_counter()-t
            strong_name='tsa8' if n==128 else 'tsa16';recipe='reference_tail' if n==128 else 'tsa16_zero';bases={};bs={}
            for name in ([strong_name,'tsa4'] if index%2==0 else ['tsa4',strong_name]):
                sync();t=time.perf_counter();bases[name]=build_base(low,name);sync();bs[name]=time.perf_counter()-t
            strong,shallow=bases[strong_name],bases['tsa4'];wrappers={};capture={}
            def original_literal(rhs,kind):
                original_g=shallow(rhs)
                original_d=rhs-low.A_op(original_g)
                return original_g+shallow.reference.reference_transport_solve(original_d,harmonics=1 if kind=='q3' else None)
            order=['context_full','context_segmented','direct_full','direct_segmented','tsa4']
            if index%2:order.reverse()
            for method in order:
                sync();t=time.perf_counter()
                if method=='context_full':w=GraphClassicalV2(strong,low.source)
                elif method=='context_segmented':w=SegmentedTSA(strong,low.source,chunk=1,capture_first=False)
                elif method=='direct_full':w=DirectFullGraph(strong,low.source)
                elif method=='direct_segmented':w=DirectSegmentedTSA(strong,low.source)
                else:w=DirectSegmentedTSA(shallow,low.source)
                sync();capture[method]=time.perf_counter()-t;wrappers[method]=w
            actions={}
            mix_order=['full','q3'] if index%2==0 else ['q3','full']
            for kind in mix_order:
                harmonics=None if kind=='full' else 1
                sync();t=time.perf_counter();a=ReferenceDefectAction(shallow,wrappers['tsa4'],harmonics=harmonics);sync();capture[kind+'_segmented']=capture['tsa4']+time.perf_counter()-t;wrappers[kind+'_segmented']=a
                sync();t=time.perf_counter();literal=ReferenceDefectAction(shallow,shallow,harmonics=harmonics);w=RawReferenceDefectGraph(literal,low.source);sync();capture[kind+'_raw']=time.perf_counter()-t;wrappers[kind+'_raw']=w;actions[kind]=literal
            sync();t=time.perf_counter();P=high.phase_matrix.to(torch.complex128)*high.weights[None,:];mu=torch.linalg.eigvalsh((P+P.mH)/2).max().real;margin=float((high.sigma_t-high.sigma_s*mu).min());qnorm=float(high.source.norm());sync();acceptance_setup=time.perf_counter()-t
            assert margin>0 and math.isfinite(margin)
            stripe=(-1.)**torch.arange(n,device='cuda');grazing=torch.zeros_like(low.source);grazing[:,0]=stripe[None,None,:];probes=[('source',low.source),('grazing',grazing),('mixed',low.source.roll(1,-2)*(1.13+.21j))]
            for dtype in (torch.complex64,torch.complex128):
                for label,probe in probes:
                    rhs=probe.to(dtype)
                    for method,w in wrappers.items():
                        expected=(strong(rhs) if method in STRONG else shallow(rhs) if method=='tsa4' else original_literal(rhs,'q3' if method.startswith('q3') else 'full')).to(torch.complex128)
                        got=w(rhs).to(torch.complex128);passed=bool(torch.equal(got,expected) and torch.isfinite(got).all())
                        checks.append(dict(size=n,index=index,method=method,probe=label,dtype=str(dtype),bitwise_equal=bool(torch.equal(got,expected)),finite=bool(torch.isfinite(got).all()),passed=passed))
            for method,w in wrappers.items():
                saved=w(low.source).to(torch.complex128);held=saved.clone();w(grazing);ownership=bool(torch.equal(saved,held));base=strong if method in STRONG else shallow;old=base.reference._reference_transport_inverse;base.reference._reference_transport_inverse=old.transpose(1,2)
                try:
                    try:w(low.source)
                    except RuntimeError:guard=True
                    else:guard=False
                finally:base.reference._reference_transport_inverse=old
                expected=(strong(low.source) if method in STRONG else shallow(low.source) if method=='tsa4' else original_literal(low.source,'q3' if method.startswith('q3') else 'full')).to(torch.complex128)
                restored=bool(torch.equal(w(low.source).to(torch.complex128),expected));checks.append(dict(size=n,index=index,method=method,kind='ownership_cache_view',ownership=ownership,guard=guard,restored=restored,passed=ownership and guard and restored))
            setups.append(dict(size=n,index=index,physical_seconds=physical,base_seconds=bs,constructor_seconds=capture,acceptance_setup_seconds=acceptance_setup,evaluated_margin=margin,strong_depth=strong.depth,
                capture_fft_counts={m:getattr(w,'capture_fft_counts',None) for m,w in wrappers.items()},mix_info={m:w.info for m,w in wrappers.items() if m.startswith(('full_','q3_'))},reference_cache_bytes=shallow.reference._reference_transport_inverse.numel()*shallow.reference._reference_transport_inverse.element_size()))
            save();assert all(c['passed'] for c in checks)
            for repeat in range(7):
                for method in METHODS:
                    sync();t=time.perf_counter();v=(strong(low.source) if method=='eager' else wrappers[method](low.source)).to(torch.complex128);sync();duration=time.perf_counter()-t
                    if repeat>=2:apps.append(dict(size=n,index=index,method=method,repeat=repeat-2,seconds=duration,finite=bool(torch.isfinite(v).all())))
            contexts.append(dict(n=n,index=index,low=low,high=high,strong=strong,shallow=shallow,wrappers=wrappers,actions=actions,strong_name=strong_name,recipe=recipe,physical=physical,bs=bs,capture=capture,acceptance_setup=acceptance_setup,margin=margin,qnorm=qnorm));save()
            print(f'matched entries N{n} ID{index} passed',flush=True)
    assert len(checks)==policy['global_entry_checks'] and all(c['passed'] for c in checks)
    # Geometry is diagnostic acquisition and never omitted from an online oracle cost claim.
    for c in contexts:
        n,index,low,high,strong,shallow,wrappers,actions=(c[k] for k in ('n','index','low','high','strong','shallow','wrappers','actions'))
        sync();t=time.perf_counter();init=None if n==256 else balance_integrated_source(high,strong.reference.reference_transport_solve(low.source).to(torch.complex128),high.source);sync();prefix_init_seconds=time.perf_counter()-t
        caps=[];sync();t=time.perf_counter();_,status=augmented_fgmres(high.A_op,high.source,lambda r:wrappers['direct_segmented'](r.to(torch.complex64)),init,maxiter=4,restart=25,capture_prefix=caps.append);sync();prefix_seconds=time.perf_counter()-t
        assert len(caps)==1 and caps[0]['image_basis'] is not None;cap=caps[0];sync();t=time.perf_counter();physical_prefix=high.source-high.A_op(cap['solution']);basis=cap['image_basis'];gram=torch.stack([torch.stack([dot(a,b) for b in basis]) for a in basis]);prefix_gap=float((physical_prefix-cap['residual']).norm()/cap['residual'].norm().clamp_min(1e-30));az_gram_defect=float((gram-torch.eye(len(basis),device=gram.device)).norm());sync();prefix_verify_seconds=time.perf_counter()-t
        prefix_total=prefix_init_seconds+prefix_seconds+prefix_verify_seconds
        prefixes.append(dict(size=n,index=index,seconds=prefix_seconds,initialization_seconds=prefix_init_seconds,independent_verification_seconds=prefix_verify_seconds,total_diagnostic_seconds=prefix_total,status={k:v for k,v in status.items() if k!='trace'},scope='four-step truncated diagnostic, not a successful complete solve',prefix_gap=prefix_gap,real_AZ_gram_defect=az_gram_defect))
        sync();t=time.perf_counter();H0=shallow.reference._reference_transport_matrix().to(torch.complex128);sync();h0_setup=time.perf_counter()-t
        def h0(v):
            spectrum=torch.fft.fftn(v.to(torch.complex128),dim=(-2,-1));return torch.fft.ifftn(torch.einsum('bhwij,bjhw->bihw',H0,spectrum),dim=(-2,-1))
        for label,r,basis in [('source',high.source,None),('prefix4',physical_prefix,cap['image_basis'])]:
            check_time();sync();target_began=time.perf_counter();t=target_began;r32=r.to(torch.complex64);g32=shallow(r32);d32=r32-low.A_op(g32);dtrue=r-high.A_op(g32.to(torch.complex128));pi=shallow.reference.angular_project(d32,harmonics=1)
            zq=shallow.reference.reference_transport_solve(d32,harmonics=1);zf=shallow.reference.reference_transport_solve(d32);zm=shallow.reference.reference_transport_solve(d32-pi)
            pq=(g32+zq).to(torch.complex128);pf=(g32+zf).to(torch.complex128);Hq=high.A_op(zq.to(torch.complex128));Hf=high.A_op(zf.to(torch.complex128));Hmq=high.A_op(pq);Hmf=high.A_op(pf);Hm=high.A_op(zm.to(torch.complex128));sync();acquire=time.perf_counter()-t
            sync();t=time.perf_counter();small=span_metrics(r,[Hmq,Hmf],basis);sync();fit_seconds=time.perf_counter()-t
            record=dict(size=n,index=index,context=label,diagnostic_acquisition_seconds=acquire,prefix_acquisition_seconds=prefix_total if basis is not None else 0.,homogeneous_diagnostic_matrix_seconds=h0_setup,oracle_small_fit_seconds=fit_seconds,
                low_vs_high_defect_relative=float((d32.to(torch.complex128)-dtrue).norm()/dtrue.norm().clamp_min(1e-30)),discarded_input_angle_fraction=float((d32-pi).norm()/d32.norm().clamp_min(1e-30)),
                q3_higher_output_angle_fraction=float((zq-shallow.reference.angular_project(zq,harmonics=1)).norm()/zq.norm().clamp_min(1e-30)),
                full_homogeneous_closure_relative=float((h0(zf)-d32.to(torch.complex128)).norm()/d32.norm().clamp_min(1e-30)),q3_homogeneous_closure_relative=float((h0(zq)-pi.to(torch.complex128)).norm()/d32.norm().clamp_min(1e-30)),
                correction_additivity_relative=float((zf.to(torch.complex128)-zq.to(torch.complex128)-zm.to(torch.complex128)).norm()/zf.norm().clamp_min(1e-30)),physical_image_additivity_relative=float((Hf-Hq-Hm).norm()/Hf.norm().clamp_min(1e-30)),
                composite_q3=direction_metrics(r,Hmq,basis),composite_full=direction_metrics(r,Hmf,basis),increment_q3=direction_metrics(dtrue,Hq,basis),increment_full=direction_metrics(dtrue,Hf,basis),two_composite_span_oracle=small,
                physical_residual_projection_gap=float((r-project(r,basis)).norm()/r.norm().clamp_min(1e-30)),oracle_deployable=False,range_optimality_claim=False)
            sync();record['total_geometry_diagnostic_seconds']=time.perf_counter()-target_began
            geometry.append(record);save()
        del H0
    # All original full-solve recipes/start/injection/accuracy budgets remain identical.
    for c in contexts:
        n,index,low,high,strong,shallow,wrappers,strong_name,recipe,physical,bs,capture,acceptance_setup,margin,qnorm=(c[k] for k in ('n','index','low','high','strong','shallow','wrappers','strong_name','recipe','physical','bs','capture','acceptance_setup','margin','qnorm'))
        methods=list(METHODS);rng=random.Random(68317+n+index)
        for repeat in range(-1,policy['timed_repeats']):
            rng.shuffle(methods);values={};batch=[]
            for method in methods:
                check_time();base=strong if method=='eager' else wrappers[method];sync();t=time.perf_counter();v,status=solve(recipe,low,high,{strong_name:base},{});sync();seconds=time.perf_counter()-t;values[method]=v
                sync();t=time.perf_counter();res=float((high.source-high.A_op(v)).norm());delta=res/margin;den=float(v.norm())-delta;bound=delta/den if den>0 else None;finite=bool(torch.isfinite(v).all());sync();acceptance=time.perf_counter()-t
                setup=physical+acceptance_setup+bs[strong_name if method in STRONG else 'tsa4']+(0. if method=='eager' else capture[method])
                batch.append(dict(size=n,index=index,method=method,repeat=repeat,warmup=repeat<0,seconds=seconds,acceptance_seconds=acceptance,accepted_seconds=seconds+acceptance,setup_seconds=setup,setup_plus_accepted_seconds=setup+seconds+acceptance,status=status,independent_true_residual=res/qnorm,evaluated_error_bound=bound,evaluated_margin=margin,finite=finite,diagnostic_acquisition_excluded_from_deployable_cost=True,training=False))
            reference=values['direct_segmented']
            for row in batch:
                row['field_difference']=float((values[row['method']]-reference).norm()/reference.norm());row['bitwise_vs_strong']=bool(torch.equal(values[row['method']],reference))
                row['accuracy_passed']=row['finite'] and row['status']['failed']==0 and row['independent_true_residual']<=1e-5 and row['evaluated_error_bound'] is not None and math.isfinite(row['evaluated_error_bound']) and 0<=row['evaluated_error_bound']<=1e-3 and row['field_difference']<=1e-4
            for a,b in [('eager','context_full'),('eager','context_segmented'),('eager','direct_full'),('eager','direct_segmented'),('full_segmented','full_raw'),('q3_segmented','q3_raw')]:
                ra=next(x for x in batch if x['method']==a);rb=next(x for x in batch if x['method']==b);equal=bool(torch.equal(values[a],values[b]) and ra['status']==rb['status']);ra.setdefault('same_recipe_backend_checks',[]).append(dict(other=b,passed=equal));assert equal
            rows.extend(batch);save();assert len(rows)<=policy['max_full_solves']
        print(f'matched full solves N{n} ID{index} complete',flush=True)
    assert len(rows)==policy['max_full_solves'] and len(geometry)==8 and len(prefixes)==4
    result=dict(complete=True,wall_seconds=time.perf_counter()-start,full_solves=len(rows),measured_full_solves=sum(not r['warmup'] for r in rows),global_entry_checks=len(checks),geometry_targets=len(geometry),truncated_prefix_solves=len(prefixes),all_accuracy_passed=all(r['accuracy_passed'] for r in rows),all_entry_passed=all(r['passed'] for r in checks),all_same_recipe_backend_checks_passed=True,training=False,test9626_touched=False,original_source_exported=False,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope='MatchedFP32M4+R0full/q3input information controls; five strong controls; actualsetup+warmaccepted costs; geometry/2spanoracle separatechargeddiagnostics, no rangeoptimality/learned claim')
    (out/'settings.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
