"""Frozen E tangent-permutation mechanism gate; one predeclared N128 material."""
import os,sys,time,json,hashlib,signal,resource,gc,subprocess
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
E=ROOT/'rte/workstation_runs/rte_angular_full_20261110_dot_v1'
sys.path[:0]=[str(E),str(ROOT),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1')]
from angular_modes import FullAngularModes
from screen_structured import medium
from layered_inverse import LayeredInverse,background_from_material
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
from rte.train_source_predictor import build_base
from rte.source_initialization import solve_from_initial
from rte.augmented_fgmres import augmented_fgmres
N=128;MULTIPLIER=3;INVERSE=43;MODES=list(range(-8,9));IDS=[0,4,7,14]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(x):return hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
def admission():return subprocess.check_output(['ps','-eo','pid,psr,pcpu,args','--sort=-pcpu'],text=True).splitlines()[:20]
class Stop(RuntimeError):
    def __init__(self,code,message):self.code=code;super().__init__(message)
def require(condition,code,message):
    if not condition:raise Stop(code,message)
def relative(error,reference):
    a,b=error.reshape(error.shape[0],-1),reference.reshape(reference.shape[0],-1)
    active=(b!=0).any(dim=1);scale=b.abs().amax(dim=1);safe=torch.where(active,scale,torch.ones_like(scale));den=(b/safe[:,None]).norm(dim=1)
    values=(a/safe[:,None]).norm(dim=1)/torch.where(active,den,torch.ones_like(den))
    return torch.where(active,values,torch.where((a!=0).any(dim=1),torch.full_like(values,float('inf')),torch.zeros_like(values)))
def project8(r):
    f=torch.fft.fft(r,dim=-1,norm='ortho');out=torch.zeros_like(f)
    for k in MODES:out[...,k]=f[...,k]
    return torch.fft.ifft(out,dim=-1,norm='ortho')
def permuted_medium():
    t,s,info=medium(N,1,10/3,'cpu');idx=(MULTIPLIER*torch.arange(N))%N
    return t.index_select(-1,idx),s.index_select(-1,idx),dict(**info,tangent_multiplier=MULTIPLIER)
def background_detail(t,s):
    axis,tb,sb,score=background_from_material(t,s);rows=[]
    for normal,reduce in [(0,-1),(1,-2)]:
        v,i=t.min(reduce,keepdim=True);g=torch.gather(s,reduce,i)
        rows.append(dict(axis=normal,score=float((t-v).square().sum()),total_profile_sha256=thash(v),scatter_profile_sha256=thash(g),gather_indices_sha256=thash(i)))
    return dict(axis=axis,total_sha256=thash(tb),scatter_sha256=thash(sb),score=score,candidate_axes=rows),tb,sb

def construct(b,depth=64):
    tick=time.perf_counter();t,s,info=permuted_medium()
    low=PeriodicRTE2D(shape=(N,N),dx=1/N,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(t,s,b.to(torch.complex64));high=high_precision_copy(low)
    phase=high.phase_matrix.double()*high.weights.double()[None];mu=torch.linalg.eigvalsh((phase+phase.T)/2).max();margin=float((high.sigma_t-high.sigma_s*mu).min());require(margin>0,'STOP_COERCIVITY','Nonpositive discrete margin')
    physical=time.perf_counter()-tick;tick=time.perf_counter();base=build_base(low,'tsa'+str(depth));base.prepare();tsa=time.perf_counter()-tick
    return low,high,base,margin,dict(physical_seconds=physical,TSA_seconds=tsa,depth=depth,material_info=info)
def background(low):
    axis,t,s,score=background_from_material(low.sigma_t,low.sigma_s)
    return dict(axis=axis,total_sha256=thash(t),scatter_sha256=thash(s),score=score),t,s
def A0_check(high,t,s,k,x,q):
    cols=x.shape[1];f=torch.zeros((cols,12,N,N),dtype=torch.complex128)
    f[...,k]=x.T.reshape(cols,12,N);field=torch.fft.ifft(f,dim=-1,norm='ortho')
    phase=(high.phase_matrix*high.weights[None]).to(torch.complex128)
    actual=high.A_op(field)-(high.sigma_t-t.double())*field+(high.sigma_s-s.double())*torch.einsum('lm,bmxy->blxy',phase,field)
    f.zero_();f[...,k]=q.T.reshape(cols,12,N);target=torch.fft.ifft(f,dim=-1,norm='ortho')
    return relative(target-actual,target)
def qualify(factor,low,high,rec,save):
    start=time.perf_counter();gen=torch.Generator().manual_seed(2026111004);info,t,s=background(low)
    require(info['axis']==0,'STOP_BACKGROUND_AXIS','E axis0 contract');rec['modes']=[];rec['boundary_cases']=[];qs={}
    for k in MODES:
        q=torch.complex(torch.randn(12*N,1,dtype=torch.float64,generator=gen),torch.randn(12*N,1,dtype=torch.float64,generator=gen));qs[k]=q
        x,st=factor.blocks[k].repaired(q,tolerance=1e-10,max_corrections=3);r=A0_check(high,t,s,k,x,q)
        row=dict(k=k,RHS_sha256=thash(q),repair=st,original_A0_relative=r.tolist());rec['modes'].append(row)
        require(st['success'] and bool((r<=1e-10).all()) and bool(torch.isfinite(x).all()),'STOP_ORIGINAL_A0',str(row))
    for k in [-8,0,8]:
        zero=torch.zeros(12*N,2,dtype=torch.complex128);q=qs[k];mix=torch.cat([zero[:,:1],q,1e-200*q,zero[:,:1]],dim=1)
        for name,rhs in [('all_zero',zero),('mixed_zero_ordinary_tiny',mix)]:
            x,st=factor.blocks[k].repaired(rhs,tolerance=1e-10,max_corrections=3);r=A0_check(high,t,s,k,x,rhs)
            inactive=~(rhs!=0).any(dim=0);strict=bool((x[:,inactive]==0).all());tiny=bool(x[:,2].abs().max()>0) if name.startswith('mixed') else None
            row=dict(k=k,kind=name,RHS_sha256=thash(rhs),repair=st,original_A0_relative=r.tolist(),strict_zero_solution=strict,tiny_nonzero_preserved=tiny);rec['boundary_cases'].append(row)
            require(st['success'] and bool((r<=1e-10).all()) and strict and tiny is not False,'STOP_BOUNDARY_QUALIFICATION',str(row))
    rec['qualification_seconds']=time.perf_counter()-start;rec['passed']=True;save()

@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);assert set(os.sched_getaffinity(0))=={0} and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    es=json.loads((E/'full_01/settings.json').read_text());inputs=json.loads((HERE/'SOURCE_AUDIT.json').read_text())
    assert digest(HERE/'FROZEN_PLAN.md')=='4b24e2d47d714a189c4b6e5ec6b2c4dae1b330409f8ae12fe3ee6c26f0b200d2'
    for f,h in es['source_hashes'].items():assert digest(E/f)==h
    sources={**es['production_hashes'],**es['dependency_hashes']}
    for f in ['rte/transport_baselines.py','rte/benchmark_tail_solver.py','rte/run_large_angular_spectral_experiments.py']:sources[f]=inputs['existing_solver_sources'][f]
    for f,h in sources.items():assert digest(ROOT/f)==h
    out=HERE/'gate_01';out.mkdir(exist_ok=False);start=time.perf_counter();current=None
    s=dict(complete=False,pid=os.getpid(),affinity=[0],threads=1,device='cpu',stage_limit_seconds=90,total_limit_seconds=240,RSS_limit_bytes=4*1024**3,grid=N,L=12,g=.65,amplitude=3,source_ids=IDS,modes=MODES,prefix=1,inner_tolerance=1e-4,inner_maxiter=8,outer_tolerance=1e-5,outer_maxiter=100,relative_bound_limit=1e-3,field_difference_limit=1e-4,block_target=1e-10,block_max_corrections=3,heldout9626=False,training=False,new_teachers=False,orders=['tsa64_zero','compact4','tsa16_zero','E'],E_source_hashes=es['source_hashes'],source_hashes=sources,own_sources={p.name:digest(p) for p in HERE.glob('*.py')},plan_sha256=digest(HERE/'FROZEN_PLAN.md'),input_audit_sha256=digest(HERE/'SOURCE_AUDIT.json'),scope='One fixed discrete tangent coefficient permutation j->3j mod128. Original four source arrays and stored streaming remain unchanged. Same-background/spectrum qualification precedes any solver; no global-material, continuous-equivalence or timing-repeat claim.',stages=[],batches=[],rows=[],qualification={},source_identity=[])
    def save():
        target=out/'results.json';temp=out/'results.json.tmp';temp.write_text(json.dumps(s,indent=2)+'\n');os.replace(temp,target)
    def expired(signum,frame):raise Stop('STOP_STAGE_WALL_OR_OWNED_TERMINATION','Signal '+str(signum))
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired)
    def stage(name):
        nonlocal current
        require(rss()<=4*1024**3,'STOP_RSS_HIGH_WATER','Recorded worker RSS exceeded4 GiB')
        if current and '_clock' in current:current.update(status='completed',seconds=time.perf_counter()-current.pop('_clock'))
        remaining=240-(time.perf_counter()-start);require(remaining>0,'STOP_TOTAL_WALL','Numerical total exhausted')
        current=dict(name=name,status='running',_clock=time.perf_counter(),admission=admission());s['stages'].append(current);signal.setitimer(signal.ITIMER_REAL,min(90,remaining));save()
    save();print(json.dumps(dict(event='worker_started',pid=os.getpid(),own_sources=s['own_sources'])),flush=True)
    try:
        stage('material_source_background_spectrum_identity');tick=time.perf_counter()
        bankpath=ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt'
        bank=torch.load(bankpath,map_location='cpu',weights_only=True,mmap=True);rhs=[]
        s['source_bank_sha256']=digest(bankpath)
        for r in inputs['source_records']:
            sid=r['id'];q=bank['bank']['q'][sid:sid+1].to(torch.complex128)
            require(thash(q)==r['source128_sha256'],'STOP_SOURCE_IDENTITY',str(sid))
            require(torch.equal(q,q.to(torch.complex64).to(torch.complex128)),'STOP_SOURCE_PRECISION','Not exact complex64 promotion')
            rhs.append(q.clone());s['source_identity'].append(dict(id=sid,q_sha256=thash(q),expected_sha256=r['source128_sha256'],unpermuted=True,neural_field_used=False))
        del bank,q;gc.collect()
        t0,s0,meta=medium(N,1,10/3,'cpu');t1,s1,meta1=permuted_medium()
        require(thash(t0)=='9d9256eedd9242ad1d13f3853a9149b5d5e29a4fbfe6c63caf5a0e012f714b74' and thash(s0)=='5539f482895689656b2ee0dc05722c6428d3f08b8a3eb3c75f7050f2e8eaf356','STOP_BASE_MATERIAL_IDENTITY','Frozen E base construction changed')
        idx=(MULTIPLIER*torch.arange(N))%N;inverse=(INVERSE*torch.arange(N))%N
        require(torch.equal(torch.sort(idx).values,torch.arange(N)) and torch.equal(idx[inverse],torch.arange(N)),'STOP_PERMUTATION_IDENTITY','Not exact bijection/inverse')
        material=dict(original_total_sha256=thash(t0),original_scatter_sha256=thash(s0),total_sha256=thash(t1),scatter_sha256=thash(s1),parameters=meta1,multiplier=MULTIPLIER,inverse=INVERSE,identity_and_inverse={},background={},spectrum={})
        s['material_identity']=material;save()
        for name,old,new in [('total',t0,t1),('scatter',s0,s1)]:
            row=dict(identity_bitwise=torch.equal(old,old.index_select(-1,torch.arange(N))),inverse_bitwise=torch.equal(old,new.index_select(-1,inverse)),row_multiset_bitwise=torch.equal(torch.sort(old,dim=-1).values,torch.sort(new,dim=-1).values))
            material['identity_and_inverse'][name]=row
            require(all(row.values()),'STOP_PERMUTATION_IDENTITY',str(row))
        oldbg,ot,osb=background_detail(t0,s0);newbg,nt,ns=background_detail(t1,s1)
        br=dict(original=oldbg,permuted=newbg,total_max_absolute_difference=float((nt-ot).abs().max()),scatter_max_absolute_difference=float((ns-osb).abs().max()),total_unequal_elements=int((nt!=ot).sum()),scatter_unequal_elements=int((ns!=osb).sum()))
        material['background']=br;save()
        require(oldbg['axis']==newbg['axis']==0 and torch.equal(ot,nt) and torch.equal(osb,ns),'STOP_BACKGROUND_IDENTITY','Actual min/gather background changed; no forced old profile')
        mask=torch.ones(N,dtype=torch.bool);mask[[k%N for k in MODES]]=False
        for name,old,new,bg0,bg1 in [('total',t0,t1,ot,nt),('scatter',s0,s1,osb,ns)]:
            f0=torch.fft.fft(old.double()-bg0.double(),dim=-1,norm='ortho');f1=torch.fft.fft(new.double()-bg1.double(),dim=-1,norm='ortho')
            err=float((f1-f0.index_select(-1,inverse)).norm()/f0.norm())
            e0=f0.abs().square();e1=f1.abs().square();outside0=float(e0[...,mask].sum()/e0.sum());outside1=float(e1[...,mask].sum()/e1.sum())
            row=dict(modular_FFT_relative_difference=err,original_outside_P8_energy_fraction=outside0,permuted_outside_P8_energy_fraction=outside1,energy_ratio=float(e1.sum()/e0.sum()),spectrum_premise=outside1>outside0)
            material['spectrum'][name]=row;save()
            require(err<=1e-12,'STOP_SPECTRUM_IDENTITY',str(row))
            require(outside1>outside0,'STOP_SPECTRUM_PREMISE','Fixed permutation did not increase outside-P8 coefficient energy; no alternate multiplier')
        material['passed']=True;s['input_identity_seconds']=time.perf_counter()-tick
        del t0,s0,t1,s1,ot,osb,nt,ns,f0,f1,e0,e1;gc.collect();save()
        print(json.dumps(dict(event='permutation_identity_passed',seconds=s['input_identity_seconds'],material=material)),flush=True)
        stage('all17_original_A0_qualification')
        low,high,base,margin,common=construct(rhs[0]);del base
        bg,tb,sb=background(low);require(bg['axis']==0,'STOP_AXIS','Permutation selected nonzero normal axis');s['background_identity']=bg
        tick=time.perf_counter();factor=FullAngularModes(low,high);s['qualification'].update(factor_seconds=time.perf_counter()-tick,factor_bytes=factor.bytes,mode_setup={str(k):b.setup for k,b in factor.blocks.items()},common=common)
        qualify(factor,low,high,s['qualification'],save);s['qualification']['peak_RSS_bytes']=rss();save()
        print(json.dumps(dict(event='all17_qualification_passed',seconds=time.perf_counter()-start,factor_seconds=s['qualification']['factor_seconds'],factor_bytes=factor.bytes,background=bg,material=material)),flush=True)
        del factor,low,high,tb,sb;gc.collect();refs={}

        def run(method,cutoff=None):
            stage('batch_'+method);tick=time.perf_counter();rec=dict(method=method,status='running');s['batches'].append(rec)
            depth=16 if method=='tsa16_zero' else 64;low,high,base,margin,common=construct(rhs[0],depth=depth)
            require(base.kind=='tsa' and base.depth==depth and common['depth']==depth,'STOP_METHOD_BINDING','Requested/effective TSA depth mismatch for '+method)
            rec['requested_depth']=depth;rec['effective_depth']=base.depth
            rec['preconditioner_object']=dict(type=type(base).__module__+'.'+type(base).__name__,kind=base.kind,actual_depth=int(base.depth),reference_shape=list(base.reference.shape),cache_shape=list(base.reference._reference_transport_inverse.shape))
            pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128)
            require(thash(low.sigma_t)==material['total_sha256'] and thash(low.sigma_s)==material['scatter_sha256'],'STOP_BATCH_MATERIAL','Fresh construction changed')
            bgi,tb,sb=background(low);require(bgi==bg,'STOP_BATCH_BACKGROUND','Fresh background changed');rec.update(common=common,background=bgi)
            factor=None
            if method in ['compact4','E']:
                ti=time.perf_counter();factor=LayeredInverse(low,scheme='compact4',nyquist=True,dtype=torch.complex64) if method=='compact4' else FullAngularModes(low,high);rec['factor_seconds']=time.perf_counter()-ti;rec['factor_bytes']=factor.factor_bytes if method=='compact4' else factor.bytes
                if method=='E':
                    rec['mode_setup']={str(k):b.setup for k,b in factor.blocks.items()}
                    if cutoff is not None and time.perf_counter()-tick>=cutoff:raise Stop('NO_GO_E_SETUP_LOWER_BOUND','E factor setup already exceeds best complete control')
                    rec['qualification']={};qualify(factor,low,high,rec['qualification'],save)
            rec['setup_observed_seconds']=time.perf_counter()-tick
            if cutoff is not None and rec['setup_observed_seconds']>=cutoff:raise Stop('NO_GO_E_SETUP_LOWER_BOUND','E qualified setup already exceeds best complete control')
            local=[];counts=dict(outer_or_native_A=0,inner_A=0,final_A=0)
            def A(v):counts['outer_or_native_A']+=1;return high.A_op(v)
            def innerA(v):counts['inner_A']+=1;return project8(high.A_op(v))
            for j,(sid,b) in enumerate(zip(IDS,rhs)):
                inner=[];before=len(factor.profiles) if method=='E' else 0;ti=time.perf_counter()
                if method in ['tsa64_zero','tsa16_zero']:
                    u,st=solve_from_initial(A,b,pre,None,tolerance=1e-5,maxiter=100);st['classical_calls']=st['iterations'];st['injection_calls']=0
                else:
                    if method=='compact4':inject=lambda x,r:factor(r).to(torch.complex128)
                    else:
                        def inject(x,r):
                            it=time.perf_counter();z,ist=solve_from_initial(innerA,project8(r),lambda v:project8(factor.apply(project8(v))),None,tolerance=1e-4,maxiter=8);inner.append(dict(**ist,seconds=time.perf_counter()-it));return z
                    u,st=augmented_fgmres(A,b,pre,None,inject=inject,inject_after=1,tolerance=1e-5,maxiter=100)
                solve=time.perf_counter()-ti;ti=time.perf_counter();counts['final_A']+=1;bn=float(b.norm());res=float((b-high.A_op(u)).norm()/bn);delta=res*bn/margin;bound=delta/(float(u.norm())-delta)
                if method=='tsa64_zero':refs[sid]=u.clone()
                diff=float((u-refs[sid]).norm()/refs[sid].norm());accepted=bool(torch.isfinite(u).all()) and not st['failed'] and res<=1e-5 and 0<=bound<=1e-3 and diff<=1e-4;check=time.perf_counter()-ti
                if j==0:rec['first_RHS_observed_seconds']=time.perf_counter()-tick
                st.pop('trace',None);row=dict(method=method,source_id=sid,solve_seconds=solve,check_seconds=check,status=st,inner=inner,true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff,accepted=accepted,ordinary_reference_self_comparison=(method=='tsa64_zero'))
                if method=='E':row['inverse_applications']=len(factor.profiles)-before
                s['rows'].append(row);local.append(row)
                require(rss()<=4*1024**3,'STOP_RSS_HIGH_WATER','Recorded worker RSS exceeded4 GiB')
                require(accepted and all(not x['failed'] and x['true_residual']<=1e-4 for x in inner),'STOP_ORIGINAL_ACCEPTANCE',str(row))
                save();print(json.dumps(dict(event='source_complete',method=method,source_id=sid,seconds=solve+check,calls=st['classical_calls'],inner=[x['iterations'] for x in inner],residual=res)),flush=True)
            rec.update(status='completed',complete_four_RHS_observed_seconds=time.perf_counter()-tick,online_phase_seconds=sum(r['solve_seconds']+r['check_seconds'] for r in local),inner_phase_seconds=sum(x['seconds'] for r in local for x in r['inner']),calls=[r['status']['classical_calls'] for r in local],inner_calls=[x['iterations'] for r in local for x in r['inner']],A_call_counts=counts,peak_process_RSS_bytes=rss())
            if method=='E':rec['runtime_inverse_profiles']=factor.profiles
            save();print(json.dumps({k:rec[k] for k in ['method','setup_observed_seconds','first_RHS_observed_seconds','complete_four_RHS_observed_seconds','calls','inner_calls','peak_process_RSS_bytes']}),flush=True)
            del A,innerA,pre,factor,low,high,base,u,tb,sb
            if method in ['compact4','E']:del inject
            gc.collect();return rec
        controls=[run(m) for m in ['tsa64_zero','compact4','tsa16_zero']]
        best=min(controls,key=lambda r:r['complete_four_RHS_observed_seconds']);s['strongest_control']=best['method'];s['strongest_control_seconds']=best['complete_four_RHS_observed_seconds'];candidate=run('E',s['strongest_control_seconds'])
        s.update(complete=True,result='PERMUTATION_PRELIMINARY_FORWARD_PASS' if candidate['complete_four_RHS_observed_seconds']<s['strongest_control_seconds'] else 'NO_GO_PERMUTATION_COMPLETE_COST')
    except Stop as exc:s.update(complete=True,result=exc.code,stop_reason=str(exc))
    except BaseException as exc:s['error']=repr(exc);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if rss()>4*1024**3:s.update(complete=True,result='STOP_RSS_HIGH_WATER',stop_reason='Recorded worker peak exceeded4 GiB')
        if current and '_clock' in current:current.update(status='completed' if s.get('result')=='PERMUTATION_PRELIMINARY_FORWARD_PASS' else 'stopped',seconds=time.perf_counter()-current.pop('_clock'))
        for rec in s['batches']:
            if rec['status']=='running':rec.update(status='stopped',reason=s.get('stop_reason',s.get('error')))
        s.update(wall_seconds=time.perf_counter()-start,peak_process_RSS_bytes=rss(),source_hashes_after={p:digest(ROOT/p) for p in sources},E_source_hashes_after={p:digest(E/p) for p in es['source_hashes']});save();print(json.dumps(dict(event='worker_finished',pid=s['pid'],result=s.get('result'),error=s.get('error'),wall_seconds=s['wall_seconds'],peak_RSS=s['peak_process_RSS_bytes'])),flush=True)

if __name__=='__main__':main()
