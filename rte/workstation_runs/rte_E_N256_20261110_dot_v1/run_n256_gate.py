"""Frozen E pure-scale N256 qualification and one four-method CPU gate."""
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
N=256;MODES=list(range(-8,9));IDS=[0,4,7,14]
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
def construct(b,depth=64):
    tick=time.perf_counter();t,s,info=medium(N,1,10/3,'cpu')
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
    es=json.loads((E/'full_01/settings.json').read_text());inputs=json.loads((HERE/'SAVED_INPUT_AUDIT.json').read_text())
    assert digest(HERE/'FROZEN_PLAN.md')=='3be997b61d1f65d10ca7c254ff7f2b3d752cfc7c5694513a5e7244f22b75ea3f'
    for f,h in es['source_hashes'].items():assert digest(E/f)==h
    sources={**es['production_hashes'],**es['dependency_hashes']}
    for f in ['rte/transport_baselines.py','rte/benchmark_tail_solver.py','rte/run_large_angular_spectral_experiments.py']:sources[f]=inputs['existing_solver_sources'][f]
    for f,h in sources.items():assert digest(ROOT/f)==h
    out=HERE/'gate_01';out.mkdir(exist_ok=False);start=time.perf_counter();current=None
    s=dict(complete=False,pid=os.getpid(),affinity=[0],threads=1,device='cpu',stage_limit_seconds=180,total_limit_seconds=600,RSS_limit_bytes=6*1024**3,grid=N,L=12,g=.65,amplitude=3,source_ids=IDS,modes=MODES,prefix=1,inner_tolerance=1e-4,inner_maxiter=8,outer_tolerance=1e-5,outer_maxiter=100,relative_bound_limit=1e-3,field_difference_limit=1e-4,block_target=1e-10,block_max_corrections=3,heldout9626=False,training=False,new_teachers=False,orders=['tsa64_zero','compact4','tsa16_zero','E'],E_source_hashes=es['source_hashes'],source_hashes=sources,own_sources={p.name:digest(p) for p in HERE.glob('*.py')},plan_sha256=digest(HERE/'FROZEN_PLAN.md'),input_audit_sha256=digest(HERE/'SAVED_INPUT_AUDIT.json'),scope='Pure N128-to-N256 scale extension of the same fixed constructed physical scene and four exactly nested saved sources. One forward gate, no general-material or repeated-timing claim.',stages=[],batches=[],rows=[],qualification={},source_identity=[])
    def save():
        target=out/'results.json';temp=out/'results.json.tmp';temp.write_text(json.dumps(s,indent=2)+'\n');os.replace(temp,target)
    def expired(signum,frame):raise Stop('STOP_STAGE_WALL_OR_OWNED_TERMINATION','Signal '+str(signum))
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired)
    def stage(name):
        nonlocal current
        require(rss()<=6*1024**3,'STOP_RSS_HIGH_WATER','Recorded worker RSS exceeded6 GiB')
        if current and '_clock' in current:current.update(status='completed',seconds=time.perf_counter()-current.pop('_clock'))
        remaining=600-(time.perf_counter()-start);require(remaining>0,'STOP_TOTAL_WALL','Numerical total exhausted')
        current=dict(name=name,status='running',_clock=time.perf_counter(),admission=admission());s['stages'].append(current);signal.setitimer(signal.ITIMER_REAL,min(180,remaining));save()
    save();print(json.dumps(dict(event='worker_started',pid=os.getpid(),own_sources=s['own_sources'])),flush=True)
    try:
        stage('identity_and_all17_qualification');tick=time.perf_counter()
        parents={n:torch.load(ROOT/path,map_location='cpu',weights_only=True,mmap=True) for n,path in inputs['development_parent_paths'].items()}
        for n,path in inputs['development_parent_paths'].items():require(digest(ROOT/path)==inputs['development_parent_sha256'][n],'STOP_PARENT_IDENTITY',path)
        require(all(torch.equal(parents['128'][k],parents['256'][k][...,::2,::2]) for k in ['sigma_t','sigma_s']),'STOP_PARENT_NESTING','Stored parent material nesting')
        require(all(torch.equal(parents['128'][k],parents['256'][k]) for k in inputs['source_parameter_arrays_equal']),'STOP_PARAMETER_IDENTITY','Source parameters differ')
        bank=torch.load(ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True,mmap=True);rhs=[]
        for r in inputs['source_records']:
            sid=r['id'];path=ROOT/inputs['source_record_root']/r['file256'];require(digest(path)==r['file256_sha256'],'STOP_SOURCE_FILE_IDENTITY',str(path));stored=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            require(stored['grid']==N and stored['index']==sid and stored['data_sha256']==inputs['development_parent_sha256']['256'],'STOP_SOURCE_PROVENANCE',str(sid))
            q=stored['source'];q128=bank['bank']['q'][sid:sid+1].to(torch.complex128)
            require(thash(q)==r['source256_sha256'] and thash(q128)==r['source128_sha256'] and torch.equal(q[...,::2,::2],q128),'STOP_SOURCE_NESTING',str(sid))
            require(torch.equal(q,q.to(torch.complex64).to(torch.complex128)),'STOP_SOURCE_PRECISION','Saved q not an exact complex64 promotion')
            rhs.append(q.clone());s['source_identity'].append(dict(id=sid,file_sha256=r['file256_sha256'],q256_sha256=thash(q),q128_sha256=thash(q128),exact_nested=True,neural_field_used=False));del stored,q,q128
        del bank,parents;gc.collect()
        t128,s128,meta128=medium(128,1,10/3,'cpu');t256,s256,meta256=medium(N,1,10/3,'cpu')
        require(thash(t128)=='9d9256eedd9242ad1d13f3853a9149b5d5e29a4fbfe6c63caf5a0e012f714b74' and thash(s128)=='5539f482895689656b2ee0dc05722c6428d3f08b8a3eb3c75f7050f2e8eaf356','STOP_N128_MATERIAL_IDENTITY','Frozen E construction changed')
        require(torch.equal(t128,t256[...,::2,::2]) and torch.equal(s128,s256[...,::2,::2]),'STOP_CONSTRUCTED_MATERIAL_NESTING','Exact even-grid nesting failed')
        material=dict(total256_sha256=thash(t256),scatter256_sha256=thash(s256),total128_sha256=thash(t128),scatter128_sha256=thash(s128),exact_nested=True,parameters=meta256);s['material_identity']=material;s['input_identity_seconds']=time.perf_counter()-tick
        del t128,s128,t256,s256;gc.collect()
        low,high,base,margin,common=construct(rhs[0]);del base
        bg,tb,sb=background(low);require(bg['axis']==0,'STOP_AXIS','N256 selected nonzero normal axis');s['background_identity']=bg
        tick=time.perf_counter();factor=FullAngularModes(low,high);s['qualification'].update(factor_seconds=time.perf_counter()-tick,factor_bytes=factor.bytes,mode_setup={str(k):b.setup for k,b in factor.blocks.items()},common=common)
        qualify(factor,low,high,s['qualification'],save);s['qualification']['peak_RSS_bytes']=rss();save()
        print(json.dumps(dict(event='N256_qualification_passed',seconds=time.perf_counter()-start,factor_seconds=s['qualification']['factor_seconds'],factor_bytes=factor.bytes,background=bg,material=material)),flush=True)
        del factor,low,high,tb,sb;gc.collect();refs={}

        def run(method,cutoff=None):
            stage('batch_'+method);tick=time.perf_counter();rec=dict(method=method,status='running');s['batches'].append(rec)
            depth=16 if method=='tsa16_zero' else 64;low,high,base,margin,common=construct(rhs[0]);pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128)
            require(thash(low.sigma_t)==material['total256_sha256'] and thash(low.sigma_s)==material['scatter256_sha256'],'STOP_BATCH_MATERIAL','Fresh construction changed')
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
                require(rss()<=6*1024**3,'STOP_RSS_HIGH_WATER','Recorded worker RSS exceeded6 GiB')
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
        s.update(complete=True,result='N256_PRELIMINARY_FORWARD_PASS' if candidate['complete_four_RHS_observed_seconds']<s['strongest_control_seconds'] else 'NO_GO_N256_COMPLETE_COST')
    except Stop as exc:s.update(complete=True,result=exc.code,stop_reason=str(exc))
    except BaseException as exc:s['error']=repr(exc);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if rss()>6*1024**3:s.update(complete=True,result='STOP_RSS_HIGH_WATER',stop_reason='Recorded worker peak exceeded6 GiB')
        if current and '_clock' in current:current.update(status='completed' if s.get('result')=='N256_PRELIMINARY_FORWARD_PASS' else 'stopped',seconds=time.perf_counter()-current.pop('_clock'))
        for rec in s['batches']:
            if rec['status']=='running':rec.update(status='stopped',reason=s.get('stop_reason',s.get('error')))
        s.update(wall_seconds=time.perf_counter()-start,peak_process_RSS_bytes=rss(),source_hashes_after={p:digest(ROOT/p) for p in sources},E_source_hashes_after={p:digest(E/p) for p in es['source_hashes']});save();print(json.dumps(dict(event='worker_finished',pid=s['pid'],result=s.get('result'),error=s.get('error'),wall_seconds=s['wall_seconds'],peak_RSS=s['peak_process_RSS_bytes'])),flush=True)

if __name__=='__main__':main()
