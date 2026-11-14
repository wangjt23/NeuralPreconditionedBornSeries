"""STATIC PREPARATION ONLY until explicit coordinator admission and GPU release."""
import os,sys,time,json,hashlib,signal,subprocess,gc,resource
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
F=ROOT/'rte/workstation_runs/rte_batched_cpu_20261110_dot_v1'
E=ROOT/'rte/workstation_runs/rte_angular_full_20261110_dot_v1'
sys.path[:0]=[str(F),str(ROOT),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/direct_capture_20261002/confirmation/shard0/code')]
from batched_angular import BatchedAngularModes,MODES
from screen_structured import medium
from layered_inverse import LayeredInverse,CapturedAction,background_from_material
from direct_segmented_tsa import DirectSegmentedTSA
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
from rte.train_source_predictor import build_base
from rte.source_initialization import solve_from_initial
from rte.augmented_fgmres import augmented_fgmres

FROZEN_BATCHED_HASH='b9aed56b4962ba1620bac5ff9693f7d63ce20e1909f8c5083b4065973b84bec7'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sync():torch.cuda.synchronize()
def guard():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    other=[int(x.strip()) for x in output.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
    if other:raise RuntimeError('GPU still owned by other processes: '+str(other))
class SetupStop(RuntimeError):pass

def project(r):
    f=torch.fft.fft(r,dim=-1,norm='ortho');out=torch.zeros_like(f)
    indices=torch.tensor([k%128 for k in MODES],device=r.device)
    out.index_copy_(-1,indices,f.index_select(-1,indices))
    return torch.fft.ifft(out,dim=-1,norm='ortho')

def neutral_warmup():
    start=time.perf_counter();torch.cuda.init();sync();context=time.perf_counter()-start;t=time.perf_counter()
    for dtype in [torch.complex64,torch.complex128]:
        a=2*torch.eye(12,dtype=dtype,device='cuda').expand(128,128,12,12).clone()
        inv=torch.linalg.inv(a);inv@a
        field=torch.ones(1,12,128,128,dtype=dtype,device='cuda')
        torch.fft.ifft2(torch.fft.fft2(field));torch.fft.ifft(torch.fft.fft(field,dim=-1),dim=-1)
    for n in [128,256]:
        a=2*torch.eye(n,dtype=torch.complex128,device='cuda').expand(17,n,n).clone()
        lu,piv=torch.linalg.lu_factor(a);b=torch.ones(17,n,2,dtype=a.dtype,device='cuda')
        torch.linalg.lu_solve(lu,piv,b);a@b
    x=torch.ones(17,12,128,1,dtype=torch.complex128,device='cuda')
    torch.fft.ifft(torch.fft.fft(x,dim=1),dim=1);torch.fft.ifft(torch.fft.fft(x,dim=2),dim=2)
    sync();elapsed=time.perf_counter()-t
    return dict(context_seconds=context,neutral_library_seconds=elapsed,convention='Device warm for both methods. Common neutral warm-up excluded equally; no measured material/RHS or retained factors.')

def move_operator(module):
    """Preserve all stored CPU values, including unregistered setup tensors."""
    assert module._reference_transport_inverse is None and not module._reference_moment_inverse_cache
    plain={k:v for k,v in vars(module).items() if torch.is_tensor(v)}
    buffers=dict(module.named_buffers());byte_count=sum(v.numel()*v.element_size() for v in list(plain.values())+list(buffers.values()))
    core=['directions','weights','phase_matrix','kx','ky','sigma_t','sigma_s']
    expected={k:getattr(module,k) for k in core}
    module.to('cuda')
    for k,v in plain.items():setattr(module,k,v.cuda())
    module.device_name='cuda'
    return byte_count,expected

def construct(raw):
    sync();start=time.perf_counter();total,scatter,_=medium(128,1,10/3,'cpu')
    low=PeriodicRTE2D(shape=(128,128),dx=1/128,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu')
    low.setup(total,scatter,raw[0]);high=high_precision_copy(low)
    phase=high.phase_matrix*high.weights[None];mu=torch.linalg.eigvalsh((phase+phase.T)/2).max();margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0
    coefficient_hashes={name:hashlib.sha256(getattr(high,name).numpy().tobytes()).hexdigest() for name in ['directions','weights','phase_matrix','kx','ky','sigma_t','sigma_s']}
    host=time.perf_counter()-start;t=time.perf_counter()
    low_bytes,low_expected=move_operator(low);high_bytes,high_expected=move_operator(high)
    rhs=[r.cuda().to(torch.complex128) for r in raw];sync();copy=time.perf_counter()-t;t=time.perf_counter();verify_bytes=0
    for op,expected in [(low,low_expected),(high,high_expected)]:
        for name,value in expected.items():
            actual=getattr(op,name).cpu();verify_bytes+=actual.numel()*actual.element_size();assert torch.equal(actual,value)
    sync();verify=time.perf_counter()-t;t=time.perf_counter()
    base=build_base(low,'tsa64');tsa=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False);sync();tsa_seconds=time.perf_counter()-t
    pre=lambda r:tsa(r.to(torch.complex64)).to(torch.complex128)
    return rhs,low,high,base,tsa,pre,margin,dict(host_physical_seconds=host,H2D_seconds=copy,H2D_bytes=low_bytes+high_bytes+sum(r.numel()*r.element_size() for r in raw),exact_value_check_D2H_seconds=verify,exact_value_check_D2H_bytes=verify_bytes,TSA_seconds=tsa_seconds,coefficient_hashes=coefficient_hashes)

def qualify(factor,low,high,record):
    # Timer begins before CPU vector generation and transfer; no uncharged preparation.
    sync();start=time.perf_counter();gen=torch.Generator().manual_seed(2026111004)
    cpu=torch.stack([torch.complex(torch.randn(1536,1,dtype=torch.float64,generator=gen),torch.randn(1536,1,dtype=torch.float64,generator=gen)) for k in MODES])
    old=json.loads((E/'full_01/mode_qualifications.json').read_text());hashes={r['k']:r['RHS_sha256'] for b in old for r in b['records']}
    for j,k in enumerate(MODES):assert hashlib.sha256(cpu[j].numpy().tobytes()).hexdigest()==hashes[k]
    q=cpu.cuda();z,st=factor.repaired(q);records=[]
    record.update(vector_H2D_bytes=cpu.numel()*cpu.element_size(),records=records,repair_status=st)
    assert st['success']
    axis,t,s,_=background_from_material(low.sigma_t,low.sigma_s);assert axis==0;phase=(high.phase_matrix*high.weights[None]).to(torch.complex128)
    for j,k in enumerate(MODES):
        f=torch.zeros_like(high.source);f[...,k]=z[j].reshape(1,12,128);field=torch.fft.ifft(f,dim=-1,norm='ortho')
        actual=high.A_op(field)-(high.sigma_t-t.double())*field+(high.sigma_s-s.double())*torch.einsum('lm,bmxy->blxy',phase,field)
        f.zero_();f[...,k]=q[j].reshape(1,12,128);target=torch.fft.ifft(f,dim=-1,norm='ortho');res=float((target-actual).norm()/target.norm())
        records.append(dict(k=k,original_A0_residual=res));assert res<=1e-10
    # GPU checks of the CPU-qualified strict-zero/tiny behavior, charged to setup.
    zero=torch.zeros_like(q);mix=torch.cat((zero,q,1e-200*q,zero),dim=2);mix[[0,8,16]]=0
    zx,zs=factor.repaired(torch.cat((zero,zero),dim=2));record['zero_status']=zs;assert zs['success'] and bool((zx==0).all())
    mx,ms=factor.repaired(mix);record['mixed_status']=ms;assert ms['success'];active=(mix!=0).any(dim=1)
    assert bool((mx.masked_select(~active[:,None])==0).all()) and bool(((mx!=0).any(dim=1)[active]).all())
    sync();record['seconds']=time.perf_counter()-start

@torch.no_grad()
def main():
    # Must remain before any CUDA call. Coordinator permission is checked by the worker, not inferred from this flag.
    if os.environ.get('RTE_GPU_ADMISSION')!='coordinator_released':raise RuntimeError('No explicit GPU admission; CUDA not initialized')
    assert set(os.sched_getaffinity(0))=={0};torch.set_num_threads(1);torch.set_num_interop_threads(1);guard()
    assert digest(F/'batched_angular.py')==FROZEN_BATCHED_HASH
    assert all(json.loads((F/stage/'results.json').read_text())['passed'] for stage in ['qualification_01','full_01'])
    out=HERE/'gpu_gate_01';out.mkdir(exist_ok=False);start=time.perf_counter()
    s=dict(complete=False,pid=os.getpid(),affinity=[0],device='cuda',wall_limit=180,total_budget_limit=600,shared_host='GB CPU32 background audit; Schur numerical work released. GPU competitor checks at each measured batch entry and exit.',source_ids=[0,4,7,14],grid=128,L=12,g=.65,amplitude=3,prefix=1,tangent_cut=8,inner_tolerance=1e-4,inner_maxiter=8,block_target=1e-10,block_max_corrections=3,outer_tolerance=1e-5,outer_maxiter=100,relative_bound_limit=1e-3,field_difference_limit=1e-4,source_sha256=digest(__file__),batched_source_sha256=FROZEN_BATCHED_HASH,orders=[['compact4','batched17'],['batched17','compact4']],rows=[],batches=[],references=[],scope='Synchronized device-warm cost gate only. Internal factor timers are host dispatch only.',admission_processes=subprocess.check_output(['ps','-eo','pid,psr,pcpu,args','--sort=-pcpu'],text=True).splitlines()[:16])
    def save():
        temp=out/'results.json.tmp';temp.write_text(json.dumps(s,indent=2)+'\n');os.replace(temp,out/'results.json')
    def expired(signum,frame):raise TimeoutError('Own bounded gate interrupted: '+str(signum))
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired);signal.setitimer(signal.ITIMER_REAL,180);save()
    try:
        old=json.loads((E/'full_01/settings.json').read_text());s['production_hashes']={p:digest(ROOT/p) for p in old['production_hashes']};assert s['production_hashes']==old['production_hashes']
        s['dependencies']={p:digest(ROOT/p) for p in old['dependency_hashes']};assert s['dependencies']==old['dependency_hashes']
        bank=torch.load(ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True);raw=[bank['bank']['q'][i:i+1].clone() for i in s['source_ids']];del bank;gc.collect()
        s['rhs_sha256']={str(i):hashlib.sha256(r.numpy().tobytes()).hexdigest() for i,r in zip(s['source_ids'],raw)};assert s['rhs_sha256']==old['rhs_sha256']
        s['warmup']=neutral_warmup();s['GPU']=torch.cuda.get_device_name();sync();t=time.perf_counter()
        rhs,low,high,base,tsa,pre,margin,common=construct(raw);frozen_coefficients=common['coefficient_hashes'];references={}
        for sid,b in zip(s['source_ids'],rhs):
            u,st=solve_from_initial(high.A_op,b,pre,None,tolerance=1e-5,maxiter=100);assert not st['failed'];references[sid]=u;s['references'].append(dict(source_id=sid,status=st))
        sync();s['reference_research_seconds']=time.perf_counter()-t
        del rhs,low,high,base,tsa,pre;gc.collect();save()

        def run(method,rep,limit=None):
            guard();torch.cuda.reset_peak_memory_stats();sync();began=time.perf_counter();record=dict(method=method,repeat=rep,status='running',GPU_allocated_at_entry=torch.cuda.memory_allocated(),GPU_reserved_at_entry=torch.cuda.memory_reserved());s['batches'].append(record)
            def checkpoint(label):
                sync();cost=time.perf_counter()-began;record['last_setup_lower_bound']=dict(phase=label,seconds=cost,compact_complete_limit=limit)
                if method=='batched17' and limit is not None and cost>=limit:raise SetupStop('Candidate setup lower bound already exceeds compact4 complete cost')
            rhs,low,high,base,tsa,pre,margin,common=construct(raw);record['base_setup']=common;assert common['coefficient_hashes']==frozen_coefficients;checkpoint('base')
            sync();t=time.perf_counter()
            if method=='compact4':factor=LayeredInverse(low,scheme='compact4',nyquist=True,dtype=torch.complex64)
            else:factor=BatchedAngularModes(low,high)
            sync();record['factor_synchronized_seconds']=time.perf_counter()-t
            if method=='batched17':record['internal_factor_host_dispatch_only']=factor.setup
            record['logical_factor_bytes']=factor.factor_bytes if method=='compact4' else factor.bytes;checkpoint('factor')
            sync();t=time.perf_counter()
            if method=='compact4':
                action=CapturedAction(lambda r:factor(r).to(torch.complex128),rhs[0]);assert action.relative_capture_difference<1e-12
                record['capture_relative_difference']=action.relative_capture_difference
            else:
                record['original_mode_qualification']={};qualify(factor,low,high,record['original_mode_qualification']);action=factor.apply
            sync();record['capture_or_qualification_synchronized_seconds']=time.perf_counter()-t;checkpoint('qualified')
            record['setup_observed_seconds']=time.perf_counter()-began;local=[]
            if method=='batched17':record['repair_profiles']=factor.profiles
            for sid,b in zip(s['source_ids'],rhs):
                inner=[];bn=float(b.norm());sync();t=time.perf_counter()
                if method=='compact4':inject=lambda x,r:action(r)
                else:
                    def inject(x,r):
                        ti=time.perf_counter();z,st=solve_from_initial(lambda v:project(high.A_op(v)),project(r),lambda v:project(action(project(v))),None,tolerance=1e-4,maxiter=8);sync();inner.append(dict(**st,seconds=time.perf_counter()-ti));return z
                u,st=augmented_fgmres(high.A_op,b,pre,None,inject=inject,inject_after=1,tolerance=1e-5,maxiter=100);sync();solve=time.perf_counter()-t;t=time.perf_counter()
                res=float((b-high.A_op(u)).norm()/bn);delta=res*bn/margin;bound=delta/(float(u.norm())-delta);diff=float((u-references[sid]).norm()/references[sid].norm());ok=bool(torch.isfinite(u).all()) and not st['failed'] and res<=1e-5 and 0<=bound<=1e-3 and diff<=1e-4
                sync();check=time.perf_counter()-t
                if sid==s['source_ids'][0]:record['first_RHS_observed_seconds']=time.perf_counter()-began
                st.pop('trace',None);row=dict(method=method,repeat=rep,source_id=sid,solve_seconds=solve,check_seconds=check,status=st,inner=inner,accepted=ok,true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff);s['rows'].append(row);local.append(row)
                assert ok and all(not z['failed'] and z['true_residual']<=1e-4 for z in inner)
            sync();record.update(status='completed',complete_four_RHS_observed_seconds=time.perf_counter()-began,online_phase_seconds=sum(r['solve_seconds']+r['check_seconds'] for r in local),calls=[r['status']['classical_calls'] for r in local],inner_calls=[i['iterations'] for r in local for i in r['inner']],peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
            guard();save();print(json.dumps({k:record[k] for k in ['method','repeat','setup_observed_seconds','first_RHS_observed_seconds','complete_four_RHS_observed_seconds','calls']}),flush=True)
            del inject,action,factor,pre,tsa,base,low,high,rhs,u;gc.collect();return record

        c0=run('compact4',0);a0=run('batched17',0,c0['complete_four_RHS_observed_seconds'])
        if a0['complete_four_RHS_observed_seconds']>=c0['complete_four_RHS_observed_seconds']:
            s.update(complete=True,result='NO_GO_FIRST_COMPLETE_PAIR');return
        a1=run('batched17',1,c0['complete_four_RHS_observed_seconds']);c1=run('compact4',1)
        s.update(complete=True,result='TWO_ORDER_COST_GATE_PASS' if a1['complete_four_RHS_observed_seconds']<c1['complete_four_RHS_observed_seconds'] else 'UNRESOLVED_ORDER_DEPENDENCE')
    except SetupStop as exc:
        s.update(complete=True,result='NO_GO_SETUP_LOWER_BOUND',stop_reason=str(exc))
        for r in s['batches']:
            if r['status']=='running':r['status']='stopped_setup';r['setup_observed_seconds']=r['last_setup_lower_bound']['seconds']
    except BaseException as exc:s['error']=repr(exc);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if torch.cuda.is_initialized():sync()
        for r in s['batches']:
            if r['status']=='running':r['status']='failed';r['error']=s.get('error','Unfinished gate')
        s.update(wall_seconds=time.perf_counter()-start,peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if 'production_hashes' in s:s['production_hashes_after']={p:digest(ROOT/p) for p in s['production_hashes']}
        save()
if __name__=='__main__':main()
