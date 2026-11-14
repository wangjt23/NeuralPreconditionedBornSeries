import os,sys,time,json,hashlib,signal,resource,subprocess
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
E=ROOT/'rte/workstation_runs/rte_angular_full_20261110_dot_v1'
sys.path.extend([str(ROOT),str(E),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1')])
from run_full_cpu import construct,project8
from angular_modes import FullAngularModes
from layered_inverse import background_from_material
from batched_angular import BatchedAngularModes,MODES

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def peak_rss_bytes():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
def current_rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024
def stable_relative(error,reference):
    """[mode,unknown,column] -> independent [mode,column], no nonzero floor."""
    active=(reference!=0).any(dim=1);scale=reference.abs().amax(dim=1)
    safe=torch.where(active,scale,torch.ones_like(scale));den=(reference/safe[:,None]).norm(dim=1)
    result=(error/safe[:,None]).norm(dim=1)/torch.where(active,den,torch.ones_like(den))
    zero_error=(error!=0).any(dim=1)
    return torch.where(active,result,torch.where(zero_error,torch.full_like(result,float('inf')),torch.zeros_like(result)))

def locked_vectors():
    gen=torch.Generator().manual_seed(2026111004)
    q=torch.stack([torch.complex(torch.randn(1536,1,dtype=torch.float64,generator=gen),torch.randn(1536,1,dtype=torch.float64,generator=gen)) for k in MODES])
    old=json.loads((E/'full_01/mode_qualifications.json').read_text())
    expected={r['k']:r['RHS_sha256'] for row in old if row['records'] for r in row['records']}
    for j,k in enumerate(MODES):assert hashlib.sha256(q[j].numpy().tobytes()).hexdigest()==expected[k]
    return q

def independent_A0_residual(high,low,k,z,target):
    axis,t,s,_=background_from_material(low.sigma_t,low.sigma_s);assert axis==0
    phase=(high.phase_matrix*high.weights[None]).to(torch.complex128)
    cols=z.shape[1];f=torch.zeros((cols,12,128,128),dtype=torch.complex128)
    f[...,k]=z.T.reshape(cols,12,128);field=torch.fft.ifft(f,dim=-1,norm='ortho')
    actual=high.A_op(field)-(high.sigma_t-t.double())*field+(high.sigma_s-s.double())*torch.einsum('lm,bmxy->blxy',phase,field)
    f.zero_();f[...,k]=target.T.reshape(cols,12,128);rhs=torch.fft.ifft(f,dim=-1,norm='ortho')
    return stable_relative((rhs-actual).reshape(cols,-1).T[None],rhs.reshape(cols,-1).T[None])[0]

class Receipt:
    def __init__(self,stage):
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert set(os.sched_getaffinity(0))=={0} and os.environ.get('CUDA_VISIBLE_DEVICES')==''
        self.started=time.perf_counter();self.out=HERE/stage;self.out.mkdir(exist_ok=False)
        old=json.loads((E/'full_01/settings.json').read_text())
        for name,h in old['source_hashes'].items():assert digest(E/name)==h
        for name,h in {**old['production_hashes'],**old['dependency_hashes']}.items():assert digest(ROOT/name)==h
        previous=[json.loads(p.read_text()) for p in HERE.glob('*_01/results.json') if p.parent!=self.out]
        assert all(x['complete'] and x.get('passed') for x in previous)
        previous_wall=sum(x['wall_seconds'] for x in previous);assert previous_wall<600
        self.data=dict(complete=False,passed=False,stage=stage,pid=os.getpid(),affinity=[0],device='cpu',threads=1,stage_wall_limit=180,total_numerical_wall_limit=600,previous_numerical_wall_seconds=previous_wall,shared_host='CPU0 only; GB GPU/CPU32,34 may remain active; not isolated performance.',source_ids=[0,4,7,14],grid=128,L=12,g=.65,inclusion_amplitude=3,inner_tolerance=1e-4,inner_maxiter=8,outer_tolerance=1e-5,outer_maxiter=100,block_target=1e-10,block_max_corrections=3,action_bridge_target=1e-12,inverse_difference_target=1e-10,E_sources=old['source_hashes'],production_hashes=old['production_hashes'],dependency_hashes=old['dependency_hashes'],own_sources={p.name:digest(p) for p in HERE.glob('*.py')},admission_processes=subprocess.check_output(['ps','-eo','pid,psr,pcpu,args','--sort=-pcpu'],text=True).splitlines()[:15],peak_RSS_start_bytes=peak_rss_bytes(),records=[])
        def expired(signum,frame):raise TimeoutError('Owned stage stopped by signal '+str(signum))
        signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired)
        signal.setitimer(signal.ITIMER_REAL,min(180,600-previous_wall))
        self.save()
    def save(self):
        self.data.update(wall_seconds=time.perf_counter()-self.started,process_peak_RSS_bytes=peak_rss_bytes(),current_RSS_bytes=current_rss_bytes())
        target=self.out/'results.json';temp=self.out/'results.json.tmp';temp.write_text(json.dumps(self.data,indent=2)+'\n');os.replace(temp,target)
    def finish(self,error=None):
        signal.setitimer(signal.ITIMER_REAL,0)
        if error is not None:self.data['error']=repr(error)
        self.data['complete']=True
        self.data['production_hashes_after']={p:digest(ROOT/p) for p in self.data['production_hashes']}
        if self.data['production_hashes_after']!=self.data['production_hashes']:self.data['passed']=False;self.data['error']='Production source changed during qualification'
        self.save()
        print(json.dumps({k:self.data[k] for k in ['stage','pid','complete','passed','wall_seconds','process_peak_RSS_bytes']},indent=2),flush=True)

def qualify_all_modes(factor,low,high,record=None):
    record={} if record is None else record
    q=locked_vectors();start=time.perf_counter();z,st=factor.repaired(q);records=[]
    record.update(records=records,repair_status=st)
    if not st['success']:raise RuntimeError('Batched qualification repair failed: '+json.dumps(st))
    for j,k in enumerate(MODES):
        r=independent_A0_residual(high,low,k,z[j],q[j]);records.append(dict(k=k,original_A0_residual=r.tolist()))
        if not bool((r<=1e-10).all()):raise RuntimeError('Original A0 validation failed: '+json.dumps(records[-1]))
    record['seconds']=time.perf_counter()-start
    return record
