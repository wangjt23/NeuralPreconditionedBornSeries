import time,os,json,hashlib,gc
from pathlib import Path
import nested_factor as nf
import packed_factor as pf
nb=nf.nb;p=nb.p;torch=nb.torch;old=nb.old;DT=nb.DT
HERE=Path(__file__).resolve().parent

def snapshot():
 info=dict(loadavg=Path('/proc/loadavg').read_text().strip(),uptime=Path('/proc/uptime').read_text().strip())
 info['memory']=[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','MemFree:','SwapFree:'))]
 info['cpu0']=next(x for x in Path('/proc/stat').read_text().splitlines() if x.startswith('cpu0 '));return info

def rss():return int(next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024

def configure():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert set(os.sched_getaffinity(0))=={0} and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
 return dict(torch_num_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),torch_parallel_info=torch.__config__.parallel_info(),affinity=list(os.sched_getaffinity(0)),environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']})

def seal():
 prior=json.loads((HERE/'PRIOR_K_RESULTS.json').read_text());prev=json.loads((HERE/'PRIOR_NESTED_RESULTS.json').read_text())
 for f,h in prior['source_hashes'].items():assert p.digest(p.ROOT/f)==h,f
 for f,h in prior['E_source_hashes'].items():assert p.digest(p.E/f)==h,f
 for f in ['nested_basis.py','nested_factor.py','operator_green.py','prior_permutation_driver.py']:assert p.digest(HERE/f)==prev['source_hashes'][f],f
 return prior,prev,dict(source_hashes={f.name:p.digest(f) for f in HERE.glob('*.py')},plan_sha256=p.digest(HERE/'LAYOUT_PLAN.md'),prior_results_sha256=p.digest(HERE/'PRIOR_NESTED_RESULTS.json'))

def unchanged(prior):return dict(production_unchanged=all(p.digest(p.ROOT/f)==h for f,h in prior['source_hashes'].items()),E_unchanged=all(p.digest(p.E/f)==h for f,h in prior['E_source_hashes'].items()))

def physical():
 n=128;l=12;total,scatter,_=p.medium(n,1,10/3,'cpu');low=p.PeriodicRTE2D(shape=(n,n),dx=1/n,n_angles=l,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(total,scatter,torch.zeros((1,l,n,n),dtype=torch.complex64));high=p.high_precision_copy(low);bg,tb,sb=p.background(low)
 return dict(low=low,high=high,tb=tb,sb=sb,tv=tb[0,0,:,0].double(),sv=sb[0,0,:,0].double(),K=(high.phase_matrix*high.weights[None]).to(DT),kn=high.kx.flatten(),normal=high.directions[:,0],tangent=high.directions[:,1],background=bg)

def operators(ctx):
 n=128;l=12;tv=ctx['tv'];sv=ctx['sv'];K=ctx['K'];kn=ctx['kn'];normal=ctx['normal'];tangent=ctx['tangent'];kt=ctx['high'].ky.flatten()[0]
 V=tv[:,None,None]*torch.eye(l,dtype=DT)-sv[:,None,None]*K[None]+1j*kt*torch.diag(tangent.to(DT))[None]
 def original(q):
  x=q.reshape(n,l,-1);der=torch.fft.ifft(1j*kn[:,None,None]*torch.fft.fft(x,dim=0),dim=0);return (der*normal[None,:,None]+torch.einsum('nij,njr->nir',V,x)).reshape_as(q)
 def approximate(q):
  x=q.reshape(n,l,-1);der=ctx['Dhat']@x.reshape(n,-1);return (der.reshape_as(x)*normal[None,:,None]+torch.einsum('nij,njr->nir',V,x)).reshape_as(q)
 return V,original,approximate

def identities(ctx):return {k:p.thash(ctx[k]) for k in ['tv','sv','K','kn','normal','tangent']}

def basis_identity(tree):
 rec=[]
 def visit(t):
  row=dict(i=t.i,n=t.n,m=t.m,r=t.r,leaf=t.leaf,root=t.root)
  for name in ['Q','B','D','U']:
   x=getattr(t,name,None)
   if x is not None:row[name]=dict(shape=list(x.shape),stride=list(x.stride()),sha256=p.thash(x))
  rec.append(row)
  if not t.leaf:visit(t.a);visit(t.b)
 visit(tree);return dict(sha256=hashlib.sha256(json.dumps(rec,sort_keys=True).encode()).hexdigest(),records=rec)

def probes():
 gen=torch.Generator().manual_seed(2026111312);q=torch.complex(torch.randn(1536,3,dtype=torch.float64,generator=gen),torch.randn(1536,3,dtype=torch.float64,generator=gen));q=q/q.norm(dim=0)
 mixed=torch.cat([torch.zeros(1536,1,dtype=DT),q,1e-200*q[:,:1]],1);return q,dict(single=q[:,:1].clone(),mixed=mixed)

def validate(q,x,original,ref=None):
 rel=p.relative((q-original(x)).T,q.T);active=(q!=0).any(dim=0);zero=bool((x[:,~active]==0).all());tiny=bool(x[:,active].abs().amax(dim=0).min()>0)
 diff=p.relative((x-ref).T,ref.T) if ref is not None else torch.zeros(q.shape[1]);passed=bool((rel<=1e-10).all()) and bool((diff<=1e-9).all()) and zero and tiny and bool(torch.isfinite(x).all())
 return dict(original_relative=rel.tolist(),reference_difference=diff.tolist(),zero=zero,all_nonzero_columns_preserved=tiny,passed=passed)
