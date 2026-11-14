"""Small FWI-only diagnostics on immutable stored development snapshots."""
import hashlib,json,pathlib,time
import numpy as np
import torch
from rte.mixed_precision_transport import high_precision_copy
from rte.run_large_angular_spectral_experiments import ExperimentConfig,make_operator

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb')as f:
  for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
 return h.hexdigest()
def aid(a):
 a=np.ascontiguousarray(a)
 return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def write(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sync():torch.cuda.synchronize()
def timing(function):
 sync();start=time.perf_counter();v=function();sync()
 return v,time.perf_counter()-start
def load_case(root,item):
 path=root/'jobA'/item['path'];assert sha(path)==item['sha256']
 with np.load(path,allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
 assert all(aid(a[k])==v['sha256']for k,v in item['arrays'].items())
 cfg=ExperimentConfig(size=item['size'],n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed')
 t,s,q=[torch.from_numpy(a[k]).to('cuda') for k in ('sigma_t','sigma_s','original_source')]
 low=make_operator(cfg,t.float(),s.float(),q.to(torch.complex64),'cuda');high=high_precision_copy(low)
 captured=dict(original_source=high.source,stream_symbol=high._stream_symbol(1,torch.complex128,'cuda'),
   sigma_t=high.sigma_t,sigma_s=high.sigma_s,phase=high.phase_matrix,weights=high.weights)
 assert all(aid(v.detach().cpu().numpy())==aid(a[k])for k,v in captured.items()),'Original high buffer mismatch'
 return low,high,a
class Snapshot:
 def __init__(self,path,maxbytes):
  self.root=path;path.mkdir(exist_ok=False);self.rows=[];self.total=0;self.maxbytes=maxbytes
 def save(self,name,v):
  """Persist exact represented fields BEFORE any physical-image check."""
  assert v.dtype==torch.complex128 and bool(torch.isfinite(v).all())
  cpu=v.detach().cpu().contiguous();path=self.root/(name+'.pt')
  with path.open('xb')as f:torch.save({'field':cpu},f)
  digest=sha(path);self.total+=path.stat().st_size;assert self.total<=self.maxbytes
  restored=torch.load(path,map_location='cpu',weights_only=True)['field']
  assert aid(cpu.numpy())==aid(restored.numpy()),'Persisted candidate bit mismatch'
  self.rows.append(dict(path=str(path),sha256=digest,field_sha256=aid(cpu.numpy()),bytes=path.stat().st_size,
    dtype=str(cpu.dtype),shape=list(cpu.shape),saved_before_physical_check=True))
  return restored.to(v.device)
 def finish(self):
  assert all(sha(r['path'])==r['sha256']for r in self.rows)
  write(self.root/'manifest.json',dict(snapshots=self.rows,total_bytes=self.total))
def best_real_line(r,y):
 den=torch.vdot(y.flatten(),y.flatten()).real
 assert bool(den>0)
 alpha=torch.vdot(y.flatten(),r.flatten()).real/den
 return float(alpha),float(torch.linalg.vector_norm(r-alpha*y)/torch.linalg.vector_norm(r))
def scalar_checks(q,x,Hx):
 return dict(source_norm=float(torch.linalg.vector_norm(q)),candidate_norm=float(torch.linalg.vector_norm(x)),
  image_norm=float(torch.linalg.vector_norm(Hx)),all_finite=bool(torch.isfinite(x).all()and torch.isfinite(Hx).all()))
