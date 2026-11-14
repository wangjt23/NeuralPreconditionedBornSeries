"""Exact stored-Fourier normal-angle background blocks: one-mode gate."""
import sys,time,json,hashlib,math,argparse,os
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
sys.path[:0]=[str(ROOT),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1')]
from screen_structured import medium
from layered_inverse import background_from_material
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
from rte.train_source_predictor import build_base
from rte.augmented_fgmres import augmented_fgmres,projected_direction_loss

class ExactModes:
 def __init__(self,low,high,ks):
  self.shape=low.source.shape;self.axis,self.t,self.s,_=background_from_material(low.sigma_t,low.sigma_s)
  n,l=low.shape[0],low.n_angles;device=low.source.device
  t=(self.t[0,0,:,0] if self.axis==0 else self.t[0,0,0,:]).double()
  s=(self.s[0,0,:,0] if self.axis==0 else self.s[0,0,0,:]).double()
  self.phase=(high.phase_matrix*high.weights[None]).to(torch.complex128)
  normal=high.directions[:,self.axis].to(torch.complex128);tangent=high.directions[:,1-self.axis].to(torch.complex128)
  nk=(high.kx if self.axis==0 else high.ky).flatten();tk=(high.ky if self.axis==0 else high.kx).flatten()
  I=torch.eye(n,device=device,dtype=torch.complex128)
  D=torch.fft.ifft(1j*nk[:,None]*torch.fft.fft(I,dim=0),dim=0)
  base=torch.kron(torch.diag(normal),D.contiguous())+torch.kron(torch.eye(l,dtype=torch.complex128,device=device),torch.diag(t).to(torch.complex128))-torch.kron(self.phase.contiguous(),torch.diag(s).to(torch.complex128))
  self.factors={};self.times={};self.matrices={}
  for k in ks:
   j=k%n;t0=time.perf_counter()
   M=base+torch.kron(torch.diag(1j*tk[j]*tangent),I)
   lu,piv=torch.linalg.lu_factor(M)
   self.factors[k]=(lu,piv);self.times[k]=time.perf_counter()-t0
   self.matrices[k]=M
 def apply(self,r,ks=None):
  ks=list(self.factors) if ks is None else ks
  r=r.transpose(-1,-2) if self.axis else r
  f=torch.fft.fft(r,dim=-1,norm='ortho');out=torch.zeros_like(f)
  for k in ks:
   lu,piv=self.factors[k];out[...,k%r.shape[-1]]=torch.linalg.lu_solve(lu,piv,f[...,k%r.shape[-1]].flatten()[:,None]).reshape(*r.shape[:-1])
  z=torch.fft.ifft(out,dim=-1,norm='ortho')
  return z.transpose(-1,-2) if self.axis else z
 def project(self,r,ks):
  r=r.transpose(-1,-2) if self.axis else r
  f=torch.fft.fft(r,dim=-1,norm='ortho');out=torch.zeros_like(f)
  for k in ks:out[...,k%r.shape[-1]]=f[...,k%r.shape[-1]]
  z=torch.fft.ifft(out,dim=-1,norm='ortho')
  return z.transpose(-1,-2) if self.axis else z
 def perturbation(self,z,high):
  return (high.sigma_t-self.t.double())*z-(high.sigma_s-self.s.double())*torch.einsum('lm,bmxy->blxy',self.phase,z)

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 out=HERE/'mode_probe_01';out.mkdir(exist_ok=False);began=time.perf_counter();rows=[]
 saved=torch.load(ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True)
 b=saved['bank']['q'][0:1].to(torch.complex128);del saved
 total,scatter,info=medium(128,1,10/3,'cpu');low=PeriodicRTE2D(shape=(128,128),dx=1/128,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(total,scatter,b.to(torch.complex64));high=high_precision_copy(low)
 t0=time.perf_counter();base=build_base(low,'tsa64');base.prepare();pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128);base_setup=time.perf_counter()-t0
 prefixes=[];augmented_fgmres(high.A_op,b,pre,None,maxiter=1,inject_after=1,capture_prefix=prefixes.append);prefix=prefixes[0]
 t0=time.perf_counter();factors=ExactModes(low,high,[0,1,8]);setup=time.perf_counter()-t0
 r=prefix['residual'];basis=prefix['image_basis'].transpose(0,1)
 for k in factors.factors:
  t0=time.perf_counter();z=factors.apply(r,[k]);action=time.perf_counter()-t0
  target=factors.project(r,[k]);az=high.A_op(z);bgaz=az-factors.perturbation(z,high)
  block_res=float((target-bgaz).norm()/target.norm())
  assert block_res<1e-10
  rows.append(dict(k=k,factor_seconds=factors.times[k],action_seconds=action,original_background_block_residual=block_res,physical_selected_rhs_defect=float((target-az).norm()/target.norm()),retained_prefix_residual_energy=float(target.norm().square()/r.norm().square()),original_quotient_residual_ratio=float(projected_direction_loss(high.A_op,z,r,basis)[0])))
 report=dict(complete=True,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),device='cpu',threads=1,source_id=0,grid=128,L=12,g=.65,modes=[0,1,8],total_factor_setup_seconds=setup,base_setup_seconds=base_setup,wall_seconds=time.perf_counter()-began,factor_bytes=sum(a.numel()*a.element_size()+p.numel()*p.element_size() for a,p in factors.factors.values()),rows=rows,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='CPU algebra/cost screen of three modes only; no full solve speed claim, no neural model, no reserved cases.')
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
