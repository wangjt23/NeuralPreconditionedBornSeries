import time
import torch
from layered_inverse import background_from_material

class ExactModes:
 def __init__(self,low,high,ks,checkpoint=None,synchronize=None):
  sync=synchronize or (lambda:None);sync();began=time.perf_counter()
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
  self.factors={};self.times={};self.profiles=[];sync();self.common_assembly_seconds=time.perf_counter()-began
  if checkpoint is not None:checkpoint(self,None)
  for k in ks:
   j=k%n;sync();t0=time.perf_counter()
   M=base+torch.kron(torch.diag(1j*tk[j]*tangent),I)
   sync();assembly=time.perf_counter()-t0;tlu=time.perf_counter();lu,piv=torch.linalg.lu_factor(M);sync();lusec=time.perf_counter()-tlu
   self.factors[k]=(lu,piv);self.times[k]=time.perf_counter()-t0
   self.profiles.append(dict(k=k,assembly_seconds=assembly,LU_seconds=lusec,total_seconds=self.times[k]))
   if checkpoint is not None:checkpoint(self,k)
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

