import time
import torch

class CyclicBlockFactor:
 def __init__(self,diag,upper,lower,corner_first_last,corner_last_first):
  self.l=len(diag);self.n=len(diag[0]);self.lower=lower;self.lu=[];self.piv=[];self.Z=[]
  t=time.perf_counter();current=diag[0]
  for i in range(self.l):
   lu,piv=torch.linalg.lu_factor(current);self.lu.append(lu);self.piv.append(piv)
   if i<self.l-1:
    z=torch.linalg.lu_solve(lu,piv,upper[i]);self.Z.append(z);current=diag[i+1]-lower[i]@z
  chain=time.perf_counter()-t;t=time.perf_counter()
  U=torch.zeros((self.l,self.n,2*self.n),dtype=diag[0].dtype,device=diag[0].device);U[0,:,:self.n]=corner_first_last;U[-1,:,self.n:]=corner_last_first
  self.W=self.open_solve(U);H=torch.eye(2*self.n,dtype=U.dtype,device=U.device)+torch.cat([self.W[-1],self.W[0]],dim=0)
  self.corner_lu,self.corner_piv=torch.linalg.lu_factor(H)
  self.setup=dict(chain_factor_seconds=chain,corner_factor_seconds=time.perf_counter()-t)
  tensors=self.lower+self.lu+self.piv+self.Z+[self.W,self.corner_lu,self.corner_piv]
  self.bytes=sum(v.numel()*v.element_size() for v in tensors)
 def open_solve(self,b):
  y=[]
  for i in range(self.l):
   value=b[i] if i==0 else b[i]-self.lower[i-1]@y[-1]
   y.append(torch.linalg.lu_solve(self.lu[i],self.piv[i],value))
  x=[None]*self.l;x[-1]=y[-1]
  for i in range(self.l-2,-1,-1):x[i]=y[i]-self.Z[i]@x[i+1]
  return torch.stack(x)
 def __call__(self,b):
  y=self.open_solve(b);c=torch.linalg.lu_solve(self.corner_lu,self.corner_piv,torch.cat([y[-1],y[0]],dim=0))
  return y-self.W@c

class AngularBandMode:
 def __init__(self,high,t,s,k):
  start=time.perf_counter();self.n=len(t);self.l=high.n_angles;self.t=t;self.s=s;self.k=k;self.nk=high.kx.flatten();self.tk=high.ky.flatten()[k]
  self.normal=high.directions[:,0];self.tangent=high.directions[:,1];self.K=(high.phase_matrix*high.weights[None]).to(torch.complex128)
  F=torch.fft.fft(torch.eye(self.l,dtype=torch.complex128),dim=0,norm='ortho');Cn=F@torch.diag(self.normal.to(torch.complex128))@F.conj().T;Ct=F@torch.diag(self.tangent.to(torch.complex128))@F.conj().T;Kf=F@self.K@F.conj().T
  mask=torch.eye(self.l,dtype=torch.bool)
  for i in range(self.l):mask[i,(i-1)%self.l]=True;mask[i,(i+1)%self.l]=True
  self.Cn=Cn*mask;self.Ct=Ct*mask;self.Kf=Kf*mask
  I=torch.eye(self.n,dtype=torch.complex128);D=torch.fft.ifft(1j*self.nk[:,None]*torch.fft.fft(I,dim=0),dim=0);S=torch.diag(s).to(torch.complex128);T=torch.diag(t).to(torch.complex128)
  def block(i,j):return Cn[i,j]*D+1j*self.tk*Ct[i,j]*I-Kf[i,j]*S+(T if i==j else 0.)
  diag=[block(i,i) for i in range(self.l)];upper=[block(i,i+1) for i in range(self.l-1)];lower=[block(i+1,i) for i in range(self.l-1)];cfl=block(0,self.l-1);clf=block(self.l-1,0)
  assembly=time.perf_counter()-start;self.factor=CyclicBlockFactor(diag,upper,lower,cfl,clf);self.setup=dict(assembly_seconds=assembly,**self.factor.setup,total_seconds=time.perf_counter()-start)
  extra=[self.t,self.s,self.nk,self.tk,self.normal,self.tangent,self.K,self.Cn,self.Ct,self.Kf]
  self.bytes=self.factor.bytes+sum(v.numel()*v.element_size() for v in extra)
 def derivative(self,x):return torch.fft.ifft(1j*self.nk[None,:,None]*torch.fft.fft(x,dim=1),dim=1)
 def original(self,q):
  x=q.reshape(self.l,self.n,-1);v=self.normal[:,None,None]*self.derivative(x)+1j*self.tk*self.tangent[:,None,None]*x+self.t[None,:,None]*x-self.s[None,:,None]*torch.einsum('lm,mnr->lnr',self.K,x)
  return v.reshape_as(q)
 def band_action(self,q):
  x=torch.fft.fft(q.reshape(self.l,self.n,-1),dim=0,norm='ortho')
  v=torch.einsum('lm,mnr->lnr',self.Cn,self.derivative(x))+1j*self.tk*torch.einsum('lm,mnr->lnr',self.Ct,x)+self.t[None,:,None]*x-self.s[None,:,None]*torch.einsum('lm,mnr->lnr',self.Kf,x)
  return torch.fft.ifft(v,dim=0,norm='ortho').reshape_as(q)
 def raw(self,q):return torch.fft.ifft(self.factor(torch.fft.fft(q.reshape(self.l,self.n,-1),dim=0,norm='ortho')),dim=0,norm='ortho').reshape_as(q)
 def repaired(self,q,tolerance=1e-10,max_corrections=3):
  # Strict zero-column handling. No absolute residual floor for nonzero RHS.
  active=(q!=0).any(dim=0);x=torch.zeros_like(q);trace=[]
  if not bool(active.any()):return x,dict(corrections=0,success=True,residual_trace=[[0.]*q.shape[1]],zero_columns=q.shape[1])
  qa=q[:,active];scale=qa.abs().amax(dim=0);den=(qa/scale).norm(dim=0)
  x[:,active]=self.raw(qa)
  for count in range(max_corrections+1):
   r=q-self.original(x);ratio=torch.zeros(q.shape[1],dtype=q.real.dtype)
   ratio[active]=(r[:,active]/scale).norm(dim=0)/den
   trace.append(ratio.tolist())
   if bool((ratio<=tolerance).all()):return x,dict(corrections=count,success=True,residual_trace=trace,zero_columns=int((~active).sum()))
   if count<max_corrections:x[:,active]+=self.raw(r[:,active])
  return x,dict(corrections=max_corrections,success=False,residual_trace=trace,zero_columns=int((~active).sum()))

class FullAngularModes:
 def __init__(self,low,high):
  from layered_inverse import background_from_material
  self.shape=low.source.shape;self.axis,self.t,self.s,_=background_from_material(low.sigma_t,low.sigma_s)
  if self.axis!=0:raise ValueError('This frozen qualification covers normal axis0 only')
  t=self.t[0,0,:,0].double();s=self.s[0,0,:,0].double();self.blocks={k:AngularBandMode(high,t,s,k) for k in range(-8,9)}
  self.profiles=[];self.bytes=sum(b.bytes for b in self.blocks.values())
 def project(self,r):
  f=torch.fft.fft(r,dim=-1,norm='ortho');out=torch.zeros_like(f)
  for k in self.blocks:out[...,k]=f[...,k]
  return torch.fft.ifft(out,dim=-1,norm='ortho')
 def apply(self,r):
  f=torch.fft.fft(r,dim=-1,norm='ortho');out=torch.zeros_like(f);record=[]
  for k,block in self.blocks.items():
   q=f[...,k].reshape(-1,1);z,st=block.repaired(q,tolerance=1e-10,max_corrections=3)
   record.append(dict(k=k,corrections=st['corrections'],success=st['success'],zero_columns=st['zero_columns'],residual_trace=st['residual_trace']))
   if not st['success']:
    self.profiles.append(record);raise RuntimeError('Original-G correction target failed for mode '+str(k))
   out[...,k]=z.reshape(*r.shape[:-1])
  self.profiles.append(record);return torch.fft.ifft(out,dim=-1,norm='ortho')
