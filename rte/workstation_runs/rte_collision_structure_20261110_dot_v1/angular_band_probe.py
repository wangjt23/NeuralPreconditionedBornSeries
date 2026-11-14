"""Three fixed mode blocks, actual stored angular geometry, max3 original-G repairs."""
import os,sys,time,json,hashlib
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
sys.path[:0]=[str(ROOT),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1')]
from screen_structured import medium
from layered_inverse import background_from_material
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy

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
  x=self.raw(q);bn=q.norm(dim=0);trace=[]
  for count in range(max_corrections+1):
   r=q-self.original(x);ratio=r.norm(dim=0)/bn;trace.append(ratio.tolist())
   if bool((ratio<=tolerance).all()):return x,dict(corrections=count,success=True,residual_trace=trace)
   if count<max_corrections:x=x+self.raw(r)
  return x,dict(corrections=max_corrections,success=False,residual_trace=trace)

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261110);start=time.perf_counter();out=HERE/'angular_band_01';out.mkdir(exist_ok=False);n=128;l=12
 t,s,_=medium(n,1,10/3,'cpu');low=PeriodicRTE2D(shape=(n,n),dx=1/n,n_angles=l,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(t,s,torch.ones(1,l,n,n,dtype=torch.complex64));high=high_precision_copy(low);axis,tbg,sbg,_=background_from_material(low.sigma_t,low.sigma_s);assert axis==0;t=tbg[0,0,:,0].double();s=sbg[0,0,:,0].double();K=(high.phase_matrix*high.weights[None]).to(torch.complex128);I=torch.eye(n,dtype=torch.complex128);D=torch.fft.ifft(1j*high.kx.flatten()[:,None]*torch.fft.fft(I,dim=0),dim=0);records=[]
 for k in [0,1,8]:
  # Same RNG draw shapes/order as audit_collision.py, no random draws elsewhere.
  q=torch.complex(torch.randn(l*n,3,dtype=torch.float64),torch.randn(l*n,3,dtype=torch.float64));qhash=hashlib.sha256(q.numpy().tobytes()).hexdigest()
  stage=time.perf_counter();T=high.directions[:,0,None,None]*D[None]+torch.diag(t).to(torch.complex128)[None]+1j*high.ky.flatten()[k]*high.directions[:,1,None,None]*I[None];G=torch.block_diag(*T.unbind(0))-torch.kron(K.contiguous(),torch.diag(s).to(torch.complex128));assembly=time.perf_counter()-stage;stage=time.perf_counter();lu,piv=torch.linalg.lu_factor(G);densesetup=time.perf_counter()-stage
  candidate=AngularBandMode(high,t,s,k);action_error=((candidate.original(q)-G@q).norm(dim=0)/(G@q).norm(dim=0)).tolist();assert max(action_error)<1e-10
  dense_times=[];candidate_times=[];diagnostics=None
  for rep in range(3):
   order=['dense','band'] if rep%2==0 else ['band','dense']
   for name in order:
    tick=time.perf_counter()
    if name=='dense':dense=torch.linalg.lu_solve(lu,piv,q);dense_res=(q-candidate.original(dense)).norm(dim=0)/q.norm(dim=0);dense_times.append(time.perf_counter()-tick)
    else:answer,status=candidate.repaired(q,tolerance=1e-10,max_corrections=3);candidate_times.append(time.perf_counter()-tick)
   diff=((answer-dense).norm(dim=0)/dense.norm(dim=0)).tolist();assert status['success'] and max(diff)<1e-9
   diagnostics=dict(status=status,dense_original_residual=dense_res.tolist(),relative_difference_vs_dense=diff)
  raw=candidate.raw(q);raw_original=((candidate.original(raw)-q).norm(dim=0)/q.norm(dim=0)).tolist();band_res=((candidate.band_action(raw)-q).norm(dim=0)/q.norm(dim=0)).tolist();assert max(band_res)<1e-10
  records.append(dict(k=k,RHS_sha256=qhash,original_matrix_free_vs_dense=action_error,dense_assembly_seconds=assembly,dense_LU_seconds=densesetup,band_setup=candidate.setup,dense_three_RHS_apply_and_check_seconds=dense_times,band_three_RHS_apply_repair_check_seconds=candidate_times,dense_factor_bytes=lu.numel()*lu.element_size()+piv.numel()*piv.element_size(),band_stored_bytes=candidate.bytes,raw_band_original_residual=raw_original,raw_band_own_operator_residual=band_res,**diagnostics))
 report=dict(complete=True,pid=os.getpid(),device='cpu',affinity=sorted(os.sched_getaffinity(0)),grid=n,L=l,g=.65,modes=[0,1,8],seed=20261110,vectors_per_mode=3,repair_target=1e-10,max_corrections=3,records=records,wall_seconds=time.perf_counter()-start,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Fixed-mode original-G qualification only. Cyclic band is an approximate representation of stored angular coefficients, repaired against original G. Not bitwise exact, no17-mode full solve or GPU claim.',GPU_used=False)
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
