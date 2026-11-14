"""Untrained anisotropic classical control for the direct-coefficient oracle.
Retain the full normal grid and all angles, coarsen only tangent to32.
The outer original Fourier A64/acceptance and native TSA16 remain unchanged.
"""
import os,sys,time,json,hashlib,subprocess,random,gc
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"rte/workstation_runs/direct_capture_20261002/confirmation/shard0/code"))
sys.path.insert(0,str(ROOT/"rte/workstation_runs/layered_background_20261108_dot_v1"))
from compressed_oracle import Encoded,safe,competitors,sync
from direct_segmented_tsa import DirectSegmentedTSA
from layered_inverse import background_from_material
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
from rte.transport_baselines import ClassicalTransportStack
from rte.run_large_angular_spectral_experiments import ExperimentConfig
from rte.train_source_predictor import material_operator,build_base
from rte.source_initialization import solve_from_initial
from rte.augmented_fgmres import augmented_fgmres

class SemiCoarse:
 def __init__(self,fine,axis,m=32):
  self.axis=axis;self.n=fine.shape[0];self.m=m
  assert self.n % m==0
  device=fine.sigma_t.device
  freq=torch.fft.fftfreq(m,device=device)*m
  self.indices=freq.long().remainder(self.n)
  def rotate(x):return x.transpose(-1,-2) if axis else x
  t=rotate(fine.sigma_t)[...,::self.n//m].contiguous()
  s=rotate(fine.sigma_s)[...,::self.n//m].contiguous()
  q=self.restrict(fine.source)
  low=PeriodicRTE2D(shape=(self.n,m),dx=fine.dx,n_angles=fine.n_angles,anisotropy_g=fine.anisotropy_g,
   eta_scale=0.,eta_floor=0.,fdtype=fine.float_dtype,device=device)
  low.directions=fine.directions[:,[axis,1-axis]].clone()
  low.weights=fine.weights.clone();low.phase_matrix=fine.phase_matrix.clone()
  kn=(fine.kx if axis==0 else fine.ky).flatten()
  kt=(fine.ky if axis==0 else fine.kx).flatten()
  low.kx=kn.view(1,1,self.n,1).clone()
  low.ky=kt[self.indices].view(1,1,1,m).clone()
  low.setup(t,s,q);self.low=low;self.high=high_precision_copy(low)
  self.base=ClassicalTransportStack(low,"tsa",16,0.)
  # Existing constructor assumes equal dx in both directions. Repair only this
  # new private reference's stored geometry before any application/capture.
  # Initial discarded reference construction remains charged in setup.
  for key in ["directions","weights","phase_matrix","kx","ky"]:
   setattr(self.base.reference,key,getattr(low,key).clone())
  self.base.reference._reference_transport_inverse=None
  self.base.reference._reference_moment_inverse_cache={}
  self.base.prepare()
  self.tsa=DirectSegmentedTSA(self.base,low.source,chunk=1,capture_first=False)
 def restrict(self,v):
  if self.axis:v=v.transpose(-1,-2)
  z=torch.fft.fft(v,dim=-1,norm="forward")
  return torch.fft.ifft(z[...,self.indices],dim=-1,norm="forward")
 def prolong(self,v):
  z=torch.fft.fft(v,dim=-1,norm="forward")
  large=torch.zeros((*z.shape[:-1],self.n),device=z.device,dtype=z.dtype)
  large[...,self.indices]=z
  out=torch.fft.ifft(large,dim=-1,norm="forward")
  return out.transpose(-1,-2) if self.axis else out
 def __call__(self,r,steps):
  rhs=self.restrict(r)
  pre=lambda x:self.tsa(x.to(torch.complex64)).to(torch.complex128)
  if steps==0:u=pre(rhs)
  else:u,_=solve_from_initial(self.high.A_op,rhs,pre,None,tolerance=1e-3,maxiter=steps)
  return self.prolong(u)

@torch.no_grad()
def main():
 torch.set_num_threads(1)
 if competitors():raise RuntimeError("Competing GPU work")
 out=HERE/"semicoarse_01";out.mkdir(exist_ok=False)
 began=time.perf_counter();rows=[];contexts=[]
 settings=dict(complete=False,pid=os.getpid(),training=False,new_teachers=False,heldout9626=False,grid=128,ids=list(range(16)),
  repeats=3,scope="Development classical countercontrol. Full normal grid, all12 angles, tangent32 collocation; Fourier R/P.",
  setup="Charge private rectangular material/operator+reference+capture. Native unchanged. No per-material large layered factor.",
  caveat="Collocation is not exact RAP for heterogeneous tangent coefficients; original fine acceptance remains mandatory.",
  code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 def save():
  for k,x in [("settings",settings),("contexts",contexts),("rows",rows)]:(out/(k+".json")).write_text(json.dumps(safe(x),indent=2)+"\n")
 save();torch.cuda.reset_peak_memory_stats()
 data=torch.load(ROOT/"rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n128.pt",map_location="cpu",weights_only=True)
 bank=torch.load(ROOT/"rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt",map_location="cpu",weights_only=True)["bank"]
 cfg=ExperimentConfig(size=128,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes="mixed")
 methods=["TSA16_native","TSA16_augmented","semi32_TSA16","semi32_GMRES2","semi32_GMRES4","oracle_layer8"]
 try:
  for index in range(16):
   if competitors():raise RuntimeError("New GPU competitor; stop own worker")
   sync();t=time.perf_counter();low,high=material_operator(cfg,data,torch.tensor([index]),.1,"cuda");sync();physical=time.perf_counter()-t
   sync();t=time.perf_counter();phase=high.phase_matrix.double()*high.weights.double()[None]
   mu=torch.linalg.eigvalsh((phase+phase.T)/2).max();margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0
   sync();accept_setup=time.perf_counter()-t
   sync();t=time.perf_counter();base=build_base(low,"tsa16");sync();bs=time.perf_counter()-t
   sync();t=time.perf_counter();tsa=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False);sync();capture=time.perf_counter()-t
   pre=lambda r:tsa(r.to(torch.complex64)).to(torch.complex128)
   sync();t=time.perf_counter();axis,*_=background_from_material(low.sigma_t,low.sigma_s);sync();axis_time=time.perf_counter()-t
   sync();t=time.perf_counter();semi=SemiCoarse(low,axis);sync();semi_time=time.perf_counter()-t
   prefix=[];augmented_fgmres(high.A_op,high.source,pre,None,maxiter=1,inject_after=1,capture_prefix=prefix.append)
   assert len(prefix)==1;prefix=prefix[0]
   for key,v in [("t",low.sigma_t),("s",low.sigma_s),("q",low.source)]:assert torch.equal(bank[key][index:index+1].cuda(),v)
   teacher=bank["target"][index:index+1].cuda();assert float((high.source-high.A_op(teacher)).norm()/high.source.norm())<2e-9
   oracle=Encoded(teacher-prefix["solution"],"all_layer8",axis)
   v=semi.restrict(prefix["residual"])
   streamed=semi.restrict(high.streaming_op(semi.prolong(v)))
   stream_error=float((streamed-semi.high.streaming_op(v)).norm()/streamed.norm())
   assert stream_error<2e-12,stream_error
   rap=semi.restrict(high.A_op(semi.prolong(v)))
   rap_error=float((rap-semi.high.A_op(v)).norm()/rap.norm())
   transfer_error=float((semi.restrict(semi.prolong(v))-v).norm()/v.norm());assert transfer_error<2e-12
   contexts.append(dict(index=index,axis=axis,physical_setup=physical,acceptance_setup=accept_setup,
    tsa_setup=bs,tsa_capture=capture,axis_setup=axis_time,semicoarse_setup=semi_time,
    streaming_commutation_error=stream_error,restriction_prolongation_error=transfer_error,
    collocation_RAP_relative_defect_on_prefix=rap_error))
   common=physical+accept_setup+bs+capture;baseline=None;bn=float(high.source.norm());rng=random.Random(811192+index)
   for repeat in range(-1,3):
    order=methods.copy();rng.shuffle(order)
    if repeat<0:order.remove("TSA16_native");order.insert(0,"TSA16_native")
    for name in order:
     sync();t=time.perf_counter()
     if name=="TSA16_native":u,st=solve_from_initial(high.A_op,high.source,pre,None,maxiter=100);st.update(classical_calls=st["iterations"],injection_calls=0)
     elif name=="TSA16_augmented":u,st=augmented_fgmres(high.A_op,high.source,pre,None,maxiter=100)
     else:
      def inject(x,r):
       if name=="oracle_layer8":
        if repeat<0:assert torch.equal(x,prefix["solution"]) and torch.equal(r,prefix["residual"])
        return oracle()
       return semi(r,0 if name.endswith("TSA16") else 2 if name.endswith("GMRES2") else 4)
      u,st=augmented_fgmres(high.A_op,high.source,pre,None,inject=inject,inject_after=1,maxiter=100)
     sync();solver=time.perf_counter()-t;st.pop("trace",None)
     if baseline is None:assert not st["failed"];baseline=u.clone()
     sync();t=time.perf_counter();r=float((high.source-high.A_op(u)).norm()/bn);d=r*bn/margin;den=float(u.norm())-d
     bound=d/den if den>0 else float("inf");diff=float((u-baseline).norm()/baseline.norm())
     accepted=bool(torch.isfinite(u).all()) and not st["failed"] and r<=1e-5 and bound<=1e-3 and diff<=1e-4
     sync();check=time.perf_counter()-t
     extra=(axis_time+semi_time) if name.startswith("semi") else axis_time if name.startswith("oracle") else 0.
     rows.append(dict(grid=128,index=index,method=name,repeat=repeat,solver_seconds=solver,acceptance_seconds=check,
      accepted_seconds=solver+check,setup_seconds=common+extra,Q1_seconds=common+extra+solver+check,
      status=st,accepted=accepted,true_residual=r,evaluated_relative_bound=bound,relative_field_difference=diff))
    save()
   print("SEMICOARSE",index,"rows",len(rows),flush=True)
   del semi,oracle,low,high,base,tsa,teacher,prefix,baseline;gc.collect();sync();torch.cuda.empty_cache()
  settings["complete"]=True
 except BaseException as exc:settings["error"]=repr(exc);raise
 finally:
  settings.update(wall_seconds=time.perf_counter()-began,rows=len(rows),failures=sum(not r["accepted"] for r in rows),
   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());save()
if __name__=="__main__":main()
