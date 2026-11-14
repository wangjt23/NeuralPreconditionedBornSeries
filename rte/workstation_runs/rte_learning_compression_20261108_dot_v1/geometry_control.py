"""No-training feasibility test of cheaper direct RTE correction representations.
Free coefficients are encoded from existing exact teachers at the actual TSA16
prefix1. Decoder, material-axis extraction, physical projection and continuation
are charged. This is an optimistic representation diagnostic, not a learned solver.
"""
import os,sys,time,json,math,hashlib,subprocess,random,gc
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"rte/workstation_runs/direct_capture_20261002/confirmation/shard0/code"))
sys.path.insert(0,str(ROOT/"rte/workstation_runs/layered_background_20261108_dot_v1"))
from direct_segmented_tsa import DirectSegmentedTSA
from layered_inverse import background_from_material
from rte.run_large_angular_spectral_experiments import ExperimentConfig
from rte.train_source_predictor import material_operator,build_base
from rte.source_initialization import solve_from_initial
from rte.augmented_fgmres import augmented_fgmres,projected_direction_loss

def sync():torch.cuda.synchronize()
def competitors():
 s=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
 return [int(v.strip()) for v in s.splitlines() if v.strip().isdigit() and int(v.strip())!=os.getpid()]
def safe(x):
 if isinstance(x,float) and not math.isfinite(x):return str(x)
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,list):return [safe(v) for v in x]
 return x

class Encoded:
 def __init__(self,e,name,axis):
  self.name=name;self.axis=axis if "layer" in name else 0;self.n=e.shape[-1];self.l=e.shape[1]
  e=e.transpose(-1,-2) if self.axis else e
  theta=torch.arange(self.l,device=e.device,dtype=torch.float64)*2*math.pi/self.l
  basis=torch.stack([torch.ones_like(theta)/math.sqrt(self.l),theta.cos()*math.sqrt(2/self.l),theta.sin()*math.sqrt(2/self.l)])
  angular=1 if name.startswith("p0") else 3 if name.startswith("p1") else self.l
  self.basis=basis[:angular].to(torch.complex64) if angular<self.l else None
  e=torch.einsum("ml,blxy->bmxy",basis[:angular].to(e.dtype),e) if angular<self.l else e
  self.dtype=torch.complex128 if name=="exact64" else torch.complex64
  self.cut=8 if "layer8" in name else 23 if "f23" in name else 32 if "f32" in name else None
  self.layer="layer" in name
  if self.cut is not None:
   f=torch.fft.fftfreq(self.n,device=e.device)*self.n
   self.iy=torch.nonzero(f.abs()<=self.cut).flatten()
   self.ix=torch.arange(self.n,device=e.device) if self.layer else self.iy
   z=torch.fft.fft2(e,norm="ortho")
   self.coeff=z[:,:,self.ix[:,None],self.iy[None,:]].to(self.dtype).contiguous()
   self.shape=(1,angular,self.n,self.n)
  else:self.coeff=e.to(self.dtype).clone()
  self.coefficient_count=self.coeff.numel()
 def __call__(self):
  if self.cut is not None:
   z=torch.zeros(self.shape,dtype=self.dtype,device=self.coeff.device)
   z[:,:,self.ix[:,None],self.iy[None,:]]=self.coeff
   v=torch.fft.ifft2(z,norm="ortho")
  else:v=self.coeff.clone()
  if self.basis is not None:v=torch.einsum("ml,bmxy->blxy",self.basis,v)
  v=v.transpose(-1,-2) if self.axis else v
  return v.to(torch.complex128)

@torch.no_grad()
def main():
 torch.set_num_threads(1)
 if competitors():raise RuntimeError("GPU competitor before admission")
 out=HERE/"geometry_01";out.mkdir(exist_ok=False)
 started=time.perf_counter();rows=[];contexts=[]
 specs=["exact64","all_layer8","all_f23","all_f32"]
 methods=["TSA16_native","TSA16_augmented","reference_inject1"]+specs
 settings=dict(complete=False,pid=os.getpid(),training=False,new_teachers=False,heldout9626=False,
  ids=list(range(16)),grid=128,prefix=1,methods=methods,repeats=3,maxiter=100,
  scope="Free teacher coefficients on full16 reused development. Not deployable ML. No N256 teacher generated.",
  target="Encode e=u_teacher-x_prefix1; coefficient acquisition is free. Pay decoder and axis extraction, original A64 projection, subsequent TSA16 and acceptance.",
  code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  representation="Matched-DOF attribution: layered26112 complex coefficients vs isotropic f23 26508 (+1.52%), and f32 50700 (+94.16%). All12 angles retained.",
  gate="Representation must save at least5% complete Q1 versus native TSA16, leaving positive budget for inference; this is necessary headroom, not sufficient learning evidence.")
 def save():
  for name,obj in [("settings",settings),("rows",rows),("contexts",contexts)]:
   (out/(name+".json")).write_text(json.dumps(safe(obj),indent=2)+"\n")
 save();torch.cuda.reset_peak_memory_stats()
 bank=torch.load(ROOT/"rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt",map_location="cpu",weights_only=True)["bank"]
 data=torch.load(ROOT/"rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n128.pt",map_location="cpu",weights_only=True)
 cfg=ExperimentConfig(size=128,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes="mixed")
 try:
  for index in range(16):
   if competitors():raise RuntimeError("New GPU competitor; stopping own worker")
   sync();t=time.perf_counter();low,high=material_operator(cfg,data,torch.tensor([index]),.1,"cuda");sync();physical=time.perf_counter()-t
   sync();t=time.perf_counter()
   phase=high.phase_matrix.double()*high.weights.double()[None];mu=torch.linalg.eigvalsh((phase+phase.T)/2).max()
   margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0
   sync();accept_setup=time.perf_counter()-t
   sync();t=time.perf_counter();base=build_base(low,"tsa16");sync();bs=time.perf_counter()-t
   sync();t=time.perf_counter();tsa=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False);sync();capture=time.perf_counter()-t
   pre=lambda r:tsa(r.to(torch.complex64)).to(torch.complex128)
   for key,new in [("t",low.sigma_t),("s",low.sigma_s),("q",low.source)]:
    assert torch.equal(bank[key][index:index+1].cuda(),new)
   teacher=bank["target"][index:index+1].cuda()
   teacher_r=float((high.A_op(teacher)-high.source).norm()/high.source.norm());assert teacher_r<2e-9
   before=[];sync();t=time.perf_counter()
   augmented_fgmres(high.A_op,high.source,pre,None,inject_after=1,maxiter=1,capture_prefix=before.append)
   sync();prefix_cost=time.perf_counter()-t
   assert len(before)==1;prefix=before[0];e=teacher-prefix["solution"]
   sync();t=time.perf_counter();axis,*_=background_from_material(low.sigma_t,low.sigma_s);sync();axis_cost=time.perf_counter()-t
   enc={name:Encoded(e,name,axis) for name in specs}
   metrics={}
   for name,obj in enc.items():
    trial=obj()
    image_basis=prefix["image_basis"].transpose(0,1)
    ratio=float(projected_direction_loss(high.A_op,trial,prefix["residual"],image_basis)[0])
    norms=[]
    for _ in range(5):
     sync();t=time.perf_counter();obj();sync();norms.append(time.perf_counter()-t)
    metrics[name]=dict(coefficient_complex_count=obj.coefficient_count,
     dof_fraction=obj.coefficient_count/e.numel(),field_relative_error=float((trial-e).norm()/e.norm()),
     raw_original_residual=float((prefix["residual"]-high.A_op(trial)).norm()/high.source.norm()),
     optimal_quotient_residual_ratio=ratio,
     decoder_seconds=sorted(norms)[2])
   ctx=dict(index=index,axis=axis,teacher_residual=teacher_r,prefix_original_residual=prefix["relative_residual"],
    physical_setup=physical,acceptance_setup=accept_setup,tsa_setup=bs,tsa_capture=capture,
    axis_setup=axis_cost,prefix_acquisition_seconds=prefix_cost,representations=metrics)
   contexts.append(ctx)
   common=physical+accept_setup+bs+capture;baseline=None;bn=float(high.source.norm())
   rng=random.Random(811091+index)
   for repeat in range(-1,3):
    order=methods.copy();rng.shuffle(order)
    if repeat<0:order.remove("TSA16_native");order.insert(0,"TSA16_native")
    for name in order:
     sync();t=time.perf_counter()
     if name=="TSA16_native":u,st=solve_from_initial(high.A_op,high.source,pre,None,maxiter=100);st.update(classical_calls=st["iterations"],injection_calls=0)
     elif name=="TSA16_augmented":u,st=augmented_fgmres(high.A_op,high.source,pre,None,maxiter=100)
     else:
      def inject(x,r):
       if repeat<0:
        assert torch.equal(x,prefix["solution"]) and torch.equal(r,prefix["residual"])
       return base.reference.reference_transport_solve(r.to(torch.complex64)).to(torch.complex128) if name=="reference_inject1" else enc[name]()
      u,st=augmented_fgmres(high.A_op,high.source,pre,None,inject=inject,inject_after=1,maxiter=100)
     sync();solver=time.perf_counter()-t;st.pop("trace",None)
     if baseline is None:assert not st["failed"];baseline=u.clone()
     sync();t=time.perf_counter();res=float((high.source-high.A_op(u)).norm()/bn)
     delta=res*bn/margin;den=float(u.norm())-delta;bound=delta/den if den>0 else float("inf")
     diff=float((u-baseline).norm()/baseline.norm())
     accepted=bool(torch.isfinite(u).all()) and not st["failed"] and res<=1e-5 and bound<=1e-3 and diff<=1e-4
     sync();check=time.perf_counter()-t;setup=common+(axis_cost if "layer" in name else 0.)
     rows.append(dict(grid=128,index=index,method=name,repeat=repeat,solver_seconds=solver,
      acceptance_seconds=check,accepted_seconds=solver+check,setup_seconds=setup,Q1_seconds=setup+solver+check,
      status=st,accepted=accepted,true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff))
    save()
   print("MATCHED_GEOMETRY",index,"rows",len(rows),flush=True)
   del enc,low,high,base,tsa,baseline,teacher,e,prefix,before;gc.collect();sync();torch.cuda.empty_cache()
  settings["complete"]=True
 except BaseException as exc:settings["error"]=repr(exc);raise
 finally:
  settings.update(wall_seconds=time.perf_counter()-started,rows=len(rows),
   failures=sum(not r["accepted"] for r in rows),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  save()
if __name__=="__main__":main()
