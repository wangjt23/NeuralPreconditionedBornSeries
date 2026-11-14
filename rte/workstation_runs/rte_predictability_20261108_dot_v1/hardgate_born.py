"""Four-source gate on the hardest interior-optimal TSA64 structured scene.
Follow-up fixes redundant duplicate G(r) evaluation in Born only. Original run retained; do not use its Born timings as the optimized comparator.
Actual reusable layered factors get setup, capture, action and acceptance costs.
"""
import os,sys,time,json,hashlib,random,gc
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(HERE))
from screen_structured import medium
from prepare_predictability import competitors,sync,safe,build_base,DirectSegmentedTSA,solve_from_initial,augmented_fgmres
from registered_frame import NoTangentFrame
from layered_inverse import LayeredInverse,CapturedAction
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
from rte.augmented_fgmres import projected_direction_loss

@torch.no_grad()
def main():
 torch.set_num_threads(1)
 if competitors():raise RuntimeError("GPU competitor before admission")
 out=HERE/"hardgate_born_02";out.mkdir(exist_ok=False);start=time.perf_counter();rows=[];contexts=[]
 settings=dict(complete=False,pid=os.getpid(),source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  neural_training=False,new_unique_teacher_cases=0,recomputed_reference_solves=0,heldout9626=False,grid=128,n_angles=12,g=.65,
  selected_condition="p1_b3.33333",selection="Hardest of4 tested scenes; best depth64 is interior and beats32 and128.",
  source_ids=[0,4,7,14],repeats=3,maxiter=100,wall_limit=600,
  teacher_contract="4 new exact solutions for the changed A, original Fourier residual <=2e-9.",
  cost="Full Q1 and actual reuse across the same4RHS. Teacher cost excluded only for explicitly oracle method. Factor/capture charged to every method that uses it.",
  methods=["tsa64"]+[f"{s}:born" for s in ["upwind","compact4"]])
 def save():
  for key,value in [("settings",settings),("rows",rows),("contexts",contexts)]:
   (out/(key+".json")).write_text(json.dumps(safe(value),indent=2)+"\n")
 save();torch.cuda.reset_peak_memory_stats()
 saved=torch.load(ROOT/"rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt",map_location="cpu",weights_only=True)
 rhs=[saved["bank"]["q"][i:i+1].cuda().to(torch.complex128) for i in settings["source_ids"]];del saved
 try:
  # Process-first common initialization is separately recorded, same for all methods.
  sync();t=time.perf_counter();total,scattering,info=medium(128,1,10/3,"cuda")
  low=PeriodicRTE2D(shape=(128,128),dx=1/128,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device="cuda")
  low.setup(total,scattering,rhs[0].to(torch.complex64));high=high_precision_copy(low)
  phase=high.phase_matrix.double()*high.weights.double()[None];mu=torch.linalg.eigvalsh((phase+phase.T)/2).max()
  margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0
  sync();common=time.perf_counter()-t;settings["common_setup_seconds"]=common;settings["coercivity_margin"]=margin
  sync();t=time.perf_counter();base=build_base(low,"tsa64");tsa=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False)
  pre=lambda r:tsa(r.to(torch.complex64)).to(torch.complex128)
  sync();tsa_setup=time.perf_counter()-t;settings["tsa64_setup_seconds"]=tsa_setup
  factors={};actions={};gsetup={};settings["factors"]={}
  for scheme in ["upwind","compact4"]:
   if competitors():raise RuntimeError("New GPU competitor")
   sync();t=time.perf_counter();factor=LayeredInverse(low,scheme=scheme,nyquist=True,dtype=torch.complex64)
   sync();factor_sec=time.perf_counter()-t
   sync();t=time.perf_counter();action=CapturedAction(lambda r,g=factor:g(r).to(torch.complex128),high.source)
   sync();capture_sec=time.perf_counter()-t;assert action.relative_capture_difference<1e-12
   factors[scheme]=factor;actions[scheme]=action;gsetup[scheme]=factor_sec+capture_sec
   settings["factors"][scheme]=dict(factor_seconds=factor_sec,capture_seconds=capture_sec,factor_bytes=factor.factor_bytes,
    capture_error=action.relative_capture_difference)
  save()
  for source_id,b in zip(settings["source_ids"],rhs):
   if competitors():raise RuntimeError("New GPU competitor")
   if time.perf_counter()-start>600:raise RuntimeError("Hard gate budget")
   bn=float(b.norm());sync();t=time.perf_counter()
   teacher,st=solve_from_initial(high.A_op,b,pre,None,tolerance=1e-10,maxiter=180);sync();teachersec=time.perf_counter()-t
   tr=float((b-high.A_op(teacher)).norm()/bn);assert not st["failed"] and tr<2e-9
   settings["recomputed_reference_solves"]+=1
   prefixes=[];augmented_fgmres(high.A_op,b,pre,None,maxiter=1,inject_after=1,capture_prefix=prefixes.append)
   prefix=prefixes[0];frame=NoTangentFrame(low);mat,x,phi,scale=frame.visible(low,prefix["residual"])
   raw=frame.encode((teacher-prefix["solution"])/phi,8).flatten();label=raw/raw.norm()
   direction=frame.decode((label*phi).reshape(1,12,128,17),8)
   ctx=dict(source_id=source_id,teacher_seconds=teachersec,teacher_residual=tr,teacher_iterations=st["iterations"],
    full_C8_quotient_ratio=float(projected_direction_loss(high.A_op,direction,prefix["residual"],prefix["image_basis"].transpose(0,1))[0]),
    retained_error_fraction=float(raw.norm()/(teacher-prefix["solution"]).norm()),G_defects={})
   for scheme in factors:
    z=actions[scheme](prefix["residual"]);defect=float((prefix["residual"]-high.A_op(z)).norm()/prefix["residual"].norm())
    ctx["G_defects"][scheme]=dict(original_A_relative_defect=defect)
   contexts.append(ctx);rng=random.Random(991123+source_id)
   for repeat in range(-1,3):
    order=settings["methods"].copy();rng.shuffle(order)
    for name in order:
     if time.perf_counter()-start>600:raise RuntimeError("Hard gate budget")
     extra=0.;sync();t=time.perf_counter()
     if name=="tsa64":u,status=solve_from_initial(high.A_op,b,pre,None,maxiter=100);status.update(classical_calls=status["iterations"],injection_calls=0)
     elif name=="tsa64_aug":u,status=augmented_fgmres(high.A_op,b,pre,None,maxiter=100)
     elif name=="oracle_C8":
      def inject(x,r):
       f=NoTangentFrame(low);_,_,p,_=f.visible(low,r)
       return f.decode((label*p).reshape(1,12,128,17),8)
      u,status=augmented_fgmres(high.A_op,b,pre,None,inject=inject,inject_after=1,maxiter=100)
     else:
      scheme,kind=name.split(":");g=actions[scheme];factor=factors[scheme];extra=gsetup[scheme]
      if kind=="standalone":u,status=solve_from_initial(high.A_op,b,g,None,maxiter=100);status.update(classical_calls=0,g_calls=status["iterations"],injection_calls=0)
      else:
       if kind=="inject":inject=lambda x,r:g(r)
       elif kind=="born":
        def inject(x,r):
         first=g(r)
         return first-g(factor.perturbation(first))
       else:inject=lambda x,r:solve_from_initial(high.A_op,r,g,None,tolerance=1e-30,maxiter=2)[0]
       u,status=augmented_fgmres(high.A_op,b,pre,None,inject=inject,inject_after=1,maxiter=100)
     sync();solve=time.perf_counter()-t
     sync();t=time.perf_counter();res=float((b-high.A_op(u)).norm()/bn);d=res*bn/margin;den=float(u.norm())-d
     bound=d/den if den>0 else float("inf");diff=float((u-teacher).norm()/teacher.norm())
     ok=bool(torch.isfinite(u).all()) and not status["failed"] and res<=1e-5 and bound<=1e-3 and diff<=1e-4
     sync();check=time.perf_counter()-t
     setup=common+extra+(0. if name.endswith(":standalone") else tsa_setup);status.pop("trace",None)
     rows.append(dict(source_id=source_id,method=name,repeat=repeat,accepted=ok,accepted_seconds=solve+check,
      Q1_seconds=setup+solve+check,shared_setup_seconds=setup,solve_seconds=solve,acceptance_seconds=check,
      true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff,status=status))
    save()
   print("HARDGATE",source_id,"rows",len(rows),flush=True)
  settings["complete"]=True
 except BaseException as exc:settings["error"]=repr(exc);raise
 finally:
  settings.update(wall_seconds=time.perf_counter()-start,rows=len(rows),failures=sum(not x["accepted"] for x in rows),
   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());save()
if __name__=="__main__":main()
