"""Structured hard-scene screen: layer count x scattering scale, fixed absorption.
Same four actual RHS arrays and original fine Fourier contract throughout.
No teachers or learner fitting. Best complete cost among existing TSA/HSS paths.
"""
import os,sys,time,json,hashlib,random,gc,math
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"rte/workstation_runs/layered_background_20261108_dot_v1"))
from prepare_predictability import competitors,sync,safe,DirectSegmentedTSA,build_base,solve_from_initial
from layered_inverse import CapturedAction
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy

def medium(n,periods,beta,device):
 x=torch.arange(n,device=device,dtype=torch.float32)/n
 xx=x[:,None];yy=x[None,:]
 transition=2/128
 delta=math.sin(math.pi*periods*transition)/math.atanh(.8)
 h=.5*(1+torch.tanh(torch.cos(2*math.pi*periods*xx)/delta)/math.tanh(1/delta))
 dx=(xx-.34).abs();dx=torch.minimum(dx,1-dx)
 dy=(yy-.60).abs();dy=torch.minimum(dy,1-dy)
 inclusion=3*torch.exp(-(dx.square()+dy.square())/.08**2)
 T=2+198*h+inclusion
 scattering=beta*(T-.05)
 total=.05+scattering
 return total[None,None],scattering[None,None],dict(periods=periods,layers=2*periods,beta=beta,
  transition_physical_width=transition,low_plateau_total=.05+beta*1.95,high_plateau_total_without_inclusion=.05+beta*199.95,
  inclusion_amplitude_before_scaling=3.,inclusion_width=.08)

@torch.no_grad()
def main():
 torch.set_num_threads(1)
 if competitors():raise RuntimeError("GPU competitor before admission")
 prior=json.loads((HERE/"matched_01/settings.json").read_text());assert prior["complete"]
 out=HERE/"depth_01";out.mkdir(exist_ok=False);began=time.perf_counter();rows=[];contexts=[]
 settings=dict(complete=False,pid=os.getpid(),training=False,new_teachers=False,heldout9626=False,grid=128,
  conditions=[dict(periods=p,beta=b) for p in [1,4] for b in [1.,10/3]],
  source_ids=[0,4,7,14],methods=["tsa16","tsa32","tsa64","tsa128"],
  repeats=3,maxiter=100,wall_limit=600,
  contract="Same original stored Fourier A64; L12 g=.65, residual1e-5, evaluated bound1e-3, paired field1e-4.",
  design="2x2:2/8 layers x scattering scale1 or10/3; effective absorption fixed .05. Same smooth weak inclusion. Four identical RHS arrays per condition. Depth follow-up because32 won endpoint of initial8/16/32 grid.",
  cost="Q1: physical+acceptance+method setup each RHS. Actual four-RHS batch: setup once plus sum of four solve/acceptance medians.",
  source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 def save():
  for k,v in [("settings",settings),("contexts",contexts),("rows",rows)]:(out/(k+".json")).write_text(json.dumps(safe(v),indent=2)+"\n")
 saved=torch.load(ROOT/"rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt",map_location="cpu",weights_only=True)
 rhs=[saved["bank"]["q"][i:i+1].cuda().to(torch.complex128) for i in settings["source_ids"]]
 settings["rhs_sha256"]={str(i):hashlib.sha256(saved["bank"]["q"][i:i+1].numpy().tobytes()).hexdigest() for i in settings["source_ids"]}
 del saved;save();torch.cuda.reset_peak_memory_stats()
 try:
  for spec in settings["conditions"]:
   if competitors():raise RuntimeError("New GPU competitor")
   sync();t=time.perf_counter();total,scattering,info=medium(128,spec["periods"],spec["beta"],"cuda")
   low=PeriodicRTE2D(shape=(128,128),dx=1/128,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device="cuda")
   low.setup(total,scattering,rhs[0].to(torch.complex64));high=high_precision_copy(low)
   phase=high.phase_matrix.double()*high.weights.double()[None]
   mu=torch.linalg.eigvalsh((phase+phase.T)/2).max();margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0
   sync();common=time.perf_counter()-t
   p1=float(torch.fft.fft(phase[0])[1].real)
   info.update(condition=f"p{spec['periods']}_b{spec['beta']:.6g}",physical_acceptance_setup=common,
    sigma_t_min=float(total.min()),sigma_t_max=float(total.max()),sigma_t_mean=float(total.mean()),
    absorption_min=float((high.sigma_t-high.sigma_s).min()),absorption_max=float((high.sigma_t-high.sigma_s).max()),
    phase_max_eigenvalue=float(mu),coercivity_margin=margin,discrete_p1_eigenvalue=p1,
    transport_optical_low=(.05+(1-p1)*spec["beta"]*1.95)/(2*spec["periods"]),
    transport_optical_high=(.05+(1-p1)*spec["beta"]*199.95)/(2*spec["periods"]))
   actions={};setups={};objects=[]
   for depth in [16,32,64,128]:
    sync();t=time.perf_counter();b=build_base(low,f"tsa{depth}")
    wrap=DirectSegmentedTSA(b,low.source,chunk=1,capture_first=False);sync();used=time.perf_counter()-t
    actions[f"tsa{depth}"]=lambda r,w=wrap:w(r.to(torch.complex64)).to(torch.complex128)
    setups[f"tsa{depth}"]=used;objects.extend([b,wrap])
   info["method_setup"]=setups
   contexts.append(info);save()
   for source_id,b in zip(settings["source_ids"],rhs):
    if competitors():raise RuntimeError("New GPU competitor")
    reference=None;bn=float(b.norm());rng=random.Random(811301+source_id+spec["periods"])
    for repeat in range(-1,3):
     order=list(actions);rng.shuffle(order)
     if repeat<0:order.remove("tsa16");order.insert(0,"tsa16")
     for name in order:
      if time.perf_counter()-began>settings["wall_limit"]:raise RuntimeError("Screen wall limit")
      sync();t=time.perf_counter();error=None
      try:
       u,st=solve_from_initial(high.A_op,b,actions[name],None,maxiter=100)
       sync();solver=time.perf_counter()-t
       sync();t=time.perf_counter();res=float((b-high.A_op(u)).norm()/bn);d=res*bn/margin;den=float(u.norm())-d
       bound=d/den if den>0 else float("inf")
       good=bool(torch.isfinite(u).all()) and not st["failed"] and res<=1e-5 and bound<=1e-3
       if reference is None and good:reference=u.clone()
       diff=float((u-reference).norm()/reference.norm()) if reference is not None else float("inf")
       accepted=good and diff<=1e-4;sync();check=time.perf_counter()-t
      except RuntimeError as exc:
       sync();solver=time.perf_counter()-t;check=0.;error=repr(exc)
       if "CUDA" in str(exc) or "cuda" in str(exc):raise
       st=dict(failed=1,iterations=None);res=bound=diff=float("inf");accepted=False
      depth=16 if name.startswith("hss") else int(name[3:])
      rows.append(dict(condition=info["condition"],source_id=source_id,method=name,repeat=repeat,
       accepted_seconds=solver+check,Q1_seconds=common+setups[name]+solver+check,
       shared_setup_seconds=common+setups[name],status=st,accepted=accepted,true_residual=res,
       evaluated_relative_bound=bound,relative_field_difference=diff,
       internal_sweeps=depth*st["iterations"] if st.get("iterations") is not None else None,error=error))
     save()
    print("SCREEN",info["condition"],source_id,"rows",len(rows),flush=True)
    del reference
   del actions,objects,low,high;gc.collect();sync();torch.cuda.empty_cache()
  settings["complete"]=True
 except BaseException as exc:settings["error"]=repr(exc);raise
 finally:
  settings.update(wall_seconds=time.perf_counter()-began,rows=len(rows),failures=sum(not r["accepted"] for r in rows),
   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());save()
if __name__=="__main__":main()
