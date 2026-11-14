"""Small-data predictability and matched real-POD classical controls.
Kernel ridge is fitted on96 existing physical teachers. Hyperparameters selected
on32 disjoint training-family IDs; final16 reused development is separate.
No neural optimization and no reserved confirmation data.
"""
import os,sys,time,json,hashlib,random,gc,math
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from prepare_predictability import ObservableFrame,competitors,sync,safe
from prepare_predictability import material_operator,build_base,ExperimentConfig,DirectSegmentedTSA,augmented_fgmres,solve_from_initial
from rte.augmented_fgmres import projected_direction_loss

class Frame(ObservableFrame):
 def visible(self,low,r):
  c=self.encode(r,16)
  flat=c.flatten();largest=flat[flat.abs().argmax()]
  phase=largest/largest.abs().clamp_min(1e-30)
  scale=c.abs().square().mean().sqrt().clamp_min(1e-30)
  return self.material_features(low).float(),(c/(phase*scale).clamp_min(1e-30) if False else c/(phase*scale)).to(torch.complex64).flatten(),phase,scale

def dist_real(x,y):
 return ((x*x).sum(-1)[:,None]+(y*y).sum(-1)[None]-2*x@y.T).clamp_min(0)/x.shape[-1]
def dist_complex(x,y):
 return (x.abs().square().sum(-1)[:,None]+y.abs().square().sum(-1)[None]-2*(x@y.conj().T).real).clamp_min(0)/x.shape[-1]

class KernelModel:
 def __init__(self,blob):
  self.b=blob;self.U=blob["basis"].to(torch.complex128)
 def coefficient(self,mat,features,length,ridge):
  b=self.b;mn=(mat-b["material_mean"])/b["material_std"]
  dm=dist_real(mn[None],b["material_fit"])/b["median_material"]
  dr=dist_complex(features[None],b["residual_fit"])/b["median_residual"]
  kernel=torch.exp(-(dm+dr)/(4*length*length)).double()
  return kernel@b["weights"][f"{length}_{ridge}"]
 def direction(self,low,high,r,kind,state,known_y=None):
  frame=Frame(low);mat,features,phase,scale=frame.visible(low,r)
  if not bool(scale>1e-25):return torch.zeros_like(r)
  if kind.startswith("krr"):
   _,rank,length,ridge=kind.split(":");rank=int(rank)
   coef=self.coefficient(mat,features,float(length),float(ridge))[0,:rank]
   c=(coef.to(self.U.dtype)@self.U[:rank]).reshape(1,12,128,17)
   return frame.decode(c*phase,8)
  if kind.startswith("oracle"):
   rank=int(kind.split(":")[1])
   if rank==0:c=known_y.to(torch.complex128)
   else:
    coef=(known_y.to(torch.complex128)[None]*self.U[:rank].conj()).real.sum(-1)
    c=coef.to(self.U.dtype)@self.U[:rank]
   return frame.decode(c.reshape(1,12,128,17)*phase,8)
  rank=int(kind.split(":")[1])
  z=frame.decode((self.U[:rank]*phase).reshape(rank,12,128,17),8)
  w=high.A_op(z)
  q=state["image_basis"][:,0]
  dots=(q[:,None].conj()*w[None]).real.flatten(2).sum(2)
  w=w-torch.einsum("kr,klxy->rlxy",dots.to(q.dtype),q)
  wr=torch.view_as_real(w).flatten(1);norms=wr.norm(dim=1).clamp_min(1e-30)
  matrix=(wr/norms[:,None]).T.contiguous()
  qr,rr=torch.linalg.qr(matrix,mode="reduced")
  uu,ss,vh=torch.linalg.svd(rr,full_matrices=False)
  cutoff=1e-10*ss.max()
  inverse=torch.where(ss>cutoff,ss.reciprocal(),torch.zeros_like(ss))
  rhs=torch.view_as_real(r).flatten()
  coefficients=(vh.T@(inverse*(uu.T@(qr.T@rhs))))/norms
  return (z*coefficients[:,None,None,None]).sum(0,keepdim=True)

def convert_items(raw,case_records):
 out={}
 for it in raw:
  record=case_records[(it["split"],it["index"])]
  phase_old=complex(*record["phase"])
  c=it["residual_features"].to(torch.complex128)*phase_old*record["residual_rms"]
  maximum=c[c.abs().argmax()];phase=maximum/maximum.abs().clamp_min(1e-30)
  scale=c.abs().square().mean().sqrt().clamp_min(1e-30)
  f=(c/(phase*scale)).to(torch.complex64)
  y=(it["target"].to(torch.complex128)*phase_old/phase).to(torch.complex64)
  out[(it["split"],it["index"])]=dict(material=it["material"],features=f,target=y)
 return out

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.manual_seed(20261109)
 if competitors():raise RuntimeError("GPU competitor before admission")
 prior=json.loads((HERE/"prepare_01/settings.json").read_text());assert prior["complete"]
 out=HERE/"matched_01";out.mkdir(exist_ok=False);began=time.perf_counter();rows=[];contexts=[]
 settings=dict(complete=False,pid=os.getpid(),neural_training=False,learning="Real kernel ridge on real-POD coefficients",
  source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  feature_protocol="V2: phase and RMS from C16r only; old saved features/labels algebraically regauged. No full residual phase/RMS sidechannel.",
  split="96 fit physical IDs;32 internal selection;16 reused development",heldout9626=False,
  ranks=[32,96],lengths=[.5,1.,2.],ridges=[.001,.01],wall_limit=900,
  classical="Same fitted real-POD basis and decoded phase; original A64 minimum residual using QR plus small-R SVD. Existing AZ retained.",
  selection="Frozen previously selected kernel parameters; no fitting or model selection. Add matched ranks32/96 and oracle96 controls.")
 def save():
  for k,v in [("settings",settings),("rows",rows),("contexts",contexts)]:(out/(k+".json")).write_text(json.dumps(safe(v),indent=2)+"\n")
 save();torch.cuda.reset_peak_memory_stats()
 try:
  prepared=torch.load(HERE/"prepare_01/features.pt",map_location="cpu",weights_only=True)
  cr={(r["split"],r["index"]):r for r in json.loads((HERE/"prepare_01/cases.json").read_text())}
  items=convert_items(prepared["items"],cr);fit=prepared["fit_ids"];internal=prepared["internal_ids"]
  settings["fit_ids"]=fit;settings["internal_ids"]=internal
  blob=torch.load(HERE/"predict_01/model.pt",map_location="cuda",weights_only=True)
  U=blob["basis"];settings["actual_rank"]=len(U)
  settings["frozen_model_sha256"]=hashlib.sha256((HERE/"predict_01/model.pt").read_bytes()).hexdigest()
  model=KernelModel(blob);settings["model_file_bytes"]=(HERE/"predict_01/model.pt").stat().st_size
  ranks=[min(r,len(U)) for r in settings["ranks"]]
  candidates=[f"krr:{rank}:{length}:{ridge}" for rank in ranks for length in settings["lengths"] for ridge in settings["ridges"]]
  selected=json.loads((HERE/"predict_01/settings.json").read_text())["selected"];settings["selected"]=selected;save()
  for split,ids in [("validation",list(range(16)))]:
   saved=torch.load(ROOT/f"rte/workstation_runs/tail_multiresolution_bank_20260926/{split}.pt",map_location="cpu",weights_only=True)
   data=saved["data"];bank=saved["bank"]
   cfg=ExperimentConfig(size=128,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes="mixed")
   methods=["TSA16_native","TSA16_augmented",selected,"krr:32:"+":".join(selected.split(":")[2:]),"oracle:0","oracle:32","oracle:96","rom:32","rom:96"]
   repeats=1 if split=="train" else 3
   for index in ids:
    if competitors():raise RuntimeError("New GPU competitor")
    if time.perf_counter()-began>settings["wall_limit"]:raise RuntimeError("Prediction study wall limit")
    sync();t=time.perf_counter();low,high=material_operator(cfg,data,torch.tensor([index]),.1,"cuda")
    phase=high.phase_matrix.double()*high.weights.double()[None];mu=torch.linalg.eigvalsh((phase+phase.T)/2).max()
    margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0
    base=build_base(low,"tsa16");tsa=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False);sync();setup=time.perf_counter()-t
    pre=lambda r:tsa(r.to(torch.complex64)).to(torch.complex128)
    prefixes=[];augmented_fgmres(high.A_op,high.source,pre,None,maxiter=1,inject_after=1,capture_prefix=prefixes.append)
    p=prefixes[0];teacher=bank["target"][index:index+1].cuda();assert float((high.A_op(teacher)-high.source).norm()/high.source.norm())<2e-9
    f=Frame(low);mat,feat,phi,scale=f.visible(low,p["residual"])
    label=f.encode((teacher-p["solution"])/phi,8).flatten();label=(label/label.norm()).to(torch.complex64)
    saved_label=items[(split,index)]["target"].cuda()
    conversion_error=float((saved_label-label).norm());assert conversion_error<2e-6
    feature_error=float((feat-items[(split,index)]["features"].cuda()).norm()/feat.norm());assert feature_error<2e-6
    _,featcut,phicut,scalecut=f.visible(low,f.project(p["residual"],16))
    visible_error=float((feat-featcut).norm()/feat.norm())
    ctx=dict(split=split,index=index,setup_seconds=setup,label_regauging_error=conversion_error,
     feature_regauging_error=feature_error,v2_full_visible_feature_counterfactual_difference=visible_error,
     phase_counterfactual_difference=float((phi-phicut).abs()),pod_projection_errors={})
    for rank in [8,16,32,96]:
     rank=min(rank,len(model.U));u=model.U[:rank]
     c=(label.to(torch.complex128)[None]*u.conj()).real.sum(-1).to(u.dtype)@u
     ctx["pod_projection_errors"][str(rank)]=float((c-label).norm())
    contexts.append(ctx);baseline=None;bn=float(high.source.norm());rng=random.Random(811199+index)
    for repeat in range(-1,repeats):
     order=methods.copy();rng.shuffle(order)
     if repeat<0:order.remove("TSA16_native");order.insert(0,"TSA16_native")
     for name in order:
      state={}
      sync();t=time.perf_counter()
      if name=="TSA16_native":u,st=solve_from_initial(high.A_op,high.source,pre,None,maxiter=100);st.update(classical_calls=st["iterations"],injection_calls=0)
      elif name=="TSA16_augmented":u,st=augmented_fgmres(high.A_op,high.source,pre,None,maxiter=100)
      else:
       def inj(x,r):
        if repeat<0:assert torch.equal(x,p["solution"]) and torch.equal(r,p["residual"])
        return model.direction(low,high,r,name,state,label)
       u,st=augmented_fgmres(high.A_op,high.source,pre,None,inject=inj,inject_after=1,maxiter=100,
        capture_prefix=state.update if name.startswith("rom") else None)
      sync();solver=time.perf_counter()-t;st.pop("trace",None)
      if baseline is None:assert not st["failed"];baseline=u.clone()
      sync();t=time.perf_counter();res=float((high.source-high.A_op(u)).norm()/bn);delta=res*bn/margin;den=float(u.norm())-delta
      bound=delta/den if den>0 else float("inf");diff=float((u-baseline).norm()/baseline.norm())
      accepted=bool(torch.isfinite(u).all()) and not st["failed"] and res<=1e-5 and bound<=1e-3 and diff<=1e-4
      sync();check=time.perf_counter()-t
      rows.append(dict(split=split,index=index,method=name,repeat=repeat,accepted_seconds=solver+check,Q1_seconds=setup+solver+check,
       setup_seconds=setup,solver_seconds=solver,acceptance_seconds=check,status=st,accepted=accepted,
       true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff))
     save()
    print("PREDICT",split,index,"rows",len(rows),flush=True)
    del low,high,base,tsa,teacher,p,prefixes,f,mat,feat,label,baseline;gc.collect();sync();torch.cuda.empty_cache()
   if split=="train":
    scores=[]
    for name in candidates:
     z=[r for r in rows if r["split"]=="train" and r["method"]==name and r["repeat"]>=0]
     scores.append(dict(method=name,failures=sum(not r["accepted"] for r in z),
      mean_seconds=sum(r["accepted_seconds"] for r in z)/len(z),
      mean_classical_calls=sum(r["status"]["classical_calls"] for r in z)/len(z)))
    selected=min(scores,key=lambda x:(x["failures"],x["mean_seconds"],x["mean_classical_calls"]))["method"]
    settings["internal_selection"]=scores;settings["selected"]=selected;save()
   del saved,data,bank;gc.collect()
  settings["complete"]=True
 except BaseException as exc:settings["error"]=repr(exc);raise
 finally:
  settings.update(wall_seconds=time.perf_counter()-began,rows=len(rows),failures=sum(not r["accepted"] for r in rows),
   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());save()
if __name__=="__main__":main()
