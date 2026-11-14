import os,sys,time,json,random,traceback,gc
from pathlib import Path
import torch
from torch.nn import functional as F
from train import Model,digest,save,sync,HERE,ROOT,OLD
sys.path.insert(0,str(OLD))
from fit_and_evaluate import Frame
from prepare_predictability import material_operator,build_base,ExperimentConfig,DirectSegmentedTSA,augmented_fgmres,solve_from_initial
@torch.no_grad()
def features(low,r):
 frame=Frame(low);_,f,phase,scale=frame.visible(low,r)
 material=frame.canonical(torch.cat([low.sigma_t.log(),(low.sigma_t-low.sigma_s).log()],1))
 material=F.avg_pool2d(material,(1,2))/4
 c=f.reshape(1,12,128,33);z=c.new_zeros(1,12,128,64);z[...,:17]=c[...,:17];z[...,-16:]=c[...,-16:]
 q=torch.fft.ifft(z,dim=-1,norm="ortho");q=q/q.abs().square().mean().sqrt().clamp_min(1e-12)
 return torch.cat([material,q.real,q.imag],1),frame,phase
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(2)
 while True:
  st=json.loads((HERE/"status.json").read_text())
  if st.get("error"):raise RuntimeError("training failure; do not silently evaluate partial")
  if st["complete"]:break
  time.sleep(30)
 out=HERE/"evaluation";out.mkdir(exist_ok=False)
 models={};load_start=time.perf_counter()
 for seed in [2026111401,2026111402]:
  m=Model().cuda();m.load_state_dict(torch.load(HERE/f"seed{seed}_8192.pt",map_location="cuda",weights_only=True)["model"]);m.eval();models[f"learned_{seed}"]=m
 sync();start=time.perf_counter();rows=[]
 result=dict(complete=False,source_hash=digest(__file__),training_source_hash=digest(HERE/"train.py"),fixed_updates=8192,model_loading_seconds=start-load_start,rows=rows,heldout9626=False,scope="32 TRAIN-family unseen-by-optimizer IDs and16 repeatedly reused DEV; no new independent test. Same fixed endpoint; no selection")
 def persist():
  p=out/"results.tmp";p.write_text(json.dumps(result,indent=2));os.replace(p,out/"results.json")
 persist()
 try:
  for split,ids in [("train",st["internal_ids"]),("validation",list(range(16)))]:
   parent=torch.load(ROOT/f"rte/workstation_runs/tail_multiresolution_bank_20260926/{split}.pt",map_location="cpu",weights_only=True);cfg=ExperimentConfig(size=128,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes="mixed")
   for index in ids:
    sync();t=time.perf_counter();low,high=material_operator(cfg,parent["data"],torch.tensor([index]),.1,"cuda")
    phase=high.phase_matrix.double()*high.weights.double()[None];mu=torch.linalg.eigvalsh((phase+phase.T)/2).max();margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0
    base=build_base(low,"tsa16");tsa=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False);sync();setup=time.perf_counter()-t
    pre=lambda r:tsa(r.to(torch.complex64)).to(torch.complex128)
    baseline=None;bn=float(high.source.norm());rng=random.Random(141199+index)
    methods=["TSA16_native","TSA16_augmented"]+list(models)
    for repeat in [-1,0,1,2]:
     order=methods.copy();rng.shuffle(order)
     if repeat==-1:order.remove("TSA16_native");order.insert(0,"TSA16_native")
     for name in order:
      injected=[]
      def inject(x,r):
       tt=time.perf_counter();inp,frame,ph=features(low,r);pred=models[name](inp).reshape(1,12,128,17);z=frame.decode(pred.to(torch.complex128)*ph,8);sync();injected.append(time.perf_counter()-tt);return z
      sync();t=time.perf_counter()
      if name=="TSA16_native":
       u,status=solve_from_initial(high.A_op,high.source,pre,None,maxiter=100);status.update(classical_calls=status["iterations"],injection_calls=0)
      elif name=="TSA16_augmented":u,status=augmented_fgmres(high.A_op,high.source,pre,None,maxiter=100)
      else:u,status=augmented_fgmres(high.A_op,high.source,pre,None,inject=inject,inject_after=1,maxiter=100)
      sync();elapsed=time.perf_counter()-t;status.pop("trace",None)
      if baseline is None:
       assert not status["failed"];baseline=u.clone()
      t=time.perf_counter();res=float((high.source-high.A_op(u)).norm()/bn);delta=res*bn/margin;den=float(u.norm())-delta;bound=delta/den if den>0 else float("inf");diff=float((u-baseline).norm()/baseline.norm())
      passed=bool(torch.isfinite(u).all()) and not status["failed"] and res<=1e-5 and bound<=1e-3 and diff<=1e-4
      sync();check=time.perf_counter()-t
      rows.append(dict(split=split,index=index,method=name,repeat=repeat,setup_seconds=setup,solver_seconds=elapsed,acceptance_seconds=check,accepted_seconds=elapsed+check,Q1_seconds=setup+elapsed+check,proposal_preprocess_infer_decode_seconds=injected,status=status,accepted=passed,true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff))
     persist()
    print("EVAL",split,index,len(rows),flush=True)
    del low,high,base,tsa,baseline;gc.collect();torch.cuda.empty_cache()
   del parent
  result["complete"]=True
 except BaseException as e:result.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:result.update(wall_seconds=time.perf_counter()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),failures=sum(not x["accepted"] for x in rows));persist()
if __name__=="__main__":main()

