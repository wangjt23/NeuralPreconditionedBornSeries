import sys,os,time,json,hashlib,traceback
from pathlib import Path
import torch
ROOT=Path("/data/wjt/NeuralPreconditionedBornSeries");OLD=ROOT/"rte/workstation_runs/rte_mcmg_born_20261114_dot_v1";FIT=ROOT/"rte/workstation_runs/rte_mcmg_fit8_20261114_dot_v1";OUT=Path(__file__).parent
sys.path.insert(0,str(OLD));from train import Cycle,ExperimentConfig,source_from_data,setup_batch,make_operator,dataset_fingerprint
from rte.models import PeriodicRTE2D
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace
from rte.collision_transport import CollisionTransportPreconditioner

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4);torch.manual_seed(145401)
 data=torch.load(OLD/"data.pt",weights_only=True)["train"];cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes="mixed");q=source_from_data(cfg,data,torch.tensor([0]),"cuda");op=make_operator(cfg,data["sigma_t"][:1],data["sigma_s"][:1],q,"cuda")
 paths={"before":OLD/"physical_2048.pt","after":FIT/"fit_4096.pt","CT4":ROOT/"rte/workstation_runs/ct4_structure_control_n128_seed402_20260913/best.pt"}
 models={n:Cycle(op).cuda().eval() for n in ["before","after"]};models["CT4"]=CollisionTransportPreconditioner(op,depth=4,learned=True,width=12).cuda().eval()
 for n,m in models.items():m.load_state_dict(torch.load(paths[n],weights_only=True)["model"])
 report=dict(pid=os.getpid(),complete=False,source_hash=sha(__file__),checkpoint_hashes={n:sha(p) for n,p in paths.items()},train_fingerprint=dataset_fingerprint(data),train_ids=list(range(8)),budget=dict(maxiter=100,restart=20,max_restarts=5,tol=1e-5),scope="TRAIN-only diagnostic, actual normalized Arnoldi input v, NOT current true residual; no timing claim; FP64 physical operator and real-stacked FGMRES; stored v/z/Az complex64 with quantization receipts",trajectories=[],cross=[],subspaces=[],failures=[])
 def save(): (OUT/"results.json").write_text(json.dumps(report,indent=2))
 save()
 with torch.no_grad():
  for i in range(8):
   q=setup_batch(cfg,op,data,torch.tensor([i]),"cuda");a=PeriodicRTE2D(shape=(64,64),dx=1/64,n_angles=12,anisotropy_g=.65,eta_scale=0.,fdtype=torch.float64,device="cuda").setup(sigma_t=data["sigma_t"][i:i+1].double().cuda(),sigma_s=data["sigma_s"][i:i+1].double().cuda(),source=q.to(torch.complex128))
   for n in ["before","after"]:models[n].cache=None;models[n].key=None
   for origin in ["before","after"]:
    capture=[]
    def wrapped(v):
     z=models[origin](v.to(torch.complex64)).to(torch.complex128);az=a.A_op(z);k=len(capture);capture.append(dict(call=k+1,restart_index=k//20,inner_index=k%20+1,v=v.clone(),z=z.clone(),Az=az.clone()));return z
    try:
     x,h,_,calls=fgmres_real_stacked_with_trace(a.A_op,a.source,wrapped,tol=1e-5,maxiter=100,restart=20,max_restarts=5)
     residual=float((a.source-a.A_op(x)).norm()/a.source.norm());file=OUT/f"train{i:02d}_{origin}_arnoldi.pt";torch.save([dict(call=c["call"],restart_index=c["restart_index"],inner_index=c["inner_index"],**{k:c[k].to(torch.complex64).cpu() for k in ["v","z","Az"]}) for c in capture],file)
     quant={k:max(float((c[k]-c[k].to(torch.complex64).to(torch.complex128)).norm()/c[k].norm().clamp_min(1e-30)) for c in capture) for k in ["v","z","Az"]}
     report["trajectories"].append(dict(index=i,origin=origin,calls=calls[-1],steps=len(h)-1,true_residual=residual,accepted=residual<=1e-5,history=h,file=file.name,sha256=sha(file),quantization_max_relative=quant))
     for target,m in models.items():
      errors=[]
      for c in capture:
       v=c["v"];z=m(v.to(torch.complex64)).to(torch.complex128);e=v-a.A_op(z);loss=float(e.abs().square().sum()/v.abs().square().sum());errors.append(e.flatten());report["cross"].append(dict(index=i,origin=origin,target=target,call=c["call"],restart_index=c["restart_index"],inner_index=c["inner_index"],squared_gain=loss,v_norm=float(v.norm())))
      # Finite real-span upper gain, not a global operator norm; up to 8 actual Arnoldi vectors per contiguous phase.
      for start in range(0,len(capture),8):
       stop=min(start+8,len(capture));V=torch.stack([c["v"].flatten() for c in capture[start:stop]]);E=torch.stack(errors[start:stop]);G=(V.conj()@V.T).real;H=(E.conj()@E.T).real;d,U=torch.linalg.eigh(G);keep=d>1e-10*d.max();W=U[:,keep]/d[keep].sqrt()[None];lam=torch.linalg.eigvalsh(W.T@H@W).max();report["subspaces"].append(dict(index=i,origin=origin,target=target,first_call=start+1,last_call=stop,rank=int(keep.sum()),worst_squared_gain=float(lam),scope="real span of these actual inputs; may straddle restart; not full-space norm"))
     save();print("DONE",i,origin,len(capture),residual,flush=True)
    except Exception as exc:
     report["failures"].append(dict(index=i,origin=origin,error=repr(exc),trace=traceback.format_exc()));save();print("FAIL",i,origin,repr(exc),flush=True)
 report["complete"]=True;save();print("COMPLETE",flush=True)
if __name__=="__main__":main()
