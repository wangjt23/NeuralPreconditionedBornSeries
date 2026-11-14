import sys,json,time,hashlib,argparse,torch
from pathlib import Path
ROOT=Path("/data/wjt/NeuralPreconditionedBornSeries");OLD=ROOT/"rte/workstation_runs/rte_mcmg_born_20261114_dot_v1";BANK=ROOT/"rte/workstation_runs/rte_arnoldi_audit_20261114_dot_v1";OUT=Path(__file__).parent;sys.path.insert(0,str(OLD));from train import Cycle,ExperimentConfig,source_from_data,make_operator,setup_batch
from rte.models import PeriodicRTE2D
from rte.collision_transport import CollisionTransportPreconditioner
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evaluate(checkpoint,name,kind="cycle"):
 output=OUT/(name+".json")
 if output.exists():raise RuntimeError("refuse overwrite "+str(output))
 d=torch.load(OLD/"data.pt",weights_only=True)["train"];cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes="mixed");q=source_from_data(cfg,d,torch.tensor([0]),"cuda");op=make_operator(cfg,d["sigma_t"][:1],d["sigma_s"][:1],q,"cuda");model=Cycle(op).cuda().eval() if kind=="cycle" else CollisionTransportPreconditioner(op,depth=4,learned=True,width=12).cuda().eval();model.load_state_dict(torch.load(checkpoint,weights_only=True)["model"])
 report=dict(complete=False,name=name,checkpoint=str(checkpoint),checkpoint_hash=sha(checkpoint),source_hash=sha(__file__),scope="only fixed saved BEFORE/AFTER Arnoldi inputs, not current-model deployment; all methods same stored complex64 vectors promoted double; no power/search/training; identical per-call and per-material weighting; finite 8-vector real spans retained",rows=[],subspaces=[],bank_hashes={})
 with torch.no_grad():
  for i in range(8):
   q=setup_batch(cfg,op,d,torch.tensor([i]),"cuda");a=PeriodicRTE2D(shape=(64,64),dx=1/64,n_angles=12,anisotropy_g=.65,eta_scale=0.,fdtype=torch.float64,device="cuda").setup(sigma_t=d["sigma_t"][i:i+1].double().cuda(),sigma_s=d["sigma_s"][i:i+1].double().cuda(),source=q.to(torch.complex128))
   if kind=="cycle":model.cache=None;model.key=None
   for origin in ["before","after"]:
    path=BANK/f"train{i:02d}_{origin}_arnoldi.pt";bank=torch.load(path,weights_only=True);report["bank_hashes"][path.name]=sha(path);vectors=[];errors=[]
    for c in bank:
     v=c["v"].cuda();z=model(v);v=v.to(torch.complex128);e=v-a.A_op(z.to(torch.complex128));gain=float(e.norm()/v.norm());vectors.append(v.flatten());errors.append(e.flatten());report["rows"].append(dict(material=i,origin=origin,call=c["call"],restart_index=c["restart_index"],inner_index=c["inner_index"],relative_norm=gain,squared_gain=gain*gain))
    for start in range(0,len(vectors),8):
     stop=min(start+8,len(vectors));V=torch.stack(vectors[start:stop]);E=torch.stack(errors[start:stop]);G=(V.conj()@V.T).real;H=(E.conj()@E.T).real;dd,U=torch.linalg.eigh(G);keep=dd>1e-10*dd.max();W=U[:,keep]/dd[keep].sqrt()[None];lam=float(torch.linalg.eigvalsh(W.T@H@W).max());report["subspaces"].append(dict(material=i,origin=origin,first_call=start+1,last_call=stop,rank=int(keep.sum()),worst_squared_gain=lam,scope="same predetermined contiguous8 real span, may straddle restart; NOT global operator norm"))
   (OUT/(name+".partial.json")).write_text(json.dumps(report,indent=2));print(name,i,flush=True)
 report["complete"]=True;output.write_text(json.dumps(report,indent=2));return output
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--name",required=True);p.add_argument("--kind",default="cycle");a=p.parse_args();torch.set_num_threads(2);print(evaluate(Path(a.checkpoint),a.name,a.kind))
