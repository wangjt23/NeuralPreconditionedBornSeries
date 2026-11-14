"""Deterministic reconstruction, NOT a previously saved input panel.
Require all four O/K endpoints before GPU work; qualify against EVERY original
before/after/CT4/G0 scalar before freezing. Original results remain untouched.
"""
import sys,json,hashlib,argparse,torch,math
from pathlib import Path
ROOT=Path("/data/wjt/NeuralPreconditionedBornSeries");OLD=ROOT/"rte/workstation_runs/rte_mcmg_born_20261114_dot_v1";FIT=ROOT/"rte/workstation_runs/rte_mcmg_fit8_20261114_dot_v1";TRAIN=ROOT/"rte/workstation_runs/rte_krylov_window_20261114_dot_v1";OUT=Path(__file__).parent;sys.path.insert(0,str(OLD));from train import Cycle,ExperimentConfig,source_from_data,make_operator,setup_batch
from rte.collision_transport import CollisionTransportPreconditioner
from rte.models import PeriodicRTE2D
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 endpoints=[(f"{a}_{s}",TRAIN/f"v2_{a}_{s}/checkpoint_2048.pt") for s in [146501,146502] for a in ["O","K"]]
 for n,p in endpoints:
  status=json.load(open(p.parent/"status.json"))
  if not status.get("complete") or not p.exists():raise RuntimeError("GPU gate: all four final endpoints must exist and complete")
  torch.load(p,map_location="cpu",weights_only=True)
 if (OUT/"reconstruction_audit.json").exists():raise RuntimeError("refuse rerun/overwrite; inspect prior reconstruction audit")
 # Identical seed and model construction/order to original gaussian_probe_audit.py.
 torch.set_num_threads(4);torch.manual_seed(146401);d=torch.load(OLD/"data.pt",weights_only=True)["train"];cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes="mixed");q=source_from_data(cfg,d,torch.tensor([0]),"cuda");op=make_operator(cfg,d["sigma_t"][:1],d["sigma_s"][:1],q,"cuda");models={}
 for name,ck in [("before",OLD/"physical_2048.pt"),("after",FIT/"fit_4096.pt")]:
  m=Cycle(op).cuda().eval();m.load_state_dict(torch.load(ck,weights_only=True)["model"]);models[name]=m
 ct=CollisionTransportPreconditioner(op,depth=4,learned=True,width=12).cuda().eval();ct.load_state_dict(torch.load(ROOT/"rte/workstation_runs/ct4_structure_control_n128_seed402_20260913/best.pt",weights_only=True)["model"]);models["CT4"]=ct;models["G0"]=op.G
 original=json.load(open(FIT/"gaussian_probe_audit.json"));reference={(x["index"],x["probe"],x["kind"],x["method"]):x for x in original["rows"]};entries=[];rows=[];checks=[]
 with torch.no_grad():
  for i in range(4):
   setup_batch(cfg,op,d,torch.tensor([i]),"cuda")
   for name in ["before","after"]:models[name].cache=None;models[name].key=None
   for j in range(16):
    z=torch.randn(1,12,64,64,device="cuda",dtype=torch.complex64)
    for kind,r in [("white_r",z),("white_Ae",op.A_op(z))]:
     entries.append(dict(material=i,probe=j,kind=kind,r=r.cpu()))
     for name,m in models.items():
      e=r-op.A_op(m(r));val=float(e.norm()/r.norm());ref=reference[(i,j,kind,name)];err=abs(val-ref["relative_norm"]);tol=1e-6+1e-5*abs(ref["relative_norm"]);row=dict(index=i,probe=j,kind=kind,method=name,relative_norm=val,squared_ratio=val*val);rows.append(row);checks.append(dict(index=i,probe=j,kind=kind,method=name,reference=ref["relative_norm"],reconstructed=val,absolute_error=err,tolerance=tol,passed=math.isfinite(val) and err<=tol))
 candidate=OUT/"reconstructed_candidate.pt";torch.save(dict(entries=entries,sigma_t=d["sigma_t"][:4],sigma_s=d["sigma_s"][:4]),candidate)
 passed=len(checks)==len(reference) and all(x["passed"] for x in checks);audit=dict(passed=passed,scope="deterministic reconstructed old 4x16 paired white-r/Ae panel, NOT original saved vectors; distinct from new 8x8 white panel",seed=146401,construction="original model/source/RNG order replayed",tolerance="abs error <= 1e-6 + 1e-5*abs(original relative norm), every scalar",original_source_hash=sha(FIT/"gaussian_probe_audit.py"),original_results_hash=sha(FIT/"gaussian_probe_audit.json"),candidate_hash=sha(candidate),source_hash=sha(__file__),checks=checks,reconstructed_baseline_FP32=rows);(OUT/"reconstruction_audit.json").write_text(json.dumps(audit,indent=2))
 if not passed:print("FAILED reconstruction, candidate retained, no qualified artifact or endpoint comparison",flush=True);return
 # Freeze only after all 512 original scalar checks pass. No replacement of old results.
 qualified=OUT/"qualified_inputs.pt";torch.save(torch.load(candidate,weights_only=True),qualified);(OUT/"qualified_manifest.json").write_text(json.dumps(dict(input_hash=sha(qualified),audit_hash=sha(OUT/"reconstruction_audit.json"),scope=audit["scope"],not_for_training_or_selection=True),indent=2));print("QUALIFIED",flush=True)
 # Evaluate all endpoint models and the four references on the identical reconstructed vectors with FP64 A.
 for name,cp in endpoints:
  m=Cycle(op).cuda().eval();m.load_state_dict(torch.load(cp,weights_only=True)["model"]);models[name]=m
 result=[]
 with torch.no_grad():
  for i in range(4):
   q=setup_batch(cfg,op,d,torch.tensor([i]),"cuda");a=PeriodicRTE2D(shape=(64,64),dx=1/64,n_angles=12,anisotropy_g=.65,eta_scale=0.,fdtype=torch.float64,device="cuda").setup(sigma_t=d["sigma_t"][i:i+1].double().cuda(),sigma_s=d["sigma_s"][i:i+1].double().cuda(),source=q.to(torch.complex128))
   for m in models.values():
    if isinstance(m,Cycle):m.cache=None;m.key=None
   for entry in entries:
    if entry["material"]!=i:continue
    r=entry["r"].cuda()
    for name,m in models.items():
     e=r.to(torch.complex128)-a.A_op(m(r).to(torch.complex128));val=float(e.norm()/r.to(torch.complex128).norm());result.append(dict(material=i,probe=entry["probe"],kind=entry["kind"],method=name,relative_norm=val,squared_ratio=val*val))
  (OUT/"endpoint_results.json").write_text(json.dumps(dict(complete=True,input_hash=sha(qualified),endpoint_hashes={n:sha(p) for n,p in endpoints},scope="same reconstructed fixed vectors, FP32 M and FP64 A, per-probe/per-material, diagnostic only; not all-panel improvement gate",rows=result),indent=2));print("COMPLETE",flush=True)
if __name__=="__main__":main()
