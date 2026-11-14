import sys,os,time,json,hashlib,math,random
from pathlib import Path
import torch
ROOT=Path("/data/wjt/NeuralPreconditionedBornSeries");OLD=ROOT/"rte/workstation_runs/rte_mcmg_born_20261114_dot_v1";OUT=Path(__file__).parent;sys.path.insert(0,str(OLD));from train import Cycle,ExperimentConfig,make_operator,dataset_fingerprint
from rte.models import PeriodicRTE2D
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace
from rte.collision_transport import CollisionTransportPreconditioner

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4);data=torch.load(OLD/"data.pt",weights_only=True)["train"];specs=[dict(name="isotropic",center=[.173,.731],sigma=.08,angular_kappa=0.,angle=0.),dict(name="beam",center=[.683,.227],sigma=.11,angular_kappa=2.,angle=math.pi/3)]
 grid=torch.arange(64,dtype=torch.float64)/64;x,y=torch.meshgrid(grid,grid,indexing="ij");angles=2*math.pi*torch.arange(12,dtype=torch.float64)/12;qs=[]
 for s in specs:
  cx,cy=s["center"];k=1/(4*math.pi**2*s["sigma"]**2);sp=torch.exp(k*(torch.cos(2*math.pi*(x-cx))+torch.cos(2*math.pi*(y-cy))-2));sp=sp/sp.mean();ang=torch.exp(s["angular_kappa"]*torch.cos(angles-s["angle"]));ang=ang/ang.mean();qs.append((ang[:,None,None]*sp[None]).to(torch.complex128))
 sources=torch.stack(qs);torch.save(dict(sources=sources,train_ids=torch.arange(8),sigma_t=data["sigma_t"][:8],sigma_s=data["sigma_s"][:8],specs=specs),OUT/"inputs.pt")
 protocol=dict(scope="heldout positive RHS on TRAIN8 materials; not independent-material confirmation; never use for training or selection",train_ids=list(range(8)),source_formula="q_l(x,y)=exp(k*(cos(2pi(x-cx))+cos(2pi(y-cy))-2))/grid_mean * exp(a*cos(theta_l-angle))/discrete_angle_mean; k=1/(4*pi^2*sigma^2)",specs=specs,size=64,n_angles=12,g=.65,eta_scale=0.,input_hash=sha(OUT/"inputs.pt"),source_hash=sha(__file__),parent_train_fingerprint=dataset_fingerprint(data),minimum=float(sources.real.min()),dtype="complex128 sources, float32 original coefficient tensors promoted to float64 for A",budget=dict(tol=1e-5,maxiter=100,restart=20,max_restarts=5),repeats=[-1,0,1,2],timing="first(-1) includes per-model cache build; 0/1/2 warm; all include independent physical residual verification; shared setup recorded separately; process/model loading excluded")
 (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2));cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes="mixed");op=make_operator(cfg,data["sigma_t"][:1],data["sigma_s"][:1],sources[:1].to(torch.complex64),"cuda");before=Cycle(op).cuda().eval();bp=OLD/"physical_2048.pt";before.load_state_dict(torch.load(bp,weights_only=True)["model"]);ct=CollisionTransportPreconditioner(op,depth=4,learned=True,width=12).cuda().eval();cp=ROOT/"rte/workstation_runs/ct4_structure_control_n128_seed402_20260913/best.pt";ct.load_state_dict(torch.load(cp,weights_only=True)["model"])
 report=dict(complete=False,pid=os.getpid(),protocol=protocol,checkpoint_hashes={"before":sha(bp),"CT4":sha(cp)},rows=[])
 with torch.no_grad():
  for i in range(8):
   for sid in range(2):
    q=sources[sid:sid+1].cuda();torch.cuda.synchronize();t=time.perf_counter();op.setup(sigma_t=data["sigma_t"][i:i+1].cuda(),sigma_s=data["sigma_s"][i:i+1].cuda(),source=q.to(torch.complex64));a=PeriodicRTE2D(shape=(64,64),dx=1/64,n_angles=12,anisotropy_g=.65,eta_scale=0.,fdtype=torch.float64,device="cuda").setup(sigma_t=data["sigma_t"][i:i+1].double().cuda(),sigma_s=data["sigma_s"][i:i+1].double().cuda(),source=q);torch.cuda.synchronize();setup=time.perf_counter()-t;before.cache=None;before.key=None
    for rep in [-1,0,1,2]:
     methods=[("before",before),("CT4",ct)];random.Random(149001+8*i+2*sid+rep).shuffle(methods)
     for name,m in methods:
      torch.cuda.synchronize();t=time.perf_counter();sol,h,_,calls=fgmres_real_stacked_with_trace(a.A_op,a.source,lambda r:m(r.to(torch.complex64)).to(torch.complex128),tol=1e-5,maxiter=100,restart=20,max_restarts=5);res=float((q-a.A_op(sol)).norm()/q.norm());torch.cuda.synchronize();elapsed=time.perf_counter()-t;report["rows"].append(dict(index=i,rhs=sid,method=name,repeat=rep,seconds=elapsed,shared_setup_seconds=setup,setup_plus_seconds=setup+elapsed,residual=res,accepted=math.isfinite(res) and res<=1e-5,calls=calls[-1],history=h))
    (OUT/"results.json").write_text(json.dumps(report,indent=2));print("DONE",i,sid,flush=True)
 report["complete"]=True;(OUT/"results.json").write_text(json.dumps(report,indent=2));print("COMPLETE",flush=True)
if __name__=="__main__":main()
