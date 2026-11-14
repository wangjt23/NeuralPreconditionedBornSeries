import sys,json,hashlib,argparse,torch
from pathlib import Path
ROOT=Path("/data/wjt/NeuralPreconditionedBornSeries");OLD=ROOT/"rte/workstation_runs/rte_mcmg_born_20261114_dot_v1";FIT=ROOT/"rte/workstation_runs/rte_mcmg_fit8_20261114_dot_v1";OUT=Path(__file__).parent;sys.path.insert(0,str(OLD));from train import Cycle,ExperimentConfig,source_from_data,make_operator,setup_batch,dataset_fingerprint
from rte.models import PeriodicRTE2D
from rte.collision_transport import CollisionTransportPreconditioner
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
 if (OUT/"inputs.pt").exists():raise RuntimeError("already frozen")
 d=torch.load(OLD/"data.pt",weights_only=True)["train"];old=torch.load(FIT/"fixed_residuals.pt",weights_only=True);gen=torch.Generator().manual_seed(147901);white=torch.randn((8,8,12,64,64),dtype=torch.complex64,generator=gen);white=white/white.abs().square().sum((2,3,4),keepdim=True).sqrt();entries=[]
 for i in range(8):
  for j in range(4):entries.append(dict(material=i,kind="old_true_residual",item=j,r=old[i,j].clone()))
  for j in range(8):entries.append(dict(material=i,kind="white_gaussian",item=j,r=white[i,j].clone()))
  if i<4:
   for origin in ["before","after","CT4"]:
    p=FIT/f"witness_{i}_{origin}.pt";v=torch.load(p,weights_only=True)["input_ri"];re,im=v.chunk(2,1);entries.append(dict(material=i,kind="stored_witness_"+origin,item=0,r=torch.complex(re,im)[0],origin_hash=sha(p)))
 torch.save(dict(entries=entries,sigma_t=d["sigma_t"][:8],sigma_s=d["sigma_s"][:8]),OUT/"inputs.pt")
 protocol=dict(scope="diagnostic fixed TRAIN8 residual panels only; never train/select; no full solver claim; no old-prefix mixed-context rollout",white="8 independent full-space complex Gaussian r per medium (NOT Ae), CPU seed147901, each L2 normalized; evaluate mean norm AND mean squared norm",old="32 frozen before-model true residuals from source/TSA1/FGMRES2/8",witness="12 saved before/after/CT4 power witnesses, 4TRAIN, no new power",input_hash=sha(OUT/"inputs.pt"),old_bank_hash=sha(FIT/"fixed_residuals.pt"),train_fingerprint=dataset_fingerprint(d),source_hash=sha(__file__),count=len(entries));(OUT/"protocol.json").write_text(json.dumps(protocol,indent=2));print("PREPARED",protocol)
def evaluate(checkpoint,name,kind):
 data=torch.load(OUT/"inputs.pt",weights_only=True);q=torch.ones(1,12,64,64,dtype=torch.complex64,device="cuda");cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes="mixed");op=make_operator(cfg,data["sigma_t"][:1],data["sigma_s"][:1],q,"cuda");model=Cycle(op).cuda().eval() if kind=="cycle" else CollisionTransportPreconditioner(op,depth=4,learned=True,width=12).cuda().eval();model.load_state_dict(torch.load(checkpoint,weights_only=True)["model"]);rows=[]
 with torch.no_grad():
  for i in range(8):
   op.setup(sigma_t=data["sigma_t"][i:i+1].cuda(),sigma_s=data["sigma_s"][i:i+1].cuda(),source=q);a=PeriodicRTE2D(shape=(64,64),dx=1/64,n_angles=12,anisotropy_g=.65,eta_scale=0.,fdtype=torch.float64,device="cuda").setup(sigma_t=data["sigma_t"][i:i+1].double().cuda(),sigma_s=data["sigma_s"][i:i+1].double().cuda(),source=q.to(torch.complex128))
   if kind=="cycle":model.cache=None;model.key=None
   for entry in data["entries"]:
    if entry["material"]!=i:continue
    r=entry["r"][None].cuda();z=model(r);e=r.to(torch.complex128)-a.A_op(z.to(torch.complex128));ratio=float(e.norm()/r.to(torch.complex128).norm());rows.append(dict(material=i,kind=entry["kind"],item=entry["item"],relative_norm=ratio,squared_ratio=ratio**2))
 report=dict(complete=True,name=name,checkpoint=str(checkpoint),checkpoint_hash=sha(checkpoint),input_hash=sha(OUT/"inputs.pt"),source_hash=sha(__file__),scope="same fixed TRAIN8 diagnostic inputs; FP32 M with FP64 A; no new power/training/selection",rows=rows);path=OUT/(name+".json")
 if path.exists():raise RuntimeError("refuse overwrite "+str(path))
 path.write_text(json.dumps(report,indent=2));print("COMPLETE",path)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--checkpoint");p.add_argument("--name");p.add_argument("--kind",choices=["cycle","ct"],default="cycle");a=p.parse_args();torch.set_num_threads(2)
 if a.prepare:prepare()
 else:evaluate(Path(a.checkpoint),a.name,a.kind)
