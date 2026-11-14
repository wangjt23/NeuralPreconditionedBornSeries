import os,sys,json,time,hashlib,subprocess
from pathlib import Path
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
ROOT=Path("/data/wjt/rte_pre");OUT=Path("/data1/wjt/mgrte_replay_20261114/cached_audit");OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
import torch
from RTE_ATFPS_func import func_inflow_Dirichlet,funcDinv,block_basis_mask
from rte_preconditioning import mgnet_preconditioner
from RTE_mgmodel import MG_precond
from gmres import fgmres
seed=146715;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4);torch.set_default_dtype(torch.float32)
device="cuda:0";M=1;N=16
sync=torch.cuda.synchronize
models={};loads={}
for name,file in [("mg1000","diffusionI16N1L4batch256epoch879d100010000tol5_multilevel.pt"),("mg2000","diffusionI16N1L4batch256epoch661d200020000tol5_multilevel.pt")]:
 sync();t=time.perf_counter();model=MG_precond(1,[16,32,64,128],[16,32,64,128],[2,2,2,4]).to(device);model.load_state_dict(torch.load(ROOT/"model_precond/CosineAnnealingLR_T_max100/lr0.001"/file,map_location=device,weights_only=True),strict=True);model.eval();sync();loads[name]=time.perf_counter()-t;models[name]=model
D=torch.load(ROOT/"discretized_parameters/diffusionN1I16d32tol5.pt",weights_only=True)
rows=[];qual=[];setups=[]
def gpu():
 return subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True)
meta={"seed":seed,"model_load_and_init_seconds":loads,"concurrent_jobs_before":gpu(),"timing":"GPU shared with GB training; NOT uncontended benchmark. Three interleaved repetitions; setup and warm separate; original coefficient-generation excluded. Cached LU is same block Jacobi, not strong multilevel baseline.","probe_tolerance":1e-5,"data":"same saved first8, not certified heldout"}
(OUT/"meta.json").write_text(json.dumps(meta,indent=2))
with torch.no_grad():
 for i in range(8):
  c,f,I,T,V=[D[n][i:i+1].to(device) for n in ["Coef","fsmLRBTC","I2A","MLRBT","VecSize"]];c,f,I,T=c.float(),f.float(),I.float(),T.float();fd,Id,Td=f.double(),I.double(),T.double()
  def A(x):
   torch.set_default_dtype(torch.float64)
   try:return func_inflow_Dirichlet(x.double(),Id,fd,Td,V,M,N,N,device)
   finally:torch.set_default_dtype(torch.float32)
  def old(x):return funcDinv(x.float(),I,f,T,V,M,N,N,device).double()
  sync();t=time.perf_counter()
  block_sol=torch.cat([torch.einsum("bklij,bloij,bomij->bkmij",T[:,k],f[:,k],I) for k in range(4)],dim=1)
  mask=block_basis_mask(V,M,device);block=torch.zeros((1,8,8,N,N),device=device);block[mask]=block_sol[mask];block[~mask]=I[~mask]
  lu,piv=torch.linalg.lu_factor(block.permute(0,3,4,1,2));sync();cached_setup=time.perf_counter()-t
  def cached(x):return torch.linalg.lu_solve(lu,piv,x.float().permute(0,2,3,1).unsqueeze(-1)).squeeze(-1).permute(0,3,1,2).double()
  probes=[]
  for j in range(8):
   x=torch.randn((1,8,N,N),device=device);a,b=old(x),cached(x);probes.append(float(torch.linalg.norm(a-b)/torch.linalg.norm(a)))
  q={"material":i,"relative_probe_errors":probes,"pass":max(probes)<=1e-5};qual.append(q);(OUT/"qualification.json").write_text(json.dumps(qual,indent=2))
  if not q["pass"]:raise RuntimeError(q)
  pres={"bj_original":old,"bj_cached":cached};setup={"bj_original":0.,"bj_cached":cached_setup}
  for name,model in models.items():
   sync();t=time.perf_counter();aa,ii=model.setup(c);sync();setup[name]=time.perf_counter()-t
   def learned(x,model=model,aa=aa,ii=ii):return mgnet_preconditioner(x.float(),model,aa,ii,I,f,T,V,M,N,device,mode="raw",form="direct").double()
   pres[name]=learned
  warm={}
  for name,pre in pres.items():
   sync();t=time.perf_counter();pre(x);sync();warm[name]=time.perf_counter()-t
  setups.append({"material":i,"setup_seconds":setup,"warm_one_apply_seconds":warm});(OUT/"setup.json").write_text(json.dumps(setups,indent=2))
  rhs=torch.load(OUT.parent/f"rhs_{i}.pt",weights_only=True)
  for kind,b in rhs.items():
   b=b.to(device)
   names=list(pres)
   for rep in range(3):
    order=names[rep:]+names[:rep]
    if (rep+i)%2:order=order[::-1]
    for name in order:
     sync();t=time.perf_counter();sol,h=fgmres(A,b,pres[name],tol=1e-6,maxiter=200,restart=50,verbose=False);sync();seconds=time.perf_counter()-t
     rr=float(torch.linalg.norm(b-A(sol))/torch.linalg.norm(b))
     row={"material":i,"rhs":kind,"repeat":rep,"order":order,"method":name,"iterations":len(h)-1,"residual":rr,"seconds":seconds,"history":h};rows.append(row)
     (OUT/"results.json").write_text(json.dumps({"complete":False,"rows":rows},indent=2));print({k:v for k,v in row.items() if k!="history"},flush=True)
(OUT/"results.json").write_text(json.dumps({"complete":True,"rows":rows},indent=2));meta["concurrent_jobs_after"]=gpu();(OUT/"meta.json").write_text(json.dumps(meta,indent=2))
