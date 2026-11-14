import os,sys,json,time,hashlib,traceback
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path('/data/wjt/rte_pre');OUT=Path('/data1/wjt/mgrte_replay_20261114')
sys.path.insert(0,str(ROOT))
import torch,scipy.io
from RTE_ATFPS_func import func_inflow_Dirichlet,funcDinv,generate_TFPS_rhs_Dirichlet_iso
from rte_preconditioning import mgnet_preconditioner
from RTE_mgmodel import MG_precond
from gmres import fgmres
from RTE_datagenerator import fun_value
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.manual_seed(146714);torch.cuda.manual_seed_all(146714)
torch.set_num_threads(4);torch.set_default_dtype(torch.float32)
device='cuda:0';M=1;N=16
paths=[ROOT/'model_precond/CosineAnnealingLR_T_max100/lr0.001'/n for n in ['diffusionI16N1L4batch256epoch879d100010000tol5_multilevel.pt','diffusionI16N1L4batch256epoch661d200020000tol5_multilevel.pt']]
models={}
for name,p in zip(['mg1000','mg2000'],paths):
 model=MG_precond(1,[16,32,64,128],[16,32,64,128],[2,2,2,4]).to(device);model.load_state_dict(torch.load(p,map_location=device,weights_only=True),strict=True);model.eval();models[name]=model
p=ROOT/'discretized_parameters/diffusionN1I16d32tol5.pt'
data=torch.load(p,map_location='cpu',weights_only=True)
meta={'seed':146714,'data_sha':sha(p),'checkpoints':{str(x):sha(x) for x in paths},'source_hashes':{n:sha(ROOT/n) for n in ['RTE_ATFPS_func.py','RTE_mgmodel.py','gmres.py','rte_preconditioning.py']},'independent_material_solves':True,'materials':'existing saved diffusion first8; not certified heldout','operator':'original ATFPS FP64 application of saved FP32 coefficients; preconditioners FP32','stop':{'tol':1e-6,'maxiter':200,'restart':50},'timing':'single GPU process, model loading excluded, per-material setup separately; no precision/speed claim against periodic backend'}
(OUT/'meta.json').write_text(json.dumps(meta,indent=2));rows=[]
quad=scipy.io.loadmat(ROOT/'discretized_parameters/quadrature2DN1.mat');omega=torch.tensor(quad['omega']).squeeze(-1).float().to(device);Kappa=torch.tensor(scipy.io.loadmat(ROOT/'discretized_parameters/KappaN1g0.mat')['Kappa']).float().to(device)
with torch.no_grad():
 for i in range(8):
  c,f,I,T,V=[data[n][i:i+1].to(device) for n in ['Coef','fsmLRBTC','I2A','MLRBT','VecSize']]
  c,f,I,T=c.float(),f.float(),I.float(),T.float()
  fd,Id,Td=f.double(),I.double(),T.double()
  def A(x):
   torch.set_default_dtype(torch.float64)
   try:return func_inflow_Dirichlet(x.double(),Id,fd,Td,V,M,N,N,device)
   finally:torch.set_default_dtype(torch.float32)
  coeff=torch.rand((1,2,3),device=device);coeff[:,:,0]+=1;coeff[:,:,1:]=2*coeff[:,:,1:]-1
  Q=fun_value(N,N,0,1,0,1,coeff,device).unsqueeze(1).repeat(1,4,1,1);ones=torch.ones((1,2,N),device=device)
  phys,unselected,_=generate_TFPS_rhs_Dirichlet_iso(Q,ones,ones,ones,ones,c,Kappa,omega,T,V,M,N,N,device)
  rhsdict={'physical_selected':phys.double(),'uniform_random':(2*torch.rand((1,8,N,N),device=device)-1).double()}
  torch.save({k:v.cpu() for k,v in rhsdict.items()},OUT/f'rhs_{i}.pt')
  # Numerical qualification of the promoted operator only, without changing its formula.
  v=torch.randn_like(phys).double();w=torch.randn_like(v);av=A(v);aw=A(w)
  lin=float(torch.linalg.norm(A(v+w)-av-aw)/(torch.linalg.norm(av)+torch.linalg.norm(aw)))
  if lin>1e-10:raise RuntimeError(('A64 linearity',lin))
  for name in ['block_jacobi','mg1000','mg2000']:
   torch.cuda.synchronize();t=time.perf_counter()
   if name=='block_jacobi':
    def pre(x):return funcDinv(x.float(),I,f,T,V,M,N,N,device).double()
   else:
    model=models[name];alist,ilist=model.setup(c)
    def pre(x):return mgnet_preconditioner(x.float(),model,alist,ilist,I,f,T,V,M,N,device,mode='raw',form='direct').double()
   pre(v);torch.cuda.synchronize();setup=time.perf_counter()-t
   for kind,b in rhsdict.items():
    torch.cuda.synchronize();t=time.perf_counter();x,h=fgmres(A,b,pre,tol=1e-6,maxiter=200,restart=50,verbose=False);torch.cuda.synchronize();elapsed=time.perf_counter()-t
    rr=float(torch.linalg.norm(b-A(x))/torch.linalg.norm(b));first=next((j for j,y in enumerate(h) if y<=1e-4),None)
    row={'material':i,'method':name,'rhs':kind,'iterations':len(h)-1,'residual':rr,'accepted':rr<=1e-6,'first_1e4':first,'seconds':elapsed,'setup_and_warm_seconds':setup,'history':h,'A_linearity':lin};rows.append(row)
    (OUT/'results.json').write_text(json.dumps({'complete':False,'rows':rows},indent=2));print({k:v for k,v in row.items() if k!='history'},flush=True)
(OUT/'results.json').write_text(json.dumps({'complete':True,'rows':rows},indent=2))
