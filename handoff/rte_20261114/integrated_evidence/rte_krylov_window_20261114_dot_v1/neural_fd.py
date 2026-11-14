import sys,time,json
from pathlib import Path
ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries');OLD=ROOT/'rte/workstation_runs/rte_mcmg_born_20261114_dot_v1';OUT=Path(__file__).parent;sys.path.insert(0,str(OLD));from train import *
from window import real,complex,prefix,window,k_loss
from rte.models import PeriodicRTE2D
torch.set_num_threads(4);torch.manual_seed(146402);data=torch.load(OLD/'data.pt',weights_only=True)['train'];cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed');q=source_from_data(cfg,data,torch.tensor([0]),'cuda');op=make_operator(cfg,data['sigma_t'][:1],data['sigma_s'][:1],q,'cuda');a64=PeriodicRTE2D(shape=(64,64),dx=1/64,n_angles=12,anisotropy_g=.65,eta_scale=0.,fdtype=torch.float64,device='cuda').setup(sigma_t=data['sigma_t'][:1].double().cuda(),sigma_s=data['sigma_s'][:1].double().cuda(),source=q.to(torch.complex128));m=Cycle(op).cuda().eval();m.load_state_dict(torch.load(OLD/'physical_2048.pt',weights_only=True)['model']);A=lambda x:real(a64.A_op(complex(x)));M=lambda x:real(m(complex(x).to(torch.complex64)).to(torch.complex128));b=real(a64.source);ctx,_=prefix(A,M,b,16);rs=b-A(ctx['x']);_,rows=window(A,M,b,ctx);loss=k_loss(rows,rs);params=list(m.parameters());grads=torch.autograd.grad(loss,params);gn=torch.sqrt(sum(g.square().sum() for g in grads));dirs=[g/gn for g in grads];saved=[p.detach().clone() for p in params];ad=float(sum((g*d).sum() for g,d in zip(grads,dirs)));results=[]
for eps in [.003,.001,.0003]:
 vals=[]
 for sign in [1,-1]:
  with torch.no_grad():
   for p,v,d in zip(params,saved,dirs):p.copy_(v+sign*eps*d)
   m.cache=None;m.key=None;_,rr=window(A,M,b,ctx);vals.append(float(k_loss(rr,rs)))
 fd=(vals[0]-vals[1])/(2*eps);err=abs(fd-ad)/max(abs(fd),abs(ad),1e-12);results.append(dict(eps=eps,autograd=ad,finite_difference=fd,relative_error=err));print(results[-1],flush=True)
with torch.no_grad():
 for p,v in zip(params,saved):p.copy_(v)
(OUT/'neural_fd.json').write_text(json.dumps(dict(scope='all-parameter gradient direction, fixed stopgrad prefix16, window crossing restart20',rows=results,pass_best=min(x['relative_error'] for x in results)<.02),indent=2))
