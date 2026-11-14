import sys,json,time
from pathlib import Path
ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries');OLD=ROOT/'rte/workstation_runs/rte_mcmg_born_20261114_dot_v1';OUT=Path(__file__).parent;sys.path.insert(0,str(OLD));from train import *
from rte.collision_transport import CollisionTransportPreconditioner

def main():
 torch.set_num_threads(4);torch.manual_seed(145401);data=torch.load(OLD/'data.pt',weights_only=True)['train'];cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed');q=source_from_data(cfg,data,torch.tensor([0]),'cuda');op=make_operator(cfg,data['sigma_t'][:1],data['sigma_s'][:1],q,'cuda');models={}
 for name,path in [('before',OLD/'physical_2048.pt'),('after',OUT/'fit_4096.pt')]:
  m=Cycle(op).cuda().eval();m.load_state_dict(torch.load(path,weights_only=True)['model']);m.requires_grad_(False);models[name]=m
 ct=CollisionTransportPreconditioner(op,depth=4,learned=True,width=12).cuda().eval();ct.load_state_dict(torch.load(ROOT/'rte/workstation_runs/ct4_structure_control_n128_seed402_20260913/best.pt',weights_only=True)['model']);ct.requires_grad_(False);models['CT4']=ct
 rows=[]
 for i in range(4):
  setup_batch(cfg,op,data,torch.tensor([i]),'cuda')
  for name,m in models.items():
   if name!='CT4':m.cache=None;m.key=None
   def E(x):
    re,im=x.chunk(2,1);r=torch.complex(re,im);y=r-op.A_op(m(r));return torch.cat([y.real,y.imag],1)
   v=torch.randn(1,24,64,64,device='cuda');v=v/v.norm();u=torch.randn_like(v);u=u/u.norm();x=v.detach().requires_grad_(True);e=E(x);adj=torch.autograd.grad((e*u).sum(),x)[0];lhs=float((e.detach()*u).sum());rhs=float((v*adj).sum());adjerr=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-8)
   normalized_dot_error=abs(lhs-rhs)/max(float(e.detach().norm()*u.norm()),1e-20)
   hist=[]
   for k in range(24):
    x=v.detach().requires_grad_(True);ev=E(x);norm=float(ev.norm());adj=torch.autograd.grad((ev*ev.detach()).sum(),x)[0];v=adj.detach()/adj.norm().clamp_min(1e-20);hist.append(norm)
   with torch.no_grad():
    ev=E(v);witness=float(ev.norm()/v.norm());torch.save({"input_ri":v.cpu(),"output_ri":ev.cpu()},OUT/("witness_"+str(i)+"_"+name+".pt"))
   rows.append(dict(witness_gain=witness,normalized_adjoint_dot_error=normalized_dot_error,index=i,method=name,adjoint_dot_relative_error=adjerr,sigma_estimate=hist[-1],history=hist));(OUT/'amplification_v3.json').write_text(json.dumps({'scope':'FP32 real-linear E=I-AM power estimate on 4TRAIN; lower bound estimate not certified maximum or FGMRES cause', 'rows':rows},indent=2));print(i,name,witness,normalized_dot_error,flush=True)
if __name__=='__main__':main()
