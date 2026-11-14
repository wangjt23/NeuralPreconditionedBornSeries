import sys,json
from pathlib import Path
ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries');OLD=ROOT/'rte/workstation_runs/rte_mcmg_born_20261114_dot_v1';OUT=Path(__file__).parent;sys.path.insert(0,str(OLD));from train import *
from rte.collision_transport import CollisionTransportPreconditioner

def main():
 torch.set_num_threads(4);torch.manual_seed(146401);d=torch.load(OLD/'data.pt',weights_only=True)['train'];cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed');q=source_from_data(cfg,d,torch.tensor([0]),'cuda');op=make_operator(cfg,d['sigma_t'][:1],d['sigma_s'][:1],q,'cuda');models={}
 for name,ck in [('before',OLD/'physical_2048.pt'),('after',OUT/'fit_4096.pt')]:
  m=Cycle(op).cuda().eval();m.load_state_dict(torch.load(ck,weights_only=True)['model']);models[name]=m
 ct=CollisionTransportPreconditioner(op,depth=4,learned=True,width=12).cuda().eval();ct.load_state_dict(torch.load(ROOT/'rte/workstation_runs/ct4_structure_control_n128_seed402_20260913/best.pt',weights_only=True)['model']);models['CT4']=ct;models['G0']=op.G;rows=[]
 with torch.no_grad():
  for i in range(4):
   setup_batch(cfg,op,d,torch.tensor([i]),'cuda')
   for name in ['before','after']:models[name].cache=None;models[name].key=None
   for j in range(16):
    z=torch.randn(1,12,64,64,device='cuda',dtype=torch.complex64)
    for kind,r in [('white_r',z),('white_Ae',op.A_op(z))]:
     for name,m in models.items():
      e=r-op.A_op(m(r));val=float(e.norm()/r.norm());rows.append(dict(index=i,probe=j,kind=kind,method=name,relative_norm=val,squared_ratio=val*val))
 (OUT/'gaussian_probe_audit.json').write_text(json.dumps(dict(seed=146401,scope='4TRAIN,16 fresh probes each; no fitting; same raw white sample for r and Ae; not full coverage',rows=rows),indent=2));print('COMPLETE')
if __name__=='__main__':main()
