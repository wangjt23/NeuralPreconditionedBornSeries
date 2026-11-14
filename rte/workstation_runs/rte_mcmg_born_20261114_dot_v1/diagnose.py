import sys,json,time,statistics as st
from pathlib import Path
P=Path(__file__).parent;sys.path.insert(0,str(P));from train import *
from rte.collision_transport import CollisionTransportPreconditioner
from rte.spectral_collision_transport import SpectralCollisionTransport

def main():
 torch.set_num_threads(4);data=torch.load(P/'data.pt',weights_only=True);cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed');d=data['validation'];q=source_from_data(cfg,d,torch.tensor([0]),'cuda');op=make_operator(cfg,d['sigma_t'][:1],d['sigma_s'][:1],q,'cuda');models={}
 for arm in ['physical','matched']:
  m=Cycle(op).cuda().eval();m.load_state_dict(torch.load(P/(arm+'_2048.pt'),weights_only=True)['model']);models[arm]=m
 root=Path('/data/wjt/NeuralPreconditionedBornSeries/rte/workstation_runs');ct=CollisionTransportPreconditioner(op,depth=4,learned=True,width=12).cuda().eval();ct.load_state_dict(torch.load(root/'ct4_structure_control_n128_seed402_20260913/best.pt',weights_only=True)['model']);models['CT4']=ct;models['G0']=op.G
 rows=[]
 with torch.no_grad():
  for i in range(8):
   q=setup_batch(cfg,op,d,torch.tensor([i]),'cuda');gen=torch.Generator().manual_seed(18200+i);r1=q-op.A_op(op.apply_preconditioner(q,kind='transport_tsa'));e=torch.randn(q.shape,dtype=q.dtype,generator=gen).cuda();probes={'source':q,'TSA1':r1,'broad_Ae':op.A_op(e)}
   for arm in ['physical','matched']:models[arm].cache=None;models[arm].key=None
   for label,r in probes.items():
    for name,m in models.items():
     y=m(r);er=r-op.A_op(y);ts=[]
     for k in range(20):
      torch.cuda.synchronize();t=time.perf_counter();y=m(r);torch.cuda.synchronize();ts.append(time.perf_counter()-t)
     rows.append(dict(index=i,probe=label,method=name,physical=float(ratio(er,r)),matched=float(ratio(op.G(er),op.G(r))),apply_median_ms=1000*st.median(ts)))
 (P/'diagnostics.json').write_text(json.dumps({'rows':rows,'scope':'8 fixed new DEV; isolated warmed application cost, not substitute for solve timing'},indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
