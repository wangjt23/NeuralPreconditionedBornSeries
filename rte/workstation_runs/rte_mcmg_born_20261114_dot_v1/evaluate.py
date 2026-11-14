import time,json,sys,hashlib,random
from pathlib import Path
P=Path(__file__).parent;sys.path.insert(0,str(P));from train import *
from rte.models import PeriodicRTE2D
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace
from rte.spectral_collision_transport import SpectralCollisionTransport
from rte.collision_transport import CollisionTransportPreconditioner

def sync():torch.cuda.synchronize()
def main():
 torch.set_num_threads(4)
 while True:
  try:
   if json.load(open(P/'status.json')).get('complete'):break
  except (FileNotFoundError,json.JSONDecodeError):pass
  time.sleep(20)
 data=torch.load(P/'data.pt',weights_only=True);cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed');dev=data['validation'];q=source_from_data(cfg,dev,torch.tensor([0]),'cuda');op=make_operator(cfg,dev['sigma_t'][:1],dev['sigma_s'][:1],q,'cuda')
 learned={}; hashes={}
 for arm in ['physical','matched']:
  p=P/(arm+'_2048.pt');m=Cycle(op).cuda().eval();m.load_state_dict(torch.load(p,weights_only=True)['model']);learned[arm]=m;hashes[arm]=hashlib.sha256(p.read_bytes()).hexdigest()
 root=Path('/data/wjt/NeuralPreconditionedBornSeries/rte/workstation_runs');sct=SpectralCollisionTransport(op,depth=4).cuda().eval();sp=root/'sct4_krylov_n64_seed402_20260912/best.pt';sct.load_state_dict(torch.load(sp,weights_only=True)['model']);ct=CollisionTransportPreconditioner(op,depth=4,learned=True,width=12).cuda().eval();cp=root/'ct4_structure_control_n128_seed402_20260913/best.pt';ct.load_state_dict(torch.load(cp,weights_only=True)['model']);hashes.update(sct=hashlib.sha256(sp.read_bytes()).hexdigest(),ct=hashlib.sha256(cp.read_bytes()).hexdigest())
 report=dict(complete=False,hashes=hashes,scope='N64 fresh development; frozen historical CT/SCT transfer controls, not matched retraining; fixed endpoint; no confirmation claim',rows=[],panels=[])
 # Fixed TRAIN8 and fresh residual panel, distinct from stochastic training logs.
 for split,ids in [('train',torch.arange(8)),('validation',torch.arange(8))]:
  q=setup_batch(cfg,op,data[split],ids,'cuda');gen=torch.Generator().manual_seed(17771)
  with torch.no_grad():
   r=residual_sample(op,q,gen,.2)
   for name,m in learned.items():
    e=r-op.A_op(m(r));report['panels'].append(dict(split=split,method=name,physical=float(ratio(e,r)),matched=float(ratio(op.G(e),op.G(r)))))
 with torch.no_grad():
  for i in range(32):
   sync();t=time.perf_counter();q=setup_batch(cfg,op,dev,torch.tensor([i]),'cuda');op64=PeriodicRTE2D(shape=(64,64),dx=1/64,n_angles=12,anisotropy_g=.65,eta_scale=0.,fdtype=torch.float64,device='cuda').setup(sigma_t=dev['sigma_t'][i:i+1].double().cuda(),sigma_s=dev['sigma_s'][i:i+1].double().cuda(),source=q.to(torch.complex128));sync();shared=time.perf_counter()-t
   methods={'G0':op.G,'TSA':lambda r:op.apply_preconditioner(r,kind='transport_tsa'),'CT4_frozen':ct,'SCT4_frozen':sct,**learned}
   for rep in [-1,0,1,2]:
    order=list(methods);random.Random(17100+4*i+rep).shuffle(order)
    for name in order:
     m=methods[name];sync();t=time.perf_counter()
     if rep==-1:
      if name in learned:learned[name].cache=None;learned[name].key=None
     x,h,eh,calls=fgmres_real_stacked_with_trace(op64.A_op,op64.source,lambda r:m(r.to(torch.complex64)).to(torch.complex128),tol=1e-5,maxiter=100,restart=20,max_restarts=5)
     residual=float((op64.source-op64.A_op(x)).norm()/op64.source.norm());sync();elapsed=time.perf_counter()-t
     report['rows'].append(dict(index=i,repeat=rep,method=name,seconds=elapsed,shared_setup_seconds=shared,setup_plus_seconds=shared+elapsed,true_residual=residual,accepted=residual<=1e-5,iterations=len(h)-1,calls=calls))
   (P/'evaluation.json').write_text(json.dumps(report,indent=2));print('EVAL',i,flush=True)
 report['complete']=True;(P/'evaluation.json').write_text(json.dumps(report,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
