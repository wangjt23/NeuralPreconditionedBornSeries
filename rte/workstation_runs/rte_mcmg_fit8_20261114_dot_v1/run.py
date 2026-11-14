import sys,json,time,hashlib,copy,os
from pathlib import Path
ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries');OLD=ROOT/'rte/workstation_runs/rte_mcmg_born_20261114_dot_v1';OUT=Path(__file__).parent;sys.path.insert(0,str(OLD));from train import *
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace
from rte.models import PeriodicRTE2D
from rte.collision_transport import CollisionTransportPreconditioner

def op64(q,t,s):return PeriodicRTE2D(shape=(64,64),dx=1/64,n_angles=12,anisotropy_g=.65,eta_scale=0.,fdtype=torch.float64,device='cuda').setup(sigma_t=t.double().cuda(),sigma_s=s.double().cuda(),source=q.to(torch.complex128))
def main():
 torch.set_num_threads(4);torch.manual_seed(144401);start=time.time();data=torch.load(OLD/'data.pt',weights_only=True)['train'];cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed');ids=torch.arange(8);q=source_from_data(cfg,data,ids,'cuda');op=make_operator(cfg,data['sigma_t'][:8],data['sigma_s'][:8],q,'cuda');model=Cycle(op).cuda();ck=torch.load(OLD/'physical_2048.pt',weights_only=True);model.load_state_dict(ck['model']);initial=copy.deepcopy(model.state_dict());model.eval();bank=[]
 with torch.no_grad():
  for i in range(8):
   qi=setup_batch(cfg,op,data,torch.tensor([i]),'cuda');a64=op64(qi,data['sigma_t'][i:i+1],data['sigma_s'][i:i+1]);model.cache=None;model.key=None
   rr=[qi,qi-op.A_op(op.apply_preconditioner(qi,kind='transport_tsa'))]
   for k in [2,8]:
    x,h,_,_=fgmres_real_stacked_with_trace(a64.A_op,a64.source,lambda r:model(r.to(torch.complex64)).to(torch.complex128),tol=0.,maxiter=k,restart=k,max_restarts=0)
    rr.append((a64.source-a64.A_op(x)).to(torch.complex64))
   bank.append(torch.cat(rr))
 bank=torch.stack(bank).detach();bank=bank/bank.abs().square().sum((2,3,4),keepdim=True).sqrt().clamp_min(1e-20);torch.save(bank.cpu(),OUT/'fixed_residuals.pt');setup_batch(cfg,op,data,ids,'cuda')
 protocol=dict(purpose='capacity/optimization diagnostic on fixed TRAIN8, not generalization',parent=str(OLD),start_checkpoint_hash=hashlib.sha256((OLD/'physical_2048.pt').read_bytes()).hexdigest(),train_ids=list(range(8)),states=['source','TSA1_true_defect','own_FGMRES2_true_defect_FP64','own_FGMRES8_true_defect_FP64'],updates=4096,loss='mean per-sample squared original-equation residual',batch='all8 materials, randomly drawn one of four fixed states',draw_seed=144402,lr=3e-4,optimizer='fresh AdamW, wd1e-5 cosine to1e-5',DEV_training=False,selection='fixed endpoints only',bank_hash=hashlib.sha256((OUT/'fixed_residuals.pt').read_bytes()).hexdigest(),source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));status=dict(pid=os.getpid(),complete=False,history=[],seconds=0);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4096,eta_min=1e-5);gen=torch.Generator().manual_seed(144402)
 def assess(step):
  model.eval();model.cache=None;model.key=None;vals=[]
  with torch.no_grad():
   for j in range(4):
    r=bank[:,j];e=r-op.A_op(model(r));vals.append(float(ratio(e,r)))
  row=dict(step=step,state_losses=vals,mean=sum(vals)/4,seconds=time.time()-start);status['history'].append(row);status['active']=row;(OUT/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(row),flush=True);model.train()
 assess(0)
 for step in range(1,4097):
  j=int(torch.randint(4,(),generator=gen));r=bank[:,j];opt.zero_grad(set_to_none=True);e=r-op.A_op(model(r));loss=ratio(e,r);assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step();sch.step()
  if step%128==0:assess(step)
  if step in [1024,2048,4096]:torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),generator=gen.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),step=step,protocol=protocol),OUT/f'fit_{step}.pt')
 status['training_complete']=True;(OUT/'status.json').write_text(json.dumps(status,indent=2));rows=[];model.eval();frozen=Cycle(op).cuda().eval();frozen.load_state_dict(initial);ct=CollisionTransportPreconditioner(op,depth=4,learned=True,width=12).cuda().eval();ct.load_state_dict(torch.load(ROOT/'rte/workstation_runs/ct4_structure_control_n128_seed402_20260913/best.pt',weights_only=True)['model'])
 with torch.no_grad():
  for i in range(8):
   torch.cuda.synchronize();t=time.perf_counter();qi=setup_batch(cfg,op,data,torch.tensor([i]),'cuda');a64=op64(qi,data['sigma_t'][i:i+1],data['sigma_s'][i:i+1]);torch.cuda.synchronize();setup=time.perf_counter()-t;model.cache=None;model.key=None;frozen.cache=None;frozen.key=None
   for rep in [-1,0,1,2]:
    methods=[('CT4_frozen',ct),('before_fit',frozen),('after_fit',model)]
    if (i+rep)%2:methods.reverse()
    for name,m in methods:
     torch.cuda.synchronize();t=time.perf_counter();x,h,_,calls=fgmres_real_stacked_with_trace(a64.A_op,a64.source,lambda r:m(r.to(torch.complex64)).to(torch.complex128),tol=1e-5,maxiter=100,restart=20,max_restarts=5);res=float((a64.source-a64.A_op(x)).norm()/a64.source.norm());torch.cuda.synchronize();elapsed=time.perf_counter()-t;rows.append(dict(index=i,method=name,repeat=rep,seconds=elapsed,setup_seconds=setup,residual=res,accepted=res<=1e-5,calls=calls[-1]))
   (OUT/'solver_results.json').write_text(json.dumps(rows,indent=2));print('EVAL',i,flush=True)
 status.update(complete=True,seconds=time.time()-start);(OUT/'status.json').write_text(json.dumps(status,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
