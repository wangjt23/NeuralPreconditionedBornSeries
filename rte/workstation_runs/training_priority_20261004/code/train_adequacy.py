"""Bounded fixed-architecture continuation; saved train/development banks only."""
import argparse,copy,csv,hashlib,importlib,json,math,pathlib,subprocess,sys,time
import torch

REPO=pathlib.Path('/data/wjt/NeuralPreconditionedBornSeries')
HELPER=REPO/'rte/workstation_runs/resume_audit_20261002/code'
sys.path.insert(0,str(HELPER))
import train_tail_multiresolution as helper
from rte.benchmark_tail_solver import load_model
from rte.global_source_predictor import FrozenTSA
from rte.mixed_precision_transport import high_precision_copy
from rte.run_large_angular_spectral_experiments import ExperimentConfig,make_operator
from rte.tail_lookahead import continuation_loss
from rte.train_tail_direction import teacher_direction_loss,validate

WARM='rte/workstation_runs/resume_audit_20261002/paired_training/mixed64_128/best.pt'
PATHS={
 'train64':'rte/workstation_runs/tail_injection_gate_20260926/train_context.pt',
 'train128':'rte/workstation_runs/tail_multiresolution_bank_20260926/train_context.pt',
 'panel64':'rte/workstation_runs/tail_injection_gate_20260926/validation_context.pt',
 'panel128':'rte/workstation_runs/tail_multiresolution_bank_20260926/validation_context.pt',
 'parent64':'rte/workstation_runs/source_tail_validation_20260926/validation.pt',
 'parent128':'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt'}

def prepare_ops(banks):
 ops={}
 for grid,bank in banks.items():
  cfg=ExperimentConfig(size=grid,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed')
  low=make_operator(cfg,*(bank[k][:4] for k in ('t','s','q')),'cuda')
  ops[grid]=(low,high_precision_copy(low))
 return ops

def batch_fields(bank,ids,ops,phase=None,depth=8):
 fields=[bank[key][ids].to('cuda') for key in ('t','s','rhs_exact','reference','target','image_basis','inverse','stream')]
 t,s,r,ref,target,basis,inverse,stream=fields
 if phase is not None:
  factor=phase[:,None,None,None]
  r,target,ref,basis=r*factor,target*factor,ref*factor,basis*factor[:,None]
 ref=ref.to(torch.complex64)
 low,high=ops
 low.setup(t,s,r.to(torch.complex64));high.setup(t.double(),s.double(),r)
 frozen=FrozenTSA(low,inverse,stream,depth)
 return fields[:2]+[r,ref,target,basis], high, lambda v:frozen(v.to(torch.complex64)).to(torch.complex128)

@torch.no_grad()
def panel(model,banks,panel_banks,ops,step,arm,include_controls=False):
 rows=[];model.eval()
 for split,source in [('train',banks),('development',panel_banks)]:
  for grid,bank in source.items():
   count=min(16,len(bank['q'])) if split=='train' else len(bank['q'])
   for start in range(0,count,4):
    ids=torch.arange(start,min(start+4,count),device=bank['q'].device)
    fields,high,pre=batch_fields(bank,ids,ops[grid]);t,s,r,ref,target,basis=fields
    proposal=model(t,s,r.to(torch.complex64),ref).to(torch.complex128)
    candidates={'learned':proposal}
    if include_controls:candidates.update(reference=ref.to(torch.complex128),exact_teacher=target)
    for method,z in candidates.items():
     ratios=continuation_loss(high.A_op,pre,z,r,basis,6)
     teacher=teacher_direction_loss(z,target)
     coef=(z.conj()*target).real.flatten(1).sum(1)/z.abs().square().flatten(1).sum(1).clamp_min(1e-30)
     field_error=(coef[:,None,None,None]*z-target).flatten(1).norm(dim=1)/target.flatten(1).norm(dim=1)
     az=high.A_op(z);image=az.clone()
     for _ in range(2):
      for q in basis.unbind(1):image-=q*(q.conj()*image).real.flatten(1).sum(1)[:,None,None,None]
     independent=image.flatten(1).norm(dim=1)>1e-12*az.flatten(1).norm(dim=1)
     cosine=((image.conj()*r).real.flatten(1).sum(1).abs()/
       (image.flatten(1).norm(dim=1)*r.flatten(1).norm(dim=1)).clamp_min(1e-30))
     for j,idx in enumerate(ids.tolist()):
      row=dict(arm=arm,step=step,split=split,grid=grid,original_parent_index=int(bank['indices'][idx]),method=method,
       teacher_direction_loss=float(teacher[j]),optimally_scaled_field_error=float(field_error[j]),
       projected_image_cosine=float(cosine[j]),independent_image=int(independent[j]),
       objective=float(ratios[j,:3].clamp_min(1e-12).log().mean()+.05*teacher[j]))
      row.update({f'squared_ratio_after_{k}':float(ratios[j,k]) for k in [0,2,4,6]})
      rows.append(row)
 return rows

def checkpoint(path,model,opt,scheduler,step,plan,args,history,best,best_step,provenance):
 state=dict(args=vars(args),model=helper.cpu_copy(model.state_dict()),optimizer=helper.cpu_copy(opt.state_dict()),
  scheduler=scheduler.state_dict(),step=step,best_score=best,best_step=best_step,
  history=history,provenance=provenance,**helper.rng_state(plan,step))
 helper.save_checkpoint(path,state)

def run_arm(args,name,updates,lr,banks,panel_banks,parents,plan,deadline,provenance):
 out=pathlib.Path(args.output)/name;out.mkdir()
 helper.reset_rng(args.seed)
 model=load_model(WARM,False,'cuda')
 opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.horizon,eta_min=lr*.05)
 ops=prepare_ops(banks);history=[];validations=[];panels=[];best=None;best_step=None
 timing={'training':0.,'validation':0.,'panel':0.,'checkpoint':0.};complete=False;step=0
 for step in range(updates+1):
  if step%250==0 or step==updates:
   began=time.perf_counter();model.eval();records=[]
   for grid,parent in parents.items():
    _,vv=validate(model,parent,'cuda');records.extend(dict(grid=grid,**r) for r in vv)
   torch.cuda.synchronize();timing['validation']+=time.perf_counter()-began
   score=helper.selection_score(records);new_best=best is None or score<best
   if new_best:best,best_step=score,step
   validations.extend(dict(arm=name,step=step,**r) for r in records)
   helper.write_rows(out/'validation.csv',validations)
   if step%500==0 or step==updates:
    began=time.perf_counter();panels.extend(panel(model,banks,panel_banks,ops,step,name,step==0));torch.cuda.synchronize()
    timing['panel']+=time.perf_counter()-began;helper.write_rows(out/'fixed_panel.csv',panels)
   began=time.perf_counter();checkpoint(out/'last.pt',model,opt,sched,step,plan,args,history,best,best_step,provenance)
   if new_best:checkpoint(out/'best.pt',model,opt,sched,step,plan,args,history,best,best_step,provenance)
   if step in [0,1000,2000,3000]:checkpoint(out/f'step{step}.pt',model,opt,sched,step,plan,args,history,best,best_step,provenance)
   timing['checkpoint']+=time.perf_counter()-began
   helper.write_json(out/'progress.json',dict(arm=name,step=step,best_step=best_step,score=score,best_score=best,timing=timing))
   print(f'{name} update {step}: work={score[1]:.5f}, best={best_step}, training={timing["training"]:.2f}s',flush=True)
  if step==updates:complete=True;break
  if time.perf_counter()>=deadline:break
  began=time.perf_counter();grid=128 if step%2 else 64;bank=banks[grid]
  ids=plan['ids'][step].to(bank['q'].device)
  angles=plan['phase_angles'][step].to('cuda');phase=torch.polar(torch.ones_like(angles),angles)
  fields,high,pre=batch_fields(bank,ids,ops[grid],phase);t,s,r,ref,target,basis=fields
  model.train();z=model(t,s,r.to(torch.complex64),ref).to(torch.complex128)
  ratios=continuation_loss(high.A_op,pre,z,r,basis,2);teacher=teacher_direction_loss(z,target).mean()
  loss=ratios.clamp_min(1e-12).log().mean()+.05*teacher
  if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite loss')
  opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  opt.step();sched.step();torch.cuda.synchronize();seconds=time.perf_counter()-began;timing['training']+=seconds
  history.append(dict(step=step+1,grid=grid,loss=float(loss),teacher_loss=float(teacher),injection_squared_ratio=float(ratios[:,0].mean()),
   terminal_squared_ratio=float(ratios[:,-1].mean()),grad=float(grad),lr=opt.param_groups[0]['lr'],seconds=seconds))
  if torch.cuda.max_memory_reserved()>args.max_gpu_gib*1024**3:raise RuntimeError('declared GPU memory guard exceeded')
  if (step+1)%50==0:helper.write_rows(out/'history.csv',history)
 if not complete:
  checkpoint(out/'last.pt',model,opt,sched,len(history),plan,args,history,best,best_step,provenance)
 helper.write_rows(out/'history.csv',history)
 helper.write_rows(out/'sampling_consumed.csv',helper.sampling_rows(plan,helper.ARMS[1],len(history)))
 result=dict(arm=name,complete=complete,updates=len(history),selected_step=best_step,best_score=best,timing=timing,
  peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
  stopping_reason='planned budget' if complete else 'wall cap; partial checkpoint preserved',
  checkpoint_sha256={str(p):helper.file_digest(p) for p in out.glob('*.pt')})
 helper.write_json(out/'receipt.json',result)
 del opt,sched,model,ops;torch.cuda.empty_cache()
 return result

def main(args):
 torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();deadline=start+args.wall_seconds
 out=pathlib.Path(args.output);out.mkdir(exist_ok=False)
 loaded={k:torch.load(v,map_location='cpu',weights_only=True) for k,v in PATHS.items()}
 helper.check_paired_identity(loaded['train64'],loaded['train128'],contexts=True)
 helper.check_paired_identity(loaded['panel64'],loaded['panel128'],contexts=True)
 helper.check_paired_identity(loaded['parent64'],loaded['parent128'],contexts=False,expected_count=16)
 args.batch=4;args.cache_mode='auto';args.width=32;args.blocks=4;args.modes=6
 provenance={'warm_start':WARM,'warm_sha256':helper.file_digest(WARM),
  'input_sha256':{k:helper.file_digest(v) for k,v in PATHS.items()},
  'source_sha256':{str(pathlib.Path(__file__).resolve()):helper.file_digest(__file__),str(HELPER/'train_tail_multiresolution.py'):helper.file_digest(HELPER/'train_tail_multiresolution.py')}}
 for name in helper.SOURCE_MODULES:
  m=importlib.import_module('rte.'+name);provenance['source_sha256'][m.__file__]=helper.file_digest(m.__file__)
 settings=dict(args=vars(args),provenance=provenance,complete=False,scope='existing train and reused development only; no test access',
  protocol='fixed architecture, TSA8-balanced contexts, two-step loss; same 3000-step cosine horizon and shared samples; weights-only warm start, state reset',
  learning_rate_control_scope='compare arms only at matched update1000; extended main endpoint is a budget-plus-rate comparison against control1000',
  fixed_panels='first16 eligible original training IDs per grid; all13 eligible development IDs per grid; phase0; controls at initialization',
  selection='all16 development media at64/128; failures, equivalent work, residual; no selection using timing or N256',
  resources='serial GPU execution, wall cap900s for phase including cache/validation/panels/checkpoints; reserved GPU guard14GiB')
 helper.write_json(out/'settings.json',settings)
 banks,cache=helper.normalize_and_cache({64:loaded['train64'],128:loaded['train128']},args,'cuda')
 panel_banks,pcache=helper.normalize_and_cache({64:loaded['panel64'],128:loaded['panel128']},args,'cuda')
 parents={64:loaded['parent64'],128:loaded['parent128']};del loaded
 plan=helper.build_sampling_plan(banks[64]['indices'].cpu(),args.horizon,4,args.seed)
 helper.save_checkpoint(out/'sampling_plan.pt',plan)
 settings.update(cache=cache,panel_cache=pcache,identity_phase_sha256=plan['identity_phase_sha256'],arms=[])
 helper.write_json(out/'settings.json',settings)
 try:
  jobs=[('continuation',3000,.00015),('low_lr_control',1000,.00005)]
  if args.smoke:jobs=[('smoke',2,.00015)]
  for name,updates,lr in jobs:
   if time.perf_counter()>=deadline:break
   result=run_arm(args,name,updates,lr,banks,panel_banks,parents,plan,deadline,provenance)
   settings['arms'].append(result);helper.write_json(out/'settings.json',settings)
   if not result['complete']:break
  settings['complete']=len(settings['arms'])==len(jobs) and all(r['complete'] for r in settings['arms'])
 finally:
  settings['wall_seconds']=time.perf_counter()-start;helper.write_json(out/'settings.json',settings)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--wall-seconds',type=float,default=900.)
 p.add_argument('--max-gpu-gib',type=float,default=14.);p.add_argument('--seed',type=int,default=41004);p.add_argument('--horizon',type=int,default=3000)
 p.add_argument('--smoke',action='store_true')
 main(p.parse_args())
