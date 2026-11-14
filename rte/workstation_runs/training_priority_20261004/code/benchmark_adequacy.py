"""Paired development-only timing of frozen adequacy checkpoints; no data draws."""
import argparse,json,pathlib,random,sys,time,subprocess
import torch
import rte.benchmark_tail_solver as protocol
from rte.train_source_predictor import material_operator,build_base
from rte.run_large_angular_spectral_experiments import ExperimentConfig
from rte.augmented_fgmres import augmented_fgmres
from rte.global_source_predictor import balance_integrated_source
from rte.run_residual_subspace_experiment import write_csv

ROOT=pathlib.Path('/data/wjt/NeuralPreconditionedBornSeries')
BASE=ROOT/'rte/workstation_runs/training_priority_20261004'

@torch.no_grad()
def main(args):
 torch.set_num_threads(1);start=time.perf_counter();out=pathlib.Path(args.output);out.mkdir(exist_ok=False)
 train=BASE/'training'
 ck={'frozen':ROOT/'rte/workstation_runs/resume_audit_20261002/paired_training/mixed64_128/best.pt',
  'continued_selected':train/'continuation/best.pt','continued_final':train/'continuation/last.pt',
  'low_lr1000':train/'low_lr_control/best.pt'}
 if args.alignment:
  aligned=pathlib.Path(args.alignment_directory) if args.alignment_directory else BASE/'alignment'
  ck.update(aligned8=aligned/'tsa8/best.pt',aligned16=aligned/'tsa16/best.pt',
   aligned8_final=aligned/'tsa8/last.pt',aligned16_final=aligned/'tsa16/last.pt')
 models={k:protocol.load_model(p,False,'cuda') for k,p in ck.items()}
 selection_path=ROOT/'rte/workstation_runs/tail_solver_benchmark_20260926/frozen_selection.json'
 selection=json.loads(selection_path.read_text());rows=[];all_cases=[];summaries=[];comparisons=[]
 settings={'args':vars(args),'complete':False,'scope':'same16 repeatedly reused development media4626;no test or new data draws',
  'checkpoint_sha256':{k:protocol.digest(p) for k,p in ck.items()},'classical_selection_sha256':protocol.digest(selection_path),
  'original_solver':'augmented_fgmres, original-RHS tolerance1e-5, max100, restart25',
  'timing':'one warmup+3 randomized synchronized repeats; per-media median; includes initialization,inference,projection,final residual check; setup separate',
  'gpu_before':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)}
 (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
 def sync():torch.cuda.synchronize()
 for grid in [64,128,256]:
  data_path=ROOT/f'rte/workstation_runs/tail_solver_benchmark_20260926/validation_n{grid}.pt'
  data=torch.load(data_path,map_location='cpu',weights_only=True)
  settings.setdefault('data_sha256',{})[str(grid)]=protocol.digest(data_path)
  cfg=ExperimentConfig(size=grid,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed')
  classical=selection[str(grid)]['classical'];methods=list(dict.fromkeys([classical,'reference_tail',*models,'continued_tsa16']))
  if args.alignment:methods += ['frozen_tsa16','aligned8_tsa16','aligned16_tsa16']
  rng=random.Random(41004+grid);gridrows=[]
  for index in range(16):
   if time.perf_counter()-start>args.wall_seconds:raise TimeoutError('benchmark cap reached; partial tables retained')
   sync();began=time.perf_counter();low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda');sync();physical=time.perf_counter()-began
   bases={};setup={}
   for name in ['tsa8','tsa16']:
    sync();began=time.perf_counter();bases[name]=build_base(low,name);sync();setup[name]=time.perf_counter()-began
   order=methods.copy()
   for repeat in [-1,0,1,2]:
    rng.shuffle(order)
    for position,method in enumerate(order):
     sync();began=time.perf_counter()
     if method in ['continued_tsa16','frozen_tsa16','aligned8_tsa16','aligned16_tsa16','aligned16_final']:
      base=bases['tsa16'];model=models['continued_selected' if method=='continued_tsa16' else method.removesuffix('_tsa16')]
      def inject(x,r):
       ref=base.reference.reference_transport_solve(r.to(torch.complex64))
       return model(low.sigma_t,low.sigma_s,r.to(torch.complex64),ref).to(torch.complex128)
      field,status=augmented_fgmres(high.A_op,high.source,lambda r:base(r.to(torch.complex64)).to(torch.complex128),None,inject=inject)
     elif method in models:
      field,status=protocol.solve('neural_tail',low,high,bases,{'tail':models[method]})
     else:field,status=protocol.solve(method,low,high,bases,{})
     sync();seconds=time.perf_counter()-began;status.pop('trace',None);status.pop('initial_residual',None)
     depth='tsa16' if method.endswith('_tsa16') or method in ['tsa16_zero','aligned16_final'] else 'tsa8'
     if repeat>=0:
      gridrows.append(dict(size=grid,index=index,method=method,repeat=repeat,order=position,seconds=seconds,
       setup_seconds=physical+setup[depth],setup_plus_seconds=physical+setup[depth]+seconds,**status))
     if repeat==0 and method=='continued_selected':
      # Keep original-RHS neural fields on FWI for later accuracy/certificate work.
      torch.save({'field':field.cpu(),'source':high.source.cpu(),'status':status,'grid':grid,'index':index,
       'checkpoint_sha256':settings['checkpoint_sha256'][method],'data_sha256':settings['data_sha256'][str(grid)]},out/f'neural_field_n{grid}_case{index}.pt')
   write_csv(out/f'runtime_n{grid}.csv',gridrows)
   if (index+1)%4==0:print(f'benchmark N{grid} {index+1}/16',flush=True)
  cases,summary=protocol.summarize(gridrows);write_csv(out/f'case_medians_n{grid}.csv',cases)
  summaries.extend(dict(size=grid,**r) for r in summary);all_cases.extend(dict(size=grid,**r) for r in cases)
  challengers=[*models,'continued_tsa16']
  if args.alignment:challengers += ['frozen_tsa16','aligned8_tsa16','aligned16_tsa16']
  for challenger in challengers:
   for comparator in list(dict.fromkeys([classical,'frozen'])):
    if comparator==challenger:continue
    cc=dict(size=grid,**protocol.paired_comparison(cases,comparator,challenger))
    cc['failed']=sum(r['failed'] for r in summary if r['method'] in [comparator,challenger]);comparisons.append(cc)
  if args.alignment:
   for baseline,challenger in [('frozen','aligned8'),('frozen','aligned16'),('frozen_tsa16','aligned8_tsa16'),('frozen_tsa16','aligned16_tsa16'),('aligned8_tsa16','aligned16_tsa16'),('aligned8','aligned16'),('frozen','aligned8_final'),('frozen_tsa16','aligned16_final'),('aligned16_tsa16','aligned16_final')]:
    cc=dict(size=grid,**protocol.paired_comparison(cases,baseline,challenger))
    cc['failed']=sum(r['failed'] for r in summary if r['method'] in [baseline,challenger]);comparisons.append(cc)
 write_csv(out/'runtime_summary.csv',summaries)
 (out/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
 selected=[r for r in comparisons if r['neural']=='continued_selected' and r['classical']!='frozen']
 decision={'passed':all(r['failed']==0 and r['saving_percent']>=5 and r['saving_percent_ci95'][0]>0 for r in selected if r['size'] in [128,256]),
  'scope':'development only;conditional intervals;no confirmatory training-seed or OOD claim','test_generated':False,'threshold':'5% savings at BOTH128/256;zero failures;interval lower>0','selected_comparisons':selected}
 (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 settings.update(complete=True,wall_seconds=time.perf_counter()-start,measured_solves=sum(r['count'] for r in summaries)*3,
  gpu_after=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))
 (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
 print(json.dumps(decision),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--wall-seconds',type=float,default=900)
 p.add_argument('--alignment',action='store_true')
 p.add_argument('--alignment-directory')
 main(p.parse_args())
