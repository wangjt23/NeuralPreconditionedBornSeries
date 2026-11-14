"""Same explicit TSA16 deployment for both fixed-endpoint loss interventions."""
import argparse,json,math,pathlib,random,subprocess,sys,time
import torch
import rte.benchmark_tail_solver as protocol
from rte.train_source_predictor import material_operator,build_base
from rte.run_large_angular_spectral_experiments import ExperimentConfig
from rte.augmented_fgmres import augmented_fgmres
from rte.run_residual_subspace_experiment import write_csv

ROOT=pathlib.Path('/data/wjt/NeuralPreconditionedBornSeries')
BASE=ROOT/'rte/workstation_runs/tolerance_aware_20261022'
OLD=ROOT/'rte/workstation_runs/training_priority_20261004'

def gate(comparisons,candidate,controls):
    selected=[]
    for grid in [128,256]:
        for comparator in controls[grid]:
            rows=[r for r in comparisons if r['size']==grid and r['classical']==comparator and r['neural']==candidate]
            if len(rows)!=1:raise ValueError('gate needs exactly one comparison per required grid/control')
            r=rows[0]
            if r['independent_media']!=16:raise ValueError('incomplete gate cohort')
            selected.append(r)
    return dict(passed=all(r['failed']==0 and r['saving_percent']>=5 and r['saving_percent_ci95'][0]>0 for r in selected),selected_comparisons=selected)

@torch.no_grad()
def main(a):
    torch.set_num_threads(1);start=time.perf_counter();deadline=start+a.wall_seconds
    train=json.loads((BASE/'training/settings.json').read_text())
    if not train['complete'] or len(train['arms'])!=2:raise RuntimeError('matched training pair incomplete')
    checkpoints={'two_step':BASE/'training/two_step/step1000.pt','tolerance':BASE/'training/tolerance/step1000.pt',
        'previous_aligned':OLD/'alignment_retry/tsa16/step1000.pt'}
    models={name:protocol.load_model(p,False,'cuda') for name,p in checkpoints.items()}
    out=BASE/'benchmark';out.mkdir(exist_ok=False)
    selection_path=ROOT/'rte/workstation_runs/tail_solver_benchmark_20260926/frozen_selection.json'
    selection=json.loads(selection_path.read_text());comparisons=[];summaries=[];measured=0;accuracy=[]
    settings=dict(complete=False,checkpoints={name:{'path':str(p),'sha256':protocol.digest(p)} for name,p in checkpoints.items()},
        classical_selection_sha256=protocol.digest(selection_path),scope='same16 reused development4626;no9626/test generation',
        explicit_neural_deployment='TSA16-zero, same four-call injection interface for all learned methods',
        repeats=5,warmup=1,order_seed=41022,primary_checkpoint='predeclared fixed update1000, not timing-selected',
        timing='synchronized warm solve includes reference/inference/projection/independent original-RHS check; setup separate; model load excluded',
        gpu_before=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    sync=lambda:torch.cuda.synchronize()
    for grid in [64,128,256]:
        p=ROOT/f'rte/workstation_runs/tail_solver_benchmark_20260926/validation_n{grid}.pt'
        data=torch.load(p,map_location='cpu',weights_only=True)
        settings.setdefault('data_sha256',{})[str(grid)]=protocol.digest(p)
        cfg=ExperimentConfig(size=grid,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed')
        classical=selection[str(grid)]['classical']
        methods=list(dict.fromkeys([classical,'tsa16_zero','reference_tail',*models]));rows=[];rng=random.Random(41022+grid)
        for index in range(16):
            if time.perf_counter()>deadline:raise TimeoutError('timing stage deadline; partial output retained')
            sync();began=time.perf_counter();low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda');sync();physical=time.perf_counter()-began
            bases={};setup={}
            for depth in ['tsa8','tsa16']:
                sync();began=time.perf_counter();bases[depth]=build_base(low,depth);sync();setup[depth]=time.perf_counter()-began
            order=methods.copy()
            for repeat in [-1,0,1,2,3,4]:
                rng.shuffle(order)
                for position,method in enumerate(order):
                    if time.perf_counter()>deadline:raise TimeoutError('timing inner deadline')
                    sync();began=time.perf_counter()
                    if method in models:
                        base=bases['tsa16'];model=models[method]
                        def inject(x,r):
                            ref=base.reference.reference_transport_solve(r.to(torch.complex64))
                            return model(low.sigma_t,low.sigma_s,r.to(torch.complex64),ref).to(torch.complex128)
                        field,status=augmented_fgmres(high.A_op,high.source,lambda r:base(r.to(torch.complex64)).to(torch.complex128),None,inject=inject)
                        depth='tsa16'
                    else:
                        field,status=protocol.solve(method,low,high,bases,{})
                        depth=method.rsplit('_',1)[0] if method.endswith(('_balanced','_zero')) else 'tsa8'
                    # Identical additional independent acceptance check for EVERY method.
                    residual=high.source-high.A_op(field);actual=float(residual.norm()/high.source.norm())
                    sync();seconds=time.perf_counter()-began
                    status.pop('trace',None);status.pop('initial_residual',None)
                    status['failed']=int(bool(status['failed']) or not math.isfinite(actual) or actual>1e-5)
                    status['true_residual']=actual
                    if repeat>=0:
                        rows.append(dict(size=grid,index=index,method=method,repeat=repeat,order=position,seconds=seconds,
                            setup_seconds=physical+setup[depth],setup_plus_seconds=physical+setup[depth]+seconds,**status));measured+=1
                    if repeat==0 and method in ['two_step','tolerance']:
                        # Cheap independent discrete coercivity estimates; not part of solve timer.
                        began=time.perf_counter();P=high.phase_matrix.double()*high.weights.double()[None,:]
                        mu=float(torch.linalg.eigvalsh((P+P.mT)/2).max())
                        margin=float((high.sigma_t-high.sigma_s*mu).min())
                        uniform=bool(torch.equal(high.weights,high.weights[0].expand_as(high.weights)))
                        nonnegative=bool((high.sigma_s>=0).all());bound=None
                        if margin>0 and uniform and nonnegative:
                            delta=float(residual.norm())/margin;norm=float(field.norm())
                            if norm>delta:bound=delta/(norm-delta)
                        sync();extra=time.perf_counter()-began
                        accuracy.append(dict(size=grid,index=index,method=method,true_residual=actual,margin=margin,
                            uniform_weights=int(uniform),nonnegative_scattering=int(nonnegative),evaluated_relative_error_bound=bound,extra_seconds=extra))
                        if index==0:
                            torch.save(dict(field=field.cpu(),source=high.source.cpu(),status=status,checkpoint_sha256=settings['checkpoints'][method]['sha256'],grid=grid,index=index),out/f'field_{method}_n{grid}_case0.pt')
                    if torch.cuda.max_memory_reserved()>14*1024**3:raise RuntimeError('14GiB benchmark memory cap')
            write_csv(out/f'runtime_n{grid}.csv',rows);write_csv(out/'accuracy.csv',accuracy)
            if (index+1)%4==0:print(f'N{grid}: {index+1}/16, measured={measured}',flush=True)
        cases,summary=protocol.summarize(rows)
        if any(r['count']!=16 for r in summary):raise ValueError('incomplete timing cohort')
        write_csv(out/f'case_medians_n{grid}.csv',cases);summaries.extend(dict(size=grid,**r) for r in summary)
        for candidate in ['two_step','tolerance']:
            for comparator in list(dict.fromkeys([classical,'tsa16_zero','reference_tail','previous_aligned','two_step'])):
                if candidate==comparator:continue
                c=dict(size=grid,**protocol.paired_comparison(cases,comparator,candidate))
                c['failed']=sum(r['failed'] for r in summary if r['method'] in [comparator,candidate]);comparisons.append(c)
    write_csv(out/'runtime_summary.csv',summaries)
    (out/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    primary={g:[selection[str(g)]['classical']] for g in [128,256]}
    strict={g:list(dict.fromkeys([selection[str(g)]['classical'],'tsa16_zero','reference_tail'])) for g in [128,256]}
    decision={name:{'original_gate':gate(comparisons,name,primary),'all_measured_classical_controls_gate':gate(comparisons,name,strict)} for name in ['two_step','tolerance']}
    decision.update(scope='conditional reused-development single-warm-start/seed experiment; no fresh/generalization claim',test_generated=False)
    (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    settings.update(complete=True,wall_seconds=time.perf_counter()-start,measured_solves=measured,
        accuracy_extra_seconds=sum(r['extra_seconds'] for r in accuracy),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu_after=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n');print(json.dumps(decision),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wall-seconds',type=float,default=540.)
    main(p.parse_args())
