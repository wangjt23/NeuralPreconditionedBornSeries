"""Development-only useful-direction and end-to-end cost diagnostic.

No training, teachers, checkpoint updates, or held-out test generation. The
original solver times the runtime controls; the isolated clone is used only
for prefix capture and separately labelled synchronized profiling.
"""

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch

import rte.benchmark_tail_solver as protocol
from rte.augmented_fgmres import augmented_fgmres
from rte.global_source_predictor import balance_integrated_source
from rte.tail_lookahead import continuation_loss
from rte.train_source_predictor import build_base, material_operator, source_family
from rte.run_residual_subspace_experiment import write_csv
from continuation_acceptance import corrected_continuation
from instrumented_fgmres import instrumented_fgmres


PHASES = (1., 1j, -1., -1j)
PREFIXES = ('tsa8_balanced', 'tsa16_zero')
MODEL_ARMS = ('n64_only', 'mixed64_128')
CUSTOM = ('tsa8_reference', 'tsa8_reference_cost', 'tsa8_reference_phase4_cost',
          'tsa8_mixed', 'tsa8_phase4', 'tsa16_plain', 'tsa16_noop_cost',
          'tsa16_reference', 'tsa16_reference_cost', 'tsa16_mixed')


def sync():
    torch.cuda.synchronize()


def gpu_processes():
    return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], text=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def inner(a, b):
    return (a.conj()*b).real.flatten(1).sum(1)


def length(value):
    return torch.linalg.vector_norm(value.flatten(1), dim=1)


def project(value, basis):
    for _ in range(2):
        for q in basis.unbind(1):
            value = value-q*inner(q, value)[:, None, None, None]
    return value


def line_angle(a, b):
    den = float(length(a)*length(b))
    if den <= 1e-30:
        return None
    cosine = min(1., abs(float(inner(a,b)))/den)
    return math.degrees(math.acos(cosine))


@torch.no_grad()
def model_proposal(model, low, residual, reference, phases=(1.,)):
    outputs = []
    for phase in phases:
        # Quarter turns only change signs and interchange FP32 components.
        rotated = model(low.sigma_t, low.sigma_s,
                        residual.to(torch.complex64)*phase,
                        reference.to(torch.complex64)*phase)
        outputs.append((rotated*complex(phase).conjugate()).to(torch.complex128))
    return torch.stack(outputs).mean(0)


def initial_for(prefix, low, high, base):
    if prefix.endswith('_zero'):
        return None
    initial = base.reference.reference_transport_solve(low.source).to(torch.complex128)
    return balance_integrated_source(high, initial, high.source)


@torch.no_grad()
def useful_record(action, precondition, proposal, residual, basis, frontier):
    image = action(proposal.to(torch.complex128))
    useful = project(image, basis)
    image_norm, useful_norm, residual_norm = map(lambda x: float(length(x)), (image, useful, residual))
    independent = useful_norm > 1e-12*max(image_norm,1e-30)
    alpha = float(inner(residual,useful))/max(useful_norm**2,1e-30)
    remaining = residual-alpha*useful if independent else residual
    immediate = float(length(remaining))**2/max(residual_norm**2,1e-30)
    corrected = corrected_continuation(action, precondition, proposal, residual,
                                       basis, frontier, steps=2)
    masked = continuation_loss(action, precondition, proposal, residual, basis, steps=2)
    record = dict(image_norm=image_norm, useful_image_norm=useful_norm,
        useful_image_fraction=useful_norm/max(image_norm,1e-30),
        image_independent=independent,
        immediate_squared_ratio=immediate,
        immediate_residual_reduction=1-math.sqrt(max(0.,immediate)),
        corrected_ratios=corrected['ratios'][0].tolist(),
        masked_ratios=masked[0].tolist(),
        proposal_accepted=bool(corrected['proposal_accepted'][0]),
        rejection_code=int(corrected['proposal_rejection_code'][0]),
        classical_calls=int(corrected['classical_calls'][0]),
        accepted_frontier_breakdown=bool(corrected['accepted_frontier_breakdown'][0]),
        masked_corrected_max_difference=float((masked-corrected['ratios']).abs().max()))
    return record, image, useful


@torch.no_grad()
def diagnostics(cfg, data, models, out, count, deadline):
    records, prefixes = [], []
    for index in range(count):
        if time.perf_counter() >= deadline:
            raise TimeoutError('soft wall budget reached before next diagnostic medium')
        low, high = material_operator(cfg, data, torch.tensor([index]), .1, 'cuda')
        bases = {name:build_base(low,name) for name in ('tsa8','tsa16')}
        for prefix in PREFIXES:
            base=bases[prefix.split('_')[0]]
            precondition=lambda r:base(r.to(torch.complex64)).to(torch.complex128)
            captures=[]
            initial=initial_for(prefix,low,high,base)
            _, status=instrumented_fgmres(high.A_op,high.source,precondition,initial,
                                         maxiter=4,capture_prefix=captures.append)
            eligible=(status['true_residual']>1e-5 and bool(captures) and
                      captures[0]['frontier'] is not None and captures[0]['image_basis'] is not None)
            receipt=dict(size=cfg.size,index=index,prefix=prefix,eligible=eligible,
                         prefix_true_residual=status['true_residual'],prefix_classical_calls=status['classical_calls'])
            if not eligible:
                prefixes.append(receipt);continue
            context=captures[0]
            r=context['residual'];frontier=context['frontier']
            basis=context['image_basis'].transpose(0,1)
            gram=torch.einsum('bk...,bj...->bkj',basis.conj(),basis).real
            receipt.update(residual_basis_overlap=float(torch.stack([inner(q,r) for q in basis.unbind(1)]).norm()/r.norm()),
                           image_gram_defect=float((gram-torch.eye(basis.shape[1],device=gram.device)).norm()),
                           frontier_norm=float(frontier.norm()))
            prefixes.append(receipt)
            reference=base.reference.reference_transport_solve(r.to(torch.complex64))
            controls={'reference':reference.to(torch.complex128),
                      'classical_residual':precondition(r),
                      'classical_frontier':precondition(frontier),
                      'zero_rejection':torch.zeros_like(r)}
            classical_useful={}
            for label,z in controls.items():
                m,_,w=useful_record(high.A_op,precondition,z,r,basis,frontier)
                records.append(dict(size=cfg.size,index=index,prefix=prefix,arm='classical',variant=label,**m))
                classical_useful[label]=w
            # The zero control deliberately triggers the known masked fallback
            # error. It is not evidence that a deployed model is often rejected.
            for arm,model in models.items():
                proposals=[model_proposal(model,low,r,reference,(phase,)) for phase in PHASES]
                useful=[];phase_rows=[]
                for phase_index,z in enumerate(proposals):
                    m,_,w=useful_record(high.A_op,precondition,z,r,basis,frontier)
                    useful.append(w)
                    phase_rows.append(dict(size=cfg.size,index=index,prefix=prefix,arm=arm,
                        variant='phase'+str(phase_index),phase_quarters=phase_index,
                        raw_phase_defect=float((z-proposals[0]).norm()/proposals[0].norm().clamp_min(1e-30)),
                        **m))
                for phase_index,row in enumerate(phase_rows):
                    row.update(useful_angle_to_phase0_deg=line_angle(useful[0],useful[phase_index]),
                               useful_angle_to_reference_deg=line_angle(classical_useful['reference'],useful[phase_index]),
                               useful_angle_to_classical_frontier_deg=line_angle(classical_useful['classical_frontier'],useful[phase_index]))
                records.extend(phase_rows)
                averaged=torch.stack(proposals).mean(0)
                m,_,w=useful_record(high.A_op,precondition,averaged,r,basis,frontier)
                # Empirical diagnostics of positive homogeneity and additivity.
                scaled=model_proposal(model,low,2*r,2*reference)
                second=frontier*r.norm()
                second_ref=base.reference.reference_transport_solve(second.to(torch.complex64))
                second_z=model_proposal(model,low,second,second_ref)
                sum_z=model_proposal(model,low,r+second,reference+second_ref)
                difference=sum_z-proposals[0]-second_z
                records.append(dict(size=cfg.size,index=index,prefix=prefix,arm=arm,variant='phase4_average',
                    phase0_raw_positive_homogeneity_defect=float((scaled-2*proposals[0]).norm()/(2*proposals[0].norm()).clamp_min(1e-30)),
                    phase0_raw_additivity_defect=float(difference.norm()/(proposals[0]+second_z).norm().clamp_min(1e-30)),
                    phase0_useful_additivity_defect=float(project(high.A_op(difference),basis).norm()/
                        project(high.A_op(proposals[0]+second_z),basis).norm().clamp_min(1e-30)),
                    useful_angle_to_phase0_deg=line_angle(w,useful[0]),**m))
        write_json(out/f'direction_metrics_n{cfg.size}.json',records)
        write_json(out/f'prefixes_n{cfg.size}.json',prefixes)
        print(f'Directions N{cfg.size}: {index+1}/{count}',flush=True)
    return records,prefixes


@torch.no_grad()
def custom_solve(method,low,high,bases,model,profile=False):
    if method.startswith('tsa16'):
        prefix='tsa16_zero';base=bases['tsa16']
    else:
        prefix='tsa8_balanced';base=bases['tsa8']
    initial=initial_for(prefix,low,high,base)
    precondition=lambda r:base(r.to(torch.complex64)).to(torch.complex128)
    callback_parts={'reference_seconds':0.,'model_seconds':0.,'model_forwards':0}
    def measured(call,key):
        if not profile:return call()
        sync();started=time.perf_counter();value=call();sync()
        callback_parts[key]+=time.perf_counter()-started
        return value
    def inject(x,r):
        r=r.to(torch.complex64)
        ref=measured(lambda:base.reference.reference_transport_solve(r),'reference_seconds')
        if method in ('tsa8_reference','tsa16_reference'):
            return ref
        if method in ('tsa8_phase4','tsa8_reference_phase4_cost'):
            count=4
            value=measured(lambda:model_proposal(model,low,r,ref,PHASES),'model_seconds')
        else:
            count=1
            value=measured(lambda:model(low.sigma_t,low.sigma_s,r,ref),'model_seconds')
        callback_parts['model_forwards']+=count
        if method in ('tsa8_reference_cost','tsa8_reference_phase4_cost','tsa16_reference_cost'):return ref
        if method=='tsa16_noop_cost':return torch.zeros_like(r)
        return value
    callback=None if method=='tsa16_plain' else inject
    solver=instrumented_fgmres if profile else augmented_fgmres
    kwargs={'profile':True} if profile else {}
    result,status=solver(high.A_op,high.source,precondition,initial,inject=callback,**kwargs)
    status.pop('trace',None)
    if profile:status['callback_parts']=callback_parts
    return result,status


@torch.no_grad()
def benchmark(cfg,data,models,out,count,repeats,classical,deadline):
    methods=['frozen_classical',*CUSTOM]
    if classical=='reference_tail':methods.remove('tsa8_reference')
    rows,profiles,equality=[],[],[]
    rng=random.Random(27702+cfg.size)
    for index in range(count):
        if time.perf_counter()>=deadline:raise TimeoutError('soft wall budget reached before next timed medium')
        sync();started=time.perf_counter()
        low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda')
        sync();physical=time.perf_counter()-started
        bases,setup={},{}
        for name in ('tsa8','tsa16'):
            sync();started=time.perf_counter();bases[name]=build_base(low,name);sync()
            setup[name]=time.perf_counter()-started
        def solve(label):
            if label=='frozen_classical':return protocol.solve(classical,low,high,bases,{})
            return custom_solve(label,low,high,bases,models['mixed64_128'])
        # Matched cost-only controls must leave the actual solution unchanged.
        reference_label='frozen_classical' if classical=='reference_tail' else 'tsa8_reference'
        for first,second in [(reference_label,'tsa8_reference_cost'),
                             (reference_label,'tsa8_reference_phase4_cost'),
                             ('tsa16_plain','tsa16_noop_cost'),
                             ('tsa16_reference','tsa16_reference_cost')]:
            a,sa=solve(first);b,sb=solve(second)
            defect=float((a-b).norm()/a.norm().clamp_min(1e-30))
            assert defect<1e-10 and sa['iterations']==sb['iterations'] and sa['failed']==sb['failed']
            equality.append(dict(size=cfg.size,index=index,baseline=first,cost_control=second,
                                 relative_solution_difference=defect,iterations=sa['iterations']))
        for repeat in range(-1,repeats):
            order=list(methods);rng.shuffle(order)
            for position,label in enumerate(order):
                sync();started=time.perf_counter();_,status=solve(label);sync()
                seconds=time.perf_counter()-started
                if repeat>=0:
                    name='tsa16' if label.startswith('tsa16') or (label=='frozen_classical' and classical=='tsa16_zero') else 'tsa8'
                    row=dict(split='development_diagnostic',size=cfg.size,index=index,repeat=repeat,
                             order=position,method=label,seconds=seconds,setup_seconds=physical+setup[name],
                             setup_plus_seconds=physical+setup[name]+seconds,**status)
                    rows.append(row)
        write_csv(out/f'runtime_n{cfg.size}.csv',rows)
        write_json(out/f'cost_control_equivalence_n{cfg.size}.json',equality)
        print(f'Runtime N{cfg.size}: {index+1}/{count}',flush=True)
    cases,summary=protocol.summarize(rows)
    write_csv(out/f'case_medians_n{cfg.size}.csv',cases)
    write_csv(out/f'summary_n{cfg.size}.csv',summary)
    comparisons=[]
    pairs=[('frozen_classical',m) for m in methods if m!='frozen_classical']
    reference_label='frozen_classical' if classical=='reference_tail' else 'tsa8_reference'
    pairs += [(reference_label,'tsa8_reference_cost'),
              (reference_label,'tsa8_reference_phase4_cost'),
              ('tsa8_reference_cost','tsa8_mixed'),
              ('tsa8_reference_phase4_cost','tsa8_phase4'),
              ('tsa8_mixed','tsa8_phase4'),('tsa16_plain','tsa16_noop_cost'),
              ('tsa16_noop_cost','tsa16_mixed'),('tsa16_reference','tsa16_reference_cost'),
              ('tsa16_reference_cost','tsa16_mixed'),('tsa16_reference','tsa16_mixed')]
    for baseline,candidate in dict.fromkeys(pairs):
        comparison=protocol.paired_comparison(cases,baseline,candidate)
        comparison.update(size=cfg.size,failed=sum(r['failed'] for r in summary if r['method'] in (baseline,candidate)))
        comparisons.append(comparison)
    write_json(out/f'comparisons_n{cfg.size}.json',comparisons)
    return rows,cases,summary,comparisons


@torch.no_grad()
def profile_cases(cfg,data,model,out,count):
    records=[]
    # Profiles are intentionally outside all warm timing windows.
    for index in range(count):
        low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda')
        bases={name:build_base(low,name) for name in ('tsa8','tsa16')}
        for method in CUSTOM:
            _,status=custom_solve(method,low,high,bases,model,profile=True)
            records.append(dict(size=cfg.size,index=index,method=method,**status))
    write_json(out/f'profiles_n{cfg.size}.json',records)
    return records


def aggregate_metrics(records,prefixes):
    grouped={}
    for row in records:grouped.setdefault((row['size'],row['prefix'],row['arm'],row['variant']),[]).append(row)
    keys=('useful_image_fraction','immediate_squared_ratio','immediate_residual_reduction',
          'raw_phase_defect','useful_angle_to_phase0_deg','phase0_raw_positive_homogeneity_defect',
          'phase0_raw_additivity_defect','phase0_useful_additivity_defect','masked_corrected_max_difference')
    summaries=[]
    for group,rows in sorted(grouped.items()):
        result=dict(size=group[0],prefix=group[1],arm=group[2],variant=group[3],count=len(rows),
                    rejected=sum(not r['proposal_accepted'] for r in rows),
                    mean_final_corrected_squared_ratio=statistics.mean(r['corrected_ratios'][-1] for r in rows))
        for key in keys:
            values=[r[key] for r in rows if r.get(key) is not None]
            if values:result['mean_'+key]=statistics.mean(values);result['max_'+key]=max(values)
        summaries.append(result)
    return dict(groups=summaries,eligible_prefixes=sum(r['eligible'] for r in prefixes),
                total_prefixes=len(prefixes),scope='eligible tails only for direction metrics; all development media in timed full solves')


def run(args):
    if not 1<=args.count<=16 or args.repeats<1 or args.profile_count<0:raise ValueError('invalid bounded counts')
    torch.set_num_threads(1);torch.manual_seed(27702)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=started+args.max_wall_seconds
    parent=Path('rte/workstation_runs/resume_audit_20261002')
    checkpoints={arm:parent/'paired_training'/arm/'best.pt' for arm in MODEL_ARMS}
    selection_path=Path('rte/workstation_runs/tail_solver_benchmark_20260926/frozen_selection.json')
    selections=json.loads(selection_path.read_text())
    settings=dict(args=vars(args),complete=False,scope='development-only; same 16 physical media nested across resolutions; zero teachers/training/test access',
        model_sha256={arm:protocol.digest(p) for arm,p in checkpoints.items()},
        classical_selection_sha256=protocol.digest(selection_path),
        phases=[0,1,2,3],prefixes=PREFIXES,custom_methods=CUSTOM,
        decision_rule='stop this fixed nonlinear one-shot mechanism unless an already specified learned candidate is zero-failure and >=5% faster than frozen classical at BOTH N128/N256 with paired intervals above zero; this reused development diagnostic never generates held-out test data',
        timing='original uninstrumented solver; reference initialization, every neural call/projection/operator/final original-RHS check included; physical/base setup separately; resident model loading separately; profiling excluded',
        profiling='synchronized perturbed execution only; callback reference/model sub-times overlap the enclosing neural_callback category',
        source_sha256={str(p):protocol.digest(p) for p in Path(__file__).parent.glob('*.py')},
        original_source_sha256={p:protocol.digest(p) for p in ['rte/augmented_fgmres.py','rte/tail_lookahead.py','rte/benchmark_tail_solver.py','rte/tail_response_network.py','rte/train_source_predictor.py']},
        gpu_processes_before=gpu_processes(),torch=str(torch.__version__))
    write_json(out/'settings.json',settings)
    models={};loading={}
    for arm,p in checkpoints.items():
        sync();began=time.perf_counter();models[arm]=protocol.load_model(p,False,'cuda');sync()
        loading[arm]=time.perf_counter()-began
    settings['model_loading_seconds']=loading;write_json(out/'settings.json',settings)
    all_metrics,all_prefixes,all_summaries,all_comparisons=[],[],[],[]
    measured=0
    try:
        for size in args.sizes:
            cfg,data=source_family(size,16,4626)
            prior_path=parent/'runtime_gate'/f'validation_n{size}.pt'
            prior=torch.load(prior_path,map_location='cpu',weights_only=True)
            for key in ('sigma_t','sigma_s','centers','phases','amplitudes','angular_codes','geometry_base64'):
                if not torch.equal(data[key],prior[key]):raise ValueError('development material/source identity changed: '+key)
            settings.setdefault('validation_parent_sha256',{})[str(size)]=protocol.digest(prior_path)
            write_json(out/'settings.json',settings)
            metrics,prefixes=diagnostics(cfg,data,models,out,args.count,deadline)
            all_metrics.extend(metrics);all_prefixes.extend(prefixes)
            rows,cases,summary,comparisons=benchmark(cfg,data,models,out,args.count,args.repeats,selections[str(size)]['classical'],deadline)
            measured+=len(rows);all_summaries.extend(dict(size=size,**r) for r in summary);all_comparisons.extend(comparisons)
            profile_cases(cfg,data,models['mixed64_128'],out,min(args.profile_count,args.count))
        write_json(out/'direction_summary.json',aggregate_metrics(all_metrics,all_prefixes))
        write_csv(out/'runtime_summary.csv',all_summaries);write_json(out/'runtime_comparisons.json',all_comparisons)
        learned=('tsa8_mixed','tsa8_phase4','tsa16_mixed')
        advances=[]
        full_decision_scope=args.count==16 and set(args.sizes)=={64,128,256}
        if full_decision_scope:
            for candidate in learned:
                comparisons=[r for r in all_comparisons if r['classical']=='frozen_classical' and r['neural']==candidate and r['size'] in (128,256)]
                if len(comparisons)==2 and all(r['failed']==0 and r['saving_percent']>=5 and r['saving_percent_ci95'][0]>0 for r in comparisons):advances.append(candidate)
        recommendation=('continue diagnostic validation of '+','.join(advances) if advances else
            'stop the tested fixed nonlinear one-shot correction as a speed mechanism; retain classical controls and the isolated semantic repair') if full_decision_scope else 'engineering smoke only; no scientific advancement decision'
        write_json(out/'decision.json',dict(learned_candidates_passing_development_rule=advances,
            full_decision_scope=full_decision_scope,recommended=recommendation,
            test_generated=False,reserved_test_seed=9626,
            qualification='single paired training seed; repeatedly reused development media; no claim of held-out or OOD superiority'))
        settings.update(complete=True,measured_solves=measured,test_generated=False,
                        measured_failures=sum(r['failed'] for r in all_summaries),
                        learned_candidates_passing_development_rule=advances)
    except BaseException as error:
        settings['error']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        settings.update(wall_seconds=time.perf_counter()-started,gpu_processes_after=gpu_processes())
        write_json(out/'settings.json',settings)
    print('Completed direction diagnostic; no test data generated',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--sizes',type=int,nargs='+',choices=(64,128,256),default=[64,128,256])
    p.add_argument('--count',type=int,default=16)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--profile-count',type=int,default=2)
    p.add_argument('--max-wall-seconds',type=float,default=900)
    run(p.parse_args())
