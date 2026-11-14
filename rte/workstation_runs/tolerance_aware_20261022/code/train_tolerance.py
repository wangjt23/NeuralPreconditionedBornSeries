"""A paired, fixed-budget loss intervention using existing TSA16 contexts only."""
import argparse,json,math,pathlib,sys,time
from types import SimpleNamespace
import torch

ROOT=pathlib.Path('/data/wjt/NeuralPreconditionedBornSeries')
PREVIOUS=ROOT/'rte/workstation_runs/training_priority_20261004'
BASE=ROOT/'rte/workstation_runs/tolerance_aware_20261022'
sys.path.insert(0,str(PREVIOUS/'code'))
import train_adequacy as ta
import train_alignment as alignment
from objective import components,original_prefix_factor,gradient_vector,gradient_comparison,TAU,BETA
from rte.tail_lookahead import continuation_loss
from rte.train_tail_direction import teacher_direction_loss

def settings_args():
    return SimpleNamespace(batch=4,cache_mode='auto',width=32,blocks=4,modes=6,seed=41022,horizon=1000,max_gpu_gib=14.)

def load_contexts(a,panels=False):
    common=torch.load(PREVIOUS/'alignment_retry/matched_original_ids.pt',map_location='cpu',weights_only=True).tolist()
    protocol=json.loads((BASE/'PROTOCOL.json').read_text())
    if common!=protocol['physical_ids']:raise ValueError('original matched cohort changed')
    contexts={};provenance={'warm_start':ta.WARM,'warm_sha256':ta.helper.file_digest(ta.WARM),'original_ids':common,'inputs':{}}
    for grid in [64,128]:
        p=PREVIOUS/f'alignment_retry/train_tsa16_n{grid}.pt'
        saved=alignment.restrict(torch.load(p,map_location='cpu',weights_only=True),common)
        bank=saved['bank'];bank['prefix_eta']=original_prefix_factor(bank['rhs_exact'],bank['original_rhs_norm'])
        contexts[grid]=saved;provenance['inputs'][str(p)]=ta.helper.file_digest(p)
    ta.helper.check_paired_identity(contexts[64],contexts[128],contexts=True)
    banks,cache=ta.helper.normalize_and_cache(contexts,a,'cuda');del contexts
    if torch.cuda.max_memory_reserved()>a.max_gpu_gib*1024**3:raise RuntimeError('cache 14GiB GPU budget exceeded')
    panel_banks={};parents={}
    if panels:
        raw={}
        for grid in [64,128]:
            p=PREVIOUS/f'alignment_retry/development_tsa16_n{grid}.pt'
            raw[grid]=torch.load(p,map_location='cpu',weights_only=True)
            b=raw[grid]['bank'];b['prefix_eta']=original_prefix_factor(b['rhs_exact'],b['original_rhs_norm'])
            provenance['inputs'][str(p)]=ta.helper.file_digest(p)
            parent=alignment.PARENTS[grid]['development']
            parents[grid]=torch.load(parent,map_location='cpu',weights_only=True)
            provenance['inputs'][parent]=ta.helper.file_digest(parent)
        panel_banks,pcache=ta.helper.normalize_and_cache(raw,a,'cuda');provenance['panel_cache']=pcache
        del raw
    for p in pathlib.Path(__file__).parent.glob('*.py'):provenance.setdefault('own_source_sha256',{})[str(p)]=ta.helper.file_digest(p)
    for name in ta.helper.SOURCE_MODULES:
        import importlib
        module=importlib.import_module('rte.'+name)
        provenance.setdefault('original_source_sha256',{})[module.__file__]=ta.helper.file_digest(module.__file__)
    return banks,panel_banks,parents,cache,provenance

def fields(bank,ids,ops,phase=None):
    return ta.batch_fields(bank,ids,ops,phase,depth=16)

def diagnostic(a,deadline):
    out=BASE/'diagnostic';out.mkdir(exist_ok=False)
    banks,_,_,cache,provenance=load_contexts(a,False)
    model=ta.load_model(ta.WARM,False,'cuda');model.train();ops=ta.prepare_ops(banks)
    parameters=list(model.parameters());totals={};rows=[];n=56
    for start in range(0,n,4):
        if time.perf_counter()>deadline:raise TimeoutError('gradient diagnostic wall cap')
        bank=banks[128];ids=torch.arange(start,start+4,device=bank['q'].device)
        batch,high,pre=fields(bank,ids,ops[128]);t,s,r,ref,target,basis=batch
        z=model(t,s,r.to(torch.complex64),ref).to(torch.complex128)
        ratios=continuation_loss(high.A_op,pre,z,r,basis,2)
        eta=bank['prefix_eta'][ids].to('cuda')
        old,excess=components(ratios,eta,'two_step');new,_=components(ratios,eta,'tolerance')
        teacher=.05*teacher_direction_loss(z,target).mean()
        values={'old_residual':old,'new_residual':new,'old_total':old+teacher,'new_total':new+teacher}
        for k,(name,value) in enumerate(values.items()):
            grads=torch.autograd.grad(value,parameters,retain_graph=k<len(values)-1,allow_unused=True)
            v=gradient_vector(grads,parameters)*(4/n)
            totals[name]=totals.get(name,torch.zeros_like(v))+v
        for j,idx in enumerate(ids.tolist()):
            for step in range(3):
                rows.append(dict(original_id=int(bank['indices'][idx]),continuation_calls=step,prefix_eta=float(eta[j]),
                    squared_ratio=float(ratios[j,step]),original_rhs_relative_residual=float(eta[j]*ratios[j,step].sqrt()),
                    log_squared_excess=float(excess[j,step]),gradient_weight=float(torch.sigmoid(excess[j,step]/BETA))))
        del values,grads,z,ratios,old,new,teacher
        if torch.cuda.max_memory_reserved()>a.max_gpu_gib*1024**3:raise RuntimeError('diagnostic 14GiB GPU budget exceeded')
    residual=gradient_comparison(totals['old_residual'],totals['new_residual'])
    total=gradient_comparison(totals['old_total'],totals['new_total'])
    admitted=total['relative_difference']>=1e-3 and total['one_minus_cosine']>=1e-6
    result=dict(complete=True,training_ids=n,grid=128,phase=0,updates=0,tau=TAU,beta=BETA,
        residual_gradient=residual,total_gradient=total,training_pair_discriminating=admitted,
        criterion='TOTAL gradient relative difference>=1e-3 AND 1-cos>=1e-6; frozen before diagnostic',
        limitation='noncollinearity is not a theorem of distinct Adam trajectories or better solver calls',cache=cache,provenance=provenance)
    ta.helper.write_rows(out/'threshold_exposure.csv',rows);ta.helper.write_json(out/'decision.json',result)
    print(json.dumps(result),flush=True)

@torch.no_grad()
def fixed_panel(model,banks,panels,ops,step,arm):
    rows=[];model.eval()
    for split,source in [('train',banks),('development',panels)]:
        for grid,bank in source.items():
            count=min(16,len(bank['q'])) if split=='train' else len(bank['q'])
            for start in range(0,count,4):
                ids=torch.arange(start,min(start+4,count),device=bank['q'].device)
                batch,high,pre=fields(bank,ids,ops[grid]);t,s,r,ref,target,basis=batch
                z=model(t,s,r.to(torch.complex64),ref).to(torch.complex128)
                ratios=continuation_loss(high.A_op,pre,z,r,basis,6)
                eta=bank['prefix_eta'][ids].to('cuda');teacher=teacher_direction_loss(z,target)
                for j,idx in enumerate(ids.tolist()):
                    row=dict(arm=arm,step=step,split=split,grid=grid,original_id=int(bank['indices'][idx]),prefix_eta=float(eta[j]),teacher_loss=float(teacher[j]))
                    row.update({f'squared_ratio_after_{k}':float(ratios[j,k]) for k in [0,1,2,4,6]})
                    row.update({f'original_rhs_residual_after_{k}':float(eta[j]*ratios[j,k].sqrt()) for k in [0,1,2,4,6]})
                    rows.append(row)
    return rows

def run_arm(a,arm,banks,panels,parents,plan,deadline,provenance):
    out=BASE/'training'/arm;out.mkdir()
    ta.helper.reset_rng(a.seed)
    model=ta.load_model(ta.WARM,False,'cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=1.5e-4,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1000,eta_min=1.5e-4*.05)
    ops=ta.prepare_ops(banks);history=[];validations=[];panels_rows=[];best=None;best_step=None;complete=False
    timing=dict(optimization=0.,validation=0.,panels=0.,checkpoint=0.)
    for step in range(1001):
        if step%250==0:
            if time.perf_counter()>deadline:break
            began=time.perf_counter();records=[]
            for grid,parent in parents.items():
                _,vv=alignment.validate_strong(model,parent,'cuda');records.extend(dict(grid=grid,**r) for r in vv)
            torch.cuda.synchronize();timing['validation']+=time.perf_counter()-began
            score=ta.helper.selection_score(records);new_best=best is None or score<best
            if new_best:best,best_step=score,step
            validations.extend(dict(arm=arm,step=step,**r) for r in records);ta.helper.write_rows(out/'validation.csv',validations)
            if step%500==0:
                began=time.perf_counter();panels_rows.extend(fixed_panel(model,banks,panels,ops,step,arm));torch.cuda.synchronize()
                timing['panels']+=time.perf_counter()-began;ta.helper.write_rows(out/'fixed_panel.csv',panels_rows)
            began=time.perf_counter()
            ta.checkpoint(out/'last.pt',model,opt,scheduler,step,plan,a,history,best,best_step,provenance)
            if new_best:ta.checkpoint(out/'best.pt',model,opt,scheduler,step,plan,a,history,best,best_step,provenance)
            if step in [0,1000]:ta.checkpoint(out/f'step{step}.pt',model,opt,scheduler,step,plan,a,history,best,best_step,provenance)
            timing['checkpoint']+=time.perf_counter()-began
            ta.helper.write_json(out/'progress.json',dict(arm=arm,step=step,best_step=best_step,score=score,timing=timing))
            print(f'{arm} step {step}: score={score}',flush=True)
        if step==1000:complete=True;break
        if time.perf_counter()>deadline:break
        began=time.perf_counter();grid=128 if step%2 else 64;bank=banks[grid]
        ids=plan['ids'][step].to(bank['q'].device);angles=plan['phase_angles'][step].to('cuda')
        phase=torch.polar(torch.ones_like(angles),angles)
        batch,high,pre=fields(bank,ids,ops[grid],phase);t,s,r,ref,target,basis=batch
        model.train();z=model(t,s,r.to(torch.complex64),ref).to(torch.complex128)
        ratios=continuation_loss(high.A_op,pre,z,r,basis,2);eta=bank['prefix_eta'][ids].to('cuda')
        objective,excess=components(ratios,eta,arm);teacher=teacher_direction_loss(z,target).mean()
        loss=objective+.05*teacher
        if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();scheduler.step();torch.cuda.synchronize();seconds=time.perf_counter()-began;timing['optimization']+=seconds
        history.append(dict(arm=arm,step=step+1,grid=grid,loss=float(loss),residual_objective=float(objective),teacher_loss=float(teacher),
            injection_squared_ratio=float(ratios[:,0].mean()),terminal_squared_ratio=float(ratios[:,-1].mean()),
            original_rhs_terminal_residual=float((eta*ratios[:,-1].sqrt()).mean()),
            terminal_above_original_tolerance=float((excess[:,-1]>0).double().mean()),
            mean_tolerance_weight=float(torch.sigmoid(excess/BETA).mean()),grad=float(grad),lr=opt.param_groups[0]['lr'],seconds=seconds))
        if torch.cuda.max_memory_reserved()>a.max_gpu_gib*1024**3:raise RuntimeError('14GiB GPU budget exceeded')
        if (step+1)%50==0:ta.helper.write_rows(out/'history.csv',history)
    if not complete:ta.checkpoint(out/'last.pt',model,opt,scheduler,len(history),plan,a,history,best,best_step,provenance)
    ta.helper.write_rows(out/'history.csv',history)
    ta.helper.write_rows(out/'sampling_consumed.csv',ta.helper.sampling_rows(plan,ta.helper.ARMS[1],len(history)))
    receipt=dict(arm=arm,complete=complete,updates=len(history),selected_step=best_step,best_score=best,timing=timing,
        primary_checkpoint='step1000.pt' if complete else None,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        checkpoint_sha256={p.name:ta.helper.file_digest(p) for p in out.glob('*.pt')})
    ta.helper.write_json(out/'receipt.json',receipt)
    del opt,scheduler,model,ops;torch.cuda.empty_cache()
    return receipt

def train(a,deadline):
    decision=json.loads((BASE/'diagnostic/decision.json').read_text())
    if not decision['complete'] or not decision['training_pair_discriminating']:raise RuntimeError('predeclared gradient identifiability gate failed')
    out=BASE/'training';out.mkdir(exist_ok=False)
    banks,panels,parents,cache,provenance=load_contexts(a,True)
    before=decision['provenance']
    for key in ['warm_start','warm_sha256','original_ids','own_source_sha256','original_source_sha256']:
        if before[key]!=provenance[key]:raise ValueError('diagnostic provenance changed: '+key)
    for name,digest in before['inputs'].items():
        if provenance['inputs'].get(name)!=digest:raise ValueError('diagnostic input changed: '+name)
    if decision['tau']!=TAU or decision['beta']!=BETA:raise ValueError('objective changed after diagnostic')
    ta.helper.reset_rng(a.seed);plan=ta.helper.build_sampling_plan(banks[64]['indices'].cpu(),1000,4,a.seed)
    ta.helper.save_checkpoint(out/'sampling_plan.pt',plan)
    provenance['sampling_identity_phase_sha256']=plan['identity_phase_sha256']
    settings=dict(args=vars(a),provenance=provenance,cache=cache,complete=False,arms=[],scope='existing train3626, reused development4626;9626 unopened',primary='fixed matched update1000 endpoints; best checkpoints secondary')
    ta.helper.write_json(out/'settings.json',settings)
    start=time.perf_counter()
    try:
        for arm in ['two_step','tolerance']:
            torch.cuda.reset_peak_memory_stats()
            if time.perf_counter()>deadline:break
            result=run_arm(a,arm,banks,panels,parents,plan,deadline,provenance)
            settings['arms'].append(result);ta.helper.write_json(out/'settings.json',settings)
            if not result['complete']:break
        settings['complete']=len(settings['arms'])==2 and all(x['complete'] for x in settings['arms'])
    finally:
        settings['wall_seconds']=time.perf_counter()-start;ta.helper.write_json(out/'settings.json',settings)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['diagnostic','train'],required=True);p.add_argument('--wall-seconds',type=float,required=True)
    cli=p.parse_args();a=settings_args();a.output=str(BASE/'training');a.stage=cli.stage
    if cli.stage=='diagnostic':
        import unittest,test_objective
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_objective))
        if not result.wasSuccessful():raise SystemExit('objective algebra guard failed')
    torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();deadline=start+cli.wall_seconds
    (diagnostic if cli.stage=='diagnostic' else train)(a,deadline)
