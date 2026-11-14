"""Frozen same-arithmetic short replay and complete classical comparison."""
import cProfile,pstats,hashlib,json,math,random,time
from pathlib import Path
import torch
from graph_classical_v2 import GraphClassicalV2
from segmented_tsa import SegmentedTSA
from direct_segmented_tsa import DirectSegmentedTSA
from direct_full_graph import DirectFullGraph
from rte.benchmark_tail_solver import solve
from rte.train_source_predictor import build_base,material_operator,source_family

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sync():torch.cuda.synchronize()

@torch.no_grad()
def run():
    torch.set_num_threads(1);root=Path(__file__).resolve().parent.parent
    policy=json.loads((root/'policy.json').read_text());assert policy['phase'] in ('direct_screen','direct_full16')
    for n,h in policy['code_sha256'].items():assert sha(root/n)==h
    out=root/'results';out.mkdir(exist_ok=False);began_process=time.perf_counter();deadline=began_process+policy['soft_seconds']
    checks=[];setup=[];applications=[];rows=[];diagnostics=[];contexts=[]
    def check_time():
        if time.perf_counter()>deadline:raise TimeoutError('bounded segmented process deadline')
    def save():
        for name,value in [('checks',checks),('setup',setup),('applications',applications),('solver',rows),('diagnostics',diagnostics)]:
            (out/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    for n in policy['grids']:
        parent=Path(f'rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{n}.pt');assert sha(parent)==policy['development_parent_sha256'][str(n)]
        data=torch.load(parent,map_location='cpu',weights_only=True);cfg,reg=source_family(n,16,4626)
        for key in ('sigma_t','sigma_s','centers','phases','amplitudes','angular_codes','geometry_base64'):assert torch.equal(data[key],reg[key])
        for index in policy['development_ids']:
            check_time();sync();t=time.perf_counter();low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda');sync();physical=time.perf_counter()-t
            name='tsa8' if n==128 else 'tsa16';depth=int(name[3:]);recipe='reference_tail' if n==128 else 'tsa16_zero'
            sync();t=time.perf_counter();base=build_base(low,name);sync();base_setup=time.perf_counter()-t
            sync();t=time.perf_counter();P=high.phase_matrix.to(torch.complex128)*high.weights[None,:];mu=torch.linalg.eigvalsh((P+P.mH)/2).max().real;margin=float((high.sigma_t-high.sigma_s*mu).min());qnorm=float(high.source.norm());sync();acceptance_setup=time.perf_counter()-t
            assert margin>0 and math.isfinite(margin) and qnorm>0 and math.isfinite(qnorm)
            # Diagnostic only: same-shape graph of one copy, no physical work.
            sync();t=time.perf_counter();copy_input=low.source.detach().clone();copy_output=torch.empty_like(copy_input);side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):copy_output.copy_(copy_input)
            torch.cuda.current_stream().wait_stream(side);sync();copy_graph=torch.cuda.CUDAGraph()
            profile=cProfile.Profile();profile.enable()
            with torch.cuda.graph(copy_graph):copy_output.copy_(copy_input)
            profile.disable()
            sync();copy_setup=time.perf_counter()-t
            copy_times=[]
            for repeat in range(7):
                sync();t=time.perf_counter();copy_graph.replay();sync()
                if repeat>=2:copy_times.append(time.perf_counter()-t)
            assert torch.equal(copy_output,copy_input)
            profile_rows=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],total_calls=v[1],internal_seconds=v[2],cumulative_seconds=v[3]) for k,v in pstats.Stats(profile).stats.items() if 'collect' in k[2] or 'empty_cache' in k[2] or 'synchronize' in k[2] or 'capture' in k[2]]
            sync();t=time.perf_counter();raw_graph=torch.cuda.CUDAGraph()
            with torch.cuda.stream(side):
                raw_graph.capture_begin(capture_error_mode='global')
                try:copy_output.copy_(copy_input)
                finally:raw_graph.capture_end()
            sync();raw_copy_setup=time.perf_counter()-t;raw_graph.replay();sync();assert torch.equal(copy_output,copy_input)
            diagnostics.append(dict(size=n,index=index,copy_only_graph_setup_seconds=copy_setup,copy_only_replay_seconds=copy_times,context_profile=profile_rows,raw_copy_capture_seconds=raw_copy_setup,diagnostic_only=True,method_specific_cold_rank=False))
            wrappers={};capture={};order=['full_graph','segmented','direct_full','direct'] if index%2==0 else ['direct','direct_full','segmented','full_graph']
            for method in order:
                sync();t=time.perf_counter();wrapper=(DirectFullGraph if method=='direct_full' else GraphClassicalV2)(base,low.source) if method in ('full_graph','direct_full') else (DirectSegmentedTSA if method=='direct' else SegmentedTSA)(base,low.source,chunk=1,capture_first=False);sync();capture[method]=time.perf_counter()-t;wrappers[method]=wrapper
            setup.append(dict(size=n,index=index,physical_seconds=physical,base_seconds=base_setup,acceptance_setup_seconds=acceptance_setup,graph_constructor_seconds=capture,constructor_order=order,evaluated_margin=margin,depth=depth,capture_fft_counts={m:w.capture_fft_counts for m,w in wrappers.items()},segmented_info=wrappers['segmented'].info,direct_info=wrappers['direct'].info))
            stripe=(-1.)**torch.arange(n,device='cuda');grazing=torch.zeros_like(low.source);grazing[:,0]=stripe[None,None,:]
            probes=[('source',low.source),('grazing',grazing),('mixed',low.source.roll(1,-2)*(1.13+.21j))]
            for dtype in (torch.complex64,torch.complex128):
                for label,probe in probes:
                    rhs=probe.to(dtype);expected=base(rhs).to(torch.complex128)
                    for method,wrapper in wrappers.items():
                        got=wrapper(rhs);bitwise=bool(torch.equal(got,expected));finite=bool(torch.isfinite(got).all())
                        image_error=float((high.A_op(got)-high.A_op(expected)).norm()/high.A_op(expected).norm().clamp_min(1e-30))
                        checks.append(dict(size=n,index=index,method=method,probe=label,dtype=str(dtype),bitwise_equal=bitwise,finite=finite,image_error=image_error,passed=bitwise and finite and image_error==0.))
            for candidate in ('segmented','direct_full','direct'):
                wrapper=wrappers[candidate];held=wrapper(low.source);saved=held.clone();wrapper(grazing);ownership=bool(torch.equal(held,saved))
                old=base.reference._reference_transport_inverse;base.reference._reference_transport_inverse=old.transpose(1,2)
                try:
                    try:wrapper(low.source)
                    except RuntimeError:guard=True
                    else:guard=False
                finally:base.reference._reference_transport_inverse=old
                restored=bool(torch.equal(wrapper(low.source),base(low.source).to(torch.complex128)))
                checks.append(dict(size=n,index=index,kind='ownership_view_guard',method=candidate,ownership=ownership,same_pointer_square_view_guard=guard,restored=restored,passed=ownership and guard and restored))
            save();assert all(r['passed'] for r in checks)
            for repeat in range(7):
                for method in ['eager']+order:
                    sync();t=time.perf_counter();v=base(low.source).to(torch.complex128) if method=='eager' else wrappers[method](low.source);sync();duration=time.perf_counter()-t
                    if repeat>=2:applications.append(dict(size=n,index=index,method=method,repeat=repeat-2,seconds=duration,finite=bool(torch.isfinite(v).all())))
            save()
            contexts.append(dict(n=n,index=index,low=low,high=high,base=base,wrappers=wrappers,name=name,recipe=recipe,physical=physical,base_setup=base_setup,acceptance_setup=acceptance_setup,capture=capture,margin=margin,qnorm=qnorm))
            print(f'segmented entry N{n} ID{index} passed',flush=True)
            del copy_graph,raw_graph,copy_input,copy_output
    assert len(checks)==27*len(contexts) and all(r['passed'] for r in checks)
    for ctx in contexts:
        n,index,low,high,base,wrappers,name,recipe,physical,base_setup,acceptance_setup,capture,margin,qnorm=(ctx[k] for k in ('n','index','low','high','base','wrappers','name','recipe','physical','base_setup','acceptance_setup','capture','margin','qnorm'))
        methods=['eager','full_graph','segmented','direct_full','direct'];rng=random.Random(53109+n+index)
        for repeat in range(-1,policy['timed_repeats']):
            rng.shuffle(methods);values={};batch=[]
            for method in methods:
                check_time();bs={name:base if method=='eager' else wrappers[method]};sync();t=time.perf_counter();v,status=solve(recipe,low,high,bs,{});sync();duration=time.perf_counter()-t;values[method]=v
                sync();t=time.perf_counter();res=float((high.source-high.A_op(v)).norm());delta=res/margin;den=float(v.norm())-delta;bound=delta/den if den>0 else None;finite=bool(torch.isfinite(v).all());sync();acceptance=time.perf_counter()-t
                preparation=physical+base_setup+acceptance_setup+(capture[method] if method!='eager' else 0.)
                batch.append(dict(size=n,index=index,method=method,repeat=repeat,warmup=repeat<0,seconds=duration,acceptance_seconds=acceptance,accepted_seconds=duration+acceptance,setup_seconds=preparation,setup_plus_accepted_seconds=preparation+duration+acceptance,status=status,independent_true_residual=res/qnorm,evaluated_error_bound=bound,evaluated_margin=margin,finite=finite))
            for row in batch:
                v=values[row['method']];row['field_difference']=float((v-values['eager']).norm()/values['eager'].norm());row['bitwise_vs_eager']=bool(torch.equal(v,values['eager']))
                row['accuracy_passed']=row['finite'] and row['status']['failed']==0 and row['independent_true_residual']<=1e-5 and row['evaluated_error_bound'] is not None and math.isfinite(row['evaluated_error_bound']) and 0<=row['evaluated_error_bound']<=1e-3 and row['field_difference']<=1e-4
                row['work_and_residual_equal']=row['status']==next(r['status'] for r in batch if r['method']=='eager')
            rows.extend(batch);save()
            assert len(rows)<=policy['max_full_solves']
            assert all(r['accuracy_passed'] and r['bitwise_vs_eager'] and r['work_and_residual_equal'] for r in batch)
        print(f'segmented N{n} ID{index} complete',flush=True)
    assert len(rows)==policy['max_full_solves']
    settings=dict(complete=True,wall_seconds=time.perf_counter()-began_process,total_full_solves=len(rows),measured_full_solves=sum(not r['warmup'] for r in rows),all_accuracy_passed=all(r['accuracy_passed'] for r in rows),all_bitwise_full_solutions=all(r['bitwise_vs_eager'] for r in rows),all_work_and_residual_equal=all(r['work_and_residual_equal'] for r in rows),all_checks_passed=all(r['passed'] for r in checks),training=False,test9626_touched=False,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),workload='Q1primary, recorded setup+warmaccepted estimate, processcoldstartup/IOexcluded; Q16secondaryconditional',scope='same original TSA8/TSA16recurrence: eager/contextfullgraph/contextsegmented/directfullgraph/directsegmented; GCprofilediagnosticonly')
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n');print(json.dumps(settings),flush=True)

if __name__=='__main__':run()
