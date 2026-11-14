"""Frozen48 parity then six-input classical launch-cost headroom screen."""
import hashlib,json,random,statistics,time
from pathlib import Path
import torch
from graph_classical import GraphClassical
from rte.transport_baselines import ClassicalTransportStack
from rte.train_source_predictor import source_family,material_operator,build_base
from rte.benchmark_tail_solver import solve

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sync():torch.cuda.synchronize()
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))

@torch.no_grad()
def run():
    torch.set_num_threads(1);root=Path(__file__).resolve().parent.parent
    output=root/'graph_oracle';output.mkdir(exist_ok=False)
    policy=json.loads((root/'graph_policy.json').read_text())
    for name,h in policy['code_sha256'].items():assert sha(root/name)==h
    prior=json.loads(Path('rte/workstation_runs/classical_tsa_algebra_20261002/operator_checks/settings.json').read_text())
    start=time.perf_counter();parity=[];rows=[];setups=[];applications=[]
    for n in (64,128,256):
        parent=Path(f'rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{n}.pt')
        assert sha(parent)==prior['development_parent_sha256'][str(n)]
        data=torch.load(parent,map_location='cpu',weights_only=True);cfg,reg=source_family(n,16,4626)
        for key in ('sigma_t','sigma_s','centers','phases','amplitudes','angular_codes','geometry_base64'):assert torch.equal(data[key],reg[key])
        for index in (0,3):
            low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda')
            stripe=(-1.)**torch.arange(n,device='cuda');grazing=torch.zeros_like(low.source);grazing[:,0]=stripe[None,None,:]
            for depth in (1,16):
                base=ClassicalTransportStack(low,'tsa',depth,0.)
                graph=GraphClassical(base,low.source)
                expected=4*depth-1
                assert graph.capture_fft_counts==dict(fftn=expected,ifftn=expected)
                for dtype in (torch.complex64,torch.complex128):
                    for name,probe in (('existing_physical_source',low.source),('deterministic_grazing_nyquist',grazing)):
                        rhs=probe.to(dtype);a=base(rhs).to(torch.complex128);b=graph(rhs)
                        limit=1e-4 if dtype==torch.complex64 else 1e-9
                        field=rel(b,a);image=rel(high.A_op(b),high.A_op(a));finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                        parity.append(dict(size=n,index=index,depth=depth,input_dtype=str(dtype),probe=name,field_relative=field,image_relative=image,
                            bitwise_equal=bool(torch.equal(a,b)),finite=finite,limit=limit,passed=finite and field<=limit and image<=limit,
                            original_algorithm_fft_counts=graph.capture_fft_counts,graph_replay_algorithm_fft_counts=graph.capture_fft_counts,
                            native128_fallback=dtype==torch.complex128))
                # Validate result ownership and stale-state guard without changing physics.
                held=graph(low.source);saved=held.clone();graph(grazing);assert torch.equal(held,saved)
                old_depth=base.depth;base.depth+=1
                try:graph(low.source)
                except RuntimeError:pass
                else:raise AssertionError('stale settings not rejected')
                base.depth=old_depth
                if depth==16:
                    pairs=[]
                    for repeat in range(7):
                        order=['original','graph'] if repeat%2==0 else ['graph','original']
                        for arm in order:
                            sync();t=time.perf_counter()
                            value=base(low.source).to(torch.complex128) if arm=='original' else graph(low.source)
                            sync();duration=time.perf_counter()-t
                            if repeat>=2:pairs.append(dict(arm=arm,repeat=repeat-2,seconds=duration))
                    applications.append(dict(size=n,index=index,graph_setup_seconds=graph.setup_seconds,timings=pairs))
                del graph,base
            (output/'parity_cases.json').write_text(json.dumps(parity,indent=2)+'\n')
            if not all(r['passed'] for r in parity):raise RuntimeError('parity failure blocks solver screen')
    assert len(parity)==48 and all(r['passed'] for r in parity)
    # Complete the entry gate before any full-solver comparison.
    for n in (64,128,256):
        parent=Path(f'rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{n}.pt')
        data=torch.load(parent,map_location='cpu',weights_only=True);cfg,_=source_family(n,16,4626)
        for index in (0,3):
            sync();t=time.perf_counter()
            low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda')
            sync();physics_setup_seconds=time.perf_counter()-t
            # Exact original methods; graph is an expanded traditional control.
            bases={};base_setups={}
            for name in ('tsa8','tsa16','reference1','reference2'):
                sync();t=time.perf_counter();bases[name]=build_base(low,name);sync();base_setups[name]=time.perf_counter()-t
            graph_bases=dict(bases)
            for name in ('tsa8','tsa16'):graph_bases[name]=GraphClassical(bases[name],low.source)
            graph_setup={name:graph_bases[name].setup_seconds for name in ('tsa8','tsa16')}
            setups.append(dict(size=n,index=index,physical_setup_seconds=physics_setup_seconds,base_setup_seconds=base_setups,graph_setup_seconds=graph_setup))
            original_frozen='reference_tail' if n<256 else 'tsa16_zero'
            methods=['frozen_original','frozen_graph','reference1_zero','reference2_zero']
            rng=random.Random(27003+n+index)
            for repeat in range(-1,3):
                rng.shuffle(methods)
                values={};newrows=[]
                for method in methods:
                    bs=graph_bases if method=='frozen_graph' else bases
                    name=original_frozen if method.startswith('frozen_') else method
                    sync();t=time.perf_counter();value,status=solve(name,low,high,bs,{})
                    sync();duration=time.perf_counter()-t
                    actual=float((high.source-high.A_op(value)).norm()/high.source.norm())
                    values[method]=value
                    if repeat>=0:newrows.append(dict(size=n,index=index,method=method,repeat=repeat,seconds=duration,independent_true_residual=actual,**status))
                if repeat>=0:
                    P=high.phase_matrix.to(torch.complex128)*high.weights.to(torch.float64)[None,:]
                    mu=torch.linalg.eigvalsh((P+P.mH)/2).max().real
                    margin=float((high.sigma_t-high.sigma_s*mu).min())
                    for row in newrows:
                        value=values[row['method']];error=float((high.source-high.A_op(value)).norm())/margin
                        denominator=float(value.norm())-error
                        bound=error/denominator if denominator>0 else float('inf')
                        row.update(field_difference_vs_frozen=rel(value,values['frozen_original']),evaluated_coercivity_margin=margin,
                            evaluated_relative_error_bound=bound,accuracy_passed=row['failed']==0 and row['independent_true_residual']<=1e-5 and bound<=1e-3 and rel(value,values['frozen_original'])<=1e-4)
                    rows.extend(newrows)
            (output/'solver_cases.json').write_text(json.dumps(rows,indent=2)+'\n')
            del low,high,bases,graph_bases
            print(f'graph/shallow oracle N{n} ID{index} complete',flush=True)
    sync()
    result=dict(complete=True,parity_checks=len(parity),all_parity_passed=all(r['passed'] for r in parity),all_bitwise_equal=all(r['bitwise_equal'] for r in parity),
                measured_solves=len(rows),independent_media=2,setup_records=setups,application_records=applications,wall_seconds=time.perf_counter()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),test_touched=False,training_run=False,
                scope='six grid/media inputs; traditional implementation/shallow frontier oracle, no learned advantage claim',all_solve_accepted=all(r['failed']==0 and r['independent_true_residual']<=1e-5 for r in rows),
                all_solve_accuracy_passed=all(r['accuracy_passed'] for r in rows))
    (output/'settings.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('complete','parity_checks','all_parity_passed','all_bitwise_equal','measured_solves','wall_seconds','all_solve_accepted','peak_allocated_bytes','peak_reserved_bytes')},indent=2))

if __name__=='__main__':run()
