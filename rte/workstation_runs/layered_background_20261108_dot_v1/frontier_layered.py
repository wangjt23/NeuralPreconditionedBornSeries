"""Full16 development discriminator: geometry versus cheaper early directions.

No training. No oracle or new teacher. Original Fourier A64 and qualified
TSA16/seg remain the solve/acceptance contract. This is reused development,
not a held-out confirmation or exhaustive transport-solver comparison.
"""
import json,time,sys,os,hashlib,subprocess,gc,random,math
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"rte/workstation_runs/direct_capture_20261002/confirmation/shard0/code"))
from layered_inverse import LayeredInverse,CapturedAction
from direct_segmented_tsa import DirectSegmentedTSA
from rte.run_large_angular_spectral_experiments import ExperimentConfig
from rte.train_source_predictor import material_operator,build_base
from rte.source_initialization import solve_from_initial
from rte.augmented_fgmres import augmented_fgmres

def sync():torch.cuda.synchronize()
def competitors():
    v=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
    return [int(x.strip()) for x in v.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
def safe(x):
    if isinstance(x,float) and not math.isfinite(x):return str(x)
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,list):return [safe(v) for v in x]
    return x

@torch.no_grad()
def main():
    torch.set_num_threads(1)
    if competitors():raise RuntimeError("Actual competing GPU job before admission")
    prior=json.loads((HERE/"complete_01/settings.json").read_text())
    assert prior["complete"],"Prior owned worker must finish before this stage"
    out=HERE/"frontier_01";out.mkdir(exist_ok=False)
    began=time.perf_counter();rows=[];contexts=[]
    methods=["TSA16_native","TSA16_augmented","reference_eager_inject1","reference_graph_inject1",
             "TSA1_eager_inject1","TSA1_graph_inject1","layered_G_inject1"]
    settings=dict(complete=False,pid=os.getpid(),training=False,new_teachers=False,heldout9626=False,
        ids=list(range(16)),grids=[128,256],methods=methods,repeats=3,warmups=1,
        status="Full16 reused development; exploratory low-cost classical direction comparison.",
        protocol="Native TSA16/seg decision control; same augmented entry and p1 for all direction controls. Original A64 acceptance, tol1e-5, bound1e-3, field1e-4.",
        setup="Single RHS per material; common physical+acceptance+TSA caches/capture charged to every method. Only method-specific extra setup added. Eager reference/TSA1 reuse existing cache.",
        capture="Private direct warmed raw CUDA capture and fresh output storage; no context-manager GC/cache flush.",
        code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")})
    def save():
        (out/"settings.json").write_text(json.dumps(safe(settings),indent=2)+"\n")
        (out/"rows.json").write_text(json.dumps(safe(rows),indent=2)+"\n")
        (out/"contexts.json").write_text(json.dumps(safe(contexts),indent=2)+"\n")
    save();torch.cuda.reset_peak_memory_stats()
    try:
        for grid in [128,256]:
            data=torch.load(ROOT/f"rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{grid}.pt",map_location="cpu",weights_only=True)
            cfg=ExperimentConfig(size=grid,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes="mixed")
            for index in range(16):
                other=competitors()
                if other:raise RuntimeError("New GPU competitor; own worker stops: "+str(other))
                sync();t=time.perf_counter();low,high=material_operator(cfg,data,torch.tensor([index]),.1,"cuda");sync()
                physical=time.perf_counter()-t
                sync();t=time.perf_counter()
                phase=high.phase_matrix.double()*high.weights.double()[None]
                mu=torch.linalg.eigvalsh((phase+phase.T)/2).max()
                margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0
                sync();acceptance_setup=time.perf_counter()-t
                sync();t=time.perf_counter();base=build_base(low,"tsa16");sync();base_setup=time.perf_counter()-t
                sync();t=time.perf_counter();tsa=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False);sync()
                tsa_capture=time.perf_counter()-t
                pre=lambda r:tsa(r.to(torch.complex64)).to(torch.complex128)
                ref=lambda r:base.reference.reference_transport_solve(r.to(torch.complex64)).to(torch.complex128)
                one=lambda r:base.single(r.to(torch.complex64)).to(torch.complex128)
                sync();t=time.perf_counter();ref_graph=CapturedAction(ref,high.source);sync();ref_setup=time.perf_counter()-t
                sync();t=time.perf_counter();one_graph=CapturedAction(one,high.source);sync();one_setup=time.perf_counter()-t
                sync();t=time.perf_counter();g=LayeredInverse(low,scheme="upwind",nyquist=True,dtype=torch.complex64);sync()
                g_factor=time.perf_counter()-t
                sync();t=time.perf_counter();g_graph=CapturedAction(lambda r:g(r).to(torch.complex128),high.source);sync()
                g_capture=time.perf_counter()-t
                captures={"reference":ref_graph.relative_capture_difference,"TSA1":one_graph.relative_capture_difference,"layered":g_graph.relative_capture_difference}
                assert max(captures.values())<1e-12,captures
                action={"reference_eager_inject1":ref,"reference_graph_inject1":ref_graph,
                        "TSA1_eager_inject1":one,"TSA1_graph_inject1":one_graph,"layered_G_inject1":g_graph}
                extra={"TSA16_native":0.,"TSA16_augmented":0.,"reference_eager_inject1":0.,"TSA1_eager_inject1":0.,
                       "reference_graph_inject1":ref_setup,"TSA1_graph_inject1":one_setup,"layered_G_inject1":g_factor+g_capture}
                common=physical+acceptance_setup+base_setup+tsa_capture
                ctx=dict(grid=grid,index=index,axis=g.axis,physical_setup=physical,acceptance_setup=acceptance_setup,
                    tsa_base_setup=base_setup,tsa_capture_setup=tsa_capture,extra_setup=extra,
                    layered_factor_seconds=g_factor,layered_capture_seconds=g_capture,factor_bytes=g.factor_bytes,capture_differences=captures)
                contexts.append(ctx)
                baseline=None;source_norm=float(high.source.norm())
                rng=random.Random(912080+grid+index)
                for repeat in range(-1,3):
                    order=methods.copy();rng.shuffle(order)
                    if repeat==-1:
                        order.remove("TSA16_native");order.insert(0,"TSA16_native")
                    for name in order:
                        sync();t=time.perf_counter()
                        if name=="TSA16_native":
                            value,status=solve_from_initial(high.A_op,high.source,pre,None,maxiter=100)
                            status.update(classical_calls=status["iterations"],injection_calls=0)
                        elif name=="TSA16_augmented":
                            value,status=augmented_fgmres(high.A_op,high.source,pre,None,maxiter=100)
                        else:
                            fun=action[name]
                            value,status=augmented_fgmres(high.A_op,high.source,pre,None,inject=lambda x,r:fun(r),inject_after=1,maxiter=100)
                        sync();solver=time.perf_counter()-t;status.pop("trace",None)
                        if baseline is None:
                            assert status["failed"]==0
                            baseline=value.clone()
                        sync();t=time.perf_counter()
                        residual=float((high.source-high.A_op(value)).norm()/source_norm)
                        delta=residual*source_norm/margin;den=float(value.norm())-delta
                        bound=delta/den if den>0 else float("inf")
                        diff=float((value-baseline).norm()/baseline.norm())
                        accepted=bool(torch.isfinite(value).all()) and residual<=1e-5 and bound<=1e-3 and diff<=1e-4 and not status["failed"]
                        sync();checking=time.perf_counter()-t
                        rows.append(dict(grid=grid,index=index,method=name,repeat=repeat,solver_seconds=solver,
                            acceptance_seconds=checking,accepted_seconds=solver+checking,setup_seconds=common+extra[name],
                            Q1_seconds=common+extra[name]+solver+checking,status=status,accepted=accepted,
                            true_residual=residual,evaluated_relative_bound=bound,relative_field_difference=diff))
                    save()
                print("FRONTIER_CONTEXT",grid,index,"rows",len(rows),flush=True)
                del action,g_graph,ref_graph,one_graph,g,tsa,base,low,high,baseline
                gc.collect();sync();torch.cuda.empty_cache()
        settings["complete"]=True
    except BaseException as exc:
        settings["error"]=repr(exc);raise
    finally:
        settings.update(wall_seconds=time.perf_counter()-began,rows=len(rows),contexts=len(contexts),
            measured_solves=sum(r["repeat"]>=0 for r in rows),warmups=sum(r["repeat"]<0 for r in rows),
            failures=sum(not r["accepted"] for r in rows),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save()
if __name__=="__main__":main()
