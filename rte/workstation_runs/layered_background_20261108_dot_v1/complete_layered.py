"""Complete original-equation tests of layered actions; frozen two variants.

compact4 complex64 was rejected by own-equation defects in probe_01.
Compare compact4 complex128 (discretization control) and upwind complex64.
Private raw CUDA capture removes the avoidable context-manager setup overhead.
All output acceptance remains the stored original Fourier A64.
"""
import json,time,sys,os,hashlib,statistics,subprocess,gc,random,math,traceback
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
    out=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
    return [int(x.strip()) for x in out.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
def safe(v):
    if isinstance(v,float) and not math.isfinite(v):return str(v)
    if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
    if isinstance(v,list):return [safe(x) for x in v]
    return v
def inner(a,b):return (a.conj()*b).real.sum()

@torch.no_grad()
def main():
    torch.set_num_threads(1)
    if competitors():raise RuntimeError("Actual GPU competitor before admission")
    output=HERE/"complete_01";output.mkdir(exist_ok=False)
    started=time.perf_counter();rows=[];contexts=[]
    settings=dict(complete=False,pid=os.getpid(),training=False,heldout9626=False,new_teachers=False,
        diagnostic_ids=[0,4,7,14],grids=[128,256],warmups=1,repeats=3,maxiter=100,
        qualification="Diagnostic subset, not the full16 or independent confirmation.",
        candidate_selection="Upwind32 and compact4_64 with Nyquist. Compact4_32 rejected by probe own-equation error.",
        code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")})
    def save():
        (output/"settings.json").write_text(json.dumps(safe(settings),indent=2)+"\n")
        (output/"rows.json").write_text(json.dumps(safe(rows),indent=2)+"\n")
        (output/"contexts.json").write_text(json.dumps(safe(contexts),indent=2)+"\n")
    def accept(high,value,reference,margin):
        qn=float(high.source.norm())
        residual=float((high.source-high.A_op(value)).norm()/qn)
        delta=residual*qn/margin;den=float(value.norm())-delta
        bound=delta/den if den>0 else float("inf")
        diff=float((value-reference).norm()/reference.norm()) if reference is not None else 0.
        valid=bool(torch.isfinite(value).all()) and residual<=1e-5 and bound<=1e-3 and diff<=1e-4
        return dict(accepted=valid,true_residual=residual,evaluated_relative_bound=bound,relative_field_difference=diff)
    settings["production_hashes"]={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ["rte/models.py","rte/mixed_precision_transport.py","rte/augmented_fgmres.py","rte/source_initialization.py",
         "rte/npbs_unet_ablation.py","rte/train_source_predictor.py","rte/transport_baselines.py",
         "rte/workstation_runs/direct_capture_20261002/confirmation/shard0/code/direct_segmented_tsa.py"]}
    settings["torch"]=torch.__version__
    save();torch.cuda.reset_peak_memory_stats()
    parent=torch.load(ROOT/"rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt",map_location="cpu",weights_only=True)
    try:
        for grid in [128,256]:
            data=torch.load(ROOT/f"rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{grid}.pt",map_location="cpu",weights_only=True)
            cfg=ExperimentConfig(size=grid,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes="mixed")
            for index in settings["diagnostic_ids"]:
                other=competitors()
                if other:raise RuntimeError("New GPU competitor; own worker stops: "+str(other))
                sync();began=time.perf_counter();low,high=material_operator(cfg,data,torch.tensor([index]),.1,"cuda");sync()
                physical_setup=time.perf_counter()-began
                sync();began=time.perf_counter()
                phase=high.phase_matrix.double()*high.weights.double()[None]
                mu=torch.linalg.eigvalsh((phase+phase.T)/2).max()
                margin=float((high.sigma_t-high.sigma_s*mu).min())
                assert margin>0
                sync();acceptance_setup=time.perf_counter()-began
                sync();began=time.perf_counter();base=build_base(low,"tsa16");sync()
                base_setup=time.perf_counter()-began
                sync();began=time.perf_counter();tsa=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False);sync()
                tsa_capture=time.perf_counter()-began
                pre=lambda r:tsa(r.to(torch.complex64)).to(torch.complex128)
                teacher=None
                if grid==128:
                    for key,new in [("t",low.sigma_t),("s",low.sigma_s),("q",low.source)]:
                        assert torch.equal(parent["bank"][key][index:index+1].cuda(),new)
                    teacher=parent["bank"]["target"][index:index+1].cuda()
                    assert float((high.A_op(teacher)-high.source).norm()/high.source.norm())<2e-9
                baseline,bs=solve_from_initial(high.A_op,high.source,pre,None,maxiter=100)
                assert bs["failed"]==0
                prefix=[]
                sync();began=time.perf_counter()
                _,ps=augmented_fgmres(high.A_op,high.source,pre,None,inject_after=1,maxiter=1,capture_prefix=prefix.append)
                sync();prefix_oracle_acquisition=time.perf_counter()-began
                assert len(prefix)==1
                prefix=prefix[0]
                for scheme,dtype in [("upwind",torch.complex64),("compact4",torch.complex128)]:
                    other=competitors()
                    if other:raise RuntimeError("New GPU competitor; own worker stops: "+str(other))
                    label=scheme+("_64" if dtype==torch.complex128 else "_32")
                    sync();began=time.perf_counter();g=LayeredInverse(low,scheme=scheme,nyquist=True,dtype=dtype);sync()
                    factor_setup=time.perf_counter()-began
                    def plain(r):return g(r).to(torch.complex128)
                    def born2(r):
                        w=plain(r)
                        return w-plain(g.perturbation(w))
                    def gmres2(r):
                        z1=plain(r);w1=high.A_op(z1)
                        ww=inner(w1,w1).clamp_min(1e-60)
                        alpha=inner(r,w1)/ww
                        z2=plain(r-alpha*w1);w2=high.A_op(z2)
                        c=inner(w1,w2)/ww;orth=w2-c*w1
                        den=inner(orth,orth)
                        beta=torch.where(den>1e-24*inner(w2,w2).clamp_min(1e-60),
                                         inner(r,orth)/den.clamp_min(1e-60),torch.zeros_like(den))
                        return (alpha-beta*c)*z1+beta*z2
                    actions={};captures={}
                    for name,fun in [("G",plain),("Born2",born2),("GMRES2",gmres2)]:
                        sync();began=time.perf_counter();actions[name]=CapturedAction(fun,high.source);sync()
                        captures[name]=time.perf_counter()-began
                        assert actions[name].relative_capture_difference<1e-12
                    oracle_info=None
                    if teacher is not None:
                        tail=teacher-prefix["solution"]
                        qknown=g.perturbation(tail)
                        # The free-information arm still performs the incident G
                        # that a predictor conditioned on the incident field needs.
                        def oracle_action(r):
                            incident=plain(r)
                            return plain(r-qknown)
                        sync();began=time.perf_counter();actions["OracleQ"]=CapturedAction(oracle_action,high.source);sync()
                        captures["OracleQ"]=time.perf_counter()-began
                        trial=actions["OracleQ"](prefix["residual"]).clone()
                        oracle_info=dict(prefix_relative_residual=prefix["relative_residual"],
                            known_q_norm=float(qknown.norm()),teacher_equation_error_original_rhs=float((high.A_op(tail)-prefix["residual"]).norm()/high.source.norm()),
                            raw_proposal_residual_ratio=float((prefix["residual"]-high.A_op(trial)).norm()/prefix["residual"].norm()),
                            raw_proposal_residual_original_rhs=float((prefix["residual"]-high.A_op(trial)).norm()/high.source.norm()),
                            information_acquisition="Existing teacher + one captured classical prefix; excluded from optimistic online oracle, recorded separately.",
                            prefix_acquisition_seconds=prefix_oracle_acquisition)
                        # Verify the dynamic deployed prefix is the one for qknown.
                        seen=[]
                        def checked(x,r):
                            seen.append((torch.equal(x,prefix["solution"]),torch.equal(r,prefix["residual"])))
                            return actions["OracleQ"](r)
                        augmented_fgmres(high.A_op,high.source,pre,None,inject=checked,inject_after=1,maxiter=100)
                        assert seen==[(True,True)],seen
                    methods=["TSA16_native","TSA16_augmented","G_solver","Born2_solver","GMRES2_solver",
                             "G_inject1","Born2_inject1","GMRES2_inject1"]
                    if teacher is not None:methods.append("OracleQ_inject1")
                    ctx=dict(grid=grid,index=index,variant=label,axis=g.axis,physical_setup_seconds=physical_setup,acceptance_setup_seconds=acceptance_setup,
                             tsa_base_setup_seconds=base_setup,tsa_capture_seconds=tsa_capture,
                             factor_setup_seconds=factor_setup,factor_bytes=g.factor_bytes,
                             capture_seconds=captures,capture_differences={k:v.relative_capture_difference for k,v in actions.items()},
                             oracle=oracle_info)
                    contexts.append(ctx);save()
                    rng=random.Random(811080+grid+index)
                    for repeat in range(-1,3):
                        order=methods.copy();rng.shuffle(order)
                        for method in order:
                            sync();began=time.perf_counter();error=None
                            try:
                                if method=="TSA16_native":
                                    value,status=solve_from_initial(high.A_op,high.source,pre,None,maxiter=100)
                                elif method=="TSA16_augmented":
                                    value,status=augmented_fgmres(high.A_op,high.source,pre,None,maxiter=100)
                                elif method.endswith("_solver"):
                                    action_name=method.removesuffix("_solver")
                                    value,status=solve_from_initial(high.A_op,high.source,actions[action_name],None,maxiter=100)
                                else:
                                    action_name=method.removesuffix("_inject1")
                                    value,status=augmented_fgmres(high.A_op,high.source,pre,None,
                                        inject=lambda x,r:actions[action_name](r),inject_after=1,maxiter=100)
                                sync();solver_time=time.perf_counter()-began
                                status.pop("trace",None)
                                sync();check_started=time.perf_counter();accuracy=accept(high,value,baseline,margin);sync()
                                check_time=time.perf_counter()-check_started
                            except BaseException as exc:
                                sync();solver_time=time.perf_counter()-began;check_time=0.
                                error=repr(exc);status=dict(failed=1);accuracy=dict(accepted=False)
                                # A CUDA error must terminate rather than reuse uncertain device state.
                                if "CUDA" in str(exc) or "cuda" in str(exc):raise
                            if method.startswith("TSA16"):
                                prep=physical_setup+acceptance_setup+base_setup+tsa_capture
                            elif method.endswith("_solver"):
                                prep=physical_setup+acceptance_setup+factor_setup+captures[action_name]
                            else:
                                prep=physical_setup+acceptance_setup+base_setup+tsa_capture+factor_setup+captures[action_name]
                            row=dict(grid=grid,index=index,variant=label,method=method,repeat=repeat,
                                solver_seconds=solver_time,acceptance_seconds=check_time,accepted_seconds=solver_time+check_time,
                                setup_seconds=prep,Q1_seconds=prep+solver_time+check_time,status=status,**accuracy)
                            if error:row["error"]=error
                            rows.append(row)
                        save()
                    print("COMPLETE_CONTEXT",grid,index,label,"rows",len(rows),"oracle",oracle_info,flush=True)
                    del actions,g;gc.collect();sync();torch.cuda.empty_cache()
                del low,high,base,tsa,baseline,teacher,prefix;gc.collect();sync();torch.cuda.empty_cache()
        settings["complete"]=True
    except BaseException as exc:
        settings["error"]=repr(exc)
        raise
    finally:
        settings.update(wall_seconds=time.perf_counter()-started,rows=len(rows),contexts=len(contexts),
            measured_solves=sum(r["repeat"]>=0 for r in rows),warmups=sum(r["repeat"]<0 for r in rows),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save()
if __name__=="__main__":main()
