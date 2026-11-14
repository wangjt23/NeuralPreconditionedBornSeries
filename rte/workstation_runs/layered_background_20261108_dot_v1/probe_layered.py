"""Stage A: actual-data layered setup/action and physical-oracle diagnostics."""
import json,time,sys,os,hashlib,statistics,subprocess,gc
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from layered_inverse import LayeredInverse,CapturedAction
from rte.run_large_angular_spectral_experiments import ExperimentConfig
from rte.train_source_predictor import material_operator

def sync():torch.cuda.synchronize()
def competitors():
    out=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
    return [int(x.strip()) for x in out.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
def timing(fun,rhs,repeats=7):
    for _ in range(2):fun(rhs)
    sync();times=[]
    for _ in range(repeats):
        sync();t=time.perf_counter();fun(rhs);sync();times.append(time.perf_counter()-t)
    return dict(median=statistics.median(times),minimum=min(times),maximum=max(times),repeats=times)
def jsonsafe(v):
    if isinstance(v,float) and not __import__("math").isfinite(v):return str(v)
    if isinstance(v,dict):return {k:jsonsafe(x) for k,x in v.items()}
    if isinstance(v,list):return [jsonsafe(x) for x in v]
    return v

@torch.no_grad()
def main():
    torch.set_num_threads(1)
    if competitors():raise RuntimeError("GPU competition before admission: "+str(competitors()))
    out=HERE/"probe_01";out.mkdir(exist_ok=False)
    start=time.perf_counter();settings=dict(complete=False,pid=os.getpid(),training=False,heldout9626=False,
        scope="Four diagnostic contexts, existing development data; no new teachers or full PDE solves.",
        code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")})
    rows=[];setup=[]
    def save():
        (out/"settings.json").write_text(json.dumps(jsonsafe(settings),indent=2)+"\n")
        (out/"rows.json").write_text(json.dumps(jsonsafe(rows),indent=2)+"\n")
    save()
    torch.cuda.reset_peak_memory_stats()
    parents=torch.load(ROOT/"rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt",map_location="cpu",weights_only=True)
    try:
        for grid,index in [(128,0),(128,14),(256,0),(256,14)]:
            other=competitors()
            if other:raise RuntimeError("New GPU competitor; stop own worker: "+str(other))
            data=torch.load(ROOT/f"rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{grid}.pt",map_location="cpu",weights_only=True)
            cfg=ExperimentConfig(size=grid,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes="mixed")
            sync();t=time.perf_counter()
            low,high=material_operator(cfg,data,torch.tensor([index]),.1,"cuda")
            sync();physical_setup=time.perf_counter()-t
            teacher=None
            if grid==128:
                bank=parents["bank"]
                assert torch.equal(bank["t"][index:index+1].cuda(),low.sigma_t)
                assert torch.equal(bank["s"][index:index+1].cuda(),low.sigma_s)
                assert torch.equal(bank["q"][index:index+1].cuda(),low.source)
                teacher=bank["target"][index:index+1].cuda()
                teacher_residual=float((high.A_op(teacher)-high.source).norm()/high.source.norm())
                assert teacher_residual<2e-9,teacher_residual
            for scheme,nyquist,dtype in [("compact4",False,torch.complex64),("compact4",True,torch.complex64),
                                          ("compact4",True,torch.complex128),("upwind",True,torch.complex64)]:
                other=competitors()
                if other:raise RuntimeError("New GPU competitor; stop own worker: "+str(other))
                name=scheme+("_nyq" if nyquist else "")+("_64" if dtype==torch.complex128 else "_32")
                sync();t=time.perf_counter()
                action=LayeredInverse(low,scheme=scheme,nyquist=nyquist,dtype=dtype)
                sync();factor_seconds=time.perf_counter()-t
                rhs=high.source.to(dtype)
                value=action(rhs).to(torch.complex128);sync()
                source_norm=high.source.norm()
                original_defect=float((high.source-high.A_op(value)).norm()/source_norm)
                slab_defect=float((high.source-(high.A_op(value)-action.perturbation(value))).norm()/source_norm)
                surrogate_defect=float((rhs-action.surrogate_action(value)).norm()/rhs.norm())
                eager=timing(action,rhs)
                sync();t=time.perf_counter();captured=CapturedAction(action,rhs);sync()
                capture_seconds=time.perf_counter()-t
                graph=timing(captured,rhs)
                row=dict(grid=grid,index=index,name=name,axis=action.axis,physical_setup=physical_setup,
                    factor_seconds=factor_seconds,capture_seconds=capture_seconds,factor_bytes=action.factor_bytes,
                    eager=eager,graph=graph,capture_difference=captured.relative_capture_difference,
                    original_source_defect=original_defect,slab_fourier_defect=slab_defect,surrogate_defect=surrogate_defect)
                if teacher is not None:
                    q=action.perturbation(teacher)
                    corrected=action(high.source-q).to(torch.complex128)
                    row.update(teacher_residual=teacher_residual,
                        free_q_original_defect=float((high.source-high.A_op(corrected)).norm()/source_norm),
                        free_q_relative_teacher_error=float((corrected-teacher).norm()/teacher.norm()))
                    def two_calls(r):
                        incident=action(r)
                        result=action(r-q.to(dtype))
                        return result
                    row["free_q_two_G_time"]=timing(two_calls,rhs)
                rows.append(row);save()
                print("PROBE",grid,index,name,"setup",factor_seconds,"G",graph["median"],"Adef",original_defect,
                      "qdef",row.get("free_q_original_defect"),flush=True)
                del action,captured,value,rhs
                gc.collect();sync();torch.cuda.empty_cache()
            del low,high,teacher;gc.collect();sync();torch.cuda.empty_cache()
        settings["complete"]=True
    except BaseException as exc:
        settings["error"]=repr(exc)
        raise
    finally:
        settings.update(wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),rows=len(rows))
        save()
if __name__=="__main__":main()
