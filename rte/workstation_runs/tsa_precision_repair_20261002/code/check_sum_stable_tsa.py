"""Operator-parity/work-count feasibility only; no training or solver timings."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import torch
from sum_stable_tsa import SumStableAlgebraicTSA
from rte.transport_baselines import ClassicalTransportStack
from rte.train_source_predictor import source_family, material_operator

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def relative(a,b):
    return float((a-b).norm()/b.norm().clamp_min(1e-30))

def counted_apply(op,rhs):
    counts={"fftn":0,"ifftn":0}
    original_fft,original_ifft=torch.fft.fftn,torch.fft.ifftn
    def fft(*a,**k):counts["fftn"]+=1;return original_fft(*a,**k)
    def ifft(*a,**k):counts["ifftn"]+=1;return original_ifft(*a,**k)
    with patch("torch.fft.fftn",fft),patch("torch.fft.ifftn",ifft):out=op(rhs)
    return out,counts

@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    old=Path("rte/workstation_runs/resume_audit_20261002")
    prior=json.loads(Path("rte/workstation_runs/direction_diagnostic_20261002/results/settings.json").read_text())
    start=time.perf_counter()
    settings=dict(scope="two fixed development IDs [0,3], three nested grids, two operator probes, two input precisions; one selective sum-precision repair; no full solves/timing comparison/training/new source dataset/test access",
        device=args.device,gate="all finite; complex64 relative field and original-physical-image differences <=1e-4; complex128-input differences <=1e-9; exact FFT counts 6d+2(d-1) versus 4d+2(d-1)",
        input_precision_note="complex128-input checks retain actual stored FP32 material/geometry/reference inverse; not a fully double-precision rebuilt operator",
        script_sha256=digest(__file__),helper_sha256=digest(Path(__file__).with_name("sum_stable_tsa.py")),
        development_parent_sha256=prior["validation_parent_sha256"],rows=[],test_generated=False,
        gpu_before=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv"],text=True))
    (out/"settings.json").write_text(json.dumps(settings,indent=2)+"\n")
    torch.cuda.reset_peak_memory_stats()
    records=[]
    setup_records=[]
    for n in (64,128,256):
        parent=old/"runtime_gate"/f"validation_n{n}.pt"
        assert digest(parent)==prior["validation_parent_sha256"][str(n)]
        saved=torch.load(parent,map_location="cpu",weights_only=True)
        cfg,regenerated=source_family(n,16,4626)
        for key in ("sigma_t","sigma_s","centers","phases","amplitudes","angular_codes","geometry_base64"):
            assert torch.equal(saved[key],regenerated[key]),key
        for index in (0,3):
            low,physical=material_operator(cfg,saved,torch.tensor([index]),.1,args.device)
            stripe=(-1.)**torch.arange(n,device=args.device)
            grazing=torch.zeros_like(low.source);grazing[:,0]=stripe[None,None,:]
            for depth in (1,16):
                torch.cuda.synchronize();t0=time.perf_counter()
                original=ClassicalTransportStack(low,"tsa",depth,0.)
                torch.cuda.synchronize();old_setup=time.perf_counter()-t0
                torch.cuda.synchronize();t0=time.perf_counter()
                alternative=SumStableAlgebraicTSA(low,depth=depth,eta_scale=0.)
                alternative._check_stored_physics()
                torch.cuda.synchronize();new_setup=time.perf_counter()-t0
                setup_records.append(dict(size=n,index=index,depth=depth,original_setup_seconds=old_setup,alternative_setup_and_guard_seconds=new_setup,
                    original_inverse_bytes=original.reference.reference_transport_inverse.numel()*original.reference.reference_transport_inverse.element_size(),
                    alternative_inverse_bytes=alternative.reference.reference_transport_inverse.numel()*alternative.reference.reference_transport_inverse.element_size(),
                    persistent_extra_precision_cache_bytes=0))
                for dtype in (torch.complex64,torch.complex128):
                    for name,probe in (("existing_physical_source",low.source),("deterministic_grazing_nyquist",grazing)):
                        rhs=probe.to(dtype)
                        a,ca=counted_apply(original,rhs);b,cb=counted_apply(alternative,rhs)
                        expected_old=3*depth+depth-1;expected_new=2*depth+depth-1
                        assert ca==dict(fftn=expected_old,ifftn=expected_old)
                        assert cb==dict(fftn=expected_new,ifftn=expected_new)
                        field=relative(b,a)
                        image=relative(physical.A_op(b.to(torch.complex128)),physical.A_op(a.to(torch.complex128)))
                        limit=1e-4 if dtype==torch.complex64 else 1e-9
                        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                        records.append(dict(size=n,index=index,depth=depth,input_dtype=str(dtype),probe=name,relative_field_difference=field,
                            relative_original_physical_image_difference=image,finite=finite,parity_limit=limit,passed=finite and field<=limit and image<=limit,
                            original_fft_counts=ca,alternative_fft_counts=cb))
                        (out/"cases.json").write_text(json.dumps(records,indent=2,allow_nan=False)+"\n")
            del low,physical,original,alternative
        print(f"Operator-parity checks N{n} complete",flush=True)
    settings.update(complete=True,checks=len(records),all_passed=all(r["passed"] for r in records),wall_seconds=time.perf_counter()-start,
        max_field_difference_by_dtype={d:max(r["relative_field_difference"] for r in records if r["input_dtype"]==d) for d in ("torch.complex64","torch.complex128")},
        max_image_difference_by_dtype={d:max(r["relative_original_physical_image_difference"] for r in records if r["input_dtype"]==d) for d in ("torch.complex64","torch.complex128")},
        spatial_transform_reduction_tsa16=32/126,
        setup_records=setup_records,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        precision_policy="G/scatter/local/R/outerA retain input dtype; stage sums and accumulator/output complex128; accumulator cast to input before outerA; FP32 inverse retained",
        extra_precision_transform_calls=0,
        extra_complex64_to_complex128_field_casts_per_tsa16=32,
        extra_complex128_to_complex64_outer_field_casts_per_tsa16=15,
        conceptual_complex64_cast_copy_read_write_bytes_per_tsa16_n256=47*24*12*256*256,
        complex128_accumulator_bytes_n256=16*12*256*256,
        accumulator_storage_increase_vs_complex64_bytes_n256=8*12*256*256,
        claim="Selective FP64 sums, original input-dtype outer defects and cached inverse; fewer transforms; no latency or whole-solver speedup claim",
        gpu_after=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv"],text=True))
    (out/"settings.json").write_text(json.dumps(settings,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:settings[k] for k in ("checks","all_passed","wall_seconds","max_field_difference_by_dtype","max_image_difference_by_dtype")},indent=2))
    if not settings["all_passed"]:raise SystemExit(2)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True);p.add_argument("--device",default="cuda")
    run(p.parse_args())
