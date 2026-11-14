"""Supplemental profiles of actual eligible branches; no runtime retuning.

The original profiles deliberately used the first two media. Those media
already converged after four TSA16 calls, so they do not explain its injection
cost. Select the first two eligible IDs per prefix, solely from the saved
eligibility flags, and retain the original profiles and unprofiled timings.
"""

import argparse
import json
from pathlib import Path
import time

import torch

from diagnose_tail_directions import (PREFIXES, custom_solve, protocol,
    source_family, material_operator, build_base, write_json, gpu_processes)


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    source=Path(args.results);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    parent=Path('rte/workstation_runs/resume_audit_20261002')
    checkpoint=parent/'paired_training/mixed64_128/best.pt'
    original=json.loads((source/'settings.json').read_text())
    assert protocol.digest(checkpoint)==original['model_sha256']['mixed64_128']
    started=time.perf_counter()
    model=protocol.load_model(checkpoint,False,'cuda')
    settings=dict(complete=False,scope='first two eligible IDs per saved prefix, not selected by speed; supplementary synchronized profiles and classical covariance only',
                  source=vars(args),checkpoint_sha256=protocol.digest(checkpoint),
                  script_sha256=protocol.digest(__file__),gpu_before=gpu_processes(),
                  profiling='perturbs timing; all original end-to-end benchmark and first-two-media profiles retained')
    write_json(out/'settings.json',settings)
    records,covariance=[],[]
    for size in (64,128,256):
        cfg,data=source_family(size,16,4626)
        prior=torch.load(parent/'runtime_gate'/f'validation_n{size}.pt',map_location='cpu',weights_only=True)
        for key in ('sigma_t','sigma_s','centers','phases','amplitudes','angular_codes','geometry_base64'):
            assert torch.equal(data[key],prior[key])
        prefix_rows=json.loads((source/f'prefixes_n{size}.json').read_text())
        for prefix in PREFIXES:
            ids=[r['index'] for r in prefix_rows if r['prefix']==prefix and r['eligible']][:2]
            settings.setdefault('selected_ids',{})[str(size)+'/'+prefix]=ids
            for index in ids:
                low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda')
                bases={name:build_base(low,name) for name in ('tsa8','tsa16')}
                methods=(('tsa8_reference','tsa8_reference_cost','tsa8_reference_phase4_cost','tsa8_mixed','tsa8_phase4')
                         if prefix=='tsa8_balanced' else
                         ('tsa16_plain','tsa16_noop_cost','tsa16_reference','tsa16_reference_cost','tsa16_mixed'))
                for method in methods:
                    _,status=custom_solve(method,low,high,bases,model,profile=True)
                    records.append(dict(size=size,index=index,prefix=prefix,method=method,**status))
                # Classical covariance is tested on a fixed physical source at
                # this material. Arbitrary complex linearity is proved from the
                # fixed Fourier/angular matrices, not from these six examples.
                q=high.source;Aq=high.A_op(q)
                base=bases[prefix.split('_')[0]]
                Bq=base(q.to(torch.complex64));Rq=base.reference.reference_transport_solve(q.to(torch.complex64))
                for phase in (1j,-1.,-1j):
                    relative=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
                    covariance.append(dict(size=size,index=index,prefix=prefix,phase=str(phase),
                        physical_action_defect=relative(high.A_op(phase*q),phase*Aq),
                        classical_precondition_defect=relative(base((phase*q).to(torch.complex64)),phase*Bq),
                        homogeneous_reference_defect=relative(base.reference.reference_transport_solve((phase*q).to(torch.complex64)),phase*Rq)))
        write_json(out/f'profiles_n{size}.json',[r for r in records if r['size']==size])
        write_json(out/'classical_phase_covariance.json',covariance)
        write_json(out/'settings.json',settings)
        print(f'Eligible branch profiles N{size} completed',flush=True)
    settings.update(complete=True,wall_seconds=time.perf_counter()-started,profiled_solves=len(records),covariance_checks=len(covariance),gpu_after=gpu_processes())
    write_json(out/'settings.json',settings)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
