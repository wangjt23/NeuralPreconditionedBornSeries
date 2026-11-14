"""Cross-evaluate CT/HSS/TSA on validation Krylov inputs and norm witnesses.

Arnoldi preconditioner inputs are labelled explicitly; they are not the same
as the current physical solver residual. Power probes are lower-bound witnesses
and do not certify a spectral norm or stationary convergence.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from rte.collision_transport import CollisionTransportPreconditioner
from rte.npbs_unet_ablation import solve_fgmres
from rte.probe_residual_amplification import amplification_probe
from rte.residual_subspace import physical_residual_loss
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator, source_from_data
from rte.run_residual_subspace_experiment import write_csv
from rte.train_collision_transport_ensemble import material_split
from rte.transport_baselines import ClassicalTransportStack


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    last = torch.load(Path(args.run)/'last.pt', map_location='cpu', weights_only=True)
    saved = SimpleNamespace(**{**last['args'], 'dataset_dir': ''})
    cfg = ExperimentConfig(size=saved.size, n_angles=saved.n_angles, regimes=saved.regimes,
                           anisotropy_g=saved.g, batch_size=saved.batch_size, eta_scale=0.)
    data = material_split(saved, cfg, 'validation')
    rows, witnesses, solves = [], [], []
    for index in range(args.count):
        ids = torch.tensor([index])
        q = source_from_data(cfg,data,ids,torch.device('cpu'))
        rte = make_operator(cfg,data['sigma_t'][ids],data['sigma_s'][ids],q,torch.device('cpu'))
        methods = {}
        for label, weights in [('selected_CT',last['best_state']),('last_CT',last['model'])]:
            m=CollisionTransportPreconditioner(rte,depth=4,learned=True,width=12).eval()
            m.load_state_dict(weights)
            methods[label]=m
        methods['HSS4']=CollisionTransportPreconditioner(rte,depth=4).eval()
        methods['TSA4']=ClassicalTransportStack(rte,'tsa',4,0.).eval()
        for origin, method in methods.items():
            inputs=[]
            def traced(residual):
                inputs.append(residual.detach().clone())
                return method(residual)
            result=solve_fgmres(rte,q,traced,30,1e-5,real_stacked=True)
            solves.append(dict(index=index,method=origin,iters=result[0],final=result[1]))
            for stage,residual in enumerate(inputs):
                for label,control in methods.items():
                    with torch.no_grad():
                        loss=float(physical_residual_loss(rte.A_op,residual,control(residual)))
                    rows.append(dict(index=index,origin=origin,probe='Arnoldi_preconditioner_input',
                                     stage=stage,method=label,physical_loss=loss,ratio=loss**.5))
            witnessed, after12, features=amplification_probe(rte.A_op,method,q,8800+index,args.power_steps)
            witnesses.append(dict(index=index,method=origin,one_step_norm_lower_bound=witnessed,
                                  stationary12_on_own_witness=after12,**features))
        write_csv(out/'krylov_inputs.csv',rows)
        write_csv(out/'norm_witnesses.csv',witnesses)
        write_csv(out/'solves.csv',solves)
        print('validation probe medium',index,flush=True)
    (out/'settings.json').write_text(json.dumps(dict(run=args.run,indices=list(range(args.count)),
        split='validation',device='cpu',power_steps=args.power_steps,
        interpretation='exploratory witnesses; no checkpoint selection or norm certificate'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--power-steps',type=int,default=30)
    p.add_argument('--output',required=True)
    run(p.parse_args())
