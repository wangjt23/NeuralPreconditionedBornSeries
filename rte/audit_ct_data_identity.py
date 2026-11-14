"""Check tensor-level split overlap and saved N16 source-action reproducibility."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from rte.collision_transport import CollisionTransportPreconditioner
from rte.residual_subspace import physical_residual_loss
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator, setup_batch, source_from_data
from rte.run_residual_subspace_experiment import write_csv
from rte.train_collision_transport_ensemble import bank_loss, material_split


def run(args):
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    roots=[Path('rte/results/collision_transport_ensemble_seed1')]
    roots += [Path(f'rte/workstation_runs/ct_n16_data256_seed{seed}_gpu_20260912') for seed in (2,3)]
    roots += [Path('rte/workstation_runs/ct_n32_data4096_seed101')]
    identities, comparisons, seen, overlaps, losses = [], [], {}, [], []
    for root in roots:
        saved=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
        a=SimpleNamespace(**{**saved['args'],'dataset_dir':'rte/handoff/2026-09-12' if saved['args']['seed']==1 else ''})
        cfg=ExperimentConfig(size=a.size,n_angles=a.n_angles,regimes=a.regimes,anisotropy_g=a.g,batch_size=a.batch_size,eta_scale=0.)
        for split in ('train','validation','test'):
            data=material_split(a,cfg,split)
            for index in range(len(data['sigma_t'])):
                digest=hashlib.sha256()
                for name in ('sigma_t','sigma_s'):
                    digest.update(data[name][index].numpy().tobytes())
                key=digest.hexdigest()
                item=dict(seed=a.seed,size=a.size,split=split,index=index,material_sha256=key)
                if key in seen:
                    overlaps.append(dict(first=seen[key],second=item))
                seen[key]=item
                identities.append(item)
            if a.size!=16:
                continue
            ids=torch.tensor([0]);q=source_from_data(cfg,data,ids,torch.device('cpu'))
            rte=make_operator(cfg,data['sigma_t'][ids],data['sigma_s'][ids],q,torch.device('cpu'))
            model=CollisionTransportPreconditioner(rte,depth=4,learned=True,width=12).eval()
            model.load_state_dict(saved['model'])
            old_losses={r['split']:float(r['physical_loss']) for r in csv.DictReader((root/'selected_losses.csv').open())}
            corrected_loss=bank_loss(model,rte,data,cfg,torch.device('cpu'),a.probe_fraction)
            losses.append(dict(seed=a.seed,split=split,saved_bank_loss=old_losses[split],
                               corrected_bank_loss=corrected_loss,absolute_difference=abs(corrected_loss-old_losses[split])))
            if split!='test':
                continue
            evaluation=Path('rte/workstation_runs') / ('ct_seed1_saved_gpu_eval_20260912' if a.seed==1 else f'ct_n16_data256_seed{a.seed}_gpu_eval_20260912')
            old=list(csv.DictReader((root/'results.csv').open()))
            lookup={int(r['index']):float(r['source_loss']) for r in old if r['method']=='learned_CT'}
            independent={int(r['index']):float(r['source_loss']) for r in csv.DictReader((evaluation/'results.csv').open()) if r['method']=='learned_CT'}
            with torch.no_grad():
                for index in range(len(data['sigma_t'])):
                    q=setup_batch(cfg,rte,data,torch.tensor([index]),torch.device('cpu'))
                    value=float(physical_residual_loss(rte.A_op,q,model(q)))
                    comparisons.append(dict(seed=a.seed,index=index,saved_source_loss=lookup[index],
                                            corrected_source_loss=value,absolute_difference=abs(value-lookup[index]),
                                            paired_evaluator_source_loss=independent[index],
                                            paired_evaluator_absolute_difference=abs(value-independent[index])))
    write_csv(out/'material_hashes.csv',identities)
    write_csv(out/'n16_source_rechecks.csv',comparisons)
    write_csv(out/'n16_bank_rechecks.csv',losses)
    (out/'summary.json').write_text(json.dumps(dict(materials=len(identities),overlaps=overlaps,
        n16_source_rechecks=len(comparisons),max_source_loss_difference=max(r['absolute_difference'] for r in comparisons),
        max_paired_evaluator_source_difference=max(r['paired_evaluator_absolute_difference'] for r in comparisons),
        max_bank_loss_difference=max(r['absolute_difference'] for r in losses),
        original_source_mismatches=[dict(seed=r['seed'],index=r['index']) for r in comparisons if r['absolute_difference']>1e-4],
        scope='exact stored coefficient tensor equality; no claim of distributional independence or geometry equivalence'),indent=2)+'\n')
    print((out/'summary.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    run(p.parse_args())
