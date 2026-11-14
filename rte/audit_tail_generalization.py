"""Separate projected-response accuracy, generalization and phase symmetry.

Training and validation prefixes only. Not a timing benchmark; exact teacher
directions are information ceilings and are never deployed on test problems.
"""

import argparse
import json
from pathlib import Path
import statistics

import torch

from rte.augmented_fgmres import projected_direction_loss
from rte.benchmark_tail_solver import digest, load_model
from rte.mixed_precision_transport import high_precision_copy
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator
from rte.run_residual_subspace_experiment import write_csv
from rte.tail_moment_space import combine_moment_space
from rte.train_tail_direction import normalized_bank


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    models = {'tail': load_model(args.tail_checkpoint, False, 'cuda'),
              'optical_final': load_model(args.optical_checkpoint, True, 'cuda')}
    records = []
    for split in ('train', 'validation'):
        bank = normalized_bank(Path(args.contexts)/f'{split}_context.pt', 'cuda')
        for index in range(len(bank['q'])):
            t, s, r, reference, target, basis = [bank[k][index:index+1] for k in
                ('t', 's', 'rhs_exact', 'reference', 'target', 'image_basis')]
            cfg = ExperimentConfig(size=r.shape[-1], n_angles=12, anisotropy_g=.65, eta_scale=0., regimes='mixed')
            low = make_operator(cfg, t, s, r.to(torch.complex64), 'cuda')
            high = high_precision_copy(low)
            proposals = dict(reference=reference.to(torch.complex128), teacher=target)
            symmetry = {}
            for name, model in models.items():
                z = model(t, s, r.to(torch.complex64), reference).to(torch.complex128)
                rotated = model(t, s, (1j*r).to(torch.complex64), 1j*reference).to(torch.complex128)
                proposals[name] = z
                symmetry[name] = float((rotated-1j*z).norm()/z.norm().clamp_min(1e-30))
            for name, z in proposals.items():
                single = projected_direction_loss(high.A_op, z, r, basis)
                _, block = combine_moment_space(high.A_op, z, r, basis)
                coefficient = (z.conj()*target).real.sum()/z.abs().square().sum().clamp_min(1e-30)
                records.append(dict(split=split, index=index, method=name,
                    projected_residual_ratio=float(single.sqrt()),
                    moment_projected_residual_ratio=float(block.sqrt()),
                    optimally_scaled_field_error=float((coefficient*z-target).norm()/target.norm()),
                    phase_equivariance_error=symmetry.get(name, 0.)))
        write_csv(out/'cases.csv', records)
    summary = []
    for split in ('train', 'validation'):
        for name in ('reference', 'teacher', *models):
            rows = [r for r in records if r['split'] == split and r['method'] == name]
            entry = dict(split=split, method=name, count=len(rows))
            for key in ('projected_residual_ratio', 'moment_projected_residual_ratio',
                        'optimally_scaled_field_error', 'phase_equivariance_error'):
                entry[key+'_mean'] = statistics.mean(r[key] for r in rows)
                entry[key+'_median'] = statistics.median(r[key] for r in rows)
            summary.append(entry)
    write_csv(out/'summary.csv', summary)
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args), complete=True,
        scope='training and validation only; eligible prefixes; not online timing or held-out speed',
        source_sha256=digest(__file__),
        checkpoint_sha256={p:digest(p) for p in (args.tail_checkpoint, args.optical_checkpoint)}), indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--contexts', default='rte/workstation_runs/tail_injection_gate_20260926')
    parser.add_argument('--tail-checkpoint', default='rte/workstation_runs/tail_direction_pilot_20260926/best.pt')
    parser.add_argument('--optical-checkpoint', default='rte/workstation_runs/tail_optical_moment_20260926/last.pt')
    run(parser.parse_args())
