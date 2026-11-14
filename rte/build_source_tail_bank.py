"""Prepare the next learning target: actual post-Krylov error, not full flux.

Training/validation-only data. Four classical steps start from the tuned analytical
initializer. Already solved cases bypass AI; other cases expose the slow
remaining error. Accurate full solutions are reused only as offline labels.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from rte.global_source_predictor import balance_integrated_source
from rte.mixed_precision_transport import high_precision_copy
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator
from rte.run_residual_subspace_experiment import write_csv
from rte.source_initialization import solve_from_initial
from rte.transport_baselines import ClassicalTransportStack


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    split = 'train'
    if args.validation_checkpoint:
        from rte.train_source_predictor import acquire, source_family
        saved = torch.load(args.validation_checkpoint, map_location='cpu', weights_only=True)['args']
        validation_cfg, validation_data = source_family(saved['size'], saved['val_count'], saved['seed']+2000)
        acquire(validation_cfg, validation_data, saved['epsilon'], torch.device(args.device), out, 'validation')
        args.teacher_bank = str(out/'validation.pt')
        split = 'validation'
    payload = torch.load(args.teacher_bank, map_location='cpu', weights_only=True)
    parent = payload['bank']
    n = parent['q'].shape[-1]
    cfg = ExperimentConfig(size=n, n_angles=parent['q'].shape[1], anisotropy_g=args.g, eta_scale=0., regimes='mixed')
    device = torch.device(args.device)
    entries = {key: [] for key in ('t', 's', 'q', 'rhs_exact', 'target', 'reference')}
    rows = []
    start = time.perf_counter()
    for index in range(len(parent['q'])):
        t, s, q, target = [parent[k][index:index+1].to(device) for k in ('t', 's', 'q', 'target')]
        low = make_operator(cfg, t, s, q, device)
        high = high_precision_copy(low)
        source_norm = high.source.norm()
        teacher_residual = float((high.source-high.A_op(target)).norm()/source_norm)
        if teacher_residual > 2e-9:
            raise ValueError('teacher/operator identity or accuracy mismatch')
        base = ClassicalTransportStack(low, 'tsa', 8, 0.)
        analytical = balance_integrated_source(high,
            base.reference.reference_transport_solve(low.source).to(torch.complex128), high.source)
        prefix, status = solve_from_initial(high.A_op, high.source,
            lambda r: base(r.to(torch.complex64)).to(torch.complex128), analytical, maxiter=args.prefix_steps)
        rhs = high.source-high.A_op(prefix)
        error = target-prefix
        reference = base.reference.reference_transport_solve(rhs.to(torch.complex64))
        eligible = status['true_residual'] > 1e-5
        equation_defect = float((high.A_op(error)-rhs).norm()/source_norm)
        rows.append(dict(index=index, prefix_iterations=status['iterations'],
            prefix_relative_residual=status['true_residual'], eligible=int(eligible),
            teacher_relative_residual=teacher_residual, tail_equation_defect_original_rhs=equation_defect,
            tail_to_solution_norm=float(error.norm()/target.norm())))
        for key, value in [('t', t), ('s', s), ('q', rhs.to(torch.complex64)), ('rhs_exact', rhs),
                           ('target', error), ('reference', reference)]:
            entries[key].append(value.cpu())
        if (index+1) % 16 == 0:
            write_csv(out/'cases.csv', rows)
            print(f'tail targets {index+1}/{len(parent["q"])}', flush=True)
    bank = {key: torch.cat(value) for key, value in entries.items()}
    bank['eligible'] = torch.tensor([row['eligible'] for row in rows], dtype=torch.bool)
    bank['prefix_relative_residual'] = torch.tensor([row['prefix_relative_residual'] for row in rows], dtype=torch.float64)
    torch.save(dict(data=payload['data'], bank=bank,
        parent_bank_sha256=hashlib.sha256(Path(args.teacher_bank).read_bytes()).hexdigest(),
        split=split, schema='post-analytical-initialization TSA8 FGMRES tail v1'), out/f'{split}_tail.pt')
    write_csv(out/'cases.csv', rows)
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args), complete=True,
        split=split, cases=len(rows), eligible=sum(row['eligible'] for row in rows),
        seconds=time.perf_counter()-start,
        scope=f'new {split.upper()} targets only; no trained tail network and no speed claim',
        accuracy='FP32 stored problem promoted unchanged; A(error)=tail RHS checked relative to ORIGINAL source',
        prefix='TSA8 with balanced analytical initialization; four FGMRES steps or earlier physical convergence',
        next_design='one learned search direction inside continuing FGMRES, no restart and no unconditional inference on solved cases'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parent = parser.add_mutually_exclusive_group(required=True)
    parent.add_argument('--teacher-bank', default='')
    parent.add_argument('--validation-checkpoint', default='')
    parser.add_argument('--output', required=True)
    parser.add_argument('--prefix-steps', type=int, default=4)
    parser.add_argument('--g', type=float, default=.65)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())
