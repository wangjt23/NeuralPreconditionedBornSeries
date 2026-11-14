"""Frozen-weight leave-component-out diagnostics on selection validation media.

These are deployment interventions, not separately trained architectural
ablations. They do not identify what a retrained reduced model could achieve.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from rte.residual_subspace import physical_residual_loss
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator, setup_batch, source_from_data
from rte.run_residual_subspace_experiment import measured_solver, write_csv
from rte.train_collision_transport_ensemble import iteration_score, material_split
from rte.train_neural_transport import make_model


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    a = SimpleNamespace(**payload['args'])
    cfg = ExperimentConfig(size=a.size, n_angles=a.n_angles, anisotropy_g=a.g,
                           regimes='mixed', batch_size=a.batch_size, eta_scale=0.)
    data = material_split(a, cfg, 'validation')
    device = torch.device('cpu')
    ids = torch.tensor([0]); q = source_from_data(cfg, data, ids, device)
    op = make_operator(cfg, data['sigma_t'][ids], data['sigma_s'][ids], q, device)
    rows, summary = [], []
    for label in ('trained', 'no_mixing', 'no_spectral', 'no_learned_shifts', 'analytic_anchor'):
        model = make_model(a, op).eval()
        model.load_state_dict(payload['model'])
        if label in ('no_mixing', 'analytic_anchor'):
            model.mixing = False
        if label in ('no_spectral', 'analytic_anchor'):
            model.spectral = False
        if label in ('no_learned_shifts', 'analytic_anchor'):
            with torch.no_grad():
                for p in model.shift_net.parameters():
                    p.zero_()
        subset = []
        for index in range(a.selection_count):
            q = setup_batch(cfg, op, data, torch.tensor([index]), device)
            solve = measured_solver(op, model, q, a)
            with torch.no_grad():
                loss = float(physical_residual_loss(op.A_op, q, model(q)))
            row = dict(method=label, index=index, source_loss=loss, **solve)
            rows.append(row); subset.append(row)
        score = iteration_score(subset, a.tol)
        summary.append(dict(method=label, media=len(subset), failures=score[0],
                            mean_iterations=score[1], mean_final=score[2],
                            mean_source_loss=sum(r['source_loss'] for r in subset)/len(subset)))
        write_csv(out/'cases.csv', rows)
        write_csv(out/'summary.csv', summary)
    (out/'settings.json').write_text(json.dumps(dict(checkpoint=args.checkpoint,
        selected_step=payload['selected_step'], split='selection_validation', device='cpu',
        interpretation='frozen-weight interventions, not matched retraining'), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
