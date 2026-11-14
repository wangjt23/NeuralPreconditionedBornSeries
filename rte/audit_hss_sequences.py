"""Validation-only four-stage HSS sequence control; no test access or training."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from rte.collision_transport import CollisionTransportPreconditioner
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator, setup_batch, source_from_data
from rte.run_residual_subspace_experiment import measured_solver, write_csv
from rte.train_collision_transport_ensemble import iteration_score, material_split


SEQUENCES = [(1., 1., 1., 1.), (.5, 1., 2., 4.), (4., 2., 1., .5),
             (.5, .5, 2., 2.), (2., 2., .5, .5), (.25, .5, 1., 2.),
             (2., 1., .5, .25), (1., 2., 4., 8.), (8., 4., 2., 1.)]


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    saved = torch.load(Path(args.run)/'best.pt', map_location='cpu', weights_only=True)
    a = SimpleNamespace(**{**saved['args'], 'dataset_dir': ''})
    cfg = ExperimentConfig(size=a.size, n_angles=a.n_angles, regimes=a.regimes,
                           anisotropy_g=a.g, batch_size=a.batch_size, eta_scale=0.)
    data = material_split(a, cfg, 'validation')
    ids = torch.tensor([0]); device = torch.device('cpu')
    q = source_from_data(cfg, data, ids, device)
    rte = make_operator(cfg, data['sigma_t'][ids], data['sigma_s'][ids], q, device)
    learned = CollisionTransportPreconditioner(rte, depth=4, learned=True, width=12).eval()
    learned.load_state_dict(saved['model'])
    rows, scores = [], []

    def evaluate(label, sequence, indices, group):
        for index in indices:
            q = setup_batch(cfg, rte, data, torch.tensor([index]), device)
            method = learned if sequence is None else CollisionTransportPreconditioner(
                rte, depth=4, stage_factors=sequence).eval()
            solve = measured_solver(rte, method, q, a)
            rows.append(dict(method=label, sequence=str(sequence), index=index, group=group, **solve))
        subset = [r for r in rows if r['method'] == label and r['group'] == group]
        score = iteration_score(subset, a.tol)
        scores.append(dict(method=label, sequence=str(sequence), group=group, media=len(subset),
                           failures=score[0], mean_iterations=score[1], mean_final=score[2]))
        write_csv(out/'cases.csv', rows)
        write_csv(out/'summary.csv', scores)
        return score

    # Fixed list and selection rule, specified before reading any results.
    choices = [(evaluate(f'HSS_sequence_{i}', seq, range(32), 'selection32'), i)
               for i, seq in enumerate(SEQUENCES)]
    selected = min(choices)[1]
    evaluate('selected_CT', None, range(32), 'selection32')
    for i in sorted({0, selected}):
        evaluate(f'HSS_sequence_{i}', SEQUENCES[i], range(32, len(data['sigma_t'])), 'remaining_validation96')
    evaluate('selected_CT', None, range(32, len(data['sigma_t'])), 'remaining_validation96')
    (out/'settings.json').write_text(json.dumps(dict(run=args.run, device='cpu',
        selected_sequence=selected, sequences=SEQUENCES, selection_indices=list(range(32)),
        selection_rule='failures, mean iterations, mean true residual',
        scope='exploratory validation-only; one solve per medium; not a timing ranking or fresh confirmatory split'), indent=2)+'\n')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
