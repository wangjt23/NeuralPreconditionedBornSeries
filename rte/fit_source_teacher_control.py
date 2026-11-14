"""Small training-set fit control, NOT a held-out AI performance experiment.

Can the chosen architecture represent accurate responses on four saved
training cases when optimized directly against their fine solutions? This
separates representational impossibility from the solver-aware training gap.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from rte.characteristic_source_predictor import CharacteristicSourcePredictor
from rte.global_source_predictor import balance_integrated_source
from rte.mixed_precision_transport import high_precision_copy
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator
from rte.run_residual_subspace_experiment import write_csv
from rte.source_initialization import solve_from_initial
from rte.transport_baselines import ClassicalTransportStack


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(26926)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = saved['args']
    payload = torch.load(args.teacher_bank, map_location='cpu', weights_only=True)
    bank = {key: value[:args.cases] for key, value in payload['bank'].items()}
    model = CharacteristicSourcePredictor(width=config['width'], blocks=config['blocks'], modes=config['modes'])
    model.load_state_dict(saved['model'])
    cfg = ExperimentConfig(size=config['size'], n_angles=12, anisotropy_g=.65, eta_scale=0., regimes='mixed')
    low = make_operator(cfg, bank['t'], bank['s'], bank['q'], 'cpu')
    high = high_precision_copy(low)
    target = bank['target']
    norm2 = target.abs().square().flatten(1).mean(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr*.05)
    history, solves = [], []
    start = time.perf_counter()
    for step in range(args.steps+1):
        prediction = balance_integrated_source(high,
            model(bank['t'], bank['s'], bank['q'], bank['reference']).to(torch.complex128), high.source)
        squared = (prediction-target).abs().square().flatten(1).mean(1)/norm2
        loss = squared.mean()
        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                history.append(dict(step=step, mean_relative_error=float(squared.sqrt().mean()),
                    max_relative_error=float(squared.sqrt().max()), seconds=time.perf_counter()-start))
                for index in range(args.cases):
                    op = make_operator(cfg, bank['t'][index:index+1], bank['s'][index:index+1], bank['q'][index:index+1], 'cpu')
                    exact = high_precision_copy(op)
                    base = ClassicalTransportStack(op, 'tsa', 8, 0.)
                    _, status = solve_from_initial(exact.A_op, exact.source,
                        lambda r: base(r.to(torch.complex64)).to(torch.complex128), prediction[index:index+1].detach())
                    solves.append(dict(step=step, index=index, **status))
                write_csv(out/'history.csv', history)
                write_csv(out/'training_solves.csv', solves)
                print(f'fit control {step}: training error={history[-1]["mean_relative_error"]:.5g}, '
                    f'iterations={sum(row["iterations"] for row in solves[-args.cases:])/args.cases:.3f}', flush=True)
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
    torch.save(dict(args=config, model=model.state_dict(), fit_control_args=vars(args),
        scope='four training cases only; not a deployable validation-selected checkpoint'), out/'fit_only.pt')
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args), complete=True,
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256(Path(args.teacher_bank).read_bytes()).hexdigest(),
        scope='CPU training-set overfit control; zero held-out speed claims; do not compare with test results',
        seconds=time.perf_counter()-start), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--teacher-bank', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cases', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--lr', type=float, default=.001)
    run(parser.parse_args())
