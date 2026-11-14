"""Validation-only decomposition of learned source-response error.

This is a numerical diagnostic, not a timing benchmark. Teacher substitutions
are explicitly marked oracles. Filtered neural fields use no teacher online.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rte.characteristic_source_predictor import CharacteristicSourcePredictor
from rte.global_source_predictor import GlobalSourcePredictor, balance_integrated_source
from rte.mixed_precision_transport import high_precision_copy
from rte.multilevel_neural_transport import prolong_field, restrict_field
from rte.run_residual_subspace_experiment import write_csv
from rte.source_initialization import characteristic_project, solve_from_initial
from rte.train_source_predictor import build_base, material_operator, source_family


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    device = torch.device(args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = saved['args']
    model_type = CharacteristicSourcePredictor if any(k.startswith('characteristic.') for k in saved['model']) else GlobalSourcePredictor
    model = model_type(width=config['width'], blocks=config['blocks'], modes=config['modes']).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    count = config['train_count'] if args.split == 'train' else config['val_count']
    seed = config['seed']+(1000 if args.split == 'train' else 2000)
    cfg, data = source_family(config['size'], count, seed)
    bank = None
    if args.split == 'train':
        if not args.teacher_bank:
            raise ValueError('training diagnostic requires the original teacher bank')
        payload = torch.load(args.teacher_bank, map_location='cpu', weights_only=True)
        for key in ('sigma_t', 'sigma_s', 'centers', 'phases', 'amplitudes', 'angular_codes', 'geometry_base64'):
            if not torch.equal(payload['data'][key], data[key]):
                raise ValueError('training teacher bank identity mismatch: '+key)
        bank = payload['bank']
    torch.save(data, out/f'{args.split}_data.pt')
    rows, errors = [], []
    for index in range(min(args.count, count)):
        low, high = material_operator(cfg, data, torch.tensor([index]), config['epsilon'], device)
        if bank is not None:
            # Reuse the exact saved GPU source values for training diagnostics;
            # do not regenerate a differently rounded CPU illumination.
            low.setup(*(bank[k][index:index+1].to(device) for k in ('t', 's', 'q')))
            high = high_precision_copy(low)
        base = build_base(low, 'tsa8')
        precondition = lambda r: base(r.to(torch.complex64)).to(torch.complex128)
        if bank is None:
            target, status = solve_from_initial(high.A_op, high.source, precondition, tolerance=1e-9, maxiter=250)
        else:
            target = bank['target'][index:index+1].to(device)
            relative = float((high.source-high.A_op(target)).norm()/high.source.norm())
            status = dict(true_residual=relative, failed=int(relative > 2e-9), iterations=0)
        if status['failed']:
            fallback = build_base(low, 'hss16s2')
            target, status = solve_from_initial(high.A_op, high.source,
                lambda r: fallback(r.to(torch.complex64)).to(torch.complex128), tolerance=1e-9, maxiter=350)
        if status['failed']:
            raise RuntimeError('validation teacher failed; preserve this case')
        torch.save(dict(solution=target.cpu(), status=status), out/f'teacher_i{index}.pt')
        reference = base.reference.reference_transport_solve(low.source)
        prediction = balance_integrated_source(high,
            model(low.sigma_t, low.sigma_s, low.source, reference).to(torch.complex128), high.source)
        radial = lambda z, band: prolong_field(restrict_field(z, band), cfg.size)
        directional = lambda z, modes: characteristic_project(high, z, modes)
        methods = dict(zero=None, reference_balanced=reference.to(torch.complex128), neural=prediction,
            neural_band16=radial(prediction, 16), neural_band32=radial(prediction, 32),
            neural_directional256=directional(prediction, 256),
            neural_directional1024=directional(prediction, 1024),
            oracle_teacher_band32=radial(target, 32),
            oracle_teacher_directional1024=directional(target, 1024),
            oracle_replace_neural_directional1024=prediction+directional(target-prediction, 1024),
            oracle_replace_neural_complement1024=target+directional(prediction-target, 1024))
        if args.compact:
            methods = {k: methods[k] for k in ('zero', 'reference_balanced', 'neural')}
        error = prediction-target
        norm2 = error.abs().square().sum().clamp_min(1e-30)
        errors.append(dict(index=index, solution_error=float(error.norm()/target.norm()),
            radial32_error_fraction=float(radial(error, 32).abs().square().sum()/norm2),
            directional1024_error_fraction=float(directional(error, 1024).abs().square().sum()/norm2)))
        for name, initial in methods.items():
            if initial is not None:
                initial = balance_integrated_source(high, initial, high.source)
            _, check = solve_from_initial(high.A_op, high.source, precondition, initial)
            rows.append(dict(index=index, method=name, oracle=int(name.startswith('oracle_')),
                solution_error=1. if initial is None else float((initial-target).norm()/target.norm()), **check))
        write_csv(out/'cases.csv', rows)
        write_csv(out/'prediction_error.csv', errors)
        print(f'{args.split} error diagnostic {index+1}/{min(args.count, count)}', flush=True)
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args), complete=True,
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        model=model_type.__name__, source_config=config,
        scope='training/validation numerical diagnostic, NOT a timing benchmark; teacher substitutions are oracles'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--split', choices=['train', 'validation'], default='validation')
    parser.add_argument('--teacher-bank', default='')
    parser.add_argument('--compact', action='store_true')
    run(parser.parse_args())
