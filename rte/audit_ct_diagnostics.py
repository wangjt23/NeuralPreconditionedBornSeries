"""Audit saved CT deployment, selection coverage, and arithmetic precision.

No optimizer updates or test-based checkpoint selection. The original test
split is explicitly reused for diagnosis; validation compares the two retained
weight states only. Timings here include diagnostic work and are not rankings.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from rte.collision_transport import CollisionTransportPreconditioner
from rte.models import PeriodicRTE2D
from rte.residual_subspace import physical_residual_loss
from rte.run_large_angular_spectral_experiments import ExperimentConfig, dataset_fingerprint, make_operator, setup_batch, source_from_data
from rte.run_residual_subspace_experiment import measured_solver, write_csv
from rte.train_collision_transport_ensemble import bank_loss, material_split
from rte.transport_baselines import ClassicalTransportStack


def summarize(rows):
    result = []
    for split, method, group in sorted({(r['split'], r['method'], r['group']) for r in rows}):
        subset = [r for r in rows if (r['split'], r['method'], r['group']) == (split, method, group)]
        result.append(dict(split=split, method=method, group=group, count=len(subset),
                           failures=sum(not r['converged'] for r in subset),
                           mean_iterations=statistics.mean(r['iters'] for r in subset),
                           mean_final=statistics.mean(r['final'] for r in subset),
                           mean_source_loss=statistics.mean(r['source_loss'] for r in subset),
                           max_source_loss=max(r['source_loss'] for r in subset)))
    return result


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    path = Path(args.run)
    last = torch.load(path / 'last.pt', map_location='cpu', weights_only=True)
    saved = SimpleNamespace(**{**last['args'], 'dataset_dir': ''})
    cfg = ExperimentConfig(size=saved.size, n_angles=saved.n_angles, regimes=saved.regimes,
                           anisotropy_g=saved.g, batch_size=saved.batch_size, eta_scale=0.)
    validation = material_split(saved, cfg, 'validation')
    test = material_split(saved, cfg, 'test')
    assert dataset_fingerprint(validation) == last['fingerprints']['validation']
    assert dataset_fingerprint(test) == (path / 'test_fingerprint.txt').read_text().strip()
    ids = torch.tensor([0])
    q = source_from_data(cfg, validation, ids, device)
    rte = make_operator(cfg, validation['sigma_t'][ids], validation['sigma_s'][ids], q, device)
    model = CollisionTransportPreconditioner(rte, depth=4, learned=True, width=12).to(device).eval()
    model.load_state_dict(last['best_state'])
    rows, losses = [], []
    old_rows = list(csv.DictReader((path / 'results.csv').open()))
    original = {int(row['index']): row for row in old_rows if row['method'] == 'learned_CT'}
    for index in range(len(test['sigma_t'])):
        source = setup_batch(cfg, rte, test, torch.tensor([index]), device)
        with torch.no_grad():
            cached = model(source)
            model.train()
            direct = model(source)
            model.eval()
            discrepancy = float((cached - direct).norm() / direct.norm())
            loss = float(physical_residual_loss(rte.A_op, source, cached))
        solve = measured_solver(rte, model, source, saved)
        rows.append(dict(split='test', group='all', method='selected_CT', index=index,
                         source_loss=loss, cached_uncached_relative=discrepancy,
                         original_source_loss=float(original[index]['source_loss']),
                         original_final=float(original[index]['final']),
                         converged=math.isfinite(solve['final']) and solve['final'] <= saved.tol, **solve))
        if index % 32 == 31:
            write_csv(out / 'cases.csv', rows)
            print(f'corrected test {index + 1}/{len(test["sigma_t"])}', flush=True)

    for label, weights in [('selected_CT', last['best_state']), ('last_CT', last['model']),
                            ('HSS4', None), ('TSA4', None)]:
        if weights is not None:
            model.load_state_dict(weights)
            model.eval()
            loss = bank_loss(model, rte, validation, cfg, device, saved.probe_fraction)
            losses.append(dict(method=label, split='validation', bank_loss=loss))
        for index in range(len(validation['sigma_t'])):
            source = setup_batch(cfg, rte, validation, torch.tensor([index]), device)
            method = model if weights is not None else (
                CollisionTransportPreconditioner(rte, depth=4).eval() if label == 'HSS4'
                else ClassicalTransportStack(rte, 'tsa', 4, 0.).eval())
            with torch.no_grad():
                loss = float(physical_residual_loss(rte.A_op, source, method(source)))
            solve = measured_solver(rte, method, source, saved)
            rows.append(dict(split='validation', group='selection_subset' if index < saved.selection_count else 'unused_validation',
                             method=label, index=index, source_loss=loss,
                             converged=math.isfinite(solve['final']) and solve['final'] <= saved.tol, **solve))
        write_csv(out / 'cases.csv', rows)
        write_csv(out / 'summary.csv', summarize(rows))
        print('validation done', label, flush=True)

    precision = []
    for index in args.precision_indices:
        ids = torch.tensor([index])
        q = source_from_data(cfg, test, ids, torch.device('cpu'))
        for dtype in (torch.float32, torch.float64):
            # Promote the stored float32 coefficients, source, directions,
            # phase and wavenumbers, to isolate arithmetic from discretization.
            op32 = make_operator(cfg, test['sigma_t'][ids], test['sigma_s'][ids], q, torch.device('cpu'))
            op = PeriodicRTE2D(shape=op32.shape, dx=op32.dx, n_angles=op32.n_angles,
                               anisotropy_g=op32.anisotropy_g, fdtype=dtype, eta_scale=0.)
            for name in ('directions', 'weights', 'phase_matrix', 'kx', 'ky'):
                setattr(op, name, getattr(op32, name).to(dtype))
            op.setup(test['sigma_t'][ids], test['sigma_s'][ids], q)
            learned = CollisionTransportPreconditioner(op, depth=4, learned=True, width=12).eval()
            learned.load_state_dict(last['best_state'])
            for label, method in [('selected_CT', learned), ('HSS4', CollisionTransportPreconditioner(op, depth=4).eval()),
                                  ('TSA4', ClassicalTransportStack(op, 'tsa', 4, 0.).eval())]:
                solve = measured_solver(op, method, op.source, saved)
                precision.append(dict(index=index, dtype=str(dtype), method=label, **solve))
    write_csv(out / 'precision.csv', precision)
    write_csv(out / 'bank_losses.csv', losses)
    write_csv(out / 'summary.csv', summarize(rows))
    (out / 'settings.json').write_text(json.dumps(dict(
        run=str(path), checkpoint_sha256=hashlib.sha256((path/'last.pt').read_bytes()).hexdigest(),
        selected_epoch=last['best_epoch'], last_epoch=last['epoch'], updates=last['updates'],
        test_fingerprint=dataset_fingerprint(test), validation_fingerprint=dataset_fingerprint(validation),
        device=str(device), torch=str(torch.__version__), precision_indices=args.precision_indices,
        scope='diagnosis on reused test; validation compares retained selected and last only',
        cache_guard='setup_generation_and_retained_inputs_v3'), indent=2) + '\n')
    print(json.dumps(summarize(rows), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--precision-indices', type=int, nargs='+', default=[4, 18, 54, 120, 152])
    p.add_argument('--output', required=True)
    run(p.parse_args())
