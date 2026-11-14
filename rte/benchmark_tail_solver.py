"""Validation-frozen, paired held-out tail-solver timing across nested grids.

No fine test solutions are acquired. All methods stop on the same original
physical residual; reference initialization and all neural work are timed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch

from rte.augmented_fgmres import augmented_fgmres
from rte.global_source_predictor import balance_integrated_source
from rte.optical_tail_network import OpticalTailNetwork
from rte.run_residual_subspace_experiment import write_csv
from rte.source_initialization import solve_from_initial
from rte.tail_moment_space import moment_injected_solve
from rte.tail_response_network import TailResponseNetwork
from rte.train_source_predictor import build_base, material_operator, source_family


CLASSICAL = [f'{kind}_balanced' for kind in
    ('tsa4', 'tsa8', 'tsa16', 'tsa32', 'reference16', 'reference32', 'hss16s2', 'hss32s2')]
CLASSICAL += ['tsa8_zero', 'tsa16_zero', 'reference_tail', 'moment_reference']
NEURAL = ['neural_tail', 'optical_tail', 'optical_moments']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(path, optical, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    cls = OpticalTailNetwork if optical else TailResponseNetwork
    model = cls(**{k:saved['args'][k] for k in ('width', 'blocks', 'modes')}).to(device)
    model.load_state_dict(saved['model'])
    return model.eval()


def summarize(rows):
    """Median repeats first; never count timing repeats as independent media."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row['method'], row['index']), []).append(row)
    cases = []
    for (method, index), group in sorted(grouped.items()):
        record = dict(method=method, index=index,
            failed=max(r['failed'] for r in group),
            true_residual=max(r['true_residual'] for r in group))
        for key in ('seconds', 'setup_seconds', 'setup_plus_seconds', 'iterations',
                    'classical_calls', 'injection_calls'):
            record[key] = statistics.median(r[key] for r in group)
        cases.append(record)
    result = []
    for method in sorted({r['method'] for r in cases}):
        group = [r for r in cases if r['method'] == method]
        record = dict(method=method, count=len(group), failed=sum(r['failed'] for r in group),
            max_true_residual=max(r['true_residual'] for r in group))
        for key in ('seconds', 'setup_seconds', 'setup_plus_seconds', 'iterations',
                    'classical_calls', 'injection_calls'):
            record[key] = statistics.mean(r[key] for r in group)
        result.append(record)
    return cases, result


def select(summary, allowed):
    eligible = [r for r in summary if r['method'] in allowed and r['failed'] == 0]
    if not eligible:
        raise RuntimeError('no reliable candidate in the frozen selection family')
    return min(eligible, key=lambda r:(r['seconds'], r['method']))['method']


@torch.no_grad()
def solve(method, low, high, bases, models):
    if method.endswith(('_balanced', '_zero')):
        name, kind = method.rsplit('_', 1)
    else:
        name, kind = 'tsa8', 'balanced'
    base = bases[name]
    initial = None
    if kind != 'zero':
        reference_base = base if hasattr(base, 'reference') else bases['tsa8']
        initial = reference_base.reference.reference_transport_solve(low.source).to(torch.complex128)
        initial = balance_integrated_source(high, initial, high.source)
    precondition = lambda r: base(r.to(torch.complex64)).to(torch.complex128)
    if method in ('moment_reference', 'optical_moments'):
        model = None if method == 'moment_reference' else models['optical']
        result, status = moment_injected_solve(high, low, base, initial, model)
    elif method in ('reference_tail', 'neural_tail', 'optical_tail'):
        model = models.get('tail' if method == 'neural_tail' else 'optical')
        def inject(x, r):
            r = r.to(torch.complex64)
            ref = base.reference.reference_transport_solve(r)
            return ref if method == 'reference_tail' else model(low.sigma_t, low.sigma_s, r, ref)
        result, status = augmented_fgmres(high.A_op, high.source, precondition, initial, inject=inject)
    else:
        result, status = solve_from_initial(high.A_op, high.source, precondition, initial)
        status.update(classical_calls=status['iterations'], injection_calls=0,
            injection_accepted=0, injection_rejected=0)
    status.pop('trace', None)
    status.pop('initial_residual', None)
    # Both solver paths already perform an independent final physical check.
    return result, status


@torch.no_grad()
def evaluate(cfg, data, methods, models, repeats, path, split, epsilon=.1):
    rows = []
    device = torch.device('cuda')
    rng = random.Random(45226+cfg.size)
    sync = lambda: torch.cuda.synchronize(device)
    names = {'tsa8'}
    names.update(method.rsplit('_', 1)[0] for method in methods if method.endswith(('_balanced', '_zero')))
    for index in range(len(data['sigma_t'])):
        sync()
        started = time.perf_counter()
        low, high = material_operator(cfg, data, torch.tensor([index]), epsilon, device)
        sync()
        physical_setup = time.perf_counter()-started
        bases, setups = {}, {}
        for name in sorted(names):
            sync()
            started = time.perf_counter()
            bases[name] = build_base(low, name)
            sync()
            setups[name] = time.perf_counter()-started
        order = list(methods)
        for repeat in range(-1, repeats):
            rng.shuffle(order)
            for position, method in enumerate(order):
                sync()
                started = time.perf_counter()
                _, status = solve(method, low, high, bases, models)
                sync()
                seconds = time.perf_counter()-started
                name = method.rsplit('_', 1)[0] if method.endswith(('_balanced', '_zero')) else 'tsa8'
                setup = physical_setup+setups[name]
                if name.startswith('hss') and method.endswith('_balanced'):
                    setup += setups['tsa8']
                if repeat >= 0:
                    rows.append(dict(split=split, size=cfg.size, index=index, repeat=repeat,
                        order=position, method=method, seconds=seconds, setup_seconds=setup,
                        setup_plus_seconds=setup+seconds, **status))
        write_csv(path, rows)
        if (index+1) % 4 == 0:
            print(f'{split} N{cfg.size}: {index+1}/{len(data["sigma_t"])}', flush=True)
    return rows


def paired_comparison(cases, classical, neural, seed=92426):
    lookup = {(r['method'], r['index']):r for r in cases}
    indices = sorted(r['index'] for r in cases if r['method'] == classical)
    c = torch.tensor([lookup[classical,i]['seconds'] for i in indices], dtype=torch.float64)
    n = torch.tensor([lookup[neural,i]['seconds'] for i in indices], dtype=torch.float64)
    draw = torch.randint(len(indices), (10000, len(indices)), generator=torch.Generator().manual_seed(seed))
    saving = 1-n[draw].mean(1)/c[draw].mean(1)
    interval = torch.quantile(saving, torch.tensor([.025, .975], dtype=saving.dtype))
    return dict(classical=classical, neural=neural, independent_media=len(indices),
        classical_ms=float(1000*c.mean()), neural_ms=float(1000*n.mean()),
        saving_percent=float(100*(1-n.mean()/c.mean())),
        saving_percent_ci95=[float(100*v) for v in interval],
        neural_faster_cases=int((n<c).sum()),
        bootstrap='paired media bootstrap, 10000 samples; timing repeats first reduced by median')


def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    source_names = ['rte/benchmark_tail_solver.py', 'rte/augmented_fgmres.py',
        'rte/tail_response_network.py', 'rte/optical_tail_network.py', 'rte/tail_moment_space.py',
        'rte/train_source_predictor.py', 'rte/source_initialization.py']
    gpu_processes = lambda: subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], text=True)
    settings = dict(args=vars(args), complete=False, torch_version=str(torch.__version__),
        gpu=torch.cuda.get_device_name(), gpu_processes_before=gpu_processes(),
        classical_candidates=CLASSICAL, neural_candidates=NEURAL,
        checkpoint_sha256={p:digest(p) for p in (args.tail_checkpoint, args.optical_checkpoint)},
        source_sha256={p:digest(p) for p in source_names},
        selection='fastest mean case-median synchronized warm solve time with zero validation failures, per grid; frozen before any test generation',
        timing='includes reference initialization, network, projection and final physical check; physical/preconditioner setup separately measured; NOT cold latency',
        accuracy='FP32 physical geometry promoted exactly to FP64, unchanged A; 1e-5 relative to original RHS, 100 directions max, restart25',
        scope='synthetic mixed periodic RTE, L12, g0.65, epsilon0.1; shared workstation; no test teachers or test tuning')
    settings['optical_checkpoint_policy'] = 'fixed final training step, versus work-proxy-selected base-tail checkpoint; pipeline chosen only on validation timing'
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    models = dict(tail=load_model(args.tail_checkpoint, False, 'cuda'),
        optical=load_model(args.optical_checkpoint, True, 'cuda'))
    selections, validation_summary = {}, []
    for size in args.sizes:
        cfg, data = source_family(size, args.validation_count, args.validation_seed)
        torch.save(data, out/f'validation_n{size}.pt')
        rows = evaluate(cfg, data, CLASSICAL+NEURAL, models, args.validation_repeats,
            out/f'validation_n{size}.csv', 'validation')
        _, summary = summarize(rows)
        validation_summary.extend(dict(size=size, **r) for r in summary)
        selections[str(size)] = dict(classical=select(summary, CLASSICAL), neural=select(summary, NEURAL))
        print(f'FROZEN N{size}: {selections[str(size)]}', flush=True)
    write_csv(out/'validation_summary.csv', validation_summary)
    (out/'frozen_selection.json').write_text(json.dumps(selections, indent=2)+'\n')
    settings['frozen_selection_sha256'] = digest(out/'frozen_selection.json')
    # First test generation happens strictly after ALL grid selections are frozen.
    test_summary, comparisons = [], []
    for size in args.sizes:
        cfg, data = source_family(size, args.test_count, args.test_seed)
        torch.save(data, out/f'test_n{size}.pt')
        selection = selections[str(size)]
        methods = list(dict.fromkeys([selection['classical'], selection['neural'],
            'tsa8_balanced', 'neural_tail', 'optical_moments', 'moment_reference']))
        rows = evaluate(cfg, data, methods, models, args.test_repeats, out/f'test_n{size}.csv', 'test')
        cases, summary = summarize(rows)
        write_csv(out/f'test_case_medians_n{size}.csv', cases)
        test_summary.extend(dict(size=size, **r) for r in summary)
        comparisons.append(dict(size=size, **paired_comparison(cases, **selection)))
        print(json.dumps(comparisons[-1]), flush=True)
    write_csv(out/'test_summary.csv', test_summary)
    (out/'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    settings.update(complete=True, gpu_processes_after=gpu_processes(),
        data_sha256={str(p):digest(p) for p in sorted(out.glob('*.pt'))})
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tail-checkpoint', default='rte/workstation_runs/tail_direction_pilot_20260926/best.pt')
    parser.add_argument('--optical-checkpoint', default='rte/workstation_runs/tail_optical_moment_20260926/last.pt')
    parser.add_argument('--sizes', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--validation-count', type=int, default=16)
    parser.add_argument('--validation-seed', type=int, default=4626)
    parser.add_argument('--validation-repeats', type=int, default=2)
    parser.add_argument('--test-count', type=int, default=32)
    parser.add_argument('--test-seed', type=int, default=8626)
    parser.add_argument('--test-repeats', type=int, default=3)
    run(parser.parse_args())
