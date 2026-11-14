"""Paired validation gate for the controlled mixed-resolution tail study.

The frozen classical settings are inherited before this study. The same 16
development media are reused at nested resolutions; this is a validation gate.
Fresh seed 9626 is generated only after a zero-failure, >=5% runtime win at BOTH
N128 and N256. All inference and original-RHS checks are included in solve time.
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import torch

import rte.benchmark_tail_solver as protocol
from rte.run_residual_subspace_experiment import write_csv
from rte.train_source_predictor import source_family


def gpu_processes():
    return subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv'], text=True)


def run(args):
    if args.repeats < 1:
        raise ValueError('at least one measured timing repeat is required')
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    selection = json.loads(Path(args.classical_selection).read_text())
    checkpoint_paths = {
        'warm_base': args.warm_checkpoint,
        'n64_only': str(Path(args.training)/'n64_only/best.pt'),
        'mixed64_128': str(Path(args.training)/'mixed64_128/best.pt'),
    }
    models = {k: protocol.load_model(v, False, 'cuda')
              for k, v in checkpoint_paths.items()}
    settings = dict(args=vars(args), complete=False,
        scope='same 16 development validation media at three nested grids; no independent validation draws',
        gate='mixed64_128 zero failures and >=5% mean warm solve saving versus frozen classical at BOTH N128 and N256',
        test_condition='fresh seed 9626 only after validation gate passes; selections frozen before generation',
        checkpoint_sha256={p: protocol.digest(p) for p in checkpoint_paths.values()},
        classical_selection_sha256=protocol.digest(args.classical_selection),
        source_sha256={p: protocol.digest(p) for p in (
            __file__, 'rte/benchmark_tail_solver.py', 'rte/augmented_fgmres.py',
            'rte/tail_response_network.py', 'rte/train_source_predictor.py')},
        timing='one warmup, synchronized shuffled repeats, then per-media median; preparation separate; no cold-start or training cost in solve seconds',
        gpu_processes_before=gpu_processes())
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    original_solve = protocol.solve

    def solve(method, low, high, bases, ignored):
        if method in models:
            return original_solve('neural_tail', low, high, bases,
                                  {'tail': models[method]})
        return original_solve(method, low, high, bases, {})

    protocol.solve = solve
    summaries, comparisons = [], []
    try:
        for size in (64, 128, 256):
            cfg, data = source_family(size, 16, 4626)
            prior_path = Path('rte/workstation_runs/tail_solver_benchmark_20260926')/f'validation_n{size}.pt'
            prior = torch.load(prior_path, map_location='cpu', weights_only=True)
            for key in ('sigma_t', 'sigma_s', 'centers', 'phases', 'amplitudes',
                        'angular_codes', 'geometry_base64'):
                if not torch.equal(prior[key], data[key]):
                    raise ValueError(f'Existing validation cohort identity changed at N{size}: {key}')
            settings.setdefault('validation_parent_sha256', {})[str(size)] = protocol.digest(prior_path)
            torch.save(data, out/f'validation_n{size}.pt')
            classical = selection[str(size)]['classical']
            methods = list(dict.fromkeys([
                classical, 'tsa8_balanced', 'warm_base', 'n64_only', 'mixed64_128']))
            rows = protocol.evaluate(cfg, data, methods, {}, args.repeats,
                out/f'validation_n{size}.csv', 'multiresolution_validation')
            cases, summary = protocol.summarize(rows)
            write_csv(out/f'validation_case_medians_n{size}.csv', cases)
            summaries.extend(dict(size=size, **r) for r in summary)
            for baseline, challenger in ((classical, 'n64_only'),
                                         (classical, 'mixed64_128'),
                                         ('n64_only', 'mixed64_128')):
                comparison = dict(size=size,
                    **protocol.paired_comparison(cases, baseline, challenger))
                comparison['failed'] = sum(r['failed'] for r in summary
                    if r['method'] in (baseline, challenger))
                comparisons.append(comparison)
                print(json.dumps(comparison), flush=True)
        write_csv(out/'validation_summary.csv', summaries)
        (out/'validation_comparisons.json').write_text(
            json.dumps(comparisons, indent=2)+'\n')
        mixed = [r for r in comparisons if r['neural'] == 'mixed64_128'
                 and r['classical'] != 'n64_only']
        passed = all(r['failed'] == 0 for r in mixed) and all(
            r['saving_percent'] >= 5. for r in mixed if r['size'] in (128, 256))
        decision = dict(passed=passed, neural='mixed64_128', test_seed=9626,
            test_count=32, classical=selection, validation_comparisons=mixed,
            selection_scope='fixed mixed-resolution hypothesis; no architecture or checkpoint retuning from test')
        (out/'frozen_gate_decision.json').write_text(
            json.dumps(decision, indent=2)+'\n')
        test_summaries, test_comparisons = [], []
        if passed and not args.validation_only:
            for size in (64, 128, 256):
                cfg, data = source_family(size, 32, 9626)
                torch.save(data, out/f'test_n{size}.pt')
                classical = selection[str(size)]['classical']
                rows = protocol.evaluate(cfg, data,
                    [classical, 'n64_only', 'mixed64_128'], {}, 3,
                    out/f'test_n{size}.csv', 'multiresolution_test')
                cases, summary = protocol.summarize(rows)
                write_csv(out/f'test_case_medians_n{size}.csv', cases)
                test_summaries.extend(dict(size=size, **r) for r in summary)
                for baseline in (classical, 'n64_only'):
                    test_comparisons.append(dict(size=size,
                        **protocol.paired_comparison(cases, baseline, 'mixed64_128')))
            write_csv(out/'test_summary.csv', test_summaries)
            (out/'test_comparisons.json').write_text(
                json.dumps(test_comparisons, indent=2)+'\n')
        measured = 0
        for size in (64, 128, 256):
            with (out/f'validation_n{size}.csv').open() as stream:
                measured += sum(1 for _ in csv.DictReader(stream))
        settings.update(complete=True, passed=passed,
            test_generated=passed and not args.validation_only,
            measured_validation_solves=measured,
            frozen_gate_sha256=protocol.digest(out/'frozen_gate_decision.json'),
            gpu_processes_after=gpu_processes())
        (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
        print(f'Mixed-resolution gate passed={passed}; test_generated={settings["test_generated"]}', flush=True)
    finally:
        protocol.solve = original_solve


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--training', required=True)
    parser.add_argument('--warm-checkpoint', default='rte/workstation_runs/tail_direction_pilot_20260926/best.pt')
    parser.add_argument('--classical-selection', default='rte/workstation_runs/tail_solver_benchmark_20260926/frozen_selection.json')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--validation-only', action='store_true')
    run(parser.parse_args())
