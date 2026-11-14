"""Predeclared validation gate for the lookahead-trained tail continuation.

Fresh held-out seed 9626 is used only after >=5% validation mean-time savings
at BOTH N128 and N256. Classical choices are inherited from the earlier frozen
validation selection, not reselected from its held-out results.
"""

import argparse
import json
from pathlib import Path
import subprocess

import torch

import rte.benchmark_tail_solver as protocol
from rte.run_residual_subspace_experiment import write_csv
from rte.train_source_predictor import source_family


def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    selection = json.loads(Path(args.classical_selection).read_text())
    old = protocol.load_model(args.old_checkpoint, False, 'cuda')
    new = protocol.load_model(args.new_checkpoint, False, 'cuda')
    settings = dict(args=vars(args), complete=False,
        scope='validation gate; fresh test cohort only after passing the predeclared runtime gate',
        gate='zero failures and at least 5% saving at BOTH N128 and N256; N64 also measured',
        source_sha256={p:protocol.digest(p) for p in
            ('rte/benchmark_tail_lookahead.py', 'rte/benchmark_tail_solver.py')},
        checkpoint_sha256={p:protocol.digest(p) for p in (args.old_checkpoint, args.new_checkpoint)},
        classical_selection_sha256=protocol.digest(args.classical_selection))
    settings['timing'] = 'same synchronized shuffled warm-solve protocol as benchmark_tail_solver; initialization and inference included, setup separate'
    gpu_processes = lambda: subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], text=True)
    settings['gpu_processes_before'] = gpu_processes()
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    identical = all(torch.equal(v, old.state_dict()[k]) for k,v in new.state_dict().items())
    if identical:
        settings.update(complete=True, passed=False, test_generated=False,
            reason='selected weights identical to the previously tested base-tail checkpoint; prior validation/runtime gate already negative')
        (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
        print(settings['reason'], flush=True)
        return
    original_solve = protocol.solve
    def solve(method, low, high, bases, models):
        if method in ('old_tail', 'lookahead_tail'):
            model = old if method == 'old_tail' else new
            return original_solve('neural_tail', low, high, bases, {'tail':model})
        return original_solve(method, low, high, bases, {'tail':new})
    protocol.solve = solve
    summaries, comparisons = [], []
    for size in (64, 128, 256):
        cfg, data = source_family(size, 16, 4626)
        torch.save(data, out/f'validation_n{size}.pt')
        classical = selection[str(size)]['classical']
        rows = protocol.evaluate(cfg, data, [classical, 'old_tail', 'lookahead_tail'], {},
            2, out/f'validation_n{size}.csv', 'lookahead_validation')
        cases, summary = protocol.summarize(rows)
        summaries.extend(dict(size=size, **r) for r in summary)
        comparison = dict(size=size, **protocol.paired_comparison(cases, classical, 'lookahead_tail'))
        comparison['failed'] = sum(r['failed'] for r in summary if r['method'] in (classical, 'lookahead_tail'))
        comparisons.append(comparison)
        print(json.dumps(comparison), flush=True)
    write_csv(out/'validation_summary.csv', summaries)
    (out/'validation_comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    passed = all(r['failed'] == 0 for r in comparisons) and all(
        r['saving_percent'] >= 5. for r in comparisons if r['size'] in (128, 256))
    decision = dict(passed=passed, test_seed=9626, test_count=32, neural='lookahead_tail',
        classical=selection, validation_comparisons=comparisons)
    (out/'frozen_gate_decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    test_summaries, test_comparisons = [], []
    if passed:
        for size in (64, 128, 256):
            cfg, data = source_family(size, 32, 9626)
            torch.save(data, out/f'test_n{size}.pt')
            classical = selection[str(size)]['classical']
            rows = protocol.evaluate(cfg, data, [classical, 'old_tail', 'lookahead_tail'], {},
                3, out/f'test_n{size}.csv', 'lookahead_test')
            cases, summary = protocol.summarize(rows)
            write_csv(out/f'test_case_medians_n{size}.csv', cases)
            test_summaries.extend(dict(size=size, **r) for r in summary)
            test_comparisons.append(dict(size=size, **protocol.paired_comparison(cases, classical, 'lookahead_tail')))
        write_csv(out/'test_summary.csv', test_summaries)
        (out/'test_comparisons.json').write_text(json.dumps(test_comparisons, indent=2)+'\n')
    settings.update(complete=True, passed=passed, test_generated=passed,
        frozen_gate_sha256=protocol.digest(out/'frozen_gate_decision.json'),
        gpu_processes_after=gpu_processes())
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    print(f'lookahead runtime gate passed={passed}; fresh test generated={passed}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--old-checkpoint', default='rte/workstation_runs/tail_direction_pilot_20260926/best.pt')
    parser.add_argument('--new-checkpoint', default='rte/workstation_runs/tail_lookahead_20260926/best.pt')
    parser.add_argument('--classical-selection', default='rte/workstation_runs/tail_solver_benchmark_20260926/frozen_selection.json')
    run(parser.parse_args())
