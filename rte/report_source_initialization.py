"""Summarize source-initialization research without selecting on test data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pool_cases(frame):
    """Repeats are timing measurements, not independent media."""
    aggregations = dict(seconds=('seconds', 'median'), iterations=('iterations', 'max'),
        failed=('failed', 'max'), true_residual=('true_residual', 'max'),
        repeats=('repeat', 'nunique'))
    for name in ('initialization_seconds', 'initializer_seconds', 'setup_seconds', 'setup_plus_seconds'):
        if name in frame:
            aggregations[name] = (name, 'median')
    return frame.groupby(['size', 'index', 'method'], as_index=False).agg(**aggregations)


def summary(cases):
    aggregations = dict(media=('index', 'nunique'), failed=('failed', 'sum'),
        mean_iterations=('iterations', 'mean'), mean_seconds=('seconds', 'mean'),
        max_true_residual=('true_residual', 'max'), min_repeats=('repeats', 'min'))
    for name in ('initialization_seconds', 'initializer_seconds', 'setup_seconds', 'setup_plus_seconds'):
        if name in cases:
            aggregations['mean_'+name] = (name, 'mean')
    return cases.groupby(['size', 'method'], as_index=False).agg(**aggregations)


def paired_comparison(cases, baseline, candidate, label):
    """Paired media bootstrap for ratio of mean times, no test tuning."""
    rows = []
    for size, group in cases.groupby('size'):
        left = group[group.method == baseline].set_index('index')
        right = group[group.method == candidate].set_index('index')
        if not left.index.equals(right.index) or not len(left):
            raise ValueError('missing paired cases')
        b, n = left.seconds.to_numpy(), right.seconds.to_numpy()
        rng = np.random.default_rng(20260926)
        ids = rng.integers(len(b), size=(10000, len(b)))
        savings = 1-n.mean()/b.mean()
        estimates = 1-n[ids].mean(1)/b[ids].mean(1)
        lo, hi = np.quantile(estimates, [.025, .975])
        row = dict(size=size, comparison=label, baseline=baseline, candidate=candidate,
            media=len(b), baseline_failed=int(left.failed.sum()), candidate_failed=int(right.failed.sum()),
            baseline_seconds=b.mean(), candidate_seconds=n.mean(), speedup=b.mean()/n.mean(),
            aggregate_savings=savings, savings_ci_low=lo, savings_ci_high=hi,
            paired_wins=int((n < b).sum()), mean_paired_savings=float((1-n/b).mean()),
            median_paired_savings=float(np.median(1-n/b)),
            baseline_iterations=float(left.iterations.mean()), candidate_iterations=float(right.iterations.mean()))
        if 'setup_plus_seconds' in group:
            row['setup_plus_estimate_savings'] = 1-right.setup_plus_seconds.mean()/left.setup_plus_seconds.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    neural = Path(args.neural)
    settings = json.loads((neural/'settings.json').read_text())
    if not settings.get('complete'):
        raise ValueError('neural test run must be complete')
    frames = [pd.read_csv(neural/f'test_cases_n{size}.csv') for size in settings['args']['test_sizes']]
    frame = pd.concat(frames, ignore_index=True)
    cases = pool_cases(frame)
    if not (cases.repeats == settings['args']['test_repeats']).all():
        raise ValueError('incomplete timing repetitions')
    if not (cases.groupby(['size', 'method'])['index'].nunique() == settings['args']['test_count']).all():
        raise ValueError('incomplete test media')
    cases.to_csv(out/'test_pooled_cases.csv', index=False)
    summary(cases).to_csv(out/'test_summary.csv', index=False)
    selected = paired_comparison(cases, settings['selected_classical'], settings['selected_neural'],
        'frozen_validation_choices')
    matched = paired_comparison(cases, 'tsa8_zero', 'tsa8_neural', 'matched_tsa8')
    analytical = paired_comparison(cases, 'tsa8_reference_balanced', 'tsa8_neural', 'matched_tsa8_analytical_initializer')
    comparisons = pd.concat([selected, matched, analytical], ignore_index=True)
    comparisons.to_csv(out/'paired.csv', index=False)
    for split in ('classical', 'neural'):
        summary(pool_cases(pd.read_csv(neural/f'validation_{split}_cases.csv'))).to_csv(
            out/f'validation_{split}_summary.csv', index=False)
    oracle_summaries = []
    measured = len(frame)
    for directory in map(Path, args.oracles):
        if not json.loads((directory/'settings.json').read_text()).get('complete'):
            raise ValueError('oracle run must be complete')
        raw = pd.read_csv(directory/'cases.csv')
        aggregate = summary(pool_cases(raw))
        aggregate.insert(0, 'run', directory.name)
        oracle_summaries.append(aggregate)
        measured += len(raw)
    pd.concat(oracle_summaries, ignore_index=True).to_csv(out/'oracle_summary.csv', index=False)
    provenance = dict(neural=settings, measured_test_and_oracle_solves=measured,
        inference='one-shot network plus original full-angle solver, true relative residual <=1e-5',
        selection='frozen validation selections from settings; no best-of-test selection',
        samples=f"{settings['args']['test_count']} independent test media shared across resolutions; repeats and resolutions are correlated",
        uncertainty='10000 paired-media bootstrap draws; exploratory uncertainty in this family, not independent replications',
        timing='mean of per-medium median synchronized warm times; inference included; setup+warm is only an estimate',
        warning='failed solves remain in summaries; speed claims require comparable success at the specified tolerance')
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    # Include preserved failed/smoke/partial runs only in provenance hashes,
    # never in scientific aggregates.
    root = Path(__file__).resolve().parents[1]
    directories = [neural, out, *map(Path, args.oracles), *map(Path, args.extra_artifacts)]
    files = {p for directory in directories for p in directory.rglob('*') if p.is_file()}
    names = ['rte/source_initialization.py', 'rte/global_source_predictor.py',
        'rte/audit_source_initialization.py', 'rte/train_source_predictor.py',
        'rte/characteristic_source_predictor.py', 'rte/train_characteristic_source.py',
        'rte/report_source_initialization.py', 'rte/tests/test_source_initialization.py',
        'rte/tests/test_characteristic_source_predictor.py', 'rte/tests/test_source_initialization_report.py',
        'rte/tests/test_global_source_predictor.py', 'rte/train_krylov_transport.py',
        'rte/transport_baselines.py', 'rte/compiled_hss.py', 'rte/mixed_precision_transport.py',
        'rte/run_large_angular_spectral_experiments.py', 'rte/run_residual_subspace_experiment.py',
        'rte/galerkin_transport.py']
    files.update(root/name for name in names)
    hashes = [dict(path=str(path.resolve().relative_to(root)), bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(files)]
    pd.DataFrame(hashes).to_csv(out/'artifact_hashes.csv', index=False)
    print(comparisons.to_string(index=False))
    print(f'Hashed {len(hashes)} files; {len(frame)} measured neural-test solves.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neural', required=True)
    parser.add_argument('--oracles', nargs='+', required=True)
    parser.add_argument('--extra-artifacts', nargs='*', default=[])
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
