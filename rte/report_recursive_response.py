"""Report recursive-response development studies; retain failed methods."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    summaries, pairs, compression, provenance = [], [], [], []
    for directory in map(Path, args.inputs):
        data = pd.read_csv(directory/'cases.csv')
        cases = data.groupby(['size', 'epsilon', 'index', 'rhs_id', 'method'], as_index=False).agg(
            iterations=('iterations', 'max'), failed=('failed', 'max'), true_residual=('true_residual', 'max'),
            warm_seconds=('warm_seconds', 'median'), setup_seconds=('setup_seconds', 'median'),
            first_estimate_seconds=('setup_plus_warm_seconds', 'median'), repeats=('repeat', 'nunique'))
        summary = cases.groupby(['size', 'epsilon', 'method'], as_index=False).agg(
            media=('index', 'nunique'), rhs_cases=('rhs_id', 'size'), failed_cases=('failed', 'sum'),
            mean_iterations=('iterations', 'mean'), mean_warm_seconds=('warm_seconds', 'mean'),
            median_warm_seconds=('warm_seconds', 'median'), mean_setup_seconds=('setup_seconds', 'mean'),
            mean_setup_plus_warm_seconds=('first_estimate_seconds', 'mean'), max_true_residual=('true_residual', 'max'),
            min_repeats=('repeats', 'min'))
        summary.insert(0, 'run', directory.name)
        summaries.append(summary)
        for (size, epsilon), group in summary.groupby(['size', 'epsilon']):
            controls = group[group.method.map(lambda name: bool(re.fullmatch(r'hss\d+s[\d.]+|reference\d+|tsa\d+', name)))]
            successful = controls[controls.failed_cases == 0]
            if successful.empty:
                continue
            # An exploratory reporting reference, NOT independently selected
            # tuning. Never optimize the baseline separately for each RHS.
            baseline = successful.loc[successful.mean_warm_seconds.idxmin(), 'method']
            selected = cases[(cases['size'] == size) & (cases.epsilon == epsilon)]
            table = selected.pivot(index=['index', 'rhs_id'], columns='method', values='warm_seconds')
            for name in table:
                ratio = table[name]/table[baseline]
                pairs.append(dict(run=directory.name, size=size, epsilon=epsilon, method=name,
                    exploratory_baseline=baseline, cases=len(ratio), wins=int((ratio < 1).sum()),
                    median_paired_warm_savings=float((1-ratio).median()),
                    mean_paired_warm_savings=float((1-ratio).mean())))
        if (directory/'compression.csv').exists():
            comp = pd.read_csv(directory/'compression.csv')
            comp.insert(0, 'run', directory.name)
            compression.append(comp)
        provenance.append(dict(run=directory.name, settings=json.loads((directory/'settings.json').read_text()),
            measured_solves=len(data), unique_rhs_cases=len(cases[['size', 'epsilon', 'index', 'rhs_id']].drop_duplicates())))
    pd.concat(summaries, ignore_index=True).to_csv(out/'summary.csv', index=False)
    pd.DataFrame(pairs).to_csv(out/'paired.csv', index=False)
    if compression:
        pd.concat(compression, ignore_index=True).to_csv(out/'compression.csv', index=False)
    (out/'provenance.json').write_text(json.dumps(dict(runs=provenance,
        scope='small development samples; media/resolutions/regimes/sources correlated; no trained AI or general speedup claim',
        timing='median over repeats before averaging cases; setup+warm is an estimate, NOT a measured cold solve',
        baseline='best successful reported classical candidate in pooled regime, chosen for exploration only; all methods/failures retained',
        mode_history='coarse_response_pilot_n64_20260919 used recover_odd_rhs=True; later --harmonic-only excludes odd RHS reset'), indent=2)+'\n')
    root = Path(__file__).resolve().parents[1]
    files = {p for directory in [*map(Path, args.inputs), out] for p in directory.rglob('*') if p.is_file()}
    names = ['rte/hierarchical_response.py', 'rte/coarse_response_transport.py',
        'rte/response_residual_repair.py',
        'rte/run_recursive_response_study.py', 'rte/run_coarse_response_study.py', 'rte/report_recursive_response.py',
        'rte/tests/test_hierarchical_response.py', 'rte/tests/test_coarse_response_transport.py',
        'rte/tests/test_response_residual_repair.py',
        'docs/research/RTE_RECURSIVE_RESPONSE_RESULTS_2026_09_19.md']
    files.update(root/name for name in names if (root/name).exists())
    hashes = []
    for path in sorted(files):
        hashes.append(dict(path=str(path.resolve().relative_to(root)), bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(hashes).to_csv(out/'artifact_hashes.csv', index=False)
    print(json.dumps(dict(method_rows=sum(len(s) for s in summaries), hashed_files=len(hashes)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
