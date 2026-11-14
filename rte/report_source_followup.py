"""Finalize the directional follow-up and its separate numerical diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from rte.report_source_initialization import run as report_main


def run(args):
    for directory in [*args.error_diagnostics, args.fit_control]:
        if not json.loads((Path(directory)/'settings.json').read_text()).get('complete'):
            raise ValueError('all diagnostic budgets must be complete before final hashing')
    args.extra_artifacts = [*args.extra_artifacts, *args.error_diagnostics, args.fit_control]
    report_main(args)
    out = Path(args.output)
    diagnostics, error_fields = [], []
    for directory in map(Path, args.error_diagnostics):
        cases = pd.read_csv(directory/'cases.csv')
        table = cases.groupby('method', as_index=False).agg(cases=('index', 'nunique'),
            iterations=('iterations', 'mean'), mean_solution_error=('solution_error', 'mean'),
            initial_residual=('initial_residual', 'mean'), failed=('failed', 'sum'))
        table.insert(0, 'run', directory.name)
        diagnostics.append(table)
        error = pd.read_csv(directory/'prediction_error.csv')
        record = dict(run=directory.name, cases=len(error))
        record.update(error.drop(columns='index').mean().to_dict())
        error_fields.append(record)
    pd.concat(diagnostics, ignore_index=True).to_csv(out/'error_diagnostic_summary.csv', index=False)
    pd.DataFrame(error_fields).to_csv(out/'error_energy_summary.csv', index=False)
    fit = pd.read_csv(Path(args.fit_control)/'training_solves.csv')
    pooled = fit.groupby('step', as_index=False).agg(training_cases=('index', 'nunique'),
        mean_iterations=('iterations', 'mean'), failed=('failed', 'sum'))
    pd.read_csv(Path(args.fit_control)/'history.csv').merge(pooled, on='step').to_csv(out/'fit_control_summary.csv', index=False)
    tails = []
    for directory in map(Path, args.extra_artifacts):
        if (directory/'train_tail.pt').exists() or (directory/'validation_tail.pt').exists():
            cases = pd.read_csv(directory/'cases.csv')
            tails.append(dict(run=directory.name, cases=len(cases), eligible=int(cases.eligible.sum()),
                median_prefix_relative_residual=cases.prefix_relative_residual.median(),
                median_tail_to_solution_norm=cases.tail_to_solution_norm.median(),
                max_tail_equation_defect_original_rhs=cases.tail_equation_defect_original_rhs.max()))
    pd.DataFrame(tails).to_csv(out/'tail_bank_summary.csv', index=False)
    root = Path(__file__).resolve().parents[1]
    manifest = pd.read_csv(out/'artifact_hashes.csv')
    paths = {root/path for path in manifest.path}
    paths.update(p for p in out.iterdir() if p.is_file() and p.name != 'artifact_hashes.csv')
    paths.update(root/name for name in ['rte/report_source_followup.py', 'rte/audit_source_prediction_error.py',
        'rte/fit_source_teacher_control.py', 'rte/build_source_tail_bank.py',
        'docs/research/RTE_SOURCE_INITIALIZATION_REVIEW_2026_09_26.md',
        'docs/research/RTE_WORKSTATION_HANDOFF_2026_09_12.md'])
    hashes = [dict(path=str(path.resolve().relative_to(root)), bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(paths)]
    pd.DataFrame(hashes).to_csv(out/'artifact_hashes.csv', index=False)
    print(f'Final manifest: {len(hashes)} files, including completed diagnostics and research handoff.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neural', required=True)
    parser.add_argument('--oracles', nargs='+', required=True)
    parser.add_argument('--extra-artifacts', nargs='*', default=[])
    parser.add_argument('--error-diagnostics', nargs='+', required=True)
    parser.add_argument('--fit-control', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
