"""Summarize the September 19 development controls without hiding failures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=args.refresh)
    rows, provenance, paired = [], [], []
    for directory in map(Path, args.inputs):
        path = directory/'samples.csv'
        if path.exists():
            data = pd.read_csv(path)
            for (size, epsilon, method), group in data.groupby(['size', 'epsilon', 'method']):
                first = group[group['repeat'] == 0]
                warm = group[group['repeat'] > 0].groupby('index').solve_seconds.median()
                rows.append(dict(run=directory.name, size=size, epsilon=epsilon, method=method,
                    cases=group['index'].nunique(), failed_cases=int(group.groupby('index').failed.max().sum()),
                    mean_iterations=first.iterations.mean(), median_warm_ms=1000*warm.median(),
                    median_first_ms=1000*first.first_total_seconds.median(),
                    median_setup_ms=1000*first.setup_seconds.median(), max_true_residual=group.true_residual.max()))
        elif (directory/'summary.csv').exists() and (directory/'results.csv').exists():
            summary = pd.read_csv(directory/'summary.csv')
            recorded = json.loads((directory/'settings.json').read_text())
            for _, row in summary.iterrows():
                rows.append(dict(run=directory.name, size=recorded['args']['size'], epsilon=1., method=row['method'],
                    cases=row['media'], failed_cases=row['failed_media'], mean_iterations=row['mean_iterations'],
                    median_warm_ms=1000*row['median_medium_warm_seconds'],
                    median_first_ms=1000*row['median_preparation_first_seconds'],
                    median_setup_ms=1000*row['median_setup_seconds'], max_true_residual=None))
            cases = pd.read_csv(directory/'results.csv')
            warm = cases.pivot(index='index', columns='method', values='warm_median_time_to_tolerance_seconds')
            first = cases.pivot(index='index', columns='method', values='cold_end_to_end_seconds')
            baseline = 'compiled_hss_d16_s1.0'
            if baseline in warm:
                for method in ('ct128_r2', 'sct128_r2'):
                    paired.append(dict(run=directory.name, method=method, baseline=baseline, media=len(warm),
                        warm_wins=int((warm[method] < warm[baseline]).sum()),
                        median_paired_warm_savings=float((1-warm[method]/warm[baseline]).median()),
                        median_paired_first_savings=float((1-first[method]/first[baseline]).median())))
        settings = directory/'settings.json'
        telemetry_path = directory/'telemetry.json'
        foreign = []
        snapshots = json.loads(telemetry_path.read_text()) if telemetry_path.exists() else []
        for number, snapshot in enumerate(snapshots):
            own = str(snapshot.get('self_pid', snapshot.get('pid', '')))
            raw = snapshot.get('processes', snapshot.get('gpu_processes', ''))
            processes = list(csv.reader(raw.splitlines()))
            other = [r for r in processes[1:] if len(r) >= 2 and r[0].strip() != own
                     and 'snapd-desktop-integration' not in r[1] and 'Xorg' not in r[1]]
            if other:
                foreign.append(dict(snapshot=number, stage=snapshot.get('stage'), processes=other))
        provenance.append(dict(directory=str(directory), complete=settings.exists(),
            settings=json.loads(settings.read_text()) if settings.exists() else None,
            telemetry_snapshots=len(snapshots), competing_process_snapshots=len(foreign), first_competing=foreign[:1]))
    pd.DataFrame(rows).to_csv(out/'summary.csv', index=False)
    pd.DataFrame(paired).to_csv(out/'paired.csv', index=False)
    (out/'provenance.json').write_text(json.dumps(dict(runs=provenance,
        scope='development sweeps; correlated resolutions/epsilons, fixed four-media survey, no neural training',
        failures='all methods retained; successful-only filtering is not used',
        uncertainty='small exploratory sample, no held-out model-selection or general speedup claim'), indent=2)+'\n')
    root = Path(__file__).resolve().parents[1]
    files = set()
    for directory in [*map(Path, args.inputs), out]:
        files.update(p for p in directory.rglob('*') if p.is_file())
    for pattern in ['rte/*.py', 'rte/parity*.py', 'rte/audit_parity*.py', 'rte/report_parity_research.py',
                    'rte/audit_hierarchical_response_oracle.py', 'rte/make_depth_review_specs.py',
                    'rte/tests/test_parity*.py', 'rte/tests/test_response_compression.py', 'docs/research/*2026_09_19.md',
                    'rte/workstation_runs/depth_review_config_20260919/specs.json']:
        files.update(root.glob(pattern))
    hashes = []
    for path in sorted(files):
        if path.resolve() == (out/'artifact_hashes.csv').resolve():
            continue
        value = path.read_bytes()
        hashes.append(dict(path=str(path.resolve().relative_to(root)), bytes=len(value), sha256=hashlib.sha256(value).hexdigest()))
    pd.DataFrame(hashes).to_csv(out/'artifact_hashes.csv', index=False)
    print(json.dumps(dict(method_rows=len(rows), hashed_files=len(hashes)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--refresh', action='store_true', help='refresh this task\'s generated report after final review')
    run(parser.parse_args())
