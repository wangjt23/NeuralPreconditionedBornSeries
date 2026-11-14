"""Evidence tables, conditional paired intervals, plots and current-state hashes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from rte.run_residual_subspace_experiment import write_csv


def load_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def foreign_processes(path):
    observations = json.loads(path.read_text())
    foreign = []
    for row in observations:
        for line in row['gpu_processes'].splitlines()[1:]:
            fields = [x.strip() for x in line.split(',')]
            if fields and fields[0] != str(row['pid']) and 'snapd-desktop' not in line:
                foreign.append(dict(stage=row['stage'], process=line))
    return dict(snapshots=len(observations), foreign_observations=len(foreign),
                foreign_processes=sorted({row['process'] for row in foreign}),
                first_foreign_stage=foreign[0]['stage'] if foreign else None)


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    comparisons, pairs, provenance = [], [], []
    panels = []
    rng = np.random.default_rng(9013)
    for directory in map(Path, args.evaluations):
        settings = json.loads((directory/'settings.json').read_text())
        specs = {s['label']: s for s in settings['specs']}
        summary = load_csv(directory/'summary.csv')
        cases = load_csv(directory/'results.csv')
        sample_count = len(load_csv(directory/'solve_samples.csv'))
        for row in summary:
            spec = specs[row['method']]
            classical = spec.get('mode') in ('classical', 'compiled_hss')
            comparisons.append(dict(run=directory.name, size=settings['args']['size'], device=settings['args']['device'],
                method=row['method'], group=spec.get('group', row['method']), classical=classical,
                media=int(row['media']), failed_media=int(row['failed_media']), cold_failures=int(row['cold_failures']),
                iterations=float(row['mean_iterations']),
                warm_ms=float(row['median_medium_warm_seconds'])*1000 if row['median_medium_warm_seconds'] else None,
                first_ms=float(row['median_preparation_first_seconds'])*1000 if row['median_preparation_first_seconds'] else None,
                warm_p95_ms=float(row['p95_time_to_tolerance_seconds'])*1000 if row['p95_time_to_tolerance_seconds'] else None,
                p95_all_attempts_unreached=row['p95_all_attempts_unreached']))
        local = [r for r in comparisons if r['run']==directory.name]
        available = [r for r in local if r['classical'] and not r['failed_media'] and r['warm_ms'] is not None]
        best = min(available, key=lambda r:r['warm_ms']) if available else None
        targets = [r for r in local if not r['classical']]
        for target in targets:
            baselines = ([best] if best else [])+[r for r in targets if r['method'] != target['method'] and r['group'].startswith('ct')]
            for baseline in baselines:
                indexed = {r['index']:r for r in cases if r['method']==baseline['method']}
                paired = []
                for row in cases:
                    if row['method'] != target['method'] or row['index'] not in indexed:
                        continue
                    other = indexed[row['index']]
                    if any(r.get('setup_failure') or int(r.get('warm_failures') or 0) or r.get('cold_converged')!='True' for r in [row, other]):
                        continue
                    paired.append([float(row['warm_median_seconds']), float(other['warm_median_seconds']),
                                   float(row['cold_end_to_end_seconds']), float(other['cold_end_to_end_seconds'])])
                if not paired:
                    continue
                values = np.array(paired)
                delta = (values[:, 0]-values[:, 1])*1000
                bootstrap = delta[rng.integers(0, len(delta), (3000, len(delta)))].mean(1)
                ratio = values[:, 1]/values[:, 0]
                pairs.append(dict(run=directory.name, target=target['method'], baseline=baseline['method'],
                    paired_successes=len(values), faster_media=int((delta<0).sum()),
                    median_paired_speedup=float(np.median(ratio)),
                    median_paired_fraction_saved=float(np.median(1-values[:, 0]/values[:, 1])),
                    mean_delta_ms=float(delta.mean()), conditional95_low_ms=float(np.quantile(bootstrap,.025)),
                    conditional95_high_ms=float(np.quantile(bootstrap,.975)),
                    median_paired_first_speedup=float(np.median(values[:, 3]/values[:, 2]))))
        telemetry = foreign_processes(directory/'telemetry.json')
        provenance.append(dict(run=directory.name, fingerprints=settings['fingerprints'],
            scope=settings['scope'], sample_rows=sample_count, **telemetry,
            gpu_timing_uncontended_claim=False if telemetry['foreign_observations'] else 'no foreign process observed; not exclusive reservation'))
        if directory.name.startswith('resolution_n'):
            sct = next((r for r in local if r['group'] in ('sct64','sct128')), None)
            ct = next((r for r in local if r['group'] in ('ct64_krylov','ct128')), None)
            if sct and ct and best:
                panels.append((settings['args']['size'], settings['args']['device'], [sct, ct, best]))
    write_csv(out/'comparison.csv', comparisons)
    write_csv(out/'paired.csv', pairs)
    (out/'provenance.json').write_text(json.dumps(dict(evaluations=provenance,
        interval_scope='paired medium bootstrap conditional on fixed trained models, not training-seed uncertainty',
        baseline_scope='fastest observed successful classical family in each comparison; not an oracle chosen per medium',
        timing_scope='initialized process/model/operator, not process-cold; shared host/device'), indent=2)+'\n')
    if panels:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        panels.sort(key=lambda p:(p[0],p[1]))
        fig, axes = plt.subplots(2, 2, figsize=(12,8), layout='constrained')
        for ax, (size, device, rows) in zip(axes.flat, panels):
            x = np.arange(3)
            ax.bar(x-.18, [r['warm_ms'] for r in rows], .36, label='Warmed solve')
            ax.bar(x+.18, [r['first_ms'] for r in rows], .36, label='Preparation + first')
            ax.set_xticks(x, ['SCT', 'Matched CT', 'Fastest classical'])
            ax.set_ylabel('Median milliseconds over 24 media')
            ax.set_title(f'N{size}, {device}'+(' (shared GPU)' if 'cuda' in device else ' (one thread)'))
            ax.legend(fontsize=8)
        fig.suptitle('Fixed true residual tolerance 1e-5; independent tuning; no universal speedup claim')
        fig.savefig(out/'comparison.png', dpi=160)
        fig.savefig(out/'comparison.pdf')
        plt.close(fig)
    tracked = set()
    for directory in [*map(Path,args.evaluations), *map(Path,args.evidence_dirs), out]:
        tracked.update(p for p in directory.rglob('*') if p.is_file())
    source_names = ['audit_transport_structure','galerkin_transport','audit_coarse_consistency',
        'characteristic_collision_transport','spectral_collision_transport','train_krylov_transport','analyze_transport_stiffness',
        'mixed_precision_transport','multilevel_neural_transport','repeated_transport',
        'run_neural_resolution_study','make_resolution_specs','report_resolution_structure',
        'train_neural_transport','neural_transport_operator','collision_transport','models','compiled_hss','evaluate_neural_transport',
        'transport_baselines','npbs_unet_ablation','run_residual_subspace_experiment',
        'run_large_angular_spectral_experiments','train_collision_transport_ensemble','evaluate_collision_transport']
    tracked.update(Path('rte')/f'{name}.py' for name in source_names)
    tracked.update(Path('rte/tests').glob('test_*transport*.py'))
    tracked.update(Path('docs/research').glob('RTE_*2026_09_13.md'))
    hashes = []
    for path in sorted(tracked):
        if path == out/'artifact_hashes.csv':
            continue
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        hashes.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest))
    write_csv(out/'artifact_hashes.csv', hashes)
    print(json.dumps(dict(method_rows=len(comparisons), paired_rows=len(pairs), hashed_files=len(hashes)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', nargs='+', required=True)
    parser.add_argument('--evidence-dirs', nargs='*', default=[])
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
