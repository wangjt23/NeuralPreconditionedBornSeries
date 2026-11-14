"""Consolidate frozen neural-operator pilot evidence, paired costs and snapshots."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from rte.analyze_collision_transport_replications import bootstrap_mean_interval
from rte.run_large_angular_spectral_experiments import ExperimentConfig, dataset_fingerprint, save_checkpoint
from rte.run_residual_subspace_experiment import write_csv
from rte.train_collision_transport_ensemble import material_split


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    evaluation = Path(args.evaluation)
    cases = read_csv(evaluation/'results.csv')
    if args.before_index is not None:
        cases = [r for r in cases if int(r['index']) < args.before_index]
    settings = json.loads((evaluation/'settings.json').read_text())
    rows, paired = [], []
    indexed = {(int(r['index']), r['method']): r for r in cases}
    for method in sorted({r['method'] for r in cases}):
        subset = [r for r in cases if r['method']==method]
        successful = [r for r in subset if not r.get('setup_failure') and int(r.get('warm_failures') or 0)==0]
        warm = [1000*float(r['warm_median_seconds']) for r in successful]
        first = [1000*float(r['cold_end_to_end_seconds']) for r in successful if r['cold_converged']=='True']
        rows.append(dict(method=method, media=len(subset), failed_media=len(subset)-len(successful),
            mean_iterations=statistics.mean(float(r['warm_median_iters']) for r in subset),
            median_warm_ms=statistics.median(warm), p95_medium_median_warm_ms=float(np.quantile(warm,.95)),
            median_setup_ms=statistics.median(1000*float(r['setup_seconds']) for r in subset),
            median_preparation_plus_first_ms=statistics.median(first),
            mean_preparation_plus_first_ms=statistics.mean(first)))
        if method == 'nto4_hss':
            continue
        comparisons = []
        for r in subset:
            ai = indexed[int(r['index']), 'nto4_hss']
            if int(r.get('warm_failures') or 0) or int(ai.get('warm_failures') or 0) or r.get('setup_failure') or ai.get('setup_failure'):
                continue
            a, b = float(ai['warm_median_seconds']), float(r['warm_median_seconds'])
            comparisons.append(dict(index=int(r['index']), baseline=method, delta_ms=1000*(a-b),
                warm_baseline_over_ai=b/a,
                preparation_first_baseline_over_ai=float(r['cold_end_to_end_seconds'])/float(ai['cold_end_to_end_seconds'])))
        low, high = bootstrap_mean_interval([r['delta_ms'] for r in comparisons], 9821)
        paired.append(dict(baseline=method, paired_successes=len(comparisons),
            ai_faster_media=sum(r['delta_ms']<0 for r in comparisons),
            mean_delta_ms_ai_minus_baseline=statistics.mean(r['delta_ms'] for r in comparisons),
            conditional_bootstrap95_low_ms=low, conditional_bootstrap95_high_ms=high,
            median_paired_warm_speedup=statistics.median(r['warm_baseline_over_ai'] for r in comparisons),
            median_paired_preparation_first_speedup=statistics.median(r['preparation_first_baseline_over_ai'] for r in comparisons)))
        write_csv(out/f'pairs_{method}.csv', comparisons)
    write_csv(out/'comparison.csv', rows)
    write_csv(out/'paired_summary.csv', paired)

    foreign = set()
    for sample in json.loads((evaluation/'telemetry.json').read_text()):
        for line in sample['gpu_processes'].splitlines()[1:]:
            fields = [s.strip() for s in line.split(',')]
            if len(fields)>=2 and fields[0] != str(sample['pid']) and 'snapd-desktop-integration' not in fields[1]:
                foreign.add((fields[0], fields[1]))
    manifest = dict(evaluation=str(evaluation), checkpoint_selection=settings['selected_steps'],
        foreign_compute_processes_observed=sorted(foreign),
        comparison_indices=sorted({int(r['index']) for r in cases}),
        subset_rule=f'indices below {args.before_index}; telemetry-based prefix' if args.before_index is not None else 'all comparison media',
        inference_scope='single trained seed; reused development validation, not confirmatory test',
        interval_scope='resample paired media conditional on one fixed trained model, not training-seed uncertainty',
        timing_scope='resident process/weights/operator; preparation plus first solve is not process-cold deployment')
    (out/'summary.json').write_text(json.dumps(manifest, indent=2)+'\n')

    payload = torch.load(evaluation/'checkpoint_nto4_hss.pt', map_location='cpu', weights_only=True)
    saved = SimpleNamespace(**payload['args'])
    cfg = ExperimentConfig(size=saved.size, n_angles=saved.n_angles, anisotropy_g=saved.g,
                           regimes='mixed', batch_size=saved.batch_size, eta_scale=0.)
    for split in ('train', 'validation'):
        data = material_split(saved, cfg, split)
        fingerprint = dataset_fingerprint(data)
        assert fingerprint == payload['fingerprints'][split]
        save_checkpoint(out/f'{split}.pt', dict(data=data, fingerprint=fingerprint))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    lookup = {r['method']:r for r in rows}
    labels = {'nto4_hss':'NTO4', 'ct4':'CT4', 'hss_compiled':'Compiled\nHSS', 'hss_tuned':'FFT HSS', 'tsa_tuned':'TSA'}
    names = [n for n in labels if n in lookup]
    positions = np.arange(len(names))
    for offset, key, label in [(-.18, 'median_warm_ms', 'Warmed solve'), (.18, 'median_preparation_plus_first_ms', 'Preparation + first solve')]:
        axes[0].bar(positions+offset, [lookup[n][key] for n in names], .36, label=label)
    axes[0].set(xticks=positions, xticklabels=[labels[n] for n in names],
                ylabel=f'Median over {len(indexed)//len(lookup)} media (ms)', title='Same true residual tolerance: 1e-5')
    axes[0].legend(fontsize=8)
    for path in args.training_runs:
        path = Path(path)
        config = json.loads((path/'settings.json').read_text())['args']
        history = read_csv(path/'history.csv')
        label = config['model'].upper()+str(config['depth'])
        axes[1].plot([int(r['step']) for r in history], [float(r['validation_iterations']) for r in history], 'o-', ms=3, label=label)
    axes[1].set(xlabel='Optimizer updates', ylabel='Mean FGMRES iterations (selection 16)',
                title='Matched data, sampler, objective and update budget')
    axes[1].legend(fontsize=8)
    fig.savefig(out/'comparison.png', dpi=180)
    fig.savefig(out/'comparison.pdf')
    plt.close(fig)

    paths = list(evaluation.iterdir()) + list(out.iterdir())
    for root in args.training_runs:
        paths.extend(Path(root)/name for name in ('settings.json', 'history.csv', 'best.pt', 'last.pt'))
    paths += list(Path('rte').glob('*neural*.py')) + [Path('rte/compiled_hss.py')]
    paths += list(Path('rte/tests').glob('test_neural*.py'))
    paths += list(Path('docs/research').glob('RTE_NEURAL_TRANSPORT*.md'))
    hashes = []
    for path in sorted(set(paths)):
        if path.is_file():
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            hashes.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest))
    write_csv(out/'artifact_hashes.csv', hashes)
    print(json.dumps(dict(comparison=rows, paired=paired, provenance=manifest), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--training-runs', nargs='+', required=True)
    parser.add_argument('--before-index', type=int, default=None)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
