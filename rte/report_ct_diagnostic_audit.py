"""Build the workstation audit figure and a hash manifest from saved evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from rte.run_residual_subspace_experiment import write_csv


RUNS = Path('rte/workstation_runs')
INPUT_DIRS = [
    'ct_n32_diagnostic_audit_20260912', 'ct_n32_corrected_gpu_eval_20260912',
    'ct_n32_validation_witnesses_20260912', 'ct_data_identity_audit_v2_20260912',
    'ct_n16_replications_audited_v2_20260912', 'sparse_stability_audit_n16_20260912',
    'ct_n32_hss_sequences_20260912', 'sparse_n16_seed1_final_20260912',
    'sparse_n32_seed1_limit16g_20260912',
]


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    diagnostic = RUNS / INPUT_DIRS[0]
    benchmark = RUNS / INPUT_DIRS[1]
    training = RUNS / 'ct_n32_data4096_seed101'
    cases = [r for r in read_csv(diagnostic/'cases.csv') if r['split'] == 'test']
    summary = read_csv(benchmark/'summary.csv')
    assert len(cases) == 256 and len(summary) == 6
    methods = {r['method']: r for r in summary}
    old_failures = sum(float(r['original_final']) > 1e-5 or not np.isfinite(float(r['original_final'])) for r in cases)
    changed = [int(r['index']) for r in cases if abs(float(r['source_loss']) - float(r['original_source_loss'])) > 1e-4]
    checks = dict(
        original_ct_failed_media=old_failures,
        corrected_ct_failed_media=int(methods['learned_CT']['failed_media']),
        changed_source_loss_indices=changed,
        max_cached_uncached_relative=max(float(r['cached_uncached_relative']) for r in cases),
        n32_timing_status='shared GPU: another training process observed; do not use as uncontended performance claim',
        identity_audit=json.loads((RUNS/INPUT_DIRS[3]/'summary.json').read_text()),
    )
    (out/'summary.json').write_text(json.dumps(checks, indent=2)+'\n')
    table = []
    for name in ('learned_CT', 'hss_tuned', 'tsa_tuned', 'reference_tuned', 'dsa_tuned', 'dsa_transport_tuned'):
        r = methods[name]
        table.append(dict(method=name, media=int(r['media']), failed_media=int(r['failed_media']),
                          mean_iterations=float(r['mean_iterations']), mean_A_calls=float(r['mean_operator_calls']),
                          shared_gpu_successful_median_ms=1000*float(r['median_time_to_tolerance_seconds']),
                          shared_gpu_all_attempt_p95_ms=1000*float(r['p95_all_attempts_time_to_tolerance_seconds'])))
    write_csv(out/'n32_methods.csv', table)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    x = np.array([float(r['original_source_loss']) for r in cases])
    y = np.array([float(r['source_loss']) for r in cases])
    different = np.abs(x-y) > 1e-4
    ax.scatter(x[~different], y[~different], s=13, alpha=.6, label='reproduced')
    ax.scatter(x[different], y[different], s=22, color='#cf493d', label='28 changed cases')
    bounds = [min(x.min(), y.min())*.7, max(x.max(), y.max())*1.4]
    ax.plot(bounds, bounds, '--', color='.55', lw=1)
    ax.set(xscale='log', yscale='log', xlim=bounds, ylim=bounds,
           xlabel='Original source loss', ylabel='Cache-corrected source loss',
           title='Stale coefficients corrupted the N32 evaluation')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    names = ['Original CT', 'Corrected CT', 'HSS', 'TSA']
    values = [old_failures] + [int(methods[k]['failed_media']) for k in ('learned_CT', 'hss_tuned', 'tsa_tuned')]
    bars = ax.bar(names, values, color=['#cf493d', '#2679ae', '.55', '.7'])
    ax.bar_label(bars)
    ax.set(ylabel='Failed media / 256', ylim=(0, 31), title='True residual > 1e-5 after 30 FGMRES steps')

    ax = axes[1, 0]
    history = [r for r in read_csv(training/'history.csv') if int(r['epoch']) > 0]
    epochs = [int(r['epoch']) for r in history]
    left, = ax.plot(epochs, [float(r['validation_loss']) for r in history], color='#2679ae', label='128-medium bank loss')
    other = ax.twinx()
    right, = other.plot(epochs, [float(r['validation_iterations']) for r in history], color='#b66521', alpha=.75, label='32-medium FGMRES iterations')
    ax.axvline(200, ls='--', color='.4', lw=1)
    ax.set(xlabel='Epoch (128 updates each)', ylabel='Validation physical loss', title='Loss reduction is not solver improvement')
    other.set_ylabel('Mean iterations')
    ax.legend(handles=[left, right], fontsize=8, loc='upper right')

    ax = axes[1, 1]
    witnesses = read_csv(RUNS/INPUT_DIRS[2]/'norm_witnesses.csv')
    for name in ('selected_CT', 'last_CT', 'HSS4', 'TSA4'):
        subset = [r for r in witnesses if r['method'] == name]
        ax.plot([int(r['index']) for r in subset], [float(r['one_step_norm_lower_bound']) for r in subset], 'o-', label=name)
    ax.axhline(1., ls='--', color='.4', lw=1)
    ax.set(xticks=range(4), xlabel='Validation medium', ylabel='Witnessed one-step residual ratio',
           title='Power probes: lower bounds, not stability certificates')
    ax.legend(fontsize=8)
    fig.savefig(out/'diagnostic_audit.png', dpi=180)
    fig.savefig(out/'diagnostic_audit.pdf')
    plt.close(fig)

    inputs = [p for name in INPUT_DIRS for p in (RUNS/name).iterdir() if p.is_file()]
    inputs += [training/name for name in ('best.pt', 'last.pt', 'settings.json', 'history.csv', 'selected_losses.csv', 'classical_tuning.csv', 'results.csv')]
    inputs += list(Path('rte').glob('*ct_*.py'))
    inputs += [Path('rte')/name for name in ('models.py', 'collision_transport.py', 'transport_baselines.py',
               'spectral_sparse_reference.py', 'evaluate_collision_transport.py',
               'analyze_collision_transport_replications.py', 'train_collision_transport_ensemble.py',
               'audit_sparse_stability.py', 'audit_hss_sequences.py')]
    inputs += list(Path('rte/tests').glob('test_collision_transport*.py'))
    report = Path('docs/research/RTE_DIAGNOSTIC_AUDIT_2026_09_12.md')
    if report.exists():
        inputs.append(report)
    inputs += [out/name for name in ('summary.json', 'n32_methods.csv', 'diagnostic_audit.png', 'diagnostic_audit.pdf')]
    records = []
    for path in sorted(set(inputs)):
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        records.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest))
    write_csv(out/'artifact_hashes.csv', records)
    print(json.dumps(checks, indent=2))
    print('Hashed', len(records), 'evidence/code files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
