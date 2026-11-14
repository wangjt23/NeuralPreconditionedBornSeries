"""Compare frozen architecture pilots against validation-tuned classical solvers.

Tuning uses the pilot's solver-selection subset. Paired comparison uses the
remaining validation media, never a newly generated confirmatory test split.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from rte.evaluate_collision_transport import (
    CLASSICAL_FAMILIES, aggregate_summary, candidate_score, evaluate_method, load_tuning, tune_classical,
)
from rte.run_large_angular_spectral_experiments import ExperimentConfig, dataset_fingerprint, make_operator, setup_batch, source_from_data
from rte.run_large_angular_spectral_experiments import save_checkpoint
from rte.run_residual_subspace_experiment import write_csv
from rte.train_collision_transport_ensemble import material_split
from rte.train_neural_transport import make_model
from rte.compiled_hss import CompiledHSS


def gpu_processes():
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory,gpu_uuid',
                                 '--format=csv'], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def tune_compiled_hss(op, data, cfg, args, device, out):
    specs = [(depth, scale) for depth in (1, 2, 4) for scale in (.5, 1., 2., 4.)]
    cases, samples = [], []
    generator = torch.Generator().manual_seed(args.seed+9780)
    for index in range(args.validation_count):
        q = setup_batch(cfg, op, data, torch.tensor([index]), device)
        for order, position in enumerate(torch.randperm(len(specs), generator=generator).tolist()):
            depth, scale = specs[position]
            model = CompiledHSS(op, depth, scale)
            case, raw = evaluate_method(dict(family='learned_CT', depth=depth, scale=scale),
                op, q, args, device, model, args.tuning_repeats, 'selection_validation', index, order)
            for row in [case, *raw]:
                row.update(method='hss_compiled', family='compiled_hss')
            cases.append(case); samples.extend(raw)
        write_csv(out/'compiled_hss_tuning_cases.csv', cases)
        write_csv(out/'compiled_hss_tuning_samples.csv', samples)
    summaries = []
    for depth, scale in specs:
        subset = [r for r in cases if r['depth']==depth and r['scale']==scale]
        score = candidate_score(subset, args.tol)
        summaries.append(dict(depth=depth, scale=scale, validation_failures=score[0],
                              validation_quality=score[1], validation_seconds=score[2]))
    selected = min(summaries, key=lambda r: (r['validation_failures'], r['validation_quality'], r['validation_seconds']))
    write_csv(out/'compiled_hss_tuning.csv', summaries)
    print('selected compiled HSS', selected, flush=True)
    return CompiledHSS(op, selected['depth'], selected['scale'])


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    payloads = [torch.load(path, map_location='cpu', weights_only=True) for path in args.checkpoints]
    saved = SimpleNamespace(**payloads[0]['args'])
    cfg = ExperimentConfig(size=saved.size, n_angles=saved.n_angles, anisotropy_g=saved.g,
                           regimes='mixed', batch_size=saved.batch_size, eta_scale=0.)
    data = material_split(saved, cfg, 'validation')
    fingerprint = dataset_fingerprint(data)
    if any(p['fingerprints']['validation'] != fingerprint for p in payloads):
        raise ValueError('checkpoint validation fingerprints differ')
    args.seed, args.tol, args.solver_steps = saved.seed, saved.tol, saved.solver_steps
    args.validation_count = saved.selection_count
    ids = torch.tensor([0])
    q = source_from_data(cfg, data, ids, device)
    op = make_operator(cfg, data['sigma_t'][ids], data['sigma_s'][ids], q, device)
    models = []
    labels = []
    for path, payload in zip(args.checkpoints, payloads):
        a = SimpleNamespace(**payload['args'])
        model = make_model(a, op).to(device).eval()
        model.load_state_dict(payload['model'])
        label = f'{a.model}{a.depth}_{getattr(a, "base", "reference")}' if a.model == 'nto' else f'ct{a.depth}'
        if label in labels:
            raise ValueError('duplicate model labels; compare unique architecture arms')
        labels.append(label)
        models.append((label, model, a.model))
        save_checkpoint(out/f'checkpoint_{label}.pt', payload)
    telemetry = [dict(stage='start', pid=os.getpid(), gpu_processes=gpu_processes())]
    settings = dict(args=vars(args), validation_fingerprint=fingerprint,
                    checkpoint_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.checkpoints},
                    selected_steps={label: p['selected_step'] for label, p in zip(labels, payloads)},
                    scope='remaining validation media; exploratory architecture selection, not a final test',
                    timing='initialized process; preparation and first/warm solve separately; CPU one thread',
                    torch=str(torch.__version__), cuda=torch.version.cuda,
                    tuning_indices=list(range(saved.selection_count)),
                    comparison_indices=list(range(saved.selection_count, len(data['sigma_t']))))
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    selected = load_tuning(args.tuning) if args.tuning else tune_classical(op, data, cfg, args, device, models[0][1], out)
    specs = [selected[name] for name in CLASSICAL_FAMILIES]
    entries = models + [(s['family']+'_tuned', s, 'classical') for s in specs]
    if args.compiled_hss:
        compiled = tune_compiled_hss(op, data, cfg, args, device, out)
        entries.append(('hss_compiled', compiled, 'compiled_hss'))
    generator = torch.Generator().manual_seed(saved.seed+9700)
    cases, samples = [], []
    for index in range(saved.selection_count, len(data['sigma_t'])):
        q = setup_batch(cfg, op, data, torch.tensor([index]), device)
        for order, position in enumerate(torch.randperm(len(entries), generator=generator).tolist()):
            label, value, family = entries[position]
            if family == 'classical':
                spec, learned = value, models[0][1]
            else:
                # Adapt the existing evaluator's generic learned-model slot;
                # overwrite its legacy CT labels with the actual architecture.
                spec, learned = dict(family='learned_CT', depth=value.depth, scale=''), value
            case, raw = evaluate_method(spec, op, q, args, device, learned, args.repeats,
                                        'remaining_validation', index, order)
            if family != 'classical':
                for row in [case, *raw]:
                    row.update(method=label, family=family)
            cases.append(case); samples.extend(raw)
        write_csv(out/'results.csv', cases)
        write_csv(out/'solve_samples.csv', samples)
        telemetry.append(dict(stage=f'medium_{index}', pid=os.getpid(), gpu_processes=gpu_processes()))
        (out/'telemetry.json').write_text(json.dumps(telemetry, indent=2)+'\n')
        print('compared validation medium', index, flush=True)
    summary = aggregate_summary(cases, samples, args.tol)
    write_csv(out/'summary.csv', summary)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tuning-repeats', type=int, default=1)
    parser.add_argument('--tuning', default='')
    parser.add_argument('--compiled-hss', action='store_true')
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    if a.repeats < 1 or a.tuning_repeats < 1:
        parser.error('positive repeats required')
    run(a)
