"""Resolution transfer and multilevel benchmarks on saved independent datasets.

Candidate specifications are explicit JSON, including classical tuning groups.
Selection and comparison use separate random seeds. No training occurs here.
"""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from rte.compiled_hss import CompiledHSS
from rte.evaluate_collision_transport import aggregate_summary, candidate_score, evaluate_method, make_method
from rte.evaluate_neural_transport import gpu_processes
from rte.multilevel_neural_transport import MultilevelNeuralTransport
from rte.mixed_precision_transport import MixedPrecisionPreconditioner, high_precision_copy
from rte.run_large_angular_spectral_experiments import ExperimentConfig, dataset_fingerprint, make_operator, save_checkpoint, setup_batch, source_from_data
from rte.run_residual_subspace_experiment import write_csv
from rte.train_collision_transport_ensemble import material_data
from rte.spectral_collision_transport import make_resolution_model
from rte.repeated_transport import RepeatedTransportCorrection


def build_model(spec, op, payload=None):
    mode = spec.get('mode', 'fine')
    if mode == 'classical':
        return make_method(spec, op, None)
    if mode == 'compiled_hss':
        return CompiledHSS(op, spec.get('depth', 4), spec.get('scale', 1.))
    def factory(operator):
        if payload is None:
            return CompiledHSS(operator, spec.get('coarse_depth', 4), spec.get('coarse_scale', 1.))
        a = SimpleNamespace(**payload['args'])
        model = make_resolution_model(a, operator).to(device=operator.directions.device, dtype=operator.float_dtype)
        model.load_state_dict(payload['model'], strict=True)
        return model
    if mode == 'multilevel':
        return MultilevelNeuralTransport(op, factory, spec.get('coarse_size', 32),
            spec.get('pre', 0), spec.get('post', 2), spec.get('scale', 1.))
    if mode != 'fine' or payload is None:
        raise ValueError('fine mode requires a checkpoint')
    model = factory(op)
    return RepeatedTransportCorrection(op, model, spec['repeats']) if spec.get('repeats', 1) > 1 else model


def compare(spec, model, op, source, args, phase, index, order, repeats):
    case, raw = evaluate_method(dict(family='learned_CT', depth=model.depth, scale=spec.get('scale', '')),
        op, source, args, source.device, model, repeats, phase, index, order)
    inner = model.model if isinstance(model, MixedPrecisionPreconditioner) else model
    for row in [case, *raw]:
        row.update(method=spec['label'], family=spec.get('group', spec['label']))
        if isinstance(inner, MultilevelNeuralTransport) and 'preconditioner_calls' in row:
            row['coarse_operator_calls'] = (inner.coarse_model.depth-1)*row['preconditioner_calls']
        if isinstance(model, MixedPrecisionPreconditioner) and 'preconditioner_calls' in row:
            # Outer counts are FP64 A calls. The generic counter cannot see
            # nested A calls on a separate FP32 operator: record them separately.
            stages = inner.pre+inner.post if isinstance(inner, MultilevelNeuralTransport) else inner.depth-1
            if spec.get('mode') == 'classical' and spec.get('family') in ('tsa', 'dsa', 'dsa_transport'):
                stages = 2*inner.depth-1
            row['fine_fp32_operator_calls'] = stages*row['preconditioner_calls']
    return case, raw


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    specs = json.loads(Path(args.specs).read_text())
    if len({s['label'] for s in specs}) != len(specs):
        raise ValueError('unique candidate labels required')
    payloads = {}
    for spec in specs:
        path = spec.get('checkpoint')
        if path and path not in payloads:
            payloads[path] = torch.load(path, map_location='cpu', weights_only=True)
        if path:
            saved = payloads[path]['args']
            if saved['n_angles'] != args.n_angles or saved['g'] != args.g:
                raise ValueError('checkpoint angular discretization mismatch')
            save_checkpoint(out/f"checkpoint_{spec['label']}.pt", payloads[path])
    cfg = ExperimentConfig(size=args.size, n_angles=args.n_angles, anisotropy_g=args.g,
                           regimes='mixed', batch_size=1, eta_scale=0.)
    datasets = {'comparison': material_data(cfg, args.count, args.seed+100000)}
    if args.selection_count:
        datasets['selection'] = material_data(cfg, args.selection_count, args.seed)
    for name, data in datasets.items():
        save_checkpoint(out/f'{name}.pt', dict(schema=1, data=data, fingerprint=dataset_fingerprint(data), g=args.g))
    device = torch.device(args.device)
    data = datasets['comparison']
    ids = torch.tensor([0])
    q = source_from_data(cfg, data, ids, device)
    op = make_operator(cfg, data['sigma_t'][ids], data['sigma_s'][ids], q, device)
    models = {s['label']: build_model(s, op, payloads.get(s.get('checkpoint'))).eval() for s in specs}
    if args.precision == 'mixed':
        high = high_precision_copy(op)
        models = {label: MixedPrecisionPreconditioner(high, op, model).eval() for label, model in models.items()}
        op = high
    settings = dict(args=vars(args), specs=specs, fingerprints={k: dataset_fingerprint(v) for k, v in datasets.items()},
        checkpoint_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in payloads},
        torch=str(torch.__version__), cuda=torch.version.cuda,
        timing='resident process/operator; preparation + first and warmed solve separate; CPU one thread',
        scope=args.scope, transfer='independently generated higher-resolution media, not same material refined',
        coarse_operator='cell-average rediscretization, not Galerkin; all angles, strict coarse spatial band')
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    telemetry = []
    def observe(stage):
        telemetry.append(dict(stage=stage, pid=os.getpid(), gpu_processes=gpu_processes()))
        (out/'telemetry.json').write_text(json.dumps(telemetry, indent=2)+'\n')
    observe('start')
    generator = torch.Generator().manual_seed(args.seed+977)
    selected = specs
    if args.selection_count:
        cases, samples = [], []
        for index in range(args.selection_count):
            setup_batch(cfg, op, datasets['selection'], torch.tensor([index]), device)
            q = op.source
            observe(f'selection_{index}_before')
            for order, position in enumerate(torch.randperm(len(specs), generator=generator).tolist()):
                spec = specs[position]
                case, raw = compare(spec, models[spec['label']], op, q, args, 'selection', index, order, args.tuning_repeats)
                cases.append(case); samples.extend(raw)
            write_csv(out/'tuning_cases.csv', cases)
            write_csv(out/'tuning_samples.csv', samples)
            observe(f'selection_{index}_after')
            print('tuned medium', index, flush=True)
        scores = []
        for spec in specs:
            rows = [r for r in cases if r['method'] == spec['label']]
            failures, quality, seconds = candidate_score(rows, args.tol)
            failures += sum(not r.get('cold_converged', False) for r in rows)
            scores.append(dict(label=spec['label'], group=spec.get('group', spec['label']),
                               failures=failures, quality=quality, seconds=seconds))
        chosen = {min((s for s in scores if s['group']==group), key=lambda s: (s['failures'], s['quality'], s['seconds']))['label']
                  for group in {s['group'] for s in scores}}
        selected = [s for s in specs if s['label'] in chosen]
        write_csv(out/'tuning_summary.csv', scores)
        (out/'selected_specs.json').write_text(json.dumps(selected, indent=2)+'\n')
        print('selected', sorted(chosen), flush=True)
    cases, samples = [], []
    for index in range(args.count):
        setup_batch(cfg, op, data, torch.tensor([index]), device)
        q = op.source
        for order, position in enumerate(torch.randperm(len(selected), generator=generator).tolist()):
            spec = selected[position]
            observe(f'comparison_{index}_{spec["label"]}_before')
            case, raw = compare(spec, models[spec['label']], op, q, args, 'comparison', index, order, args.repeats)
            cases.append(case); samples.extend(raw)
            observe(f'comparison_{index}_{spec["label"]}_after')
        write_csv(out/'results.csv', cases)
        write_csv(out/'solve_samples.csv', samples)
        print('compared medium', index, flush=True)
    summary = aggregate_summary(cases, samples, args.tol)
    for row in summary:
        matched = [r for r in cases if r['method']==row['method'] and not r.get('setup_failure')]
        # Keep failure counts alongside success-conditioned timing summaries.
        warm = [r['warm_median_time_to_tolerance_seconds'] for r in matched if isinstance(r.get('warm_median_time_to_tolerance_seconds'), (float, int))]
        first = [r['cold_end_to_end_seconds'] for r in matched if r.get('cold_converged')]
        row['median_medium_warm_seconds'] = statistics.median(warm) if warm else None
        row['median_preparation_first_seconds'] = statistics.median(first) if first else None
    write_csv(out/'summary.csv', summary)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specs', required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--n-angles', type=int, default=12)
    parser.add_argument('--g', type=float, default=.65)
    parser.add_argument('--selection-count', type=int, default=0)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=541)
    parser.add_argument('--solver-steps', type=int, default=60)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--tuning-repeats', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=['float32', 'mixed'], default='float32')
    parser.add_argument('--scope', default='exploratory resolution/architecture screen, not confirmatory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.size, args.count, args.repeats, args.tuning_repeats, args.solver_steps) < 1 or args.selection_count < 0:
        parser.error('positive grid/count/repeats/budget required')
    run(args)
