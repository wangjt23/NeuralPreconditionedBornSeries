"""Actual one-shot injection gate, plus regenerated exact prefix contexts.

Teacher directions are diagnostic information ceilings, not deployable AI.
The teacher-with-network-cost control executes the real neural forward too.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import torch

from rte.augmented_fgmres import augmented_fgmres
from rte.global_source_predictor import balance_integrated_source
from rte.mixed_precision_transport import high_precision_copy
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator
from rte.run_residual_subspace_experiment import write_csv
from rte.source_initialization import solve_from_initial
from rte.tail_response_network import TailResponseNetwork
from rte.transport_baselines import ClassicalTransportStack


def operator_from_bank(bank, index, device):
    t, s, q = [bank[key][index:index+1].to(device) for key in ('t', 's', 'q')]
    cfg = ExperimentConfig(size=q.shape[-1], n_angles=q.shape[1], anisotropy_g=.65, eta_scale=0., regimes='mixed')
    low = make_operator(cfg, t, s, q, device)
    high = high_precision_copy(low)
    base = ClassicalTransportStack(low, 'tsa', 8, 0.)
    analytical = balance_integrated_source(high,
        base.reference.reference_transport_solve(low.source).to(torch.complex128), high.source)
    return low, high, base, analytical


@torch.no_grad()
def prepare_contexts(parent_path, output, device):
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    bank = parent['bank']
    entries = {key: [] for key in ('t', 's', 'q', 'rhs_exact', 'reference', 'target', 'image_basis')}
    ids, norms, rows = [], [], []
    for index in range(len(bank['q'])):
        low, high, base, initial = operator_from_bank(bank, index, device)
        target = bank['target'][index:index+1].to(device)
        teacher_error = float((high.A_op(target)-high.source).norm()/high.source.norm())
        if teacher_error > 2e-9:
            raise ValueError('stored teacher/operator identity mismatch')
        contexts = []
        _, status = augmented_fgmres(high.A_op, high.source,
            lambda r: base(r.to(torch.complex64)).to(torch.complex128), initial,
            maxiter=4, capture_prefix=contexts.append)
        eligible = status['true_residual'] > 1e-5
        record = dict(index=index, eligible=int(eligible), prefix_iterations=status['iterations'],
            prefix_residual=status['true_residual'], teacher_residual=teacher_error)
        if eligible:
            context = contexts[0]
            basis = context['image_basis'].transpose(0, 1)
            rhs = context['residual']
            reference = base.reference.reference_transport_solve(rhs.to(torch.complex64))
            error = target-context['solution']
            projection = (basis.conj()*rhs[:, None]).real.flatten(2).sum(2)
            record['residual_image_overlap'] = float(projection.norm()/rhs.norm())
            record['tail_equation_error_original_rhs'] = float((high.A_op(error)-rhs).norm()/high.source.norm())
            for key, value in [('t', low.sigma_t), ('s', low.sigma_s), ('q', rhs.to(torch.complex64)),
                ('rhs_exact', rhs), ('reference', reference), ('target', error), ('image_basis', basis)]:
                entries[key].append(value.cpu())
            ids.append(index)
            norms.append(float(high.source.norm()))
        rows.append(record)
        if (index+1) % 16 == 0:
            print(f'{output.stem}: {index+1}/{len(bank["q"])} prefix contexts', flush=True)
    prepared = {key: torch.cat(value) for key, value in entries.items()}
    prepared['indices'] = torch.tensor(ids)
    prepared['original_rhs_norm'] = torch.tensor(norms, dtype=torch.float64)
    torch.save(dict(bank=prepared, data=parent['data'], parent=str(parent_path),
        parent_sha256=hashlib.sha256(Path(parent_path).read_bytes()).hexdigest(),
        scope='eligible actual FGMRES tails and search-image basis; no test cases'), output)
    write_csv(output.with_suffix('.csv'), rows)
    return parent


@torch.no_grad()
def gate(parent, model, output, device, repeats):
    bank, rows = parent['bank'], []
    rng = random.Random(26927)
    for index in range(len(bank['q'])):
        low, high, base, initial = operator_from_bank(bank, index, device)
        teacher = bank['target'][index:index+1].to(device)
        precondition = lambda r: base(r.to(torch.complex64)).to(torch.complex128)
        def expensive_teacher(x, r):
            reference = base.reference.reference_transport_solve(r.to(torch.complex64))
            model(low.sigma_t, low.sigma_s, r.to(torch.complex64), reference)
            return teacher-x
        methods = dict(augmented_tsa8=None,
            reference_injection=lambda x,r: base.reference.reference_transport_solve(r.to(torch.complex64)).to(torch.complex128),
            tsa8_injection=lambda x,r: precondition(r),
            teacher_injection=lambda x,r: teacher-x,
            teacher_with_network_cost=expensive_teacher)
        labels = ['legacy_tsa8', *methods]
        for repeat in range(-1, repeats):
            rng.shuffle(labels)
            for order, label in enumerate(labels):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                if label == 'legacy_tsa8':
                    _, status = solve_from_initial(high.A_op, high.source, precondition, initial)
                    status.update(classical_calls=status['iterations'], injection_calls=0,
                                  injection_accepted=0, injection_rejected=0)
                else:
                    _, status = augmented_fgmres(high.A_op, high.source, precondition, initial,
                        inject=methods[label])
                    status.pop('trace')
                torch.cuda.synchronize(device)
                if repeat >= 0:
                    rows.append(dict(index=index, repeat=repeat, order=order, method=label,
                        seconds=time.perf_counter()-start, **status))
        write_csv(output, rows)
        if (index+1) % 4 == 0:
            print(f'injection gate {index+1}/{len(bank["q"])} validation media', flush=True)


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(26927)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    settings = dict(args=vars(args), complete=False,
        gpu_before=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], text=True),
        scope='validation information ceiling and prefix-bank preparation; no trained AI speed claim',
        timing='shuffled synchronized warm solves; common analytical initialization/setup excluded for this integration gate',
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ['rte/augmented_fgmres.py','rte/tail_response_network.py','rte/audit_tail_injection.py']})
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    prepare_contexts(args.train_bank, out/'train_context.pt', device)
    val = prepare_contexts(args.validation_bank, out/'validation_context.pt', device)
    model = TailResponseNetwork().to(device).eval()
    gate(val, model, out/'gate.csv', device, args.repeats)
    settings['complete'] = True
    settings['gpu_after'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], text=True)
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-bank', default='rte/workstation_runs/source_predictor_pilot_20260926/train.pt')
    parser.add_argument('--validation-bank', default='rte/workstation_runs/source_tail_validation_20260926/validation.pt')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
