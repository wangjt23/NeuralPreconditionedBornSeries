"""Compact fine-solution initializer opportunity, with classical coarse controls."""

import argparse
import json
from pathlib import Path
import random
import subprocess
import time

import torch
from torch.nn import functional as F

from rte.compiled_hss import CompiledHSS
from rte.mixed_precision_transport import high_precision_copy
from rte.multilevel_neural_transport import prolong_field, restrict_field
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_medium_dataset, make_operator, source_from_data
from rte.run_residual_subspace_experiment import write_csv
from rte.source_initialization import characteristic_project, constant_initial, solve_from_initial
from rte.transport_baselines import ClassicalTransportStack


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    randomizer = random.Random(args.seed)
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    settings = dict(args=vars(args), torch_version=torch.__version__,
        scope='development opportunity oracle, not AI; fine teacher acquisition excluded from oracle online time, explicitly reported',
        tolerance='unchanged FP64 physical A, 1e-5 relative to original RHS', dataset_count=4,
        controls='TSA8 fixed stiff-regime refinement; zero/constant/coarse RTE initialization',
        timing='paired shuffled synchronized solves; initialization cost included; teacher upper bounds are NOT deployable speedups')
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    rows, teachers = [], []
    for size in args.sizes:
        cfg = ExperimentConfig(size=size, n_angles=12, anisotropy_g=.65, eta_scale=0., regimes='mixed')
        data = make_medium_dataset(cfg, 4, args.seed)
        torch.save(data, out/f'data_n{size}.pt')
        for index in range(args.count):
            ids = torch.tensor([index])
            rhs = source_from_data(cfg, data, ids, device)
            t = data['sigma_t'][ids]/args.epsilon
            s = t-args.epsilon*(data['sigma_t'][ids]-data['sigma_s'][ids])
            low = make_operator(cfg, t, s, rhs, device)
            fine = high_precision_copy(low)
            tsa = ClassicalTransportStack(low, 'tsa', 8, 0.)
            precondition = lambda r: tsa(r.to(torch.complex64)).to(torch.complex128)
            sync()
            start = time.perf_counter()
            exact, check = solve_from_initial(fine.A_op, fine.source, precondition, tolerance=1e-10, maxiter=300)
            sync()
            seconds = time.perf_counter()-start
            if check['failed']:
                raise RuntimeError(f'fine teacher failed: {check}')
            teachers.append(dict(size=size, index=index, seconds=seconds, **check))
            methods = {'zero': lambda: None, 'constant': lambda: constant_initial(fine, fine.source)}
            for band in args.bands:
                methods[f'fine_teacher_band{band}'] = lambda band=band: prolong_field(restrict_field(exact, band), size)
                methods[f'fine_teacher_directional{band**2}'] = lambda band=band: characteristic_project(fine, exact, band**2)
                def coarse_initial(band=band):
                    ratio = size//band
                    tc, sc = F.avg_pool2d(t.to(device), ratio), F.avg_pool2d(s.to(device), ratio)
                    coarse_cfg = ExperimentConfig(size=band, n_angles=12, anisotropy_g=.65, eta_scale=0., regimes='mixed')
                    qc = restrict_field(fine.source, band).to(torch.complex64)
                    cl = make_operator(coarse_cfg, tc, sc, qc, device)
                    ch = high_precision_copy(cl)
                    cb = ClassicalTransportStack(cl, 'tsa', 8, 0.)
                    result, status = solve_from_initial(ch.A_op, ch.source,
                        lambda r: cb(r.to(torch.complex64)).to(torch.complex128), tolerance=1e-7, maxiter=100)
                    if status['failed']:
                        return torch.zeros_like(fine.source)
                    return prolong_field(result, size)
                methods[f'classical_coarse{band}'] = coarse_initial
            labels = list(methods)
            for repeat in range(-1, args.repeats):
                randomizer.shuffle(labels)
                for order, label in enumerate(labels):
                    sync()
                    start = time.perf_counter()
                    initial = methods[label]()
                    sync()
                    init_seconds = time.perf_counter()-start
                    result, check = solve_from_initial(fine.A_op, fine.source, precondition, initial)
                    sync()
                    seconds = time.perf_counter()-start
                    if repeat >= 0:
                        rows.append(dict(size=size, epsilon=args.epsilon, index=index, repeat=repeat, order=order,
                            method=label, seconds=seconds, initializer_seconds=init_seconds,
                            source_error=0. if initial is None else float(torch.linalg.vector_norm(initial-exact)/torch.linalg.vector_norm(exact)),
                            **check))
            torch.save(dict(t=t, s=s, source=fine.source.cpu(), solution=exact.cpu()), out/f'teacher_n{size}_i{index}.pt')
            write_csv(out/'cases.csv', rows)
            write_csv(out/'teachers.csv', teachers)
            group = [r for r in rows if r['size'] == size and r['index'] == index and r['repeat'] == 0]
            print(f'N{size} medium{index}: '+', '.join(f'{r["method"]}={r["iterations"]}' for r in group), flush=True)
    if device.type == 'cuda':
        settings['gpu_processes_after'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], text=True)
    settings['complete'] = True
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--bands', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1919)
    parser.add_argument('--epsilon', type=float, default=.1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
