"""Exploratory same-medium refinement and physical-stiffness survey.

All methods solve the unchanged promoted-FP32 physical operator to 1e-5.
No hyperparameters are selected on these development cases for confirmation.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from rte.compiled_hss import CompiledHSS
from rte.evaluate_neural_transport import gpu_processes
from rte.mixed_precision_transport import high_precision_copy
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace
from rte.parity_transport import ParityPreconditioner, ParitySchur
from rte.run_large_angular_spectral_experiments import (
    ExperimentConfig, dataset_fingerprint, make_medium_dataset, make_operator, save_checkpoint, source_from_data,
)
from rte.run_neural_resolution_study import build_model
from rte.run_residual_subspace_experiment import write_csv
from rte.transport_baselines import ClassicalTransportStack


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    rows, traces, diagnostics, telemetry = [], [], [], []
    rng = random.Random(args.seed)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    if payload is not None:
        save_checkpoint(out/'checkpoint.pt', payload)
    previous = None
    for size in args.sizes:
        cfg = ExperimentConfig(size=size, n_angles=12, anisotropy_g=.65, eta_scale=0., regimes='mixed')
        data = make_medium_dataset(cfg, args.count, args.seed)
        if previous is not None and size % previous['sigma_t'].shape[-1] == 0:
            stride = size//previous['sigma_t'].shape[-1]
            for key in ('sigma_t', 'sigma_s'):
                torch.testing.assert_close(data[key][..., ::stride, ::stride], previous[key], atol=0, rtol=0)
        previous = data
        save_checkpoint(out/f'data_n{size}.pt', dict(data=data, fingerprint=dataset_fingerprint(data)))
        for epsilon in args.epsilons:
            for index in range(args.count):
                ids = torch.tensor([index])
                source = source_from_data(cfg, data, ids, device)
                t = data['sigma_t'][ids]/epsilon
                s = t-epsilon*(data['sigma_t'][ids]-data['sigma_s'][ids])
                low = make_operator(cfg, t, s, source, device)
                high = high_precision_copy(low)
                factories = {}
                for depth in (4, 8, 16):
                    for scale in args.shifts:
                        factories[f'hss{depth}_s{scale:g}'] = lambda d=depth, a=scale: CompiledHSS(low, d, a).prepare()
                for kind in ('reference', 'tsa', 'dsa_transport'):
                    for depth in (4, 8):
                        factories[f'{kind}{depth}'] = lambda k=kind, d=depth: ClassicalTransportStack(low, k, d, 0.)
                for precision, op in [('fp32', low), ('fp64', high)]:
                    for steps in (4, 8, 16):
                        factories[f'parity_{precision}_{steps}'] = lambda o=op, k=steps: ParityPreconditioner(ParitySchur(o), k)
                if payload:
                    factories['sct_frozen_r2'] = lambda: build_model(dict(repeats=2), low, payload).eval()
                methods = list(factories)
                rng.shuffle(methods)
                for label in methods:
                    sync()
                    start = time.perf_counter()
                    model = factories[label]()
                    sync()
                    setup = time.perf_counter()-start
                    dtype = torch.complex128 if label.startswith('parity_fp64') else torch.complex64
                    action = lambda r: model(r.to(dtype)).to(torch.complex128)
                    for repeat in range(args.repeats+1):
                        if isinstance(model, ParityPreconditioner):
                            model.inner_iterations = 0
                        sync()
                        start = time.perf_counter()
                        x, trace, _, _ = fgmres_real_stacked_with_trace(high.A_op, high.source, action,
                            tol=1e-5, maxiter=100, restart=25, max_restarts=3)
                        sync()
                        elapsed = time.perf_counter()-start
                        residual = float(torch.linalg.vector_norm(high.source-high.A_op(x))/torch.linalg.vector_norm(high.source))
                        common = dict(size=size, epsilon=epsilon, index=index, method=label, repeat=repeat)
                        rows.append(dict(**common, setup_seconds=setup, solve_seconds=elapsed,
                            first_total_seconds=elapsed+setup if repeat == 0 else '', iterations=len(trace)-1,
                            inner_pcg_iterations=model.inner_iterations if isinstance(model, ParityPreconditioner) else 0,
                            true_residual=residual, failed=int(not residual <= 1e-5)))
                        traces.extend(dict(**common, step=step, residual=r) for step, r in enumerate(trace))
                    del action, model
                parity = ParitySchur(high)
                probe = torch.randn_like(high.source)
                defect = float(torch.linalg.vector_norm(high.A_op(probe)-parity.full_reference_action(probe))
                    /torch.linalg.vector_norm(high.A_op(probe)))
                x, schur_residual, steps = parity.solve(high.source, 400, 1e-9)
                residual = float(torch.linalg.vector_norm(high.source-high.A_op(x))/torch.linalg.vector_norm(high.source))
                diagnostics.append(dict(size=size, epsilon=epsilon, index=index, parity_reference_relative_defect=defect,
                    direct_pcg_iterations=steps, direct_schur_residual=float(schur_residual.max()),
                    direct_true_residual=residual))
                telemetry.append(dict(size=size, epsilon=epsilon, index=index, self_pid=os.getpid(), processes=gpu_processes()))
                write_csv(out/'samples.csv', rows)
                write_csv(out/'traces.csv', traces)
                write_csv(out/'diagnostics.csv', diagnostics)
                (out/'telemetry.json').write_text(json.dumps(telemetry, indent=2)+'\n')
                print(f'N{size} eps={epsilon} medium={index}: direct parity PCG {steps}, physical residual {residual:.2e}', flush=True)
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args),
        scope='development survey, not held-out tuned-method confirmation; no training',
        refinement='same continuous realizations, exact nested-grid coefficient agreement checked',
        scaling='t=t0/eps; s=t-eps*(t0-s0), rounded once to FP32 then physical operator promoted unchanged',
        timing='preparation plus first, followed by warmed same-RHS solves; excludes initial process/operator construction',
        parity='symmetric reference only, original physical A unchanged and checked in FP64'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--epsilons', type=float, nargs='+', default=[1., .25, .1])
    parser.add_argument('--shifts', type=float, nargs='+', default=[1.])
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1919)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
