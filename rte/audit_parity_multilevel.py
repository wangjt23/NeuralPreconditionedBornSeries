"""Exact-setup coarse and analytic scaling controls for even-Schur learning."""

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
from rte.parity_coarse import CharacteristicSchurCorrection, GalerkinSchurCorrection, SchurMultilevelControl
from rte.parity_transport import ParitySchur
from rte.run_large_angular_spectral_experiments import ExperimentConfig, dataset_fingerprint, make_operator, source_from_data
from rte.run_residual_subspace_experiment import write_csv
from rte.transport_baselines import ClassicalTransportStack


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(1919)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    rows, diagnostics, telemetry = [], [], []
    rng = random.Random(919)
    for path in args.datasets:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        data = payload['data']
        if payload['fingerprint'] != dataset_fingerprint(data):
            raise ValueError('dataset fingerprint mismatch')
        size = data['sigma_t'].shape[-1]
        cfg = ExperimentConfig(size=size, n_angles=12, anisotropy_g=.65, eta_scale=0.)
        for epsilon in args.epsilons:
            for index in range(args.count):
                ids = torch.tensor([index])
                q = source_from_data(cfg, data, ids, device)
                t = data['sigma_t'][ids]/epsilon
                s = t-epsilon*(data['sigma_t'][ids]-data['sigma_s'][ids])
                low = make_operator(cfg, t, s, q, device)
                fine = high_precision_copy(low)
                specs = [('hss8', 0, False), ('hss16_shift2', 0, False), ('reference8', 0, False),
                         ('schur_reference', 0, False), ('schur_scaled', 0, True)]
                specs += [(f'schur_coarse{m}_scaled{int(scaled)}', m, scaled)
                          for m in args.coarse_sizes for scaled in (False, True)]
                if args.characteristic_count:
                    specs = [s for s in specs if not s[2]]
                    specs += [(f'schur_characteristic_w{w:g}', -1, w) for w in (.001, .01, .1)]
                rng.shuffle(specs)
                for label, coarse_size, scaled in specs:
                    sync()
                    start = time.perf_counter()
                    if label.startswith('hss'):
                        model = CompiledHSS(low, 8 if label == 'hss8' else 16, 1. if label == 'hss8' else 2.).prepare()
                        action = lambda r: model(r.to(torch.complex64)).to(torch.complex128)
                    elif label == 'reference8':
                        model = ClassicalTransportStack(low, 'reference', 8, 0.)
                        action = lambda r: model(r.to(torch.complex64)).to(torch.complex128)
                    else:
                        parity = ParitySchur(fine)
                        coarse = (CharacteristicSchurCorrection(parity, args.characteristic_count, scaled) if coarse_size == -1
                            else GalerkinSchurCorrection(parity, coarse_size) if coarse_size else None)
                        model = SchurMultilevelControl(parity, coarse, scaled if coarse_size >= 0 else False)
                        action = lambda r: parity.solve(r, args.inner_steps, 1e-8, model)[0]
                    sync()
                    setup = time.perf_counter()-start
                    for repeat in range(args.repeats+1):
                        sync()
                        start = time.perf_counter()
                        x, trace, _, _ = fgmres_real_stacked_with_trace(fine.A_op, fine.source, action,
                            tol=1e-5, maxiter=100, restart=25, max_restarts=3)
                        sync()
                        elapsed = time.perf_counter()-start
                        residual = float(torch.linalg.vector_norm(fine.source-fine.A_op(x))/torch.linalg.vector_norm(fine.source))
                        rows.append(dict(size=size, epsilon=epsilon, index=index, method=label, repeat=repeat,
                            setup_seconds=setup, solve_seconds=elapsed, first_total_seconds=elapsed+setup if repeat == 0 else '',
                            iterations=len(trace)-1, true_residual=residual, failed=int(not residual <= 1e-5)))
                    if label.startswith('schur'):
                        _, residual, used = parity.solve(fine.source, 400, 1e-9, model)
                        if coarse:
                            shape = (1, len(coarse.indices)) if coarse_size == -1 else (1, parity.half, coarse_size, coarse_size)
                            probe = torch.randn(*shape,
                                dtype=fine.complex_dtype, device=device)
                            direct = coarse.direct_action(probe)
                            coarse_error = float(torch.linalg.vector_norm(coarse.action(probe)-direct)/torch.linalg.vector_norm(direct))
                        else:
                            coarse_error = 0.
                        diagnostics.append(dict(size=size, epsilon=epsilon, index=index, method=label,
                            direct_pcg_iterations=used, schur_residual=float(residual.max()),
                            coarse_symmetry_defect=coarse.symmetry_defect if coarse else 0.,
                            coarse_direct_relative_error=coarse_error))
                    del action, model
                write_csv(out/'samples.csv', rows)
                write_csv(out/'diagnostics.csv', diagnostics)
                telemetry.append(dict(size=size, epsilon=epsilon, index=index, self_pid=os.getpid(), processes=gpu_processes()))
                (out/'telemetry.json').write_text(json.dumps(telemetry, indent=2)+'\n')
                print(f'completed N{size} eps={epsilon} medium={index}', flush=True)
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args),
        scope='development control, no learned weights; exact coarse setup cost included',
        precision='FP64 parity/reference arithmetic, unchanged FP64 physical operator; existing full-angle controls FP32',
        coarse='positive additive R0+P(RSP)^-1R; analytic sqrt(t/mean(t)) congruence optional'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--epsilons', type=float, nargs='+', default=[1., .1])
    parser.add_argument('--coarse-sizes', type=int, nargs='+', default=[8])
    parser.add_argument('--characteristic-count', type=int, default=0)
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--inner-steps', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
