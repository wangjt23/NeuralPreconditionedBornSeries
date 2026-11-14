"""Compare FP32 and FP64 physical residuals without changing the target tolerance."""

import argparse
import json
from pathlib import Path

import torch

from rte.mixed_precision_transport import MixedPrecisionPreconditioner, high_precision_copy
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace
from rte.run_large_angular_spectral_experiments import ExperimentConfig, dataset_fingerprint, make_operator, save_checkpoint, source_from_data
from rte.run_neural_resolution_study import build_model
from rte.run_residual_subspace_experiment import write_csv
from rte.train_collision_transport_ensemble import material_data


def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    specs = json.loads(Path(args.specs).read_text())
    (out/'settings.json').write_text(json.dumps(vars(args), indent=2)+'\n')
    for size in args.sizes:
        cfg = ExperimentConfig(size=size, n_angles=12, anisotropy_g=.65, regimes='mixed', eta_scale=0.)
        data = material_data(cfg, args.count, args.seed)
        save_checkpoint(out/f'data_n{size}.pt', dict(data=data, fingerprint=dataset_fingerprint(data)))
        for index in range(args.count):
            ids = torch.tensor([index])
            q = source_from_data(cfg, data, ids, torch.device(args.device))
            low = make_operator(cfg, data['sigma_t'][ids], data['sigma_s'][ids], q, torch.device(args.device))
            high = high_precision_copy(low)
            for spec in specs:
                payload = torch.load(spec['checkpoint'], map_location='cpu', weights_only=True) if spec.get('checkpoint') else None
                model = build_model(spec, low, payload).eval()
                for precision in ('float32', 'mixed'):
                    op = low if precision=='float32' else high
                    method = model if precision=='float32' else MixedPrecisionPreconditioner(high, low, model)
                    with torch.no_grad():
                        if hasattr(method, 'prepare'):
                            method.prepare()
                        x, history, elapsed, calls = fgmres_real_stacked_with_trace(op.A_op, op.source, method,
                            tol=1e-5, maxiter=60, restart=20, max_restarts=2)
                        residual64 = float(torch.linalg.vector_norm(high.source-high.A_op(x.to(torch.complex128)))/torch.linalg.vector_norm(high.source))
                        rounded = x.to(torch.complex64)
                        rounded64 = float(torch.linalg.vector_norm(high.source-high.A_op(rounded.to(torch.complex128)))/torch.linalg.vector_norm(high.source))
                        rounded32 = float(torch.linalg.vector_norm(low.source-low.A_op(rounded))/torch.linalg.vector_norm(low.source))
                    row = dict(size=size, index=index, method=spec['label'], precision=precision,
                        iterations=len(history)-1, solver_residual=history[-1], recomputed64=residual64,
                        rounded_solution_residual64=rounded64, rounded_solution_residual32=rounded32)
                    rows.append(row)
                    save_checkpoint(out/f'trace_n{size}_{index}_{spec["label"]}_{precision}.pt',
                        dict(history=history, elapsed=elapsed, calls=calls, solution=x.cpu()))
                    write_csv(out/'summary.csv', rows)
                    print(row, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specs', required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[64, 128])
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2402)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
