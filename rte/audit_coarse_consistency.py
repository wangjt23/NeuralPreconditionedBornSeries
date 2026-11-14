"""Separate coarse range, coarse-operator consistency, and approximate inversion."""

import argparse
import json
from pathlib import Path

import torch

from rte.compiled_hss import CompiledHSS
from rte.galerkin_transport import GalerkinTransport
from rte.mixed_precision_transport import high_precision_copy
from rte.multilevel_neural_transport import MultilevelNeuralTransport, prolong_field, restrict_field
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator, save_checkpoint, source_from_data
from rte.run_residual_subspace_experiment import write_csv
from rte.train_krylov_transport import short_fgmres
from rte.transport_baselines import ClassicalTransportStack


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    data = torch.load(args.dataset, map_location='cpu', weights_only=True)['data']
    size = data['sigma_t'].shape[-1]
    cfg = ExperimentConfig(size=size, n_angles=12, anisotropy_g=.65, eta_scale=0.)
    rows = []
    norm = torch.linalg.vector_norm
    for index in range(args.count):
        ids = torch.tensor([index])
        q = source_from_data(cfg, data, ids, torch.device('cpu'))
        low = make_operator(cfg, data['sigma_t'][ids], data['sigma_s'][ids], q, torch.device('cpu'))
        fine = high_precision_copy(low)
        _, x3 = short_fgmres(fine.A_op, CompiledHSS(low, 4), fine.source, 3)
        reference, reference_trace, _, _ = fgmres_real_stacked_with_trace(fine.A_op, fine.source,
            CompiledHSS(fine, 8), tol=1e-10, maxiter=120, restart=20, max_restarts=5)
        if reference_trace[-1] > 1e-9:
            raise RuntimeError('reference solve did not converge')
        error, residual = reference-x3, fine.source-fine.A_op(x3)
        transfer = MultilevelNeuralTransport(fine, lambda op: CompiledHSS(op, 4), 32)
        transfer.setup_coarse()
        coarse = transfer.coarse_operator
        rhs = restrict_field(residual, 32)
        galerkin = GalerkinTransport(fine, 32)
        projected_error = restrict_field(error, 32)
        consistency = float(norm(coarse.A_op(projected_error)-galerkin(projected_error))/norm(galerkin(projected_error)))
        alias_check = float(norm(galerkin(projected_error)-galerkin.direct(projected_error))/norm(galerkin.direct(projected_error)))
        proposals = [('oracle_band_error', projected_error, 0., 0),
                     ('coarse_hss4', CompiledHSS(coarse, 4)(rhs), None, 4)]
        reference_preconditioner = ClassicalTransportStack(coarse, 'reference', 1, 0.)
        def projected_reference(value):
            band = restrict_field(value, 32)
            return restrict_field(reference_preconditioner(band), 32)+(value-band)/galerkin.mean_t
        for label, action, preconditioner in [('rediscretized_solve', coarse.A_op, CompiledHSS(coarse, 8)),
            ('galerkin_hss8_solve', galerkin, CompiledHSS(coarse, 8)),
            ('galerkin_projected_reference_solve', galerkin, projected_reference)]:
            solution, trace, _, _ = fgmres_real_stacked_with_trace(action, rhs, preconditioner,
                tol=1e-9, maxiter=120, restart=20, max_restarts=5)
            proposals.append((label, solution, trace[-1], len(trace)-1))
        for label, correction, solve_residual, steps in proposals:
            updated = x3+prolong_field(correction, size)
            post = updated+CompiledHSS(fine, 2)(fine.source-fine.A_op(updated))
            rows.append(dict(index=index, method=label, consistency_error=consistency, galerkin_alias_check=alias_check,
                inner_relative_residual=solve_residual, inner_iterations=steps,
                error_ratio=float(norm(reference-updated)/norm(error)),
                residual_ratio=float(norm(fine.source-fine.A_op(updated))/norm(residual)),
                post_hss2_error_ratio=float(norm(reference-post)/norm(error)),
                post_hss2_residual_ratio=float(norm(fine.source-fine.A_op(post))/norm(residual))))
        write_csv(out/'summary.csv', rows)
        save_checkpoint(out/f'reference_{index}.pt', dict(reference=reference, x3=x3))
        print('coarse consistency case', index, 'relative discrepancy', consistency, flush=True)
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args),
        scope='development oracle diagnostic, not a practical timed solver',
        inner_failures='inspect inner residual; a capped inner solve is not exact'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
