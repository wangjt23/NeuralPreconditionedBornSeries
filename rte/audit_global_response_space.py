"""Development-only oracle test of material-specific global response spaces.

No network is trained. Calibration reference solves are expensive information,
not free setup. Translated sources are withheld from basis construction, but
the materials and source shapes are development data, not confirmation data.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from rte.compiled_hss import CompiledHSS
from rte.mixed_precision_transport import high_precision_copy
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace
from rte.run_large_angular_spectral_experiments import (
    ExperimentConfig, dataset_fingerprint, make_operator, save_checkpoint, source_from_data,
)
from rte.run_residual_subspace_experiment import write_csv
from rte.train_krylov_transport import short_fgmres


class VerifiedResponseSpace:
    """Fixed-medium augmentation: AW=Q, Q*Q=I; SVD filters dependent columns.

    The action is W Q* r + M0 (I-Q Q*) r. It retains the full base range;
    exactness on span(Q) is not a general convergence or speed guarantee.
    """

    def __init__(self, action, base, columns, field_shape, rtol=1e-10):
        self.base, self.field_shape = base, tuple(field_shape)
        images = action(columns.T.reshape(-1, *field_shape)).flatten(1).T
        q, singular, vh = torch.linalg.svd(images, full_matrices=False)
        keep = singular > singular.max()*rtol
        self.q = q[:, keep]
        self.w = (columns @ vh.mH[:, keep])/singular[keep]
        self.rank = int(keep.sum())
        verified = action(self.w.T.reshape(-1, *field_shape)).flatten(1).T
        self.verification_error = float(torch.linalg.vector_norm(verified-self.q)
            / torch.linalg.vector_norm(self.q).clamp_min(1e-300))

    def __call__(self, residual):
        vectors = residual.flatten(1).T
        weights = self.q.mH @ vectors
        correction = (self.w @ weights).T.reshape_as(residual)
        remainder = (vectors-self.q @ weights).T.reshape_as(residual)
        return correction+self.base(remainder)


def leading_columns(snapshots):
    matrix = torch.cat(snapshots).flatten(1).T
    # Normalize snapshots so source amplitudes do not decide importance.
    matrix = matrix/torch.linalg.vector_norm(matrix, dim=0).clamp_min(1e-300)
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    return u, singular


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    payload = torch.load(args.dataset, map_location='cpu', weights_only=True)
    data = payload['data']
    if dataset_fingerprint(data) != payload['fingerprint']:
        raise ValueError('dataset fingerprint mismatch')
    size = data['sigma_t'].shape[-1]
    cfg = ExperimentConfig(size=size, n_angles=12, anisotropy_g=.65, eta_scale=0.)
    # Distinct shifts; no held-out evaluation translation is used as a teacher.
    shifts = [(0, 0), (1, 0), (2, 1), (3, 2), (0, 3), (1, 2), (2, 3), (3, 0),
              (1, 1), (2, 2), (3, 3), (0, 2), (2, 0), (3, 1)]
    rows, setups, traces = [], [], []
    for index in range(args.count):
        ids = torch.tensor([index])
        source = source_from_data(cfg, data, ids, torch.device('cpu'))
        low = make_operator(cfg, data['sigma_t'][ids], data['sigma_s'][ids], source, torch.device('cpu'))
        fine = high_precision_copy(low)
        inner = CompiledHSS(low, 8).prepare()
        base = lambda r: inner(r.to(torch.complex64)).to(torch.complex128)
        sources = [torch.roll(fine.source, (i*size//4, j*size//4), (-2, -1)) for i, j in shifts]
        errors, captured = [], []
        start = time.perf_counter()
        reference_steps = 0
        for rhs in sources[:8]:
            exact, trace, _, _ = fgmres_real_stacked_with_trace(fine.A_op, rhs, CompiledHSS(fine, 8),
                tol=1e-10, maxiter=120, restart=20, max_restarts=5)
            if trace[-1] > 1e-9:
                raise RuntimeError('calibration reference did not converge')
            reference_steps += len(trace)-1
            _, approximation = short_fgmres(fine.A_op, inner, rhs, 2)
            errors.append(exact-approximation)
        oracle_seconds = time.perf_counter()-start
        basis, singular = leading_columns(errors)
        start = time.perf_counter()
        def capture(residual):
            value = base(residual)
            captured.append(value.clone())
            return value
        fgmres_real_stacked_with_trace(fine.A_op, sources[0], capture,
            tol=0., maxiter=8, restart=8, max_restarts=0)
        recycle_seconds = time.perf_counter()-start
        recycle_basis, recycle_singular = leading_columns(captured)
        methods = [('hss8', base)]
        for kind, vectors, acquisition in [('oracle_error', basis, oracle_seconds),
                                           ('first_rhs_krylov', recycle_basis, recycle_seconds)]:
            for rank in (2, 4, 8):
                if rank > vectors.shape[1]:
                    continue
                start = time.perf_counter()
                model = VerifiedResponseSpace(fine.A_op, base, vectors[:, :rank], fine.source.shape[1:])
                setup_seconds = time.perf_counter()-start
                methods.append((f'{kind}_r{rank}', model))
                setups.append(dict(index=index, method=methods[-1][0], rank=model.rank,
                    verification_error=model.verification_error,
                    information_acquisition_seconds=acquisition, verification_setup_seconds=setup_seconds,
                    setup_A_vectors=2*rank, teacher_reference_iterations=reference_steps if kind == 'oracle_error' else 0))
        save_checkpoint(out/f'basis_{index}.pt', dict(oracle_basis=basis, oracle_singular=singular,
            recycle_basis=recycle_basis, recycle_singular=recycle_singular, shifts=shifts))
        for rhs_index, rhs in enumerate(sources[8:], 8):
            # These held-out diagnostics occur only after the bases are fixed.
            exact, reference_trace, _, _ = fgmres_real_stacked_with_trace(fine.A_op, rhs, CompiledHSS(fine, 8),
                tol=1e-10, maxiter=120, restart=20, max_restarts=5)
            if reference_trace[-1] > 1e-9:
                raise RuntimeError('held-out reference did not converge')
            _, x2 = short_fgmres(fine.A_op, inner, rhs, 2)
            error_vector = (exact-x2).flatten()
            residual_vector = (rhs-fine.A_op(x2)).flatten()
            for label, model in methods:
                start = time.perf_counter()
                solution, trace, _, _ = fgmres_real_stacked_with_trace(fine.A_op, rhs, model,
                    tol=1e-5, maxiter=60, restart=20, max_restarts=2)
                elapsed = time.perf_counter()-start
                true_residual = float(torch.linalg.vector_norm(rhs-fine.A_op(solution))/torch.linalg.vector_norm(rhs))
                if label == 'hss8':
                    error_capture = residual_capture = 0.
                else:
                    u, _, _ = torch.linalg.svd(model.w, full_matrices=False)
                    error_capture = float((u.mH @ error_vector).abs().square().sum()/error_vector.abs().square().sum())
                    residual_capture = float((model.q.mH @ residual_vector).abs().square().sum()/residual_vector.abs().square().sum())
                rows.append(dict(index=index, rhs_index=rhs_index, method=label, iterations=len(trace)-1,
                    true_residual=true_residual, failed=int(true_residual > 1e-5), solve_seconds=elapsed,
                    heldout_error_energy_captured=error_capture, heldout_residual_energy_captured=residual_capture))
                traces.extend(dict(index=index, rhs_index=rhs_index, method=label, step=step, residual=value)
                    for step, value in enumerate(trace))
            write_csv(out/'cases.csv', rows)
            write_csv(out/'traces.csv', traces)
        write_csv(out/'setup.csv', setups)
        print(f'completed N{size} development medium {index}', flush=True)
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args), fingerprint=payload['fingerprint'],
        scope='oracle diagnostic, no learned model; single sequential timing, not a benchmark',
        sources='8 calibration and 6 held-out periodic translations of each original source; same angular shape',
        baseline='compiled HSS8; first-RHS Krylov span is a simple recycling control, not tuned GCRO-DR',
        setup='2k physical A vectors includes second verification; teacher cost reported separately',
        limitations='same development materials, limited source family; no unseen-material or speedup claim'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
