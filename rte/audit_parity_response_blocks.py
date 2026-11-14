"""Exact small-grid response-block audit; no neural or speedup claim.

Dense inverses here are expensive diagnostic teachers only. Probe the even
Schur inverse and its correction to the homogeneous Fourier inverse, across
near/far spatial blocks. Small-grid rank is not a large-grid guarantee.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from rte.mixed_precision_transport import high_precision_copy
from rte.parity_transport import ParitySchur
from rte.run_large_angular_spectral_experiments import (
    ExperimentConfig, make_medium_dataset, make_operator, save_checkpoint, source_from_data,
)
from rte.run_residual_subspace_experiment import write_csv


def dense_action(action, channels, size, device, chunk=96):
    dim = channels*size*size
    columns = []
    for start in range(0, dim, chunk):
        end = min(start+chunk, dim)
        value = torch.zeros(end-start, dim, dtype=torch.complex128, device=device)
        value[torch.arange(end-start, device=device), torch.arange(start, end, device=device)] = 1
        columns.append(action(value.reshape(-1, channels, size, size)).flatten(1).T)
    return torch.cat(columns, dim=1)


def rank_for_frobenius(singular, tolerance):
    square = singular.square()
    if float(square.sum()) == 0:
        return 0
    captured = square.cumsum(0)/square.sum()
    return int(torch.searchsorted(captured, 1-tolerance*tolerance))+1


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    rows, checks, spectra = [], [], {}
    for size in args.sizes:
        cfg = ExperimentConfig(size=size, n_angles=12, anisotropy_g=.65, eta_scale=0., regimes='mixed')
        data = make_medium_dataset(cfg, args.count, args.seed)
        save_checkpoint(out/f'data_n{size}.pt', dict(data=data))
        for epsilon in args.epsilons:
            for index in range(min(args.count, args.limit or args.count)):
                ids = torch.tensor([index])
                q = source_from_data(cfg, data, ids, device)
                t = data['sigma_t'][ids]/epsilon
                s = t-epsilon*(data['sigma_t'][ids]-data['sigma_s'][ids])
                fine = high_precision_copy(make_operator(cfg, t, s, q, device))
                parity = ParitySchur(fine)
                sync()
                start = time.perf_counter()
                matrix = dense_action(parity.schur, parity.half, size, device)
                symmetry = float(torch.linalg.vector_norm(matrix-matrix.mH)/torch.linalg.vector_norm(matrix))
                # Symmetrization removes only measured assembly roundoff.
                symmetric = (matrix+matrix.mH)/2
                inverse = torch.cholesky_inverse(torch.linalg.cholesky(symmetric))
                reference = dense_action(parity.reference, parity.half, size, device)
                identity = torch.eye(len(matrix), dtype=matrix.dtype, device=device)
                inverse_check = float(torch.linalg.vector_norm(matrix@inverse-identity)/torch.linalg.vector_norm(identity))
                correction = inverse-reference
                sync()
                acquisition = time.perf_counter()-start
                checks.append(dict(size=size, epsilon=epsilon, index=index, symmetry_defect=symmetry,
                    inverse_check=inverse_check, dense_teacher_seconds=acquisition))
                grid = torch.arange(parity.half*size*size, device=device).reshape(parity.half, size, size)
                for divisions in args.divisions:
                    width = size//divisions
                    if width*divisions != size:
                        continue
                    source_ids = grid[:, :width, :width].flatten()
                    for tx, ty in ((0, 0), (1, 0), (divisions//2, divisions//2)):
                        target_ids = grid[:, tx*width:(tx+1)*width, ty*width:(ty+1)*width].flatten()
                        distance = max(min(tx, divisions-tx), min(ty, divisions-ty))
                        for name, operator in [('inverse', inverse), ('correction', correction)]:
                            block = operator[target_ids[:, None], source_ids[None, :]]
                            singular = torch.linalg.svdvals(block)
                            key = f'n{size}_e{epsilon}_i{index}_d{divisions}_p{tx}_{ty}_{name}'
                            spectra[key] = singular.cpu()
                            rows.append(dict(size=size, epsilon=epsilon, index=index, divisions=divisions,
                                target_x=tx, target_y=ty, periodic_patch_distance=distance,
                                operator=name, dimension=len(source_ids), frobenius_norm=float(torch.linalg.vector_norm(block)),
                                rank_1percent=rank_for_frobenius(singular, .01),
                                rank_5percent=rank_for_frobenius(singular, .05),
                                rank_10percent=rank_for_frobenius(singular, .1)))
                write_csv(out/'blocks.csv', rows)
                write_csv(out/'checks.csv', checks)
                save_checkpoint(out/'spectra.pt', spectra)
                print(f'N{size} eps={epsilon} medium={index}: inverse check {inverse_check:.2e}', flush=True)
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args),
        scope='dense small-grid structural diagnostic; compression errors are block Frobenius, not solver errors',
        operators='even Schur inverse and inverse minus homogeneous reference; all retained angular channels',
        limitation='sampled source patch only; no whole-operator hierarchical approximation or network tested'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[16])
    parser.add_argument('--divisions', type=int, nargs='+', default=[4])
    parser.add_argument('--epsilons', type=float, nargs='+', default=[1., .1])
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0, help='evaluate only this prefix without changing the dataset RNG stream')
    parser.add_argument('--seed', type=int, default=1919)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
