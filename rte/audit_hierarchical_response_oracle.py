"""Small-grid end-to-end oracle: does block compression survive a true solve?

Only well-separated blocks are truncated; near blocks remain exact. Dense
oracle setup/application is not a deployable solver and no runtime win is
claimed. This is a two-scale block test, not a complete H/H2 implementation.
"""

import argparse
import json
from pathlib import Path

import torch

from rte.audit_parity_response_blocks import dense_action, rank_for_frobenius
from rte.compiled_hss import CompiledHSS
from rte.mixed_precision_transport import high_precision_copy
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace
from rte.parity_transport import ParitySchur
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_medium_dataset, make_operator, source_from_data
from rte.run_residual_subspace_experiment import write_csv
from rte.transport_baselines import ClassicalTransportStack


def reconstruct_block(u, singular, vh, rank, triangular=None):
    block = (u[:, :rank]*singular[:rank]) @ vh[:rank]
    return block if triangular is None else torch.linalg.solve_triangular(triangular, block, upper=True)


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    size = args.size
    if size % 4:
        raise ValueError('grid must be divisible by four')
    cfg = ExperimentConfig(size=size, n_angles=12, anisotropy_g=.65, eta_scale=0., regimes='mixed')
    data = make_medium_dataset(cfg, 4, args.seed)
    rows, diagnostics = [], []
    for epsilon in (1., .1):
        for index in range(args.count):
            ids = torch.tensor([index])
            source = source_from_data(cfg, data, ids, device)
            t = data['sigma_t'][ids]/epsilon
            s = t-epsilon*(data['sigma_t'][ids]-data['sigma_s'][ids])
            low = make_operator(cfg, t, s, source, device)
            fine = high_precision_copy(low)
            parity = ParitySchur(fine)
            matrix = dense_action(parity.schur, parity.half, size, device)
            inverse = torch.cholesky_inverse(torch.linalg.cholesky((matrix+matrix.mH)/2))
            dim = len(matrix)
            grid = torch.arange(dim, device=device).reshape(parity.half, size, size)
            width = size//4
            blocks = [grid[:, x*width:(x+1)*width, y*width:(y+1)*width].flatten()
                      for x in range(4) for y in range(4)]
            pairs = []
            triangular = [torch.linalg.qr(matrix[:, block], mode='r')[1] for block in blocks] if args.metric == 'residual' else None
            for i in range(16):
                for j in range(16) if triangular is not None else range(i+1, 16):
                    x, y = abs(i//4-j//4), abs(i%4-j%4)
                    if max(min(x, 4-x), min(y, 4-y)) < 2:
                        continue
                    block = inverse[blocks[i][:, None], blocks[j][None, :]]
                    weighted = triangular[i] @ block if triangular is not None else block
                    pairs.append((i, j, torch.linalg.svd(weighted, full_matrices=False)))
            versions = [('exact_schur_inverse', inverse)]
            for tolerance in args.tolerances:
                approximation = inverse.clone()
                stored = dim*dim
                ranks = []
                for i, j, (u, singular, vh) in pairs:
                    rank = min(rank_for_frobenius(singular, tolerance), len(singular))
                    ranks.append(rank)
                    block = reconstruct_block(u, singular, vh, rank, triangular[i] if triangular is not None else None)
                    approximation[blocks[i][:, None], blocks[j][None, :]] = block
                    if triangular is None:
                        approximation[blocks[j][:, None], blocks[i][None, :]] = block.mH
                    # Count storing both directed blocks; no symmetry-storage
                    # bonus, and do not replace a dense block by larger factors.
                    dense = len(blocks[i])*len(blocks[j])
                    stored -= (2 if triangular is None else 1)*(dense-min(dense, rank*(len(blocks[i])+len(blocks[j]))))
                label = f'far_block_tol{tolerance:g}'
                versions.append((label, approximation))
                diagnostics.append(dict(size=size, epsilon=epsilon, index=index, method=label,
                    mean_far_rank=sum(ranks)/len(ranks), max_far_rank=max(ranks),
                    stored_scalar_fraction=stored/(dim*dim),
                    relative_inverse_error=float(torch.linalg.vector_norm(approximation-inverse)/torch.linalg.vector_norm(inverse)),
                    inverse_defect=float(torch.linalg.vector_norm(torch.eye(dim, dtype=matrix.dtype, device=device)-matrix@approximation)/(dim**.5))))
            hss = CompiledHSS(low, 8)
            reference = ClassicalTransportStack(low, 'reference', 8, 0.)
            actions = [('hss8', lambda r: hss(r.to(torch.complex64)).to(torch.complex128)),
                       ('reference8', lambda r: reference(r.to(torch.complex64)).to(torch.complex128))]
            for label, approximation in versions:
                def action(rhs, inverse=approximation):
                    reduced = parity.reduced_rhs(rhs)
                    even = (inverse @ reduced.flatten(1).T).T.reshape_as(reduced)
                    return parity.recover(even, rhs)
                actions.append((label, action))
            for rhs_id, shift in enumerate(((0, 0), (size//4, size//3), (size//2, size//4))):
                rhs = torch.roll(fine.source, shift, (-2, -1))
                for label, action in actions:
                    x, trace, _, _ = fgmres_real_stacked_with_trace(fine.A_op, rhs, action,
                        tol=1e-5, maxiter=100, restart=25, max_restarts=3)
                    residual = float(torch.linalg.vector_norm(rhs-fine.A_op(x))/torch.linalg.vector_norm(rhs))
                    rows.append(dict(size=size, epsilon=epsilon, index=index, rhs_id=rhs_id, method=label,
                        iterations=len(trace)-1, true_residual=residual, failed=int(not residual <= 1e-5)))
            write_csv(out/'cases.csv', rows)
            write_csv(out/'compression.csv', diagnostics)
            print(f'N{size} eps={epsilon} medium={index} block-compression solve check complete', flush=True)
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args), dataset_count=4,
        scope='small-grid exact-inverse oracle; no NN, no timing claim, no O(N) hierarchical implementation',
        near='periodic 4x4 patches, same/adjacent patches exact; separated blocks truncated',
        metric='Euclidean with adjoint pairing, or output-row weighted by QR of H[:, output_patch]; weighted result need not be Hermitian',
        limitations='SPD is not guaranteed; outer FGMRES checks unchanged physical residual'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=16)
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1919)
    parser.add_argument('--metric', choices=['euclidean', 'residual'], default='euclidean')
    parser.add_argument('--tolerances', type=float, nargs='+', default=[.1, .05, .01, .001])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
