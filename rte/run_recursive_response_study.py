"""Bounded recursive response oracle: real online action, explicit teacher cost.

Not an AI benchmark. Dataset/material selection is fixed before measurements;
classical candidates are reported separately, not tuned on these test cases.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import torch

from rte.audit_parity_response_blocks import dense_action
from rte.compiled_hss import CompiledHSS
from rte.hierarchical_response import HierarchicalParityPreconditioner, HierarchicalResponse
from rte.mixed_precision_transport import high_precision_copy
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace
from rte.parity_transport import ParitySchur
from rte.response_residual_repair import RepairedResponse, residual_repair_basis
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_medium_dataset, make_operator, source_from_data
from rte.run_residual_subspace_experiment import write_csv
from rte.transport_baselines import ClassicalTransportStack


@torch.no_grad()
def run(args):
    if args.size > 32:
        raise ValueError('dense teachers are deliberately bounded to N <= 32')
    if not 1 <= args.count <= 4 or args.repeats < 1 or args.probes < 1:
        raise ValueError('invalid count/repeats/probes')
    torch.manual_seed(args.seed)
    randomizer = random.Random(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = ExperimentConfig(size=args.size, n_angles=12, anisotropy_g=.65, eta_scale=0., regimes='mixed')
    data = make_medium_dataset(cfg, 4, args.seed)
    torch.save(data, out/'dataset.pt')
    fingerprint = hashlib.sha256((out/'dataset.pt').read_bytes()).hexdigest()
    settings = dict(args=vars(args), torch_version=torch.__version__, platform=platform.platform(),
        dataset_count=4, dataset_sha256=fingerprint, device='cpu',
        timing='one CPU thread; paired randomized warmed full solves; teacher setup is separately reported',
        scope='directed recursive H-matrix oracle, not trained AI; all true residuals use unchanged physical FP64 promotion',
        grouping='dense leaves and low-rank far blocks; padding AND int64 index bytes counted',
        limitation='small fixed sample; shared host; no scalability or NN speedup claim; QR block bounds are not global bounds')
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    cases, compression, teachers, blocks = [], [], [], []
    size = args.size
    for epsilon in args.epsilons:
        for index in range(args.count):
            torch.set_num_threads(args.setup_threads)
            ids = torch.tensor([index])
            source = source_from_data(cfg, data, ids, torch.device('cpu'))
            t = data['sigma_t'][ids]/epsilon
            s = t-epsilon*(data['sigma_t'][ids]-data['sigma_s'][ids])
            low = make_operator(cfg, t, s, source, torch.device('cpu'))
            fine = high_precision_copy(low)
            started = time.perf_counter()
            parity = ParitySchur(fine)
            parity_setup = time.perf_counter()-started
            started = time.perf_counter()
            matrix = dense_action(parity.schur, parity.half, size, torch.device('cpu'))
            inverse = torch.cholesky_inverse(torch.linalg.cholesky((matrix+matrix.mH)/2))
            teacher_seconds = time.perf_counter()-started
            dimension = len(matrix)
            generator = torch.Generator().manual_seed(8911+index)
            probes = torch.randn(args.probes, dimension, generator=generator, dtype=torch.complex128)
            exact_probe = probes@inverse.T
            exact_defect = float(torch.linalg.vector_norm(exact_probe@matrix.T-probes)/torch.linalg.vector_norm(probes))
            base = dict(size=size, epsilon=epsilon, index=index)
            teachers.append(dict(**base, parity_setup_seconds=parity_setup,
                dense_teacher_seconds=teacher_seconds, probe_inverse_defect=exact_defect))
            print(f'N{size} eps={epsilon} medium={index}: teacher {teacher_seconds:.2f}s, defect {exact_defect:.2e}', flush=True)
            actions = {}
            def exact(rhs):
                reduced = parity.reduced_rhs(rhs)
                even = (inverse@reduced.flatten(1).T).T.reshape_as(reduced)
                return parity.recover(even, rhs)
            actions['dense_oracle'] = (exact, parity_setup+teacher_seconds)
            for metric in args.metrics:
                for tolerance in args.tolerances:
                    started = time.perf_counter()
                    response = HierarchicalResponse.from_dense(inverse, parity.half, size,
                        matrix=matrix if metric != 'euclidean' else None, tolerance=tolerance,
                        leaf_size=args.leaf_size, rank_multiple=args.rank_multiple, metric=metric)
                    compress_seconds = time.perf_counter()-started
                    label = f'hierarchy_{metric}_tol{tolerance:g}'
                    for record in response.records:
                        blocks.append(dict(**base, method=label, **record))
                    versions = [(label, response, 0.)]
                    if args.repair_ranks:
                        started = time.perf_counter()
                        correction, basis, singular = residual_repair_basis(matrix, inverse, response,
                            rank=max(args.repair_ranks), seed=418+index)
                        repair_seconds = time.perf_counter()-started
                        torch.save(dict(singular=singular, rank=max(args.repair_ranks)),
                            out/f'repair_spectrum_e{epsilon}_i{index}_{label}.pt')
                        for rank in args.repair_ranks:
                            versions.append((label+f'_repair{rank}', RepairedResponse(response, correction[:, :rank], basis[:, :rank]), repair_seconds))
                    for version, response_version, extra_seconds in versions:
                        for precision, candidate in [('fp64', response_version), ('fp32', response_version.to(dtype=torch.complex64))]:
                            name = version+'_'+precision
                            approximated = candidate(probes.to(candidate.groups[0]['factors'][0].dtype)).to(probes.dtype)
                            diagnostics = candidate.diagnostics()
                            compression.append(dict(**base, method=name, compression_seconds=compress_seconds,
                                repair_seconds=extra_seconds, **diagnostics,
                                relative_inverse_probe_error=float(torch.linalg.vector_norm(approximated-exact_probe)/torch.linalg.vector_norm(exact_probe)),
                                residual_probe_defect=float(torch.linalg.vector_norm(approximated@matrix.T-probes)/torch.linalg.vector_norm(probes))))
                            actions[name] = (HierarchicalParityPreconditioner(parity, candidate), parity_setup+teacher_seconds+compress_seconds+extra_seconds)
                    base_diagnostics = response.diagnostics()
                    print(f'  {label}: {compress_seconds:.2f}s, base coefficients {base_diagnostics["stored_scalar_fraction"]:.3f} dense, {base_diagnostics["groups"]} groups', flush=True)
            # Shared host is not an isolated performance machine. These bounded
            # CPU runs leave the busy GPU untouched and separate teacher setup.
            torch.set_num_threads(1)
            for name, build in [
                ('hss8s1', lambda: CompiledHSS(low, 8, 1.).prepare()),
                ('hss16s1', lambda: CompiledHSS(low, 16, 1.).prepare()),
                ('hss16s2', lambda: CompiledHSS(low, 16, 2.).prepare()),
                ('hss32s2', lambda: CompiledHSS(low, 32, 2.).prepare()),
                ('reference8', lambda: ClassicalTransportStack(low, 'reference', 8, 0.)),
                ('tsa8', lambda: ClassicalTransportStack(low, 'tsa', 8, 0.))]:
                started = time.perf_counter()
                model = build()
                setup = time.perf_counter()-started
                actions[name] = (lambda rhs, model=model: model(rhs.to(torch.complex64)).to(torch.complex128), setup)
            def solve(action, rhs):
                return fgmres_real_stacked_with_trace(fine.A_op, rhs, action,
                    tol=1e-5, maxiter=100, restart=25, max_restarts=3)
            for rhs_id, shift in enumerate(((0, 0), (size//4, size//3), (size//2, size//4))):
                rhs = torch.roll(fine.source, shift, (-2, -1))
                labels = list(actions)
                randomizer.shuffle(labels)
                for name in labels:
                    solve(actions[name][0], rhs)
                for repeat in range(args.repeats):
                    randomizer.shuffle(labels)
                    for order, name in enumerate(labels):
                        action, setup = actions[name]
                        started = time.perf_counter()
                        x, trace, _, _ = solve(action, rhs)
                        seconds = time.perf_counter()-started
                        residual = float(torch.linalg.vector_norm(rhs-fine.A_op(x))/torch.linalg.vector_norm(rhs))
                        cases.append(dict(**base, rhs_id=rhs_id, repeat=repeat, order=order, method=name,
                            iterations=len(trace)-1, true_residual=residual, failed=int(not residual <= 1e-5),
                            warm_seconds=seconds, setup_seconds=setup, setup_plus_warm_seconds=setup+seconds))
            write_csv(out/'cases.csv', cases)
            write_csv(out/'compression.csv', compression)
            write_csv(out/'teachers.csv', teachers)
            write_csv(out/'blocks.csv', blocks)
            print(f'N{size} eps={epsilon} medium={index}: true-physical solves complete', flush=True)
            del actions, response, candidate, matrix, inverse, exact_probe
    print(f'Completed {len(cases)} measured solves; no experiment remains in this process.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=16)
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1919)
    parser.add_argument('--epsilons', type=float, nargs='+', default=[1., .1])
    parser.add_argument('--metrics', choices=['residual', 'energy', 'euclidean'], nargs='+', default=['residual'])
    parser.add_argument('--tolerances', type=float, nargs='+', default=[.001])
    parser.add_argument('--leaf-size', type=int, default=2)
    parser.add_argument('--rank-multiple', type=int, default=4)
    parser.add_argument('--setup-threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--probes', type=int, default=16)
    parser.add_argument('--repair-ranks', type=int, nargs='*', default=[])
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
