"""Fixed-coarse-response opportunity check on N64/N128 physical systems."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from rte.coarse_response_transport import CachedCoarseResponse, CoarseResponseTransport
from rte.compiled_hss import CompiledHSS
from rte.hierarchical_response import HierarchicalResponse
from rte.mixed_precision_transport import high_precision_copy
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace
from rte.parity_coarse import GalerkinSchurCorrection
from rte.parity_transport import ParitySchur
from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_medium_dataset, make_operator, source_from_data
from rte.run_residual_subspace_experiment import write_csv
from rte.transport_baselines import ClassicalTransportStack


@torch.no_grad()
def run(args):
    if max(args.coarse_sizes) > 16 or not 1 <= args.count <= 4:
        raise ValueError('bounded coarse oracle requires m <= 16 and 1 <= count <= 4')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    randomizer = random.Random(args.seed)
    cases, setups = [], []
    settings = dict(args=vars(args), device='cpu', torch_version=torch.__version__, dataset_count=4,
        scope='development opportunity check, not AI; exact Galerkin response with fine HSS complement',
        timing='one CPU thread, shuffled paired warmed solves, shared host; teacher costs separately reported',
        physics='original stored FP32 geometry promoted unchanged to FP64; true outer tolerance 1e-5',
        datasets={})
    for size in args.sizes:
        cfg = ExperimentConfig(size=size, n_angles=12, anisotropy_g=.65, eta_scale=0., regimes='mixed')
        data = make_medium_dataset(cfg, 4, args.seed)
        torch.save(data, out/f'dataset_n{size}.pt')
        settings['datasets'][str(size)] = hashlib.sha256((out/f'dataset_n{size}.pt').read_bytes()).hexdigest()
        (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
        for epsilon in args.epsilons:
            for index in range(args.count):
                torch.set_num_threads(args.setup_threads)
                ids = torch.tensor([index])
                q = source_from_data(cfg, data, ids, torch.device('cpu'))
                t = data['sigma_t'][ids]/epsilon
                s = t-epsilon*(data['sigma_t'][ids]-data['sigma_s'][ids])
                low = make_operator(cfg, t, s, q, torch.device('cpu'))
                fine = high_precision_copy(low)
                started = time.perf_counter()
                parity = ParitySchur(fine)
                parity_seconds = time.perf_counter()-started
                base = dict(size=size, epsilon=epsilon, index=index)
                actions = {}
                for m in args.coarse_sizes:
                    started = time.perf_counter()
                    coarse = GalerkinSchurCorrection(parity, size=m)
                    gal_seconds = time.perf_counter()-started
                    started = time.perf_counter()
                    cache = CachedCoarseResponse(coarse)
                    inverse_seconds = time.perf_counter()-started
                    responses = [('dense', cache, inverse_seconds)]
                    # The factorized classical solve is not hidden behind the
                    # more expensive inverse-building oracle baseline.
                    responses.append(('cholesky', coarse, 0.))
                    if args.hierarchy:
                        started = time.perf_counter()
                        response = HierarchicalResponse.from_dense(torch.cholesky_inverse(coarse.factor),
                            parity.half, m, matrix=coarse.matrix, tolerance=args.tolerance,
                            leaf_size=2, rank_multiple=4).to(dtype=torch.complex64)
                        compressed_seconds = time.perf_counter()-started
                        responses.append(('hierarchy', CachedCoarseResponse(coarse, response), compressed_seconds))
                    for kind, response_action, extra_seconds in responses:
                        if kind not in args.cache_kinds:
                            continue
                        for depth, post in [(4, False), (4, True), (8, False)]:
                            started = time.perf_counter()
                            hss = CompiledHSS(low, depth, 2.).prepare()
                            hss_seconds = time.perf_counter()-started
                            name = f'hss{depth}_coarse{m}_{kind}'+('_post' if post else '')
                            actions[name] = (CoarseResponseTransport(fine, parity, response_action, hss, post,
                                recover_odd_rhs=not args.harmonic_only, safeguard=args.safeguard),
                                parity_seconds+gal_seconds+extra_seconds+hss_seconds)
                    details = dict(**base, coarse_size=m, parity_setup_seconds=parity_seconds,
                        galerkin_seconds=gal_seconds, inverse_seconds=inverse_seconds,
                        dense_cache_bytes=cache.inverse.numel()*cache.inverse.element_size(), symmetry_defect=coarse.symmetry_defect)
                    if args.hierarchy:
                        details.update(hierarchy_seconds=compressed_seconds, **response.diagnostics())
                    setups.append(details)
                    print(f'N{size} eps={epsilon} medium={index}: m{m} Galerkin {gal_seconds:.2f}s', flush=True)
                torch.set_num_threads(1)
                for name, build in [
                    ('hss16s2', lambda: CompiledHSS(low, 16, 2.).prepare()),
                    ('hss32s2', lambda: CompiledHSS(low, 32, 2.).prepare()),
                    ('reference8', lambda: ClassicalTransportStack(low, 'reference', 8, 0.)),
                    ('tsa8', lambda: ClassicalTransportStack(low, 'tsa', 8, 0.))]:
                    started = time.perf_counter()
                    model = build()
                    elapsed = time.perf_counter()-started
                    actions[name] = (lambda r, model=model: model(r.to(torch.complex64)).to(torch.complex128), elapsed)
                def solve(action, rhs):
                    return fgmres_real_stacked_with_trace(fine.A_op, rhs, action,
                        tol=1e-5, maxiter=100, restart=25, max_restarts=3)
                for rhs_id, shift in enumerate(((0, 0), (size//4, size//3), (size//2, size//4))[:args.sources]):
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
                            elapsed = time.perf_counter()-started
                            residual = float(torch.linalg.vector_norm(rhs-fine.A_op(x))/torch.linalg.vector_norm(rhs))
                            cases.append(dict(**base, rhs_id=rhs_id, repeat=repeat, order=order, method=name,
                                iterations=len(trace)-1, true_residual=residual, failed=int(not residual <= 1e-5),
                                warm_seconds=elapsed, setup_seconds=setup, setup_plus_warm_seconds=setup+elapsed))
                            print(f'  {name}: {len(trace)-1} iterations, residual {residual:.2e}', flush=True)
                    write_csv(out/'cases.csv', cases)
                    write_csv(out/'setups.csv', setups)
                print(f'N{size} eps={epsilon} medium={index}: solves complete', flush=True)
    print(f'Completed {len(cases)} measured solves.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--coarse-sizes', type=int, nargs='+', default=[8, 16])
    parser.add_argument('--epsilons', type=float, nargs='+', default=[.1])
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1919)
    parser.add_argument('--sources', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--setup-threads', type=int, default=4)
    parser.add_argument('--hierarchy', action='store_true')
    parser.add_argument('--harmonic-only', action='store_true')
    parser.add_argument('--safeguard', action='store_true')
    parser.add_argument('--cache-kinds', nargs='+', choices=['dense', 'cholesky', 'hierarchy'], default=['dense', 'cholesky'])
    parser.add_argument('--tolerance', type=float, default=.001)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
