"""Validation-only audit of the classical direction immediately after injection.

The default flexible Arnoldi path follows the new neural image. This alternative
uses the pre-injection residual for the next classical application, then resumes
ordinary Arnoldi. It tests whether that local trajectory choice wastes the
network's immediate residual improvement. It is not an exact block-GMRES claim.
"""

import argparse
import json
from pathlib import Path
import random
import time

import torch

from rte.augmented_fgmres import augmented_fgmres
from rte.audit_tail_injection import operator_from_bank
from rte.benchmark_tail_solver import digest, load_model, summarize
from rte.run_residual_subspace_experiment import write_csv


def frontier_fgmres(action, rhs, precondition, initial=None, inject=None, **kwargs):
    pending = []
    def proposal(x, r):
        z = inject(x, r)
        if bool(torch.isfinite(z).all()) and bool(z.norm() > 0):
            pending.append(r/r.norm())
        return z
    def classical(v):
        return precondition(pending.pop() if pending else v)
    return augmented_fgmres(action, rhs, classical, initial,
        inject=proposal if inject is not None else None, **kwargs)


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    parent = torch.load(args.validation_parent, map_location='cpu', weights_only=True)
    models = dict(tail=load_model(args.tail_checkpoint, False, 'cuda'),
        optical=load_model(args.optical_checkpoint, True, 'cuda'))
    rows, rng = [], random.Random(26928)
    methods = ['classical', 'tail_default', 'tail_frontier', 'optical_default',
               'optical_frontier', 'reference_frontier']
    for index in range(len(parent['bank']['q'])):
        low, high, base, initial = operator_from_bank(parent['bank'], index, 'cuda')
        def precondition(r):
            return base(r.to(torch.complex64)).to(torch.complex128)
        for repeat in range(-1, args.repeats):
            rng.shuffle(methods)
            for order, method in enumerate(methods):
                name = method.split('_')[0]
                def inject(x, r):
                    r = r.to(torch.complex64)
                    ref = base.reference.reference_transport_solve(r)
                    return ref if name == 'reference' else models[name](low.sigma_t, low.sigma_s, r, ref)
                torch.cuda.synchronize()
                started = time.perf_counter()
                solver = frontier_fgmres if method.endswith('_frontier') else augmented_fgmres
                _, status = solver(high.A_op, high.source, precondition, initial,
                    inject=None if method == 'classical' else inject)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-started
                status.pop('trace')
                if repeat >= 0:
                    rows.append(dict(method=method, index=index, repeat=repeat, order=order,
                        seconds=seconds, setup_seconds=0., setup_plus_seconds=seconds, **status))
        write_csv(out/'cases.csv', rows)
    _, summary = summarize(rows)
    write_csv(out/'summary.csv', summary)
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args), complete=True,
        scope='validation-only trajectory audit; shared analytical initialization and setup excluded',
        source_sha256=digest(__file__),
        checkpoint_sha256={p:digest(p) for p in (args.tail_checkpoint, args.optical_checkpoint)}), indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--validation-parent', default='rte/workstation_runs/source_tail_validation_20260926/validation.pt')
    parser.add_argument('--tail-checkpoint', default='rte/workstation_runs/tail_direction_pilot_20260926/best.pt')
    parser.add_argument('--optical-checkpoint', default='rte/workstation_runs/tail_optical_moment_20260926/last.pt')
    parser.add_argument('--repeats', type=int, default=2)
    run(parser.parse_args())
