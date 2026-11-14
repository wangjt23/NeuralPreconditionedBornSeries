"""Validation-only timing of learned versus analytical moment-response shapes."""

import argparse
import json
from pathlib import Path
import random
import time

import torch

from rte.augmented_fgmres import augmented_fgmres
from rte.audit_tail_injection import operator_from_bank
from rte.run_residual_subspace_experiment import write_csv
from rte.tail_moment_space import moment_injected_solve
from rte.tail_response_network import TailResponseNetwork


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = saved['args']
    model = TailResponseNetwork(width=cfg['width'], blocks=cfg['blocks'], modes=cfg['modes']).cuda().eval()
    model.load_state_dict(saved['model'])
    bank = torch.load(args.validation_parent, map_location='cpu', weights_only=True)['bank']
    rows, rng = [], random.Random(92726)
    for index in range(len(bank['q'])):
        low, high, base, initial = operator_from_bank(bank, index, 'cuda')
        def inject(x, r):
            ref = base.reference.reference_transport_solve(r.to(torch.complex64))
            return model(low.sigma_t, low.sigma_s, r.to(torch.complex64), ref).to(torch.complex128)
        for repeat in range(-1, args.repeats):
            methods = ['classical', 'single_neural', 'moment_reference', 'moment_neural']
            rng.shuffle(methods)
            for name in methods:
                torch.cuda.synchronize()
                start = time.perf_counter()
                if name.startswith('moment'):
                    _, status = moment_injected_solve(high, low, base, initial,
                        model if name == 'moment_neural' else None)
                else:
                    _, status = augmented_fgmres(high.A_op, high.source,
                        lambda r: base(r.to(torch.complex64)).to(torch.complex128), initial,
                        inject=inject if name == 'single_neural' else None)
                torch.cuda.synchronize()
                if repeat >= 0:
                    status.pop('trace')
                    rows.append(dict(index=index, repeat=repeat, method=name,
                        seconds=time.perf_counter()-start, **status))
        write_csv(out/'cases.csv', rows)
    (out/'settings.json').write_text(json.dumps(dict(args=vars(args), complete=True,
        scope='validation only; one measured repeat after warmup; no held-out claim'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--validation-parent', default='rte/workstation_runs/source_tail_validation_20260926/validation.pt')
    parser.add_argument('--repeats', type=int, default=1)
    run(parser.parse_args())
