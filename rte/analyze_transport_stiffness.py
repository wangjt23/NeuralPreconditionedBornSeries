"""Analytic collision/HSS bounds under diffusive coefficient scaling; no solves."""

import argparse
from pathlib import Path

import torch

from rte.models import PeriodicRTE2D
from rte.run_residual_subspace_experiment import write_csv


def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    op = PeriodicRTE2D(shape=(8,8), n_angles=12, anisotropy_g=.65, fdtype=torch.float32)
    phase = torch.fft.fft(op.phase_matrix[0]*op.weights).real.double()
    rows = []
    for path in args.datasets:
        data = torch.load(path, map_location='cpu', weights_only=True)['data']
        t, s = data['sigma_t'].double(), data['sigma_s'].double()
        for index in range(min(args.count, len(t))):
            for epsilon in (1., .25, .1):
                extinction = t[index]/epsilon
                scattering = extinction-epsilon*(t[index]-s[index])
                rates = extinction-scattering*phase[:, None, None]
                lower, upper = float(rates.min()), float(rates.max())
                ratio = (upper/lower)**.5
                rows.append(dict(dataset=path, size=t.shape[-1], index=index, epsilon=epsilon,
                    collision_min=lower, collision_max=upper, collision_condition=upper/lower,
                    optimal_constant_shift=(lower*upper)**.5,
                    hss_bound_one_stage=(ratio-1)/(ratio+1), hss_bound_eight_stages=((ratio-1)/(ratio+1))**8,
                    scope='analytic ideal-HSS bound, not observed solver iterations or learned OOD performance'))
    write_csv(out/'bounds.csv', rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
