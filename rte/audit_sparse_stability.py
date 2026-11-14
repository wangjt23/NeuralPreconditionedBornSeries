"""Probe ILU factor quality before attributing downstream least-squares errors."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.sparse.linalg import spilu

from rte.models import PeriodicRTE2D
from rte.run_hard_case_experiments import make_case
from rte.run_residual_subspace_experiment import measured_solver, sources, write_csv
from rte.spectral_sparse_reference import spectral_matrix


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    op32, _ = make_case('mixed', args.size, 12, torch.float32, .1)
    qs = sources(args.size, 12, 3, 3001, torch.device('cpu'))
    matrix32 = spectral_matrix(op32)
    rows = []
    # These extra controls are exploratory on the old fixture, not selected
    # using the ensemble's test split and not a GPU performance comparison.
    settings = [(np.complex64, .001, 4., 'COLAMD'), (np.complex128, .001, 4., 'COLAMD')]
    settings += [(np.complex128, 1e-4, fill, 'COLAMD') for fill in (8., 16., 32., 64.)]
    settings += [(np.complex128, 1e-4, 8., order) for order in ('MMD_AT_PLUS_A', 'NATURAL')]
    for dtype, drop, fill, order in settings:
        a = matrix32.astype(dtype)
        common = dict(size=args.size, factor_dtype=np.dtype(dtype).name, drop_tol=drop,
                      fill_factor=fill, ordering=order, matrix_nnz=a.nnz)
        start = time.perf_counter()
        try:
            factor = spilu(a, drop_tol=drop, fill_factor=fill, permc_spec=order)
        except Exception as error:
            rows.append({**common, 'factor_failure': str(error), 'factor_seconds': time.perf_counter()-start})
            write_csv(out/'results.csv', rows)
            continue
        common.update(factor_seconds=time.perf_counter()-start, factor_nnz=factor.L.nnz+factor.U.nnz,
                      min_abs_u_diagonal=float(np.abs(factor.U.diagonal()).min()),
                      max_abs_u=float(np.abs(factor.U.data).max()), max_abs_l=float(np.abs(factor.L.data).max()))
        fdtype = torch.float32 if dtype == np.complex64 else torch.float64
        op = PeriodicRTE2D(shape=op32.shape, dx=op32.dx, n_angles=12, anisotropy_g=op32.anisotropy_g, fdtype=fdtype)
        for name in ('directions','weights','phase_matrix','kx','ky'):
            setattr(op,name,getattr(op32,name).to(fdtype))
        op.setup(op32.sigma_t,op32.sigma_s,op32.source)

        def method(residual):
            result = factor.solve(residual.flatten(1).T.numpy().astype(dtype))
            if not np.isfinite(result).all():
                raise FloatingPointError('ILU triangular solve produced nonfinite entries')
            return torch.from_numpy(result.T.copy()).reshape_as(residual)

        for i, source in enumerate(qs):
            row = {**common, 'source': i}
            q = source.to(op.complex_dtype)
            try:
                z = method(q)
                row['first_action_max_abs'] = float(z.abs().max())
                row['first_action_norm_in_working_precision'] = float(z.norm())
                zn = z.flatten().numpy().astype(np.complex128)
                rhs = q.flatten().numpy().astype(np.complex128)
                row['first_action_relative_residual_float64'] = float(np.linalg.norm(a.astype(np.complex128) @ zn-rhs)/np.linalg.norm(rhs))
                if not torch.isfinite(z.norm()):
                    raise FloatingPointError('norm overflow despite finite ILU action entries')
                solve = measured_solver(op,method,q,SimpleNamespace(solver_steps=30,tol=1e-5))
                row.update(solve)
                row['converged'] = np.isfinite(solve['final']) and solve['final'] <= 1e-5
            except Exception as error:
                row.update(converged=False, solve_failure=f'{type(error).__name__}: {error}')
            rows.append(row)
        write_csv(out/'results.csv', rows)
        print({**common,'converged_sources':sum(bool(r.get('converged')) for r in rows[-3:])},flush=True)
    (out/'settings.json').write_text(json.dumps(dict(size=args.size, source_seed=3001,
        operator='same float32 pseudo-spectral matrix, promoted for float64 arm',
        fixture='make_case mixed; separate from CT ensemble test media',
        exploratory=True),indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size',type=int,default=16)
    p.add_argument('--output',required=True)
    run(p.parse_args())
