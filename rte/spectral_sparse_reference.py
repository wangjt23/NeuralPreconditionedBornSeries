"""Independent sparse representation of the same periodic spectral RTE matrix.

Dense 1-D spectral derivative matrices still give a sparse tensor-product
operator in (angle,x,y) ordering. This is a CPU classical reference, not a finite
difference substitution. Factorization memory can grow substantially with size.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spilu, splu
import torch

from rte.run_hard_case_experiments import make_case
from rte.run_residual_subspace_experiment import measured_solver, sources, write_csv


def spectral_matrix(rte):
    if rte.sigma_t.shape[0] != 1:
        raise ValueError("sparse reference requires one medium")
    nx, ny = rte.shape
    # Use the actual stored wave numbers, including the even-grid Nyquist mode.
    kx = rte.kx.detach().cpu().numpy().reshape(nx)
    ky = rte.ky.detach().cpu().numpy().reshape(ny)
    dx = sparse.csc_matrix(np.fft.ifft(1j*kx[:, None]*np.fft.fft(np.eye(nx), axis=0), axis=0))
    dy = sparse.csc_matrix(np.fft.ifft(1j*ky[:, None]*np.fft.fft(np.eye(ny), axis=0), axis=0))
    directions = rte.directions.detach().cpu().numpy()
    st = sparse.diags(rte.sigma_t.detach().cpu().numpy().reshape(-1))
    ss = sparse.diags(rte.sigma_s.detach().cpu().numpy().reshape(-1))
    phase = (rte.phase_matrix * rte.weights[None]).detach().cpu().numpy()
    streaming = sparse.kron(sparse.diags(directions[:, 0]), sparse.kron(dx, sparse.eye(ny)))
    streaming += sparse.kron(sparse.diags(directions[:, 1]), sparse.kron(sparse.eye(nx), dy))
    matrix = streaming + sparse.kron(sparse.eye(rte.n_angles), st) - sparse.kron(phase, ss)
    matrix = matrix.astype(np.complex128 if rte.complex_dtype == torch.complex128 else np.complex64).tocsc()
    matrix.eliminate_zeros()
    return matrix


class SparseSpectralFactor:
    def __init__(self, rte, drop_tol=1e-3, fill_factor=4., exact=False):
        if rte.sigma_t.device.type != "cpu":
            raise ValueError("SciPy factor is a CPU reference")
        self.rte = rte
        start = time.perf_counter()
        self.matrix = spectral_matrix(rte)
        self.assembly_seconds = time.perf_counter() - start
        start = time.perf_counter()
        self.factor = splu(self.matrix) if exact else spilu(self.matrix, drop_tol=drop_tol, fill_factor=fill_factor)
        self.factor_seconds = time.perf_counter() - start
        self.factor_nnz = self.factor.L.nnz + self.factor.U.nnz
        self._coefficients = tuple((x.data_ptr(), x._version) for x in (rte.sigma_t, rte.sigma_s))

    def __call__(self, residual):
        if residual.device.type != "cpu" or residual.requires_grad:
            raise ValueError("sparse factor supports no-gradient CPU residuals only")
        if self._coefficients != tuple((x.data_ptr(), x._version) for x in (self.rte.sigma_t, self.rte.sigma_s)):
            raise RuntimeError("medium changed; refactor the sparse matrix")
        rhs = np.asarray(residual.flatten(1).T.numpy(), dtype=self.matrix.dtype)
        solution = self.factor.solve(rhs)
        return torch.from_numpy(solution.T.copy()).reshape_as(residual).to(residual.dtype)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="mixed")
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--n-angles", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--exact", action="store_true")
    p.add_argument("--solver-steps", type=int, default=30)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--output", default="rte/results/collision_transport_sparse_audit")
    args = p.parse_args()
    torch.set_num_threads(1)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    rte, _ = make_case(args.case, args.size, args.n_angles, torch.float32, .1)
    qs = sources(args.size, args.n_angles, 3, args.seed + 3000, torch.device("cpu"))
    settings = [(False, drop, fill) for drop in (.01, .001) for fill in (2., 4., 8.)]
    if args.exact:
        settings.append((True, 0., 0.))
    rows = []
    for exact, drop, fill in settings:
        try:
            method = SparseSpectralFactor(rte, drop, fill, exact)
        except Exception as error:
            rows.append({"exact": exact, "drop_tol": drop, "fill_factor": fill, "factor_failure": str(error)})
            write_csv(out / "results.csv", rows)
            continue
        for i, q in enumerate(qs):
            common = {"exact": exact, "drop_tol": drop, "fill_factor": fill, "source": i,
                      "assembly_seconds": method.assembly_seconds, "factor_seconds": method.factor_seconds,
                      "matrix_nnz": method.matrix.nnz, "factor_nnz": method.factor_nnz}
            try:
                measured_solver(rte, method, q, args)
                repeats = [measured_solver(rte, method, q, args) for _ in range(3)]
            except Exception as error:
                # Aggressive incomplete factors can produce nonfinite Krylov
                # systems.  They are failed controls, not a reason to discard
                # the remaining fill/drop settings (or the exact-LU control).
                rows.append({**common, "solve_failure": f"{type(error).__name__}: {error}"})
                print(exact, drop, fill, rows[-1], flush=True)
                write_csv(out / "results.csv", rows)
                continue
            row = repeats[0]
            row["seconds"] = float(np.median([r["seconds"] for r in repeats]))
            row["converged"] = math.isfinite(row["final"]) and row["final"] <= args.tol
            if not row["converged"]:
                row["solve_failure"] = "nonfinite_residual" if not math.isfinite(row["final"]) else "iteration_budget"
            rows.append({**common,
                         "first_solve_seconds": method.assembly_seconds + method.factor_seconds + row["seconds"], **row})
        print(exact, drop, fill, rows[-1], flush=True)
        write_csv(out / "results.csv", rows)


if __name__ == "__main__":
    main()
