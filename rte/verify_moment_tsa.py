import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rte.run_hard_case_experiments import make_case


def relerr(x, y):
    return float(torch.linalg.norm(x - y) / torch.clamp(torch.linalg.norm(y), min=1e-30))


def project_with_basis(rte, rhs, harmonics):
    basis = rte.angular_fourier_basis(harmonics, dtype=rhs.dtype, device=rhs.device)
    coeff = torch.einsum("nm,bnhw->bmhw", basis.conj(), rhs)
    return torch.einsum("nm,bmhw->bnhw", basis, coeff)


def moment_galerkin_residual(rte, rhs, harmonics):
    basis = rte.angular_fourier_basis(harmonics, dtype=rhs.dtype, device=rhs.device)
    n_modes = basis.shape[1]
    u = rte.reference_moment_transport_solve(rhs, harmonics=harmonics)
    u_hat = torch.fft.fftn(u, dim=(-2, -1), norm="backward")
    rhs_hat = torch.fft.fftn(rhs, dim=(-2, -1), norm="backward")
    u_vec = u_hat.permute(0, 2, 3, 1).unsqueeze(-1)
    rhs_vec = rhs_hat.permute(0, 2, 3, 1).unsqueeze(-1)
    matrix = rte._reference_transport_matrix().to(dtype=rhs.dtype, device=rhs.device)
    if matrix.shape[0] == 1 and rhs.shape[0] > 1:
        matrix = matrix.expand(rhs.shape[0], -1, -1, -1, -1)
    residual_vec = torch.matmul(matrix, u_vec) - rhs_vec
    low_residual = torch.matmul(
        basis.conj().transpose(0, 1).view(1, 1, 1, n_modes, rte.n_angles),
        residual_vec,
    )
    low_rhs = torch.matmul(
        basis.conj().transpose(0, 1).view(1, 1, 1, n_modes, rte.n_angles),
        rhs_vec,
    )
    return float(torch.linalg.norm(low_residual) / torch.clamp(torch.linalg.norm(low_rhs), min=1e-30))


def mode_energy(rte, residual):
    coeff = torch.fft.fft(residual, dim=1, norm="backward")
    energy = torch.sum(coeff.real.square() + coeff.imag.square(), dim=(0, 2, 3))
    energy = energy / torch.clamp(torch.sum(energy), min=1e-30)
    return energy.detach().cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Verify closed angular moment TSA implementation.")
    parser.add_argument("--case", default="anisotropic")
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--n-angles", type=int, default=12)
    parser.add_argument("--harmonics", type=int, default=2)
    parser.add_argument("--eta-scale", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=5e-10)
    args = parser.parse_args()

    torch.manual_seed(20260519)
    rte, _ = make_case(args.case, args.size, args.n_angles, torch.float64, args.eta_scale)
    rhs = torch.randn_like(rte.source) + 1j * torch.randn_like(rte.source)

    projected = project_with_basis(rte, rhs, args.harmonics)
    projection_error = relerr(projected, rte.angular_project(rhs, harmonics=args.harmonics))

    gal_res = moment_galerkin_residual(rte, rhs, args.harmonics)

    a = 0.7 - 0.2j
    b = -0.3 + 0.4j
    rhs2 = torch.randn_like(rhs) + 1j * torch.randn_like(rhs)
    lhs = rte.reference_moment_transport_solve(a * rhs + b * rhs2, harmonics=args.harmonics)
    rhs_lin = a * rte.reference_moment_transport_solve(rhs, harmonics=args.harmonics) + b * rte.reference_moment_transport_solve(
        rhs2,
        harmonics=args.harmonics,
    )
    linearity_error = relerr(lhs, rhs_lin)

    residual = rte.source - rte.A_op(torch.zeros_like(rte.source))
    dsa_residual = residual - rte.A_op(rte.apply_preconditioner(residual, kind="transport_dsa"))
    moment_residual = residual - rte.A_op(
        rte.apply_preconditioner(residual, kind="transport_tsa_moment", tsa_harmonics=args.harmonics)
    )

    print(f"projection_error={projection_error:.3e}")
    print(f"galerkin_residual={gal_res:.3e}")
    print(f"linearity_error={linearity_error:.3e}")
    print("source_mode_energy=" + ",".join(f"{v:.3e}" for v in mode_energy(rte, residual)))
    print("dsa_next_mode_energy=" + ",".join(f"{v:.3e}" for v in mode_energy(rte, dsa_residual)))
    print("moment_next_mode_energy=" + ",".join(f"{v:.3e}" for v in mode_energy(rte, moment_residual)))

    if projection_error > args.tol or gal_res > args.tol or linearity_error > args.tol:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
