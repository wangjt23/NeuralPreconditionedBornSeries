import argparse

import torch

from models.cbs import ConvergentBornSeries2d
from models.gmres import fgmres


def relative_norm(x, y):
    denom = torch.clamp(torch.linalg.norm(y), min=1e-30)
    return (torch.linalg.norm(x - y) / denom).item()


def check_born_residual_identity(args):
    dtype = torch.float32 if args.float32 else torch.float64
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    torch.manual_seed(args.seed)

    model = ConvergentBornSeries2d(
        freq=args.freq,
        N=[args.size, args.size],
        dx=args.dx,
        boundary_width=[args.boundary_width, args.boundary_width],
        boundary_type=args.boundary_type,
        fdtype=dtype,
        device="cpu",
    )
    grid = torch.linspace(0.0, 1.0, args.size, dtype=dtype)
    x, y = torch.meshgrid(grid, grid, indexing="ij")
    sos = 3000.0 * (1.0 + 0.15 * torch.sin(2.0 * torch.pi * x) * torch.cos(2.0 * torch.pi * y))
    sos = sos.unsqueeze(0).unsqueeze(0)

    src = torch.zeros_like(sos)
    src[..., args.size // 3, args.size // 2] = 1.0
    model.setup(sos=sos, src=src)

    u = torch.randn_like(model.b, dtype=cdtype)
    lhs = model.G(model.V * u + model.b / (model.dx**2), model.g0_tilde) - u
    rhs = model.G((model.b - model.A_op(u)) / (model.dx**2), model.g0_tilde)
    err = relative_norm(lhs, rhs)
    print(f"CBS Born residual identity relative error: {err:.3e}")
    if err > args.tol:
        raise SystemExit(f"Born residual identity failed: {err:.3e} > {args.tol:.3e}")


def check_fgmres_right_preconditioner(args):
    dtype = torch.complex64 if args.float32 else torch.complex128
    torch.manual_seed(args.seed + 1)
    diagonal = torch.linspace(1.0, 3.0, 16, dtype=torch.float64).reshape(1, 1, 4, 4).to(dtype)
    b = torch.randn(1, 1, 4, 4, dtype=dtype)

    def A(x):
        return diagonal * x

    def M(r):
        return r / diagonal

    _, history = fgmres(A=A, b=b, M=M, tol=args.tol, maxiter=4, verbose=False)
    print(f"FGMRES exact-preconditioner final relative residual: {history[-1]:.3e}")
    if history[-1] > args.tol:
        raise SystemExit(f"FGMRES smoke check failed: {history[-1]:.3e} > {args.tol:.3e}")


def main():
    parser = argparse.ArgumentParser(description="Verify core Helmholtz/CBS operator identities.")
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--boundary-width", type=int, default=6)
    parser.add_argument("--boundary-type", default="pbl", choices=["pbl", "abc"])
    parser.add_argument("--freq", type=float, default=6.0e4)
    parser.add_argument("--dx", type=float, default=1.0 / 256.0)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--float32", action="store_true", help="Run checks in single precision.")
    args = parser.parse_args()

    check_born_residual_identity(args)
    check_fgmres_right_preconditioner(args)
    print("All operator identity checks passed.")


if __name__ == "__main__":
    main()
