import argparse

import torch

from models.gmres import fgmres
from models.shifted_laplacian import ShiftedLaplacianVcycle2d


def make_moderate_contrast_kappa(n, dtype, device):
    """Return squared slowness kappa = 1 / c^2 with smooth moderate contrast."""
    x = torch.linspace(0.0, 1.0, n, dtype=dtype, device=device)
    y = torch.linspace(0.0, 1.0, n, dtype=dtype, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    velocity = (
        1.0
        + 0.08 * torch.sin(2.0 * torch.pi * grid_x) * torch.cos(2.0 * torch.pi * grid_y)
        - 0.08 * torch.exp(-((grid_x - 0.35) ** 2 + (grid_y - 0.55) ** 2) / 0.025)
        + 0.07 * torch.exp(-((grid_x - 0.72) ** 2 + (grid_y - 0.38) ** 2) / 0.018)
    )
    return (1.0 / velocity.square()).unsqueeze(0).unsqueeze(0)


def solve_case(args):
    device = torch.device(args.device)
    fdtype = torch.float64 if args.fdtype == "float64" else torch.float32
    cdtype = torch.complex128 if fdtype == torch.float64 else torch.complex64

    kappa = make_moderate_contrast_kappa(args.n, dtype=fdtype, device=device)
    src = torch.zeros(1, 1, args.n, args.n, dtype=fdtype, device=device)
    src[..., args.n // 4, args.n // 2] = 1.0

    model = ShiftedLaplacianVcycle2d(
        freq=args.freq,
        N=(args.n, args.n),
        dx=1.0 / args.n,
        boundary_width=(args.boundary_width, args.boundary_width),
        boundary_type=args.boundary_type,
        shift=args.shift,
        precond_discretization=args.sl_discretization,
        num_levels=args.num_levels,
        min_coarse_size=args.min_coarse_size,
        pre_smooth=args.pre_smooth,
        post_smooth=args.post_smooth,
        coarse_smooth=args.coarse_smooth,
        direct_coarse=True,
        smoother=args.smoother,
        vanka_patch_size=args.vanka_patch_size,
        vanka_stride=args.vanka_stride,
        vanka_weight=args.vanka_weight,
        jacobi_weight=args.jacobi_weight,
        num_cycles=args.num_cycles,
        fdtype=fdtype,
        device=str(device),
    )
    model.setup_slowness(kappa, src=src, squared=True)
    rhs = model.b.to(cdtype)

    def helmholtz_operator(u):
        return model.A_op(u)

    def shifted_laplacian_preconditioner(r):
        return model.apply_preconditioner(r)

    solution, history = fgmres(
        A=helmholtz_operator,
        b=rhs,
        M=shifted_laplacian_preconditioner,
        tol=args.tol,
        maxiter=args.maxiter,
        restart=args.restart,
        max_restarts=args.max_restarts,
        verbose=args.verbose,
        log_step=args.log_step,
    )
    final_relres = (torch.norm(rhs - helmholtz_operator(solution)) / torch.norm(rhs)).item()

    identity_relres = None
    if args.compare_identity:
        _, identity_history = fgmres(
            A=helmholtz_operator,
            b=rhs,
            M=lambda r: r,
            tol=args.tol,
            maxiter=args.identity_maxiter,
            restart=args.restart,
            max_restarts=args.max_restarts,
            verbose=False,
            log_step=args.log_step,
        )
        identity_relres = identity_history[-1]

    return {
        "kappa_min": kappa.min().item(),
        "kappa_max": kappa.max().item(),
        "kappa_contrast": (kappa.max() / kappa.min()).item(),
        "levels": model.level_shapes,
        "iterations": len(history) - 1,
        "final_relres": final_relres,
        "history": history,
        "identity_relres": identity_relres,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify shifted-Laplacian V-cycle preconditioning on a generated slowness field."
    )
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--boundary-width", type=int, default=8)
    parser.add_argument("--freq", type=float, default=4.0)
    parser.add_argument("--boundary-type", type=str, default="pbl", choices=["pbl", "abc"])
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--sl-discretization", type=str, default="fft", choices=["fft", "fd"])
    parser.add_argument("--num-levels", type=int, default=5)
    parser.add_argument("--min-coarse-size", type=int, default=7)
    parser.add_argument("--pre-smooth", type=int, default=4)
    parser.add_argument("--post-smooth", type=int, default=4)
    parser.add_argument("--coarse-smooth", type=int, default=100)
    parser.add_argument("--smoother", type=str, default="jacobi", choices=["jacobi", "vanka"])
    parser.add_argument("--vanka-patch-size", type=int, default=2)
    parser.add_argument("--vanka-stride", type=int, default=1)
    parser.add_argument("--vanka-weight", type=float, default=0.7)
    parser.add_argument("--jacobi-weight", type=float, default=0.6)
    parser.add_argument("--num-cycles", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--restart", type=int, default=100)
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument("--identity-maxiter", type=int, default=200)
    parser.add_argument("--fdtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-step", type=int, default=25)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-compare-identity", dest="compare_identity", action="store_false")
    parser.set_defaults(compare_identity=True)
    args = parser.parse_args()

    result = solve_case(args)
    print("Shifted-Laplacian V-cycle verification")
    print(f"  squared slowness kappa range: [{result['kappa_min']:.6e}, {result['kappa_max']:.6e}]")
    print(f"  kappa contrast: {result['kappa_contrast']:.3f}")
    print(f"  multigrid levels: {result['levels']}")
    print(f"  FGMRES iterations: {result['iterations']}")
    print(f"  final true relative residual: {result['final_relres']:.6e}")
    if result["identity_relres"] is not None:
        print(
            f"  identity-preconditioned residual after {args.identity_maxiter} iterations: "
            f"{result['identity_relres']:.6e}"
        )

    if result["final_relres"] >= args.tol:
        raise SystemExit(
            f"verification failed: final relative residual {result['final_relres']:.3e} "
            f"did not reach tolerance {args.tol:.3e}"
        )
    print("  status: PASS")


if __name__ == "__main__":
    main()
