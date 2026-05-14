import argparse
import json
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from helmholtz.local_revisions.compare_cbs_shifted_laplacian import make_moderate_contrast_velocity, synchronize
from models.cbs import ConvergentBornSeries2d
from models.gmres import fgmres
from models.shifted_laplacian import (
    FiniteDifferenceShiftedLaplacianVcycle2d,
    MatrixFreeShiftedLaplacian2d,
    SparseShiftedLaplacian2d,
)


def elapsed(device, fn):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    return result, time.perf_counter() - start


def run_fgmres(name, operator, rhs, preconditioner, args, device, setup_time=0.0):
    def solve():
        return fgmres(
            A=operator,
            b=rhs,
            M=preconditioner,
            tol=args.tol,
            maxiter=args.krylov_maxiter,
            restart=args.restart,
            max_restarts=args.max_restarts,
            verbose=False,
            log_step=50,
        )

    (solution, history), solve_time = elapsed(device, solve)
    return {
        "name": name,
        "solution": solution,
        "history": history,
        "iterations": len(history) - 1,
        "final_relres": history[-1],
        "setup_time": float(setup_time),
        "solve_time": solve_time,
        "wall_time": float(setup_time) + solve_time,
        "converged": history[-1] < args.tol,
    }


def retune_cbs_preconditioner(cbs, k0_scale=1.0, epsilon_scale=1.0, gamma_scale=1.0):
    k0 = cbs.k0 * float(k0_scale)
    k2 = cbs.inv_sos2 * (cbs.omega0**2)
    epsilon = torch.amax(torch.abs(k2 - k0**2), dim=[-1, -2]).unsqueeze(-1).unsqueeze(-1)
    epsilon = epsilon * float(epsilon_scale)
    epsilon = torch.clamp(epsilon.real, min=torch.finfo(cbs.float_dtype).eps)
    cbs.k0 = k0
    cbs.epsilon = epsilon
    cbs.V = k2 - k0**2 - 1.0j * epsilon
    cbs.g0_tilde = 1.0 / (cbs.p_dist2 - k0**2 - 1.0j * epsilon)
    cbs.gamma = float(gamma_scale) * 1.0j / epsilon * cbs.V
    return cbs


def cbs_kernel_preconditioner(cbs):
    def apply(r):
        return (cbs.gamma * cbs.G(r / (cbs.dx**2), cbs.g0_tilde)).to(cbs.complex_dtype)

    return apply


def cbs_inner_krylov_preconditioner(operator, cbs, args):
    inner_m = cbs_kernel_preconditioner(cbs)

    def apply(r):
        z, _ = fgmres(
            A=operator,
            b=r,
            M=inner_m,
            tol=args.cbs_inner_tol,
            maxiter=args.cbs_inner_maxiter,
            restart=args.cbs_inner_restart,
            max_restarts=args.cbs_inner_max_restarts,
            verbose=False,
            log_step=50,
        )
        return z

    return apply


def crop_physical(u, boundary_width):
    if boundary_width <= 0:
        return u
    return u[..., boundary_width:-boundary_width, boundary_width:-boundary_width]


def selected_solution_results(results, max_count=5):
    preferred = [r for r in results if "identity" not in r["name"].lower()]
    if len(preferred) < max_count:
        preferred_ids = {id(r) for r in preferred}
        preferred.extend(r for r in results if id(r) not in preferred_ids)
    return preferred[:max_count]


def plot(kappa, results, args, out_path):
    solution_results = selected_solution_results(results)
    real_solutions = [
        crop_physical(r["solution"], args.boundary_width).real.squeeze().detach().cpu()
        for r in solution_results
    ]
    vmax = max(torch.quantile(torch.abs(u), 0.995).item() for u in real_solutions)
    vmax = max(vmax, 1e-12)

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes = axes.reshape(-1)
    for ax in axes:
        ax.set_visible(False)

    axes[0].set_visible(True)
    im = axes[0].imshow(kappa.squeeze().detach().cpu(), cmap="viridis")
    axes[0].set_title(r"Squared slowness $\kappa=1/c^2$")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for ax, result in zip(axes[1:-2], solution_results):
        ax.set_visible(True)
        sol = crop_physical(result["solution"], args.boundary_width).real.squeeze().detach().cpu()
        im = ax.imshow(sol, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Re solution: {result['name']}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[-2]
    ax.set_visible(True)
    for result in results:
        ax.semilogy(result["history"], label=result["name"])
    ax.axhline(args.tol, color="k", linestyle="--", linewidth=1, label=f"tol={args.tol:g}")
    ax.set_xlabel("Outer FGMRES iteration")
    ax.set_ylabel(r"$\|b-A_hu\|_2/\|b\|_2$")
    ax.set_title("FD Helmholtz Convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[-1]
    ax.set_visible(True)
    x = list(range(len(results)))
    setup_times = [r["setup_time"] for r in results]
    solve_times = [r["solve_time"] for r in results]
    setup_bars = ax.bar(x, setup_times, label="setup", color="#9ecae1")
    solve_bars = ax.bar(x, solve_times, bottom=setup_times, label="solve", color="#3182bd")
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in results], rotation=25, ha="right")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Setup + Solve Time")
    ax.legend(fontsize=8)
    for setup_bar, solve_bar, result in zip(setup_bars, solve_bars, results):
        total = result["wall_time"]
        top = setup_bar.get_height() + solve_bar.get_height()
        ax.text(
            solve_bar.get_x() + solve_bar.get_width() / 2,
            top,
            f"{result['iterations']} it\n{total:.2f}s",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.suptitle(
        f"Finite-difference Helmholtz, L={args.domain_length:g}, ppw={args.ppw:g}, freq={args.freq:.6g}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_vcycle_solver(args, velocity, kappa, src, dx, fdtype, device):
    solver = FiniteDifferenceShiftedLaplacianVcycle2d(
        freq=args.freq,
        N=(args.n, args.n),
        dx=dx,
        boundary_width=(args.boundary_width, args.boundary_width),
        boundary_type=args.boundary_type,
        shift=args.vcycle_shift,
        num_levels=args.num_levels,
        min_coarse_size=args.min_coarse_size,
        pre_smooth=args.pre_smooth,
        post_smooth=args.post_smooth,
        coarse_smooth=args.coarse_smooth,
        direct_coarse=True,
        smoother=args.smoother,
        vanka_stride=args.vanka_stride,
        vanka_weight=args.vanka_weight,
        jacobi_weight=args.jacobi_weight,
        num_cycles=args.num_cycles,
        fdtype=fdtype,
        device=str(device),
    )
    solver.setup_slowness(kappa, src=src, squared=True)
    return solver


def build_matrix_free_solver(args, kappa, src, dx, fdtype, device):
    solver = MatrixFreeShiftedLaplacian2d(
        freq=args.freq,
        N=(args.n, args.n),
        dx=dx,
        boundary_width=(args.boundary_width, args.boundary_width),
        boundary_type=args.boundary_type,
        shift=args.mf_shift,
        num_levels=args.num_levels,
        min_coarse_size=args.min_coarse_size,
        pre_smooth=args.pre_smooth,
        post_smooth=args.post_smooth,
        coarse_smooth=args.coarse_smooth,
        direct_coarse=False,
        smoother=args.smoother,
        vanka_stride=args.vanka_stride,
        vanka_weight=args.vanka_weight,
        jacobi_weight=args.jacobi_weight,
        num_cycles=args.num_cycles,
        inner_maxiter=args.mf_inner_maxiter,
        inner_preconditioner=args.mf_inner_preconditioner,
        inner_jacobi_sweeps=args.mf_inner_jacobi_sweeps,
        inner_vcycles=args.mf_inner_vcycles,
        fdtype=fdtype,
        device=str(device),
    )
    solver.setup_slowness(kappa, src=src, squared=True)
    return solver


def build_sparse_solver(args, kappa, src, dx, fdtype, device):
    solver = SparseShiftedLaplacian2d(
        freq=args.freq,
        N=(args.n, args.n),
        dx=dx,
        boundary_width=(args.boundary_width, args.boundary_width),
        boundary_type=args.boundary_type,
        shift=args.sparse_shift,
        fdtype=fdtype,
        device=str(device),
    )
    solver.setup_slowness(kappa, src=src, squared=True)
    return solver


def build_cbs_preconditioner(args, velocity, src, dx, fdtype, device):
    solver = ConvergentBornSeries2d(
        freq=args.freq,
        N=(args.n, args.n),
        dx=dx,
        boundary_width=(args.boundary_width, args.boundary_width),
        boundary_type=args.boundary_type,
        fdtype=fdtype,
        device=str(device),
    )
    solver.setup(velocity.unsqueeze(0).unsqueeze(0), src=src)
    return retune_cbs_preconditioner(
        solver,
        k0_scale=args.cbs_k0_scale,
        epsilon_scale=args.cbs_epsilon_scale,
        gamma_scale=args.cbs_gamma_scale,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare preconditioners on a matching finite-difference Helmholtz operator."
    )
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--domain-length", type=float, default=1.0)
    parser.add_argument("--boundary-width", type=int, default=16)
    parser.add_argument("--boundary-type", type=str, default="pbl", choices=["pbl", "abc"])
    parser.add_argument("--ppw", type=float, default=12.0)
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--fdtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--krylov-maxiter", type=int, default=400)
    parser.add_argument("--restart", type=int, default=100)
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument("--cbs-k0-scale", type=float, default=1.0)
    parser.add_argument("--cbs-epsilon-scale", type=float, default=1.0)
    parser.add_argument("--cbs-gamma-scale", type=float, default=1.0)
    parser.add_argument("--skip-cbs", action="store_true")
    parser.add_argument("--include-cbs-inner", action="store_true")
    parser.add_argument("--cbs-inner-maxiter", type=int, default=4)
    parser.add_argument("--cbs-inner-restart", type=int, default=None)
    parser.add_argument("--cbs-inner-max-restarts", type=int, default=0)
    parser.add_argument("--cbs-inner-tol", type=float, default=0.0)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--vcycle-shift", type=float, default=None)
    parser.add_argument("--mf-shift", type=float, default=None)
    parser.add_argument("--sparse-shift", type=float, default=None)
    parser.add_argument("--num-levels", type=int, default=5)
    parser.add_argument("--min-coarse-size", type=int, default=7)
    parser.add_argument("--pre-smooth", type=int, default=4)
    parser.add_argument("--post-smooth", type=int, default=4)
    parser.add_argument("--coarse-smooth", type=int, default=100)
    parser.add_argument("--smoother", type=str, default="jacobi", choices=["jacobi", "vanka"])
    parser.add_argument("--vanka-stride", type=int, default=2)
    parser.add_argument("--vanka-weight", type=float, default=0.8)
    parser.add_argument("--jacobi-weight", type=float, default=0.6)
    parser.add_argument("--num-cycles", type=int, default=3)
    parser.add_argument("--mf-inner-maxiter", type=int, default=8)
    parser.add_argument(
        "--mf-inner-preconditioner",
        type=str,
        default="jacobi",
        choices=["identity", "jacobi", "vcycle"],
    )
    parser.add_argument("--mf-inner-jacobi-sweeps", type=int, default=1)
    parser.add_argument("--mf-inner-vcycles", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="./outputs/fd_sl_benchmark")
    parser.add_argument("--include-identity", action="store_true")
    parser.add_argument("--skip-vcycle", action="store_true")
    parser.add_argument("--skip-matrix-free", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    parser.add_argument("--include-fd-lu-reference", action="store_true")
    args = parser.parse_args()
    args.vcycle_shift = args.shift if args.vcycle_shift is None else args.vcycle_shift
    args.mf_shift = args.shift if args.mf_shift is None else args.mf_shift
    args.sparse_shift = args.shift if args.sparse_shift is None else args.sparse_shift

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    fdtype = torch.float64 if args.fdtype == "float64" else torch.float32
    dx = args.domain_length / args.n

    velocity = make_moderate_contrast_velocity(args.n, dtype=fdtype, device=device)
    kappa = (1.0 / velocity.square()).unsqueeze(0).unsqueeze(0)
    if args.freq is None:
        args.freq = velocity.mean().item() / (args.ppw * dx)

    src = torch.zeros(1, 1, args.n, args.n, dtype=fdtype, device=device)
    src[..., args.n // 4, args.n // 2] = 1.0

    fd_solver, fd_setup_time = elapsed(
        device,
        lambda: build_vcycle_solver(args, velocity, kappa, src, dx, fdtype, device),
    )
    rhs = fd_solver.b.to(fd_solver.complex_dtype)
    operator = lambda u: fd_solver.A_op(u)

    cbs_precond = None
    cbs_setup_time = 0.0
    if not args.skip_cbs:
        cbs_precond, cbs_setup_time = elapsed(
            device,
            lambda: build_cbs_preconditioner(args, velocity, src, dx, fdtype, device),
        )

    matrix_free_solver = None
    matrix_free_setup_time = 0.0
    if not args.skip_matrix_free:
        matrix_free_solver, matrix_free_setup_time = elapsed(
            device,
            lambda: build_matrix_free_solver(args, kappa, src, dx, fdtype, device),
        )

    sparse_solver = None
    sparse_setup_time = 0.0
    sparse_error = None
    if not args.skip_sparse:
        try:
            sparse_solver, sparse_setup_time = elapsed(
                device,
                lambda: build_sparse_solver(args, kappa, src, dx, fdtype, device),
            )
        except Exception as exc:
            sparse_error = str(exc)

    fd_lu_solver = None
    fd_lu_setup_time = 0.0
    fd_lu_error = None
    if args.include_fd_lu_reference:
        try:
            fd_lu_solver, fd_lu_setup_time = elapsed(
                device,
                lambda: SparseShiftedLaplacian2d(
                    freq=args.freq,
                    N=(args.n, args.n),
                    dx=dx,
                    boundary_width=(args.boundary_width, args.boundary_width),
                    boundary_type=args.boundary_type,
                    shift=0.0,
                    fdtype=fdtype,
                    device=str(device),
                ),
            )
            _, fd_lu_setup_call_time = elapsed(
                device,
                lambda: fd_lu_solver.setup_slowness(kappa, src=src, squared=True),
            )
            fd_lu_setup_time += fd_lu_setup_call_time
        except Exception as exc:
            fd_lu_error = str(exc)

    results = []
    if args.include_identity:
        results.append(
            run_fgmres(
                "FGMRES + identity",
                operator,
                rhs,
                lambda r: r,
                args,
                device,
                setup_time=0.0,
            )
        )

    if cbs_precond is not None:
        results.append(
            run_fgmres(
                "FGMRES + CBS tuned",
                operator,
                rhs,
                cbs_kernel_preconditioner(cbs_precond),
                args,
                device,
                setup_time=cbs_setup_time,
            )
        )

    if args.include_cbs_inner and cbs_precond is not None:
        results.append(
            run_fgmres(
                f"FGMRES + CBS inner({args.cbs_inner_maxiter})",
                operator,
                rhs,
                cbs_inner_krylov_preconditioner(operator, cbs_precond, args),
                args,
                device,
                setup_time=cbs_setup_time,
            )
        )

    if not args.skip_vcycle:
        results.append(
            run_fgmres(
                "FGMRES + FD-SL V-cycle",
                operator,
                rhs,
                lambda r: fd_solver.apply_preconditioner(r),
                args,
                device,
                setup_time=fd_setup_time,
            )
        )

    if matrix_free_solver is not None:
        results.append(
            run_fgmres(
                "FGMRES + FD-SL MF",
                operator,
                rhs,
                lambda r: matrix_free_solver.apply_preconditioner(r),
                args,
                device,
                setup_time=matrix_free_setup_time,
            )
        )

    if sparse_solver is not None:
        results.append(
            run_fgmres(
                "FGMRES + FD-SL sparse LU",
                operator,
                rhs,
                lambda r: sparse_solver.apply_preconditioner(r),
                args,
                device,
                setup_time=sparse_setup_time,
            )
        )

    if fd_lu_solver is not None:
        results.append(
            run_fgmres(
                "FGMRES + FD Helmholtz LU",
                operator,
                rhs,
                lambda r: fd_lu_solver.apply_preconditioner(r),
                args,
                device,
                setup_time=fd_lu_setup_time,
            )
        )

    plot_path = os.path.join(args.out_dir, "fd_cbs_vs_sl.png")
    plot(kappa, results, args, plot_path)

    summary = {
        "operator": "finite_difference_5_point_zero_exterior",
        "n": args.n,
        "domain_length": args.domain_length,
        "boundary_width": args.boundary_width,
        "boundary_type": args.boundary_type,
        "dx": dx,
        "ppw": args.ppw,
        "freq": args.freq,
        "velocity_range": [velocity.min().item(), velocity.max().item()],
        "kappa_range": [kappa.min().item(), kappa.max().item()],
        "kappa_contrast": (kappa.max() / kappa.min()).item(),
        "sl_levels": fd_solver.level_shapes,
        "sl_params": {
            "default_shift": args.shift,
            "vcycle_shift": args.vcycle_shift,
            "mf_shift": args.mf_shift,
            "sparse_shift": args.sparse_shift,
            "smoother": args.smoother,
            "vanka_stride": args.vanka_stride,
            "vanka_weight": args.vanka_weight,
            "jacobi_weight": args.jacobi_weight,
            "pre_smooth": args.pre_smooth,
            "post_smooth": args.post_smooth,
            "num_cycles": args.num_cycles,
            "mf_inner_maxiter": args.mf_inner_maxiter,
            "mf_inner_preconditioner": args.mf_inner_preconditioner,
            "mf_inner_jacobi_sweeps": args.mf_inner_jacobi_sweeps,
            "mf_inner_vcycles": args.mf_inner_vcycles,
        },
        "cbs_params": {
            "k0_scale": args.cbs_k0_scale,
            "epsilon_scale": args.cbs_epsilon_scale,
            "gamma_scale": args.cbs_gamma_scale,
            "include_cbs_inner": args.include_cbs_inner,
            "skip_cbs": args.skip_cbs,
            "cbs_inner_maxiter": args.cbs_inner_maxiter,
            "cbs_inner_restart": args.cbs_inner_restart,
            "cbs_inner_max_restarts": args.cbs_inner_max_restarts,
            "cbs_inner_tol": args.cbs_inner_tol,
        },
        "sparse_error": sparse_error,
        "fd_lu_error": fd_lu_error,
        "methods": [
            {
                "name": result["name"],
                "iterations": result["iterations"],
                "final_relres": result["final_relres"],
                "setup_time": result["setup_time"],
                "solve_time": result["solve_time"],
                "wall_time": result["wall_time"],
                "converged": result["converged"],
            }
            for result in results
        ],
        "plot": plot_path,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Finite-difference Helmholtz benchmark")
    print(f"  domain: [0, {args.domain_length:g}]^2, n={args.n}, dx={dx:.6g}")
    print(f"  freq: {args.freq:.6g} from ppw={args.ppw:g}")
    print(f"  kappa contrast: {summary['kappa_contrast']:.3f}")
    print(f"  SL levels: {fd_solver.level_shapes}")
    print(f"  SL params: {summary['sl_params']}")
    print(f"  CBS params: {summary['cbs_params']}")
    if sparse_error is not None:
        print(f"  sparse SL skipped: {sparse_error}")
    if fd_lu_error is not None:
        print(f"  FD Helmholtz LU skipped: {fd_lu_error}")
    for result in results:
        status = "PASS" if result["converged"] else "NO-CONV"
        print(
            f"  {result['name']:<26} {status:<7} "
            f"iters={result['iterations']:4d} "
            f"relres={result['final_relres']:.3e} "
            f"setup={result['setup_time']:.4f}s "
            f"solve={result['solve_time']:.4f}s "
            f"total={result['wall_time']:.4f}s"
        )
    print(f"  plot: {plot_path}")
    print(f"  summary: {summary_path}")

    sl_results = [r for r in results if "FD-SL" in r["name"]]
    if sl_results and not any(result["converged"] for result in sl_results):
        raise SystemExit("No finite-difference shifted-Laplacian preconditioner converged")


if __name__ == "__main__":
    main()
