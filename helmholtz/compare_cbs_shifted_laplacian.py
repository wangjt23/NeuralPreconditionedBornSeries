import argparse
import json
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.cbs import ConvergentBornSeries2d
from models.gmres import fgmres
from models.shifted_laplacian import ShiftedLaplacianVcycle2d


def make_moderate_contrast_velocity(n, dtype, device):
    x = torch.linspace(0.0, 1.0, n, dtype=dtype, device=device)
    y = torch.linspace(0.0, 1.0, n, dtype=dtype, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    return (
        1.0
        + 0.08 * torch.sin(2.0 * torch.pi * grid_x) * torch.cos(2.0 * torch.pi * grid_y)
        - 0.08 * torch.exp(-((grid_x - 0.35) ** 2 + (grid_y - 0.55) ** 2) / 0.025)
        + 0.07 * torch.exp(-((grid_x - 0.72) ** 2 + (grid_y - 0.38) ** 2) / 0.018)
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_solve(device, fn):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    return result, time.perf_counter() - start


def run_cbs_stationary(model, rhs, tol, maxiter, device):
    rhs_norm = torch.clamp(torch.norm(rhs), min=1e-30)

    def solve():
        u = torch.zeros_like(rhs)
        history = [1.0]
        for _ in range(maxiter):
            u = model(u)
            relres = (torch.norm(rhs - model.A_op(u)) / rhs_norm).item()
            history.append(relres)
            if relres < tol:
                break
        return u, history

    (u, history), wall_time = elapsed_solve(device, solve)
    return {
        "name": "CBS stationary",
        "solution": u,
        "history": history,
        "iterations": len(history) - 1,
        "final_relres": history[-1],
        "wall_time": wall_time,
        "converged": history[-1] < tol,
    }


def run_fgmres(name, operator, rhs, preconditioner, tol, maxiter, restart, max_restarts, device):
    def solve():
        return fgmres(
            A=operator,
            b=rhs,
            M=preconditioner,
            tol=tol,
            maxiter=maxiter,
            restart=restart,
            max_restarts=max_restarts,
            verbose=False,
            log_step=50,
        )

    (u, history), wall_time = elapsed_solve(device, solve)
    return {
        "name": name,
        "solution": u,
        "history": history,
        "iterations": len(history) - 1,
        "final_relres": history[-1],
        "wall_time": wall_time,
        "converged": history[-1] < tol,
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


def plot_results(kappa, results, args, out_path):
    histories = {r["name"]: r["history"] for r in results}
    solutions = {r["name"]: crop_physical(r["solution"], args.boundary_width) for r in results}

    real_solutions = [u.real.squeeze().detach().cpu() for u in solutions.values()]
    vmax = max(torch.quantile(torch.abs(u), 0.995).item() for u in real_solutions)
    vmax = max(vmax, 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = axes[0, 0]
    im = ax.imshow(kappa.squeeze().detach().cpu(), cmap="viridis")
    ax.set_title(r"Squared slowness $\kappa = 1/c^2$")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax, result in zip(axes[0, 1:].flat, results[:2]):
        im = ax.imshow(solutions[result["name"]].real.squeeze().detach().cpu(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Re solution: {result['name']}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    if len(results) > 2:
        result = results[2]
        im = ax.imshow(solutions[result["name"]].real.squeeze().detach().cpu(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Re solution: {result['name']}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    for name, history in histories.items():
        ax.semilogy(range(len(history)), history, label=name)
    ax.axhline(args.tol, color="k", linestyle="--", linewidth=1.0, label=f"tol={args.tol:g}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|b - Au\|_2 / \|b\|_2$")
    ax.set_title("Convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    names = [r["name"] for r in results]
    times = [r["wall_time"] for r in results]
    bars = ax.bar(range(len(results)), times, color=["#4c78a8", "#f58518", "#54a24b"][: len(results)])
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Solve wall time")
    for bar, result in zip(bars, results):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{result['iterations']} it",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(
        f"CBS FFT Helmholtz, L={args.domain_length:g}, ppw={args.ppw:g}, freq={args.freq:.6g}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare CBS against shifted-Laplacian V-cycle FGMRES on a generated slowness field."
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
    parser.add_argument("--cbs-maxiter", type=int, default=4000)
    parser.add_argument("--skip-cbs-stationary", action="store_true")
    parser.add_argument("--krylov-maxiter", type=int, default=400)
    parser.add_argument("--restart", type=int, default=100)
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument("--cbs-k0-scale", type=float, default=1.0)
    parser.add_argument("--cbs-epsilon-scale", type=float, default=1.0)
    parser.add_argument("--cbs-gamma-scale", type=float, default=1.0)
    parser.add_argument("--include-cbs-inner", action="store_true")
    parser.add_argument("--cbs-inner-maxiter", type=int, default=4)
    parser.add_argument("--cbs-inner-restart", type=int, default=None)
    parser.add_argument("--cbs-inner-max-restarts", type=int, default=0)
    parser.add_argument("--cbs-inner-tol", type=float, default=0.0)
    parser.add_argument("--shift", type=float, default=1.5)
    parser.add_argument("--sl-discretization", type=str, default="fft", choices=["fft", "fd"])
    parser.add_argument("--num-levels", type=int, default=5)
    parser.add_argument("--min-coarse-size", type=int, default=7)
    parser.add_argument("--pre-smooth", type=int, default=6)
    parser.add_argument("--post-smooth", type=int, default=6)
    parser.add_argument("--coarse-smooth", type=int, default=100)
    parser.add_argument("--smoother", type=str, default="jacobi", choices=["jacobi", "vanka"])
    parser.add_argument("--vanka-patch-size", type=int, default=2)
    parser.add_argument("--vanka-stride", type=int, default=1)
    parser.add_argument("--vanka-weight", type=float, default=0.7)
    parser.add_argument("--jacobi-weight", type=float, default=0.6)
    parser.add_argument("--num-cycles", type=int, default=3)
    parser.add_argument("--skip-sl", action="store_true")
    parser.add_argument("--out-dir", type=str, default="./outputs/sl_vcycle_benchmark")
    args = parser.parse_args()

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

    cbs = ConvergentBornSeries2d(
        freq=args.freq,
        N=(args.n, args.n),
        dx=dx,
        boundary_width=(args.boundary_width, args.boundary_width),
        boundary_type=args.boundary_type,
        fdtype=fdtype,
        device=str(device),
    )
    cbs.setup(velocity.unsqueeze(0).unsqueeze(0), src=src)
    retune_cbs_preconditioner(
        cbs,
        k0_scale=args.cbs_k0_scale,
        epsilon_scale=args.cbs_epsilon_scale,
        gamma_scale=args.cbs_gamma_scale,
    )
    rhs = cbs.b.to(cbs.complex_dtype)

    sl = ShiftedLaplacianVcycle2d(
        freq=args.freq,
        N=(args.n, args.n),
        dx=dx,
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
    sl.setup_slowness(kappa, src=src, squared=True)

    results = []
    if not args.skip_cbs_stationary:
        results.append(run_cbs_stationary(cbs, rhs, args.tol, args.cbs_maxiter, device))

    operator = lambda u: cbs.A_op(u)
    results.append(
        run_fgmres(
            name="FGMRES + CBS",
            operator=operator,
            rhs=rhs,
            preconditioner=cbs_kernel_preconditioner(cbs),
            tol=args.tol,
            maxiter=args.krylov_maxiter,
            restart=args.restart,
            max_restarts=args.max_restarts,
            device=device,
        )
    )

    if args.include_cbs_inner:
        results.append(
            run_fgmres(
                name=f"FGMRES + CBS inner({args.cbs_inner_maxiter})",
                operator=operator,
                rhs=rhs,
                preconditioner=cbs_inner_krylov_preconditioner(operator, cbs, args),
                tol=args.tol,
                maxiter=args.krylov_maxiter,
                restart=args.restart,
                max_restarts=args.max_restarts,
                device=device,
            )
        )

    if not args.skip_sl:
        results.append(
            run_fgmres(
                name="FGMRES + SL V-cycle",
                operator=operator,
                rhs=rhs,
                preconditioner=lambda r: sl.apply_preconditioner(r),
                tol=args.tol,
                maxiter=args.krylov_maxiter,
                restart=args.restart,
                max_restarts=args.max_restarts,
                device=device,
            )
        )

    plot_path = os.path.join(args.out_dir, "cbs_vs_sl_vcycle.png")
    plot_results(kappa, results, args, plot_path)

    summary = {
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
        "sl_levels": sl.level_shapes,
        "sl_discretization": args.sl_discretization,
        "sl_smoother": args.smoother,
        "cbs_params": {
            "k0_scale": args.cbs_k0_scale,
            "epsilon_scale": args.cbs_epsilon_scale,
            "gamma_scale": args.cbs_gamma_scale,
            "include_cbs_inner": args.include_cbs_inner,
            "cbs_inner_maxiter": args.cbs_inner_maxiter,
            "cbs_inner_restart": args.cbs_inner_restart,
            "cbs_inner_max_restarts": args.cbs_inner_max_restarts,
            "cbs_inner_tol": args.cbs_inner_tol,
        },
        "methods": [
            {
                "name": result["name"],
                "iterations": result["iterations"],
                "final_relres": result["final_relres"],
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

    print("CBS vs shifted-Laplacian benchmark")
    print(f"  domain: [0, {args.domain_length:g}]^2, n={args.n}, dx={dx:.6g}")
    print(f"  freq: {args.freq:.6g} from ppw={args.ppw:g}")
    print(f"  kappa contrast: {summary['kappa_contrast']:.3f}")
    print(f"  SL levels: {sl.level_shapes}")
    print(f"  SL discretization: {args.sl_discretization}")
    print(f"  SL smoother: {args.smoother}")
    print(f"  CBS params: {summary['cbs_params']}")
    for result in results:
        status = "PASS" if result["converged"] else "NO-CONV"
        print(
            f"  {result['name']:<22} {status:<7} "
            f"iters={result['iterations']:4d} "
            f"relres={result['final_relres']:.3e} "
            f"time={result['wall_time']:.4f}s"
        )
    print(f"  plot: {plot_path}")
    print(f"  summary: {summary_path}")

    if (not args.skip_sl) and not results[-1]["converged"]:
        raise SystemExit("shifted-Laplacian V-cycle FGMRES did not converge")


if __name__ == "__main__":
    main()
