import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rte.models import PeriodicRTE2D


sys.path.append(str(ROOT / "helmholtz" / "models"))
from gmres import fgmres  # noqa: E402


METHODS = (
    ("transport", "Born", None),
    ("transport_dsa", "DSA", None),
    ("transport_tsa", "TSA-H2", 2),
    ("transport_tsa_moment", "TSA-M2", 2),
    ("transport_tsa_moment", "TSA-M4", 4),
    ("transport_tsa", "TSA-full", None),
)


def _grid(size, dtype):
    x = torch.arange(size, dtype=dtype) / size
    return torch.meshgrid(x, x, indexing="ij")


def _gaussian_source(size, n_angles, dtype, angular="isotropic"):
    x, y = _grid(size, dtype)
    source_xy = torch.exp(-100.0 * ((x - 0.32) ** 2 + (y - 0.47) ** 2))
    angles = torch.arange(n_angles, dtype=dtype) * (2.0 * math.pi / n_angles)
    if angular == "beam":
        angular_weight = torch.exp(2.0 * torch.cos(angles))
        angular_weight = angular_weight / angular_weight.mean()
    elif angular == "dipole":
        angular_weight = 1.0 + 0.8 * torch.cos(angles)
    else:
        angular_weight = torch.ones_like(angles)
    source = angular_weight.view(1, n_angles, 1, 1) * source_xy.view(1, 1, size, size)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return source.to(cdtype)


def make_case(name, size, n_angles, dtype, eta_scale, device="cpu"):
    x, y = _grid(size, dtype)
    if name == "diffusive":
        sigma_t = torch.full((size, size), 20.0, dtype=dtype)
        albedo = torch.full_like(sigma_t, 0.99)
        anisotropy_g = 0.0
        source = _gaussian_source(size, n_angles, dtype, angular="isotropic")
        description = "homogeneous optically thick high-albedo isotropic medium"
    elif name == "heterogeneous":
        inclusion = torch.exp(-70.0 * ((x - 0.58) ** 2 + (y - 0.50) ** 2))
        channel = (torch.sin(2.0 * math.pi * y) > 0.35).to(dtype)
        sigma_t = 1.0 + 9.0 * inclusion + 2.5 * channel
        albedo = 0.82 + 0.15 * torch.exp(-50.0 * ((x - 0.28) ** 2 + (y - 0.62) ** 2))
        anisotropy_g = 0.0
        source = _gaussian_source(size, n_angles, dtype, angular="isotropic")
        description = "high-contrast isotropic heterogeneous medium"
    elif name == "anisotropic":
        sigma_t = torch.full((size, size), 5.0, dtype=dtype)
        albedo = torch.full_like(sigma_t, 0.96)
        anisotropy_g = 0.85
        source = _gaussian_source(size, n_angles, dtype, angular="beam")
        description = "forward-peaked high-albedo medium with beamed source"
    elif name == "mixed":
        left = x < 0.48
        sigma_t = torch.where(left, torch.full_like(x, 0.35), torch.full_like(x, 16.0))
        sigma_t = sigma_t + 2.0 * torch.exp(-120.0 * ((x - 0.50) ** 2))
        albedo = torch.where(left, torch.full_like(x, 0.20), torch.full_like(x, 0.985))
        anisotropy_g = 0.65
        source = _gaussian_source(size, n_angles, dtype, angular="beam")
        description = "mixed thin streaming region coupled to thick scattering region"
    else:
        raise ValueError(f"unknown case {name}")

    sigma_s = torch.clamp(albedo * sigma_t, max=0.995 * sigma_t)
    rte = PeriodicRTE2D(
        shape=(size, size),
        dx=1.0 / size,
        n_angles=n_angles,
        anisotropy_g=anisotropy_g,
        eta_scale=eta_scale,
        fdtype=dtype,
        device=device,
    ).setup(
        sigma_t=sigma_t.unsqueeze(0).unsqueeze(0),
        sigma_s=sigma_s.unsqueeze(0).unsqueeze(0),
        source=source,
    )
    meta = {
        "case": name,
        "description": description,
        "albedo_min": float(albedo.min()),
        "albedo_max": float(albedo.max()),
        "sigma_t_min": float(sigma_t.min()),
        "sigma_t_max": float(sigma_t.max()),
        "anisotropy_g": anisotropy_g,
    }
    return rte, meta


def contraction(history):
    if len(history) <= 1:
        return 0.0
    if history[-1] <= 0.0 or history[0] <= 0.0:
        return float("nan")
    return float((history[-1] / history[0]) ** (1.0 / (len(history) - 1)))


def run_stationary(rte, method_kind, harmonics, relaxations, maxiter, tol):
    best = None
    for relaxation in relaxations:
        start = time.perf_counter()
        _, history = rte.preconditioned_richardson(
            maxiter=maxiter,
            tol=tol,
            relaxation=relaxation,
            preconditioner=method_kind,
            tsa_harmonics=harmonics,
        )
        elapsed = time.perf_counter() - start
        final = history[-1]
        result = {
            "relaxation": relaxation,
            "stationary_iters": len(history) - 1,
            "stationary_final": final,
            "stationary_factor": contraction(history),
            "stationary_time": elapsed,
        }
        if best is None:
            best = result
            continue
        best_converged = best["stationary_final"] < tol
        result_converged = result["stationary_final"] < tol
        if result_converged and not best_converged:
            best = result
        elif result_converged == best_converged:
            if result["stationary_iters"] < best["stationary_iters"]:
                best = result
            elif result["stationary_iters"] == best["stationary_iters"] and result["stationary_final"] < best["stationary_final"]:
                best = result
    return best


def run_fgmres(rte, method_kind, harmonics, maxiter, tol):
    def A(x):
        return rte.A_op(x)

    def M(r):
        return rte.apply_preconditioner(r, kind=method_kind, tsa_harmonics=harmonics)

    start = time.perf_counter()
    _, history = fgmres(
        A=A,
        b=rte.source,
        M=M,
        tol=tol,
        maxiter=maxiter,
        restart=min(maxiter, 30),
        max_restarts=2,
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    return {
        "fgmres_iters": len(history) - 1,
        "fgmres_final": history[-1],
        "fgmres_factor": contraction(history),
        "fgmres_time": elapsed,
    }


def format_float(value):
    if isinstance(value, str):
        return value
    return f"{value:.3e}"


def write_tex(rows, path):
    case_names = {
        "diffusive": "Diffusive",
        "heterogeneous": "Heterogeneous",
        "anisotropic": "Anisotropic",
        "mixed": "Mixed",
    }
    lines = [
        "% Auto-generated by rte/run_hard_case_experiments.py",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Case & Method & $\\omega$ & Stat. it. & Stat. final & Stat. fac. & FGMRES it. & FGMRES final\\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {:.1f} & {} & {:.1e} & {:.3f} & {} & {:.1e}\\\\".format(
                case_names[row["case"]],
                row["method_label"],
                row["relaxation"],
                row["stationary_iters"],
                row["stationary_final"],
                row["stationary_factor"],
                row["fgmres_iters"],
                row["fgmres_final"],
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run hard-regime RTE synthetic-acceleration experiments.")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--n-angles", type=int, default=24)
    parser.add_argument("--eta-scale", type=float, default=0.1)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--fgmres-maxiter", type=int, default=45)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--output-csv", default="rte/hard_case_results.csv")
    parser.add_argument("--output-tex", default="rte/hard_case_results_table.tex")
    args = parser.parse_args()

    torch.manual_seed(20260518)
    dtype = torch.float64
    relaxations = (0.2, 0.4, 0.6, 0.8, 1.0)
    rows = []
    cases = ("diffusive", "heterogeneous", "anisotropic", "mixed")

    for case in cases:
        rte, meta = make_case(case, args.size, args.n_angles, dtype, args.eta_scale)
        print(f"case={case}: {meta['description']}")
        for kind, label, harmonics in METHODS:
            stationary = run_stationary(rte, kind, harmonics, relaxations, args.maxiter, args.tol)
            krylov = run_fgmres(rte, kind, harmonics, args.fgmres_maxiter, args.tol)
            row = {**meta, "method_kind": kind, "method_label": label, "tsa_harmonics": harmonics}
            row.update(stationary)
            row.update(krylov)
            rows.append(row)
            print(
                f"  {label:8s} omega={row['relaxation']:.1f} "
                f"stat={row['stationary_iters']:3d}/{row['stationary_final']:.2e} "
                f"fgmres={row['fgmres_iters']:3d}/{row['fgmres_final']:.2e}"
            )

    csv_path = ROOT / args.output_csv
    tex_path = ROOT / args.output_tex
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_tex(rows, tex_path)
    print(f"wrote {csv_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
