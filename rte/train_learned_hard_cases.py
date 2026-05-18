import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from rte.run_hard_case_experiments import contraction, make_case, run_fgmres, run_stationary


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "helmholtz" / "models"))
from gmres import fgmres  # noqa: E402


class ResidualPolynomialPreconditioner(nn.Module):
    """
    Per-sample learned defect network:

        h_0 = B r,
        e_0 = r - A h_0,
        h_j = B e_{j-1},
        e_j = e_{j-1} - A h_j,
        M_theta r = sum_j alpha_j h_j.

    If all alpha_j are one, this is a fixed number of inner Richardson
    corrections with base preconditioner B. Training alpha_j learns a residual
    polynomial for the hard sample.
    """

    def __init__(self, rte, base_kind="transport_dsa", degree=3, tsa_harmonics=None):
        super().__init__()
        self.rte = rte
        self.base_kind = base_kind
        self.degree = int(degree)
        self.tsa_harmonics = tsa_harmonics
        self.alpha_re = nn.Parameter(torch.ones(self.degree + 1, dtype=rte.float_dtype))
        self.alpha_im = nn.Parameter(torch.zeros(self.degree + 1, dtype=rte.float_dtype))

    @property
    def alpha(self):
        return torch.complex(self.alpha_re, self.alpha_im).to(dtype=self.rte.complex_dtype)

    def base(self, residual):
        return self.rte.apply_preconditioner(
            residual,
            kind=self.base_kind,
            tsa_harmonics=self.tsa_harmonics,
        )

    def basis(self, residual):
        basis = []
        defect = residual
        for _ in range(self.degree + 1):
            correction = self.base(defect)
            basis.append(correction)
            defect = defect - self.rte.A_op(correction)
        return basis

    def forward(self, residual):
        basis = self.basis(residual)
        alpha = self.alpha.to(device=residual.device)
        out = torch.zeros_like(residual)
        for j, correction in enumerate(basis):
            out = out + alpha[j] * correction
        return out


def smooth_random_like(reference, modes=6):
    noise = torch.randn_like(reference)
    coeff = torch.fft.fftn(noise, dim=(-2, -1), norm="backward")
    h, w = reference.shape[-2:]
    kx = torch.fft.fftfreq(h, device=reference.device).view(h, 1)
    ky = torch.fft.fftfreq(w, device=reference.device).view(1, w)
    mask = ((kx.abs() * h <= modes) & (ky.abs() * w <= modes)).to(reference.dtype)
    return torch.fft.ifftn(coeff * mask.view(1, 1, h, w), dim=(-2, -1), norm="backward")


def collect_training_residuals(rte, base_kind, tsa_harmonics, n_random=4, n_rollout=6):
    residuals = []
    psi = torch.zeros_like(rte.source)
    for _ in range(n_rollout):
        residual = rte.source - rte.A_op(psi)
        residuals.append(residual.detach())
        correction = rte.apply_preconditioner(residual, kind=base_kind, tsa_harmonics=tsa_harmonics)
        psi = psi + correction

    for _ in range(n_random):
        exact_correction = smooth_random_like(rte.source)
        residuals.append(rte.A_op(exact_correction).detach())

    return residuals


def metric_loss(model, residuals):
    losses = []
    for residual in residuals:
        correction = model(residual)
        metric_residual = model.base(model.rte.A_op(correction) - residual)
        denom = torch.clamp(torch.linalg.norm(model.base(residual)), min=1e-30)
        losses.append((torch.linalg.norm(metric_residual) / denom) ** 2)
    return torch.stack(losses).mean()


def train_model(model, residuals, steps=80, lr=5e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = metric_loss(model, residuals)
        reg = 1e-5 * (torch.sum(model.alpha_re**2) + torch.sum(model.alpha_im**2))
        total = loss + reg
        total.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def solve_stationary_learned(rte, model, maxiter, tol):
    psi = torch.zeros_like(rte.source)
    source_norm = torch.clamp(torch.linalg.norm(rte.source), min=1e-30)
    history = []
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(maxiter):
            residual = rte.source - rte.A_op(psi)
            relres = float(torch.linalg.norm(residual) / source_norm)
            history.append(relres)
            if relres < tol:
                break
            psi = psi + model(residual)
        residual = rte.source - rte.A_op(psi)
        history.append(float(torch.linalg.norm(residual) / source_norm))
    return {
        "stationary_iters": len(history) - 1,
        "stationary_final": history[-1],
        "stationary_factor": contraction(history),
        "stationary_time": time.perf_counter() - start,
    }


def solve_fgmres_learned(rte, model, maxiter, tol):
    def A(x):
        return rte.A_op(x)

    def M(r):
        with torch.no_grad():
            return model(r)

    start = time.perf_counter()
    with torch.no_grad():
        _, history = fgmres(
            A=A,
            b=rte.source,
            M=M,
            tol=tol,
            maxiter=maxiter,
            restart=min(30, maxiter),
            max_restarts=2,
            verbose=False,
        )
    return {
        "fgmres_iters": len(history) - 1,
        "fgmres_final": history[-1],
        "fgmres_factor": contraction(history),
        "fgmres_time": time.perf_counter() - start,
    }


def write_tex(rows, path):
    lines = [
        "% Auto-generated by rte/train_learned_hard_cases.py",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Case & Model & Inner & Train loss & Stat. it. & Stat. final & FGMRES it. & FGMRES final\\\\",
        "\\midrule",
    ]
    names = {
        "diffusive": "Diffusive",
        "heterogeneous": "Heterogeneous",
        "anisotropic": "Anisotropic",
        "mixed": "Mixed",
    }
    for row in rows:
        lines.append(
            "{} & {} & {} & {:.1e} & {} & {:.1e} & {} & {:.1e}\\\\".format(
                names[row["case"]],
                row["model_label"],
                row["inner_calls"],
                row["train_loss_final"],
                row["stationary_iters"],
                row["stationary_final"],
                row["fgmres_iters"],
                row["fgmres_final"],
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train per-case learned RTE defect preconditioners.")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--n-angles", type=int, default=24)
    parser.add_argument("--eta-scale", type=float, default=0.1)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--fgmres-maxiter", type=int, default=45)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--output-csv", default="rte/learned_hard_case_results.csv")
    parser.add_argument("--output-tex", default="rte/learned_hard_case_results_table.tex")
    args = parser.parse_args()

    torch.manual_seed(20260518)
    dtype = torch.float64
    rows = []
    cases = ("diffusive", "heterogeneous", "anisotropic", "mixed")
    configs = (
        ("transport_dsa", "Poly-DSA", None),
        ("transport_tsa", "Poly-TSA-H2", 2),
    )

    for case in cases:
        rte, _ = make_case(case, args.size, args.n_angles, dtype, args.eta_scale)
        print(f"case={case}")
        for base_kind, label, harmonics in configs:
            residuals = collect_training_residuals(rte, base_kind, harmonics)
            model = ResidualPolynomialPreconditioner(
                rte=rte,
                base_kind=base_kind,
                degree=args.degree,
                tsa_harmonics=harmonics,
            )
            train_history = train_model(model, residuals, steps=args.steps, lr=args.lr)
            stationary = solve_stationary_learned(rte, model, args.maxiter, args.tol)
            krylov = solve_fgmres_learned(rte, model, args.fgmres_maxiter, args.tol)
            alpha = model.alpha.detach().cpu()
            row = {
                "case": case,
                "model_label": label,
                "base_kind": base_kind,
                "tsa_harmonics": harmonics,
                "degree": args.degree,
                "inner_calls": args.degree + 1,
                "train_loss_initial": train_history[0],
                "train_loss_final": train_history[-1],
                "alpha": ";".join(f"{z.real:.6g}+{z.imag:.6g}j" for z in alpha),
            }
            row.update(stationary)
            row.update(krylov)
            rows.append(row)
            print(
                f"  {label:11s} loss={row['train_loss_final']:.2e} "
                f"stat={row['stationary_iters']:3d}/{row['stationary_final']:.2e} "
                f"fgmres={row['fgmres_iters']:3d}/{row['fgmres_final']:.2e}"
            )

    csv_path = ROOT / args.output_csv
    tex_path = ROOT / args.output_tex
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_tex(rows, tex_path)
    print(f"wrote {csv_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
