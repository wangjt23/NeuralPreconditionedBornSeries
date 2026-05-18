import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rte.run_hard_case_experiments import contraction, make_case


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "helmholtz" / "models"))
from gmres import fgmres  # noqa: E402


def circular_conv(in_channels, out_channels, kernel_size=3, bias=False):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.ZeroPad2d(0),
        _CircularConv2d(in_channels, out_channels, kernel_size, padding, bias=bias),
    )


class _CircularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super().__init__()
        self.padding = int(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="circular")
        return self.conv(x)


class DefectBlock(nn.Module):
    def __init__(self, width, coeff_channels):
        super().__init__()
        self.conv = circular_conv(width, width, bias=False)
        self.gate = circular_conv(coeff_channels, width, kernel_size=1, bias=True)
        # Start as an identity-conditioned linear residual block.
        nn.init.zeros_(self.gate[1].conv.weight)
        nn.init.zeros_(self.gate[1].conv.bias)

    def forward(self, x, coeff):
        update = self.conv(x)
        gate = 1.0 + 0.25 * torch.tanh(self.gate(coeff))
        return x + gate * update


class CoeffConditionedDefectCNN(nn.Module):
    """
    Small coefficient-conditioned defect corrector.

    For a residual r and base preconditioner B_h:

        z0 = B_h r,
        z1 = B_h(r - A z0),
        M_theta r = z0 + z1 + s C_theta(z0, z1, coefficients).

    C_theta is linear in residual features for fixed coefficients, because
    coefficient maps only gate linear residual convolutions. The output
    projection is initialized to zero, so the initial preconditioner is exactly
    the equal-cost two-step DSA map.
    """

    def __init__(self, rte, width=24, blocks=3, base_kind="transport_dsa", tsa_harmonics=None, delta_scale=0.1):
        super().__init__()
        self.rte = rte
        self.base_kind = base_kind
        self.tsa_harmonics = tsa_harmonics
        self.n_angles = rte.n_angles
        self.delta_scale = float(delta_scale)
        residual_channels = 4 * rte.n_angles
        coeff_channels = 4
        self.lift = circular_conv(residual_channels, width, bias=False)
        self.blocks = nn.ModuleList([DefectBlock(width, coeff_channels) for _ in range(blocks)])
        self.proj = circular_conv(width, 2 * rte.n_angles, bias=False)
        nn.init.zeros_(self.proj[1].conv.weight)

    def base(self, residual):
        return self.rte.apply_preconditioner(
            residual,
            kind=self.base_kind,
            tsa_harmonics=self.tsa_harmonics,
        )

    def coeff_features(self, batch_size):
        sigma_t = self.rte.sigma_t.to(dtype=self.rte.float_dtype)
        sigma_s = self.rte.sigma_s.to(dtype=self.rte.float_dtype)
        albedo = sigma_s / torch.clamp(sigma_t, min=1e-12)
        st_mean = torch.mean(sigma_t, dim=(-2, -1), keepdim=True)
        st_std = torch.clamp(torch.std(sigma_t, dim=(-2, -1), keepdim=True), min=1e-6)
        st_norm = (sigma_t - st_mean) / st_std
        dx = torch.roll(sigma_t, shifts=-1, dims=-1) - torch.roll(sigma_t, shifts=1, dims=-1)
        dy = torch.roll(sigma_t, shifts=-1, dims=-2) - torch.roll(sigma_t, shifts=1, dims=-2)
        grad = torch.sqrt(dx * dx + dy * dy) / torch.clamp(st_mean, min=1e-12)
        g = torch.full_like(sigma_t, float(self.rte.anisotropy_g))
        coeff = torch.cat([st_norm, albedo, grad, g], dim=1)
        if coeff.shape[0] == 1 and batch_size > 1:
            coeff = coeff.expand(batch_size, -1, -1, -1)
        return coeff

    @staticmethod
    def _ri(x):
        return torch.cat([x.real, x.imag], dim=1).to(torch.float32)

    def forward(self, residual):
        z0 = self.base(residual)
        defect = residual - self.rte.A_op(z0)
        z1 = self.base(defect)
        features = torch.cat([self._ri(z0), self._ri(z1)], dim=1)
        coeff = self.coeff_features(residual.shape[0]).to(device=residual.device, dtype=features.dtype)
        x = self.lift(features)
        for block in self.blocks:
            x = block(x, coeff)
        delta_ri = self.proj(x)
        real, imag = torch.chunk(delta_ri, 2, dim=1)
        delta = torch.complex(real, imag).to(dtype=residual.dtype)
        return z0 + z1 + self.delta_scale * delta


def make_source(size, n_angles, dtype, center=(0.32, 0.47), angular="isotropic"):
    x = torch.arange(size, dtype=dtype) / size
    y = torch.arange(size, dtype=dtype) / size
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    source_xy = torch.exp(-100.0 * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2))
    angles = torch.arange(n_angles, dtype=dtype) * (2.0 * math.pi / n_angles)
    if angular == "beam":
        angular_weight = torch.exp(2.0 * torch.cos(angles - 0.4))
        angular_weight = angular_weight / angular_weight.mean()
    elif angular == "dipole":
        angular_weight = 1.0 + 0.7 * torch.cos(angles - 1.1)
    else:
        angular_weight = torch.ones_like(angles)
    source = angular_weight.view(1, n_angles, 1, 1) * source_xy.view(1, 1, size, size)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return source.to(cdtype)


def smooth_random_like(reference, modes=5):
    noise = torch.randn_like(reference)
    coeff = torch.fft.fftn(noise, dim=(-2, -1), norm="backward")
    h, w = reference.shape[-2:]
    kx = torch.fft.fftfreq(h, device=reference.device).view(h, 1)
    ky = torch.fft.fftfreq(w, device=reference.device).view(1, w)
    mask = ((kx.abs() * h <= modes) & (ky.abs() * w <= modes)).to(reference.dtype)
    return torch.fft.ifftn(coeff * mask.view(1, 1, h, w), dim=(-2, -1), norm="backward")


def collect_residuals(rte, base_kind, tsa_harmonics=None, n_rollout=5, n_random=5):
    residuals = []
    psi = torch.zeros_like(rte.source)
    for _ in range(n_rollout):
        residual = (rte.source - rte.A_op(psi)).detach()
        residuals.append(residual)
        correction = rte.apply_preconditioner(residual, kind=base_kind, tsa_harmonics=tsa_harmonics)
        psi = psi + correction
    for _ in range(n_random):
        exact = smooth_random_like(rte.source)
        residuals.append(rte.A_op(exact).detach())
    return residuals


def metric_loss(model, residuals, physical_weight=0.2, correction_weight=1e-5):
    losses = []
    for residual in residuals:
        correction = model(residual)
        error = model.rte.A_op(correction) - residual
        metric_error = model.base(error)
        denom_metric = torch.clamp(torch.linalg.norm(model.base(residual)), min=1e-30)
        denom_phys = torch.clamp(torch.linalg.norm(residual), min=1e-30)
        losses.append(
            (torch.linalg.norm(metric_error) / denom_metric) ** 2
            + physical_weight * (torch.linalg.norm(error) / denom_phys) ** 2
            + correction_weight * (torch.linalg.norm(correction) / denom_metric) ** 2
        )
    return torch.stack(losses).mean()


def train_model(model, residuals, steps=150, lr=2e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = metric_loss(model, residuals)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def inner_base_apply(rte, residual, depth=1, kind="transport_dsa", tsa_harmonics=None):
    out = torch.zeros_like(residual)
    defect = residual
    for _ in range(depth):
        correction = rte.apply_preconditioner(defect, kind=kind, tsa_harmonics=tsa_harmonics)
        out = out + correction
        defect = defect - rte.A_op(correction)
    return out


def solve_stationary(rte, preconditioner, maxiter, tol):
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
            psi = psi + preconditioner(residual)
        residual = rte.source - rte.A_op(psi)
        history.append(float(torch.linalg.norm(residual) / source_norm))
    return len(history) - 1, history[-1], contraction(history), time.perf_counter() - start


def solve_fgmres(rte, preconditioner, maxiter, tol):
    def A(x):
        return rte.A_op(x)

    def M(r):
        with torch.no_grad():
            return preconditioner(r)

    start = time.perf_counter()
    with torch.no_grad():
        _, history = fgmres(
            A=A,
            b=rte.source,
            M=M,
            tol=tol,
            maxiter=maxiter,
            restart=min(20, maxiter),
            max_restarts=2,
            verbose=False,
        )
    return len(history) - 1, history[-1], contraction(history), time.perf_counter() - start


def evaluate(rte, model, source, maxiter, fgmres_maxiter, tol):
    rte.source = source.to(device=rte.sigma_t.device, dtype=rte.complex_dtype)

    def dsa1(residual):
        return inner_base_apply(rte, residual, depth=1)

    def dsa2(residual):
        return inner_base_apply(rte, residual, depth=2)

    def learned(residual):
        return model(residual)

    rows = []
    for label, precond, b_calls in (("DSA", dsa1, 1), ("DSA2", dsa2, 2), ("CNN-DSA2", learned, 2)):
        si, sf, sc, st = solve_stationary(rte, precond, maxiter, tol)
        gi, gf, gc, gt = solve_fgmres(rte, precond, fgmres_maxiter, tol)
        rows.append(
            {
                "method": label,
                "b_calls_per_apply": b_calls,
                "stationary_iters": si,
                "stationary_b_calls": si * b_calls,
                "stationary_final": sf,
                "stationary_factor": sc,
                "stationary_time": st,
                "fgmres_iters": gi,
                "fgmres_b_calls": gi * b_calls,
                "fgmres_final": gf,
                "fgmres_factor": gc,
                "fgmres_time": gt,
            }
        )
    return rows


def write_tex(rows, path):
    lines = [
        "% Auto-generated by rte/train_defect_cnn_small.py",
        "\\begin{tabular}{lllrrrrrr}",
        "\\toprule",
        "Case & Split & Method & Stat. it. & Stat. $B_h$ & Stat. final & FGMRES it. & FGMRES final\\\\",
        "\\midrule",
    ]
    names = {"diffusive": "Diffusive", "anisotropic": "Anisotropic", "mixed": "Mixed"}
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {:.1e} & {} & {:.1e}\\\\".format(
                names.get(row["case"], row["case"]),
                row["split"],
                row["method"],
                row["stationary_iters"],
                row["stationary_b_calls"],
                row["stationary_final"],
                row["fgmres_iters"],
                row["fgmres_final"],
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Small coefficient-conditioned RTE defect CNN experiment.")
    parser.add_argument("--cases", nargs="+", default=["diffusive", "anisotropic", "mixed"])
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--n-angles", type=int, default=12)
    parser.add_argument("--eta-scale", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--delta-scale", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--fgmres-maxiter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--output-csv", default="rte/defect_cnn_small_results.csv")
    parser.add_argument("--output-tex", default="rte/defect_cnn_small_results_table.tex")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float32
    rows = []
    for case in args.cases:
        rte, _ = make_case(case, args.size, args.n_angles, dtype, args.eta_scale)
        train_source = rte.source.clone()
        heldout_source = make_source(
            args.size,
            args.n_angles,
            dtype,
            center=(0.68, 0.34),
            angular="beam" if case in ("anisotropic", "mixed") else "dipole",
        )
        residuals = collect_residuals(rte, "transport_dsa")
        model = CoeffConditionedDefectCNN(
            rte=rte,
            width=args.width,
            blocks=args.blocks,
            base_kind="transport_dsa",
            delta_scale=args.delta_scale,
        )
        print(f"case={case}: training CNN-DSA2 on {len(residuals)} residuals")
        history = train_model(model, residuals, steps=args.steps, lr=args.lr)
        print(f"  loss {history[0]:.3e} -> {history[-1]:.3e}")

        for split, source in (("train", train_source), ("heldout", heldout_source)):
            for result in evaluate(rte, model, source, args.maxiter, args.fgmres_maxiter, args.tol):
                row = {
                    "case": case,
                    "split": split,
                    "train_loss_initial": history[0],
                    "train_loss_final": history[-1],
                    "size": args.size,
                    "n_angles": args.n_angles,
                    "width": args.width,
                    "blocks": args.blocks,
                    "delta_scale": args.delta_scale,
                }
                row.update(result)
                rows.append(row)
                print(
                    f"  {split:7s} {result['method']:7s} "
                    f"stat={result['stationary_iters']:3d}/{result['stationary_final']:.2e} "
                    f"fgmres={result['fgmres_iters']:3d}/{result['fgmres_final']:.2e}"
                )

    csv_path = ROOT / args.output_csv
    tex_path = ROOT / args.output_tex
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_tex(rows, tex_path)
    print(f"wrote {csv_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
