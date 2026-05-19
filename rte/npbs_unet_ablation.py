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
from rte.train_defect_cnn_small import make_source, smooth_random_like


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "helmholtz" / "models"))
from gmres import fgmres  # noqa: E402


def circular_pad_conv(in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False):
    return _CircularConv2d(in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=bias)


class _CircularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=False):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        if self.padding:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="circular")
        return self.conv(x)


def norm_groups(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


class CoeffEncoder(nn.Module):
    def __init__(self, coeff_channels, width, levels):
        super().__init__()
        groups = norm_groups(width)
        self.lift = nn.Sequential(
            circular_pad_conv(coeff_channels, width, bias=True),
            nn.GroupNorm(groups, width),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    circular_pad_conv(width, width, bias=True),
                    nn.GroupNorm(groups, width),
                    nn.SiLU(),
                    circular_pad_conv(width, width, bias=True),
                    nn.GroupNorm(groups, width),
                    nn.SiLU(),
                )
                for _ in range(levels)
            ]
        )
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    circular_pad_conv(width, width, kernel_size=5, stride=2, bias=True),
                    nn.GroupNorm(groups, width),
                    nn.SiLU(),
                )
                for _ in range(levels - 1)
            ]
        )

    def forward(self, coeff):
        x = self.lift(coeff)
        features = []
        for level, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            if level < len(self.down):
                x = self.down[level](x)
        return features


class LinearGatedUNet(nn.Module):
    """
    U-Net correction that is exactly linear in residual features for fixed
    coefficient features.  All residual-path convolutions are bias-free and
    activation-free; nonlinearity enters only through coefficient gates.
    """

    def __init__(self, in_channels, out_channels, width=16, levels=3, gate_scale=0.1):
        super().__init__()
        self.levels = int(levels)
        self.gate_scale = float(gate_scale)
        self.lift = circular_pad_conv(in_channels, width, kernel_size=1, bias=False)
        self.enc_mix = nn.ModuleList([circular_pad_conv(width, width, bias=False) for _ in range(levels)])
        self.enc_gate = nn.ModuleList([circular_pad_conv(width, width, kernel_size=1, bias=True) for _ in range(levels)])
        self.down = nn.ModuleList([circular_pad_conv(width, width, kernel_size=5, stride=2, bias=False) for _ in range(levels - 1)])
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(width, width, kernel_size=4, stride=2, padding=1, bias=False) for _ in range(levels - 1)]
        )
        self.dec_mix1 = nn.ModuleList([circular_pad_conv(2 * width, width, bias=False) for _ in range(levels - 1)])
        self.dec_gate = nn.ModuleList([circular_pad_conv(width, width, kernel_size=1, bias=True) for _ in range(levels - 1)])
        self.dec_mix2 = nn.ModuleList([circular_pad_conv(width, width, bias=False) for _ in range(levels - 1)])
        self.proj = circular_pad_conv(width, out_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.proj.conv.weight)

    @staticmethod
    def resize_like(x, reference):
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        diff_h = reference.size(-2) - x.size(-2)
        diff_w = reference.size(-1) - x.size(-1)
        if diff_h >= 0 and diff_w >= 0:
            return F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        top = max((-diff_h) // 2, 0)
        left = max((-diff_w) // 2, 0)
        return x[..., top : top + reference.size(-2), left : left + reference.size(-1)]

    def gate(self, x, coeff, projector):
        if coeff.shape[-2:] != x.shape[-2:]:
            coeff = F.interpolate(coeff, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * (1.0 + self.gate_scale * torch.tanh(projector(coeff)))

    def forward(self, residual_features, coeff_features):
        x = self.lift(residual_features)
        skips = []
        for level in range(self.levels):
            x = self.enc_mix[level](x)
            x = self.gate(x, coeff_features[level], self.enc_gate[level])
            skips.append(x)
            if level < self.levels - 1:
                x = self.down[level](x)
        for step in range(self.levels - 1):
            level = self.levels - 2 - step
            x = self.resize_like(self.up[step](x), skips[level])
            x = torch.cat([skips[level], x], dim=1)
            x = self.dec_mix1[step](x)
            x = self.gate(x, coeff_features[level], self.dec_gate[step])
            x = self.dec_mix2[step](x)
        return self.proj(x)


class NPBSUNetPreconditioner(nn.Module):
    def __init__(
        self,
        rte,
        form="residual",
        base_kind="transport_dsa",
        tsa_harmonics=None,
        width=16,
        levels=3,
        gate_scale=0.1,
        delta_scale=1.0,
        defect_relaxation=1.0,
    ):
        super().__init__()
        if form not in {"residual", "defect2"}:
            raise ValueError("form must be 'residual' or 'defect2'")
        self.rte = rte
        self.form = form
        self.base_kind = base_kind
        self.tsa_harmonics = tsa_harmonics
        self.n_angles = rte.n_angles
        self.delta_scale = float(delta_scale)
        self.defect_relaxation = float(defect_relaxation)
        in_channels = 2 * self.n_angles if form == "residual" else 4 * self.n_angles
        self.coeff_encoder = CoeffEncoder(coeff_channels=4, width=width, levels=levels)
        self.unet = LinearGatedUNet(in_channels, 2 * self.n_angles, width=width, levels=levels, gate_scale=gate_scale)

    def base(self, residual):
        return self.rte.apply_preconditioner(residual, kind=self.base_kind, tsa_harmonics=self.tsa_harmonics)

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
    def ri(x):
        return torch.cat([x.real, x.imag], dim=1).to(torch.float32)

    def correction_from_features(self, features, batch_size, dtype, device):
        coeff = self.coeff_features(batch_size).to(device=device, dtype=features.dtype)
        coeff_features = self.coeff_encoder(coeff)
        delta_ri = self.unet(features, coeff_features)
        real, imag = torch.chunk(delta_ri, 2, dim=1)
        return torch.complex(real, imag).to(dtype=dtype)

    def forward(self, residual):
        z0 = self.base(residual)
        if self.form == "residual":
            features = self.ri(z0)
            delta = self.correction_from_features(features, residual.shape[0], residual.dtype, residual.device)
            return z0 + self.delta_scale * delta
        defect = residual - self.rte.A_op(z0)
        z1 = self.base(defect)
        features = torch.cat([self.ri(z0), self.ri(z1)], dim=1)
        delta = self.correction_from_features(features, residual.shape[0], residual.dtype, residual.device)
        return z0 + self.defect_relaxation * z1 + self.delta_scale * delta


def collect_residuals(rte, base_kind, tsa_harmonics=None, n_rollout=6, n_random=6):
    residuals = []
    psi = torch.zeros_like(rte.source)
    for _ in range(n_rollout):
        residual = (rte.source - rte.A_op(psi)).detach()
        residuals.append(residual)
        correction = rte.apply_preconditioner(residual, kind=base_kind, tsa_harmonics=tsa_harmonics)
        psi = psi + correction
    for _ in range(n_random):
        exact = smooth_random_like(rte.source, modes=5)
        residuals.append(rte.A_op(exact).detach())
    return residuals


def inner_base_apply(rte, residual, depth=1, kind="transport_dsa", tsa_harmonics=None, relaxation=1.0):
    out = torch.zeros_like(residual)
    defect = residual
    for _ in range(depth):
        correction = rte.apply_preconditioner(defect, kind=kind, tsa_harmonics=tsa_harmonics)
        out = out + float(relaxation) * correction
        defect = defect - rte.A_op(float(relaxation) * correction)
    return out


def metric_loss(model, residuals, physical_weight=0.5, correction_weight=1e-6):
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


def train_model(model, residuals, steps, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = metric_loss(model, residuals)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def one_step_quality(rte, preconditioner, residual):
    with torch.no_grad():
        correction = preconditioner(residual)
        error = rte.A_op(correction) - residual
        true_rel = float(torch.linalg.norm(error) / torch.clamp(torch.linalg.norm(residual), min=1e-30))
        metric_error = rte.apply_preconditioner(error, kind="transport")
        metric_ref = rte.apply_preconditioner(residual, kind="transport")
        metric_rel = float(torch.linalg.norm(metric_error) / torch.clamp(torch.linalg.norm(metric_ref), min=1e-30))
    return true_rel, metric_rel


def solve_stationary(rte, source, preconditioner, maxiter, tol):
    old_source = rte.source
    rte.source = source.to(device=rte.sigma_t.device, dtype=rte.complex_dtype)
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
    rte.source = old_source
    return len(history) - 1, history[-1], contraction(history), time.perf_counter() - start


def solve_fgmres(rte, source, preconditioner, maxiter, tol):
    old_source = rte.source
    rte.source = source.to(device=rte.sigma_t.device, dtype=rte.complex_dtype)

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
    rte.source = old_source
    return len(history) - 1, history[-1], contraction(history), time.perf_counter() - start


def write_report(rows, path, settings):
    lines = [
        "# NPBS U-Net Preconditioner Ablation",
        "",
        "This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.",
        "",
        "## Settings",
        "",
    ]
    for key, value in settings.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Results", ""])
    lines.append(
        "| Size | Base | Case | Split | Method | one-step true | one-step Born metric | stat final | FGMRES final | loss0 | loss1 |"
    )
    lines.append("|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {size} | {base_kind} | {case} | {split} | {method} | {one_step_true:.3e} | {one_step_metric:.3e} | "
            "{stationary_final:.3e} | {fgmres_final:.3e} | {train_loss_initial} | {train_loss_final} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="NPBS-native matrix-free RTE U-Net preconditioner ablation.")
    parser.add_argument("--cases", nargs="+", default=["diffusive", "heterogeneous", "anisotropic", "mixed"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[24])
    parser.add_argument("--n-angles", type=int, default=12)
    parser.add_argument("--base-kinds", nargs="+", default=["transport_dsa"])
    parser.add_argument("--forms", nargs="+", default=["residual", "defect2"])
    parser.add_argument("--defect-relaxations", nargs="+", type=float, default=[1.0])
    parser.add_argument("--eta-scale", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--fgmres-maxiter", type=int, default=25)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--output-csv", default="rte/npbs_unet_ablation_results.csv")
    parser.add_argument("--output-md", default="rte/NPBS_UNET_ABLATION_REPORT.md")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float32
    rows = []
    for size in args.sizes:
        for case in args.cases:
            rte, _ = make_case(case, size, args.n_angles, dtype, args.eta_scale)
            train_source = rte.source.clone()
            heldout_source = make_source(
                size,
                args.n_angles,
                dtype,
                center=(0.68, 0.34),
                angular="beam" if case in ("anisotropic", "mixed") else "dipole",
            )
            for base_kind in args.base_kinds:
                residuals = collect_residuals(rte, base_kind)
                base_label = base_kind.replace("transport_", "")

                def base1(residual, kind=base_kind):
                    return inner_base_apply(rte, residual, depth=1, kind=kind)

                def base2(residual, kind=base_kind):
                    return inner_base_apply(rte, residual, depth=2, kind=kind)

                trained_models = []
                for form in args.forms:
                    relaxations = args.defect_relaxations if form == "defect2" else [1.0]
                    for relaxation in relaxations:
                        model = NPBSUNetPreconditioner(
                            rte,
                            form=form,
                            base_kind=base_kind,
                            width=args.width,
                            levels=args.levels,
                            defect_relaxation=relaxation,
                        )
                        history = train_model(model, residuals, args.steps, args.lr)
                        trained_models.append((f"UNet-{form}-w{relaxation:g}", model, history))
                        print(
                            f"size={size} case={case} base={base_kind} form={form} "
                            f"omega={relaxation:g} loss={history[0]:.3e}->{history[-1]:.3e}"
                        )

                methods = [(f"{base_label}1", base1, 1, "", ""), (f"{base_label}2", base2, 2, "", "")]
                for label, model, history in trained_models:
                    methods.append((label, model, 1 if model.form == "residual" else 2, history[0], history[-1]))

                for split, source in (("train", train_source), ("heldout", heldout_source)):
                    residual = source.to(dtype=rte.complex_dtype) - rte.A_op(torch.zeros_like(rte.source))
                    for label, precond, calls, loss0, loss1 in methods:
                        one_true, one_metric = one_step_quality(rte, precond, residual)
                        si, sf, sc, st = solve_stationary(rte, source, precond, args.maxiter, args.tol)
                        gi, gf, gc, gt = solve_fgmres(rte, source, precond, args.fgmres_maxiter, args.tol)
                        row = {
                            "size": size,
                            "case": case,
                            "split": split,
                            "base_kind": base_kind,
                            "method": label,
                            "base_calls_per_apply": calls,
                            "one_step_true": one_true,
                            "one_step_metric": one_metric,
                            "stationary_iters": si,
                            "stationary_final": sf,
                            "stationary_factor": sc,
                            "stationary_time": st,
                            "fgmres_iters": gi,
                            "fgmres_final": gf,
                            "fgmres_factor": gc,
                            "fgmres_time": gt,
                            "train_loss_initial": loss0,
                            "train_loss_final": loss1,
                        }
                        rows.append(row)
                        print(
                            f"  {split:7s} {label:18s} one={one_true:.2e} "
                            f"stat={si:3d}/{sf:.2e} fgmres={gi:3d}/{gf:.2e}"
                        )

    csv_path = ROOT / args.output_csv
    md_path = ROOT / args.output_md
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_report(rows, md_path, vars(args))
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
