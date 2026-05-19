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


def ensure_residual_batch(residuals):
    if torch.is_tensor(residuals):
        return residuals
    if not residuals:
        raise ValueError("residual batch is empty")
    return torch.cat(list(residuals), dim=0)


def batch_l2_norm(x):
    dims = tuple(range(1, x.ndim))
    if torch.is_complex(x):
        sq = x.real.square() + x.imag.square()
    else:
        sq = x.square()
    return torch.sqrt(torch.clamp(torch.sum(sq, dim=dims), min=1e-30))


def make_training_sources(case, size, n_angles, dtype, device, num_aux_sources=0):
    angular = "beam" if case in ("anisotropic", "mixed") else "isotropic"
    sources = [make_source(size, n_angles, dtype, center=(0.32, 0.47), angular=angular).to(device=device)]
    candidate_specs = [
        ((0.20, 0.72), "dipole"),
        ((0.70, 0.28), "beam" if case in ("anisotropic", "mixed") else "dipole"),
        ((0.52, 0.18), "isotropic"),
        ((0.80, 0.62), "beam"),
    ]
    for idx in range(int(num_aux_sources)):
        center, aux_angular = candidate_specs[idx % len(candidate_specs)]
        sources.append(make_source(size, n_angles, dtype, center=center, angular=aux_angular).to(device=device))
    return sources


def collect_residuals_from_sources(
    rte,
    base_kind,
    tsa_harmonics=None,
    sources=None,
    n_rollout=6,
    n_random=6,
    random_modes=5,
):
    residuals = []
    original_source = rte.source.clone()
    sources = [original_source] if sources is None else list(sources)
    with torch.no_grad():
        for source in sources:
            rte.source = source.to(device=original_source.device, dtype=original_source.dtype)
            psi = torch.zeros_like(rte.source)
            for _ in range(int(n_rollout)):
                residual = (rte.source - rte.A_op(psi)).detach()
                residuals.append(residual)
                correction = rte.apply_preconditioner(residual, kind=base_kind, tsa_harmonics=tsa_harmonics)
                psi = psi + correction
        if int(n_random) > 0:
            exact = smooth_random_like(original_source.expand(int(n_random), -1, -1, -1), modes=random_modes)
            residuals.extend(torch.unbind(rte.A_op(exact).detach(), dim=0))
    rte.source = original_source
    return [residual.unsqueeze(0) if residual.ndim == 3 else residual for residual in residuals]


def make_residual_sampler(
    rte,
    base_kind,
    tsa_harmonics=None,
    sources=None,
    n_rollout=6,
    residuals_per_step=16,
    rollout_samples_per_step=8,
    random_modes=5,
    seed=0,
):
    rollout_bank = collect_residuals_from_sources(
        rte,
        base_kind,
        tsa_harmonics=tsa_harmonics,
        sources=sources,
        n_rollout=n_rollout,
        n_random=0,
        random_modes=random_modes,
    )
    original_source = rte.source.clone()
    sources = [original_source] if sources is None else list(sources)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    def sample():
        batches = []
        with torch.no_grad():
            if rollout_bank:
                count = min(int(rollout_samples_per_step), len(rollout_bank), int(residuals_per_step))
                if count > 0:
                    indices = torch.randperm(len(rollout_bank), generator=generator)[:count].tolist()
                    batches.append(torch.cat([rollout_bank[idx] for idx in indices], dim=0))

            remaining = int(residuals_per_step) - sum(batch.shape[0] for batch in batches)
            if remaining > 0:
                source_indices = torch.randint(len(sources), (remaining,), generator=generator)
                source_batch = torch.cat([sources[int(idx)] for idx in source_indices.tolist()], dim=0).to(
                    device=original_source.device,
                    dtype=original_source.dtype,
                )
                probe = smooth_random_like(source_batch, modes=random_modes)
                batches.append(rte.A_op(probe).detach())

        rte.source = original_source
        return torch.cat(batches, dim=0)

    return sample, {"rollout_bank_size": len(rollout_bank), "residuals_per_step": int(residuals_per_step)}


def inner_base_apply(rte, residual, depth=1, kind="transport_dsa", tsa_harmonics=None, relaxation=1.0):
    out = torch.zeros_like(residual)
    defect = residual
    for _ in range(depth):
        correction = rte.apply_preconditioner(defect, kind=kind, tsa_harmonics=tsa_harmonics)
        out = out + float(relaxation) * correction
        defect = defect - rte.A_op(float(relaxation) * correction)
    return out


def metric_loss(model, residuals, physical_weight=0.5, correction_weight=1e-6):
    residual_batch = ensure_residual_batch(residuals)
    correction = model(residual_batch)
    error = model.rte.A_op(correction) - residual_batch
    metric_error = model.base(error)
    denom_metric = torch.clamp(batch_l2_norm(model.base(residual_batch)), min=1e-30)
    denom_phys = torch.clamp(batch_l2_norm(residual_batch), min=1e-30)
    correction_norm = batch_l2_norm(correction)
    metric_ratio = batch_l2_norm(metric_error) / denom_metric
    physical_ratio = batch_l2_norm(error) / denom_phys
    correction_ratio = correction_norm / denom_metric
    return metric_ratio, physical_ratio, correction_ratio, error


def objective_loss(
    model,
    residuals,
    physical_weight=0.5,
    correction_weight=1e-6,
    rollout_weight=0.0,
    rollout_steps=1,
    rollout_decay=1.0,
    stability_weight=0.0,
    contraction_target=0.98,
):
    metric_ratio, physical_ratio, correction_ratio, residual_after = metric_loss(
        model,
        residuals,
        physical_weight=physical_weight,
        correction_weight=correction_weight,
    )
    loss = (
        (metric_ratio.square()).mean()
        + physical_weight * (physical_ratio.square()).mean()
        + correction_weight * (correction_ratio.square()).mean()
    )
    metric_history = [metric_ratio]
    if rollout_weight > 0.0 or stability_weight > 0.0:
        current_residual = residual_after
        for rollout_idx in range(int(rollout_steps)):
            rollout_metric_ratio, _, _, current_residual = metric_loss(
                model,
                current_residual,
                physical_weight=physical_weight,
                correction_weight=correction_weight,
            )
            metric_history.append(rollout_metric_ratio)
            if rollout_weight > 0.0:
                rollout_scale = float(rollout_decay) ** rollout_idx
                loss = loss + rollout_weight * rollout_scale * (rollout_metric_ratio.square()).mean()
    if stability_weight > 0.0:
        target = float(contraction_target)
        stability_terms = [(torch.relu(entry - target).square()).mean() for entry in metric_history]
        loss = loss + stability_weight * torch.stack(stability_terms).mean()
    return loss


def train_model(
    model,
    residuals,
    steps,
    lr,
    weight_decay=0.0,
    scheduler_name="none",
    scheduler_step_size=0,
    scheduler_gamma=0.5,
    scheduler_eta_min=0.0,
    log_steps=0,
    physical_weight=0.5,
    correction_weight=1e-6,
    rollout_weight=0.0,
    rollout_steps=1,
    rollout_decay=1.0,
    stability_weight=0.0,
    contraction_target=0.98,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if scheduler_name == "step":
        step_size = int(scheduler_step_size) if int(scheduler_step_size) > 0 else max(1, int(steps) // 5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=float(scheduler_gamma))
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(steps)),
            eta_min=float(scheduler_eta_min),
        )
    elif scheduler_name != "none":
        raise ValueError(f"unsupported scheduler: {scheduler_name}")

    history = {"loss": [], "lr": []}
    total_steps = int(steps)
    for step_idx in range(total_steps):
        current_residuals = residuals() if callable(residuals) else residuals
        optimizer.zero_grad(set_to_none=True)
        loss = objective_loss(
            model,
            current_residuals,
            physical_weight=physical_weight,
            correction_weight=correction_weight,
            rollout_weight=rollout_weight,
            rollout_steps=rollout_steps,
            rollout_decay=rollout_decay,
            stability_weight=stability_weight,
            contraction_target=contraction_target,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        history["loss"].append(float(loss.detach()))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        if int(log_steps) > 0 and ((step_idx + 1) % int(log_steps) == 0 or step_idx == 0 or step_idx + 1 == total_steps):
            print(
                f"    step [{step_idx + 1:4d}/{total_steps}] loss={history['loss'][-1]:.3e} "
                f"lr={history['lr'][-1]:.2e}"
            )
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
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "step", "cosine"], default="none")
    parser.add_argument("--scheduler-step-size", type=int, default=0)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-5)
    parser.add_argument("--log-steps", type=int, default=0)
    parser.add_argument("--residual-mode", choices=["pool", "resample"], default="pool")
    parser.add_argument("--residuals-per-step", type=int, default=16)
    parser.add_argument("--rollout-samples-per-step", type=int, default=8)
    parser.add_argument("--n-rollout", type=int, default=6)
    parser.add_argument("--n-random", type=int, default=6)
    parser.add_argument("--random-modes", type=int, default=5)
    parser.add_argument("--aux-sources", type=int, default=0)
    parser.add_argument("--physical-weight", type=float, default=0.5)
    parser.add_argument("--correction-weight", type=float, default=1e-6)
    parser.add_argument("--rollout-weight", type=float, default=0.0)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--rollout-decay", type=float, default=1.0)
    parser.add_argument("--stability-weight", type=float, default=0.0)
    parser.add_argument("--contraction-target", type=float, default=0.98)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--fgmres-maxiter", type=int, default=25)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--output-csv", default="rte/npbs_unet_ablation_results.csv")
    parser.add_argument("--output-md", default="rte/NPBS_UNET_ABLATION_REPORT.md")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float32
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.benchmark = True
        if args.allow_tf32:
            torch.set_float32_matmul_precision("high")
    rows = []
    for size in args.sizes:
        for case in args.cases:
            rte, _ = make_case(case, size, args.n_angles, dtype, args.eta_scale, device=device)
            train_source = rte.source.clone()
            training_sources = make_training_sources(
                case,
                size,
                args.n_angles,
                dtype,
                device=device,
                num_aux_sources=args.aux_sources,
            )
            heldout_source = make_source(
                size,
                args.n_angles,
                dtype,
                center=(0.68, 0.34),
                angular="beam" if case in ("anisotropic", "mixed") else "dipole",
            ).to(device=device)
            for base_kind in args.base_kinds:
                if args.residual_mode == "pool":
                    residuals = collect_residuals_from_sources(
                        rte,
                        base_kind,
                        sources=training_sources,
                        n_rollout=args.n_rollout,
                        n_random=args.n_random,
                        random_modes=args.random_modes,
                    )
                    residual_info = {"rollout_bank_size": len(residuals), "residuals_per_step": len(residuals)}
                else:
                    residuals, residual_info = make_residual_sampler(
                        rte,
                        base_kind,
                        sources=training_sources,
                        n_rollout=args.n_rollout,
                        residuals_per_step=args.residuals_per_step,
                        rollout_samples_per_step=args.rollout_samples_per_step,
                        random_modes=args.random_modes,
                        seed=args.seed,
                    )
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
                        ).to(device)
                        history = train_model(
                            model,
                            residuals,
                            args.steps,
                            args.lr,
                            weight_decay=args.weight_decay,
                            scheduler_name=args.scheduler,
                            scheduler_step_size=args.scheduler_step_size,
                            scheduler_gamma=args.scheduler_gamma,
                            scheduler_eta_min=args.scheduler_eta_min,
                            log_steps=args.log_steps,
                            physical_weight=args.physical_weight,
                            correction_weight=args.correction_weight,
                            rollout_weight=args.rollout_weight,
                            rollout_steps=args.rollout_steps,
                            rollout_decay=args.rollout_decay,
                            stability_weight=args.stability_weight,
                            contraction_target=args.contraction_target,
                        )
                        trained_models.append((f"UNet-{form}-w{relaxation:g}", model, history, residual_info))
                        print(
                            f"size={size} case={case} base={base_kind} form={form} mode={args.residual_mode} "
                            f"omega={relaxation:g} bank={residual_info['rollout_bank_size']} "
                            f"loss={history['loss'][0]:.3e}->{history['loss'][-1]:.3e}"
                        )

                methods = [(f"{base_label}1", base1, 1, "", ""), (f"{base_label}2", base2, 2, "", "")]
                for label, model, history, info in trained_models:
                    methods.append(
                        (
                            label,
                            model,
                            1 if model.form == "residual" else 2,
                            history['loss'][0],
                            history['loss'][-1],
                            history['lr'][0],
                            history['lr'][-1],
                            info['residuals_per_step'],
                            info['rollout_bank_size'],
                        )
                    )

                baseline_methods = [(f"{base_label}1", base1, 1, "", "", "", "", "", ""), (f"{base_label}2", base2, 2, "", "", "", "", "", "")]

                for split, source in (("train", train_source), ("heldout", heldout_source)):
                    residual = source.to(dtype=rte.complex_dtype) - rte.A_op(torch.zeros_like(rte.source))
                    for label, precond, calls, loss0, loss1, lr0, lr1, residuals_per_step, rollout_bank_size in baseline_methods + methods[2:]:
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
                            "train_lr_initial": lr0,
                            "train_lr_final": lr1,
                            "residual_mode": args.residual_mode,
                            "residuals_per_step": residuals_per_step,
                            "rollout_bank_size": rollout_bank_size,
                            "n_rollout": args.n_rollout,
                            "n_random": args.n_random,
                            "random_modes": args.random_modes,
                            "aux_sources": args.aux_sources,
                            "scheduler": args.scheduler,
                            "scheduler_step_size": args.scheduler_step_size,
                            "scheduler_gamma": args.scheduler_gamma,
                            "scheduler_eta_min": args.scheduler_eta_min,
                            "physical_weight": args.physical_weight,
                            "correction_weight": args.correction_weight,
                            "rollout_weight": args.rollout_weight,
                            "rollout_steps": args.rollout_steps,
                            "rollout_decay": args.rollout_decay,
                            "stability_weight": args.stability_weight,
                            "contraction_target": args.contraction_target,
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
