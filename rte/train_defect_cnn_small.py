import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rte.run_hard_case_experiments import contraction, make_case
from rte.RTE_mgmodel import RTEUNetPrecond


sys.path.append(str(ROOT / "helmholtz" / "models"))
from gmres import fgmres  # noqa: E402


def batch_l2_norm(x):
    dims = tuple(range(1, x.ndim))
    if torch.is_complex(x):
        sq = x.real.square() + x.imag.square()
    else:
        sq = x.square()
    return torch.sqrt(torch.clamp(torch.sum(sq, dim=dims), min=1e-30))


def ensure_residual_batch(residuals):
    if torch.is_tensor(residuals):
        return residuals
    if not residuals:
        raise ValueError("residual batch is empty")
    return torch.cat(list(residuals), dim=0)


def build_rte_coeff_tensor(rte):
    sigma_t = rte.sigma_t.to(dtype=rte.float_dtype)
    sigma_s = rte.sigma_s.to(dtype=rte.float_dtype)
    albedo = sigma_s / torch.clamp(sigma_t, min=1e-12)
    st_mean = torch.mean(sigma_t, dim=(-2, -1), keepdim=True)
    st_std = torch.clamp(torch.std(sigma_t, dim=(-2, -1), keepdim=True), min=1e-6)
    st_norm = (sigma_t - st_mean) / st_std
    dx = torch.roll(sigma_t, shifts=-1, dims=-1) - torch.roll(sigma_t, shifts=1, dims=-1)
    dy = torch.roll(sigma_t, shifts=-1, dims=-2) - torch.roll(sigma_t, shifts=1, dims=-2)
    grad = torch.sqrt(dx * dx + dy * dy) / torch.clamp(st_mean, min=1e-12)
    return torch.cat([st_norm, albedo, grad], dim=1)


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


class RTEUNetDefectPrecond(nn.Module):
    """Residual-form wrapper around the attached NPBS-style RTE U-Net model."""

    def __init__(
        self,
        rte,
        channels=32,
        num_levels=4,
        base_kind="transport_dsa",
        base_depth=2,
        tsa_harmonics=None,
        gate_scale=0.1,
        init_output_scale=0.05,
    ):
        super().__init__()
        self.rte = rte
        self.base_kind = base_kind
        self.base_depth = int(base_depth)
        self.tsa_harmonics = tsa_harmonics
        self.core = RTEUNetPrecond(
            M=1,
            channels=channels,
            num_levels=num_levels,
            input_channels=2 * rte.n_angles,
            output_channels=2 * rte.n_angles,
            gate_scale=gate_scale,
            init_output_scale=init_output_scale,
        )
        self.register_buffer("coeff", build_rte_coeff_tensor(rte), persistent=False)

    @staticmethod
    def _ri(x):
        return torch.cat([x.real, x.imag], dim=1).to(torch.float32)

    def base(self, residual):
        return inner_base_apply(
            self.rte,
            residual,
            depth=self.base_depth,
            kind=self.base_kind,
            tsa_harmonics=self.tsa_harmonics,
        )

    def forward(self, residual):
        reference = self.base(residual)
        a_list, inv_a_list = self.core.setup(self.coeff)
        update_ri = self.core(self._ri(reference), a_list, inv_a_list)
        real, imag = torch.chunk(update_ri, 2, dim=1)
        update = torch.complex(real, imag).to(dtype=reference.dtype)
        return reference + update


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


def collect_residuals(rte, base_kind, tsa_harmonics=None, sources=None, n_rollout=5, n_random=5, random_modes=5):
    residuals = []
    original_source = rte.source.clone()
    sources = [original_source] if sources is None else list(sources)
    with torch.no_grad():
        for source in sources:
            rte.source = source.to(device=original_source.device, dtype=original_source.dtype)
            psi = torch.zeros_like(rte.source)
            for _ in range(n_rollout):
                residual = (rte.source - rte.A_op(psi)).detach()
                residuals.append(residual)
                correction = rte.apply_preconditioner(residual, kind=base_kind, tsa_harmonics=tsa_harmonics)
                psi = psi + correction
        if n_random > 0:
            exact = smooth_random_like(original_source.expand(int(n_random), -1, -1, -1), modes=random_modes)
            residuals.extend(torch.unbind(rte.A_op(exact).detach(), dim=0))
    rte.source = original_source
    return [residual.unsqueeze(0) if residual.ndim == 3 else residual for residual in residuals]


def make_residual_sampler(
    rte,
    base_kind,
    tsa_harmonics=None,
    sources=None,
    n_rollout=5,
    residuals_per_step=16,
    rollout_samples_per_step=8,
    random_modes=5,
    seed=0,
):
    rollout_bank = collect_residuals(
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


def residual_terms(model, residual):
    correction = model(residual)
    residual_after = residual - model.rte.A_op(correction)
    base_residual = model.base(residual)
    denom_metric = torch.clamp(batch_l2_norm(base_residual), min=1e-30)
    denom_phys = torch.clamp(batch_l2_norm(residual), min=1e-30)
    metric_ratio = batch_l2_norm(model.base(residual_after)) / denom_metric
    physical_ratio = batch_l2_norm(residual_after) / denom_phys
    correction_ratio = batch_l2_norm(correction) / denom_metric
    return residual_after, metric_ratio, physical_ratio, correction_ratio


def metric_loss(
    model,
    residuals,
    physical_weight=0.2,
    correction_weight=1e-5,
    rollout_weight=0.0,
    rollout_steps=1,
    rollout_decay=1.0,
    stability_weight=0.0,
    contraction_target=0.98,
):
    residual_batch = ensure_residual_batch(residuals)
    residual_after, metric_ratio, physical_ratio, correction_ratio = residual_terms(model, residual_batch)
    loss = (
        (metric_ratio ** 2).mean()
        + physical_weight * (physical_ratio ** 2).mean()
        + correction_weight * (correction_ratio ** 2).mean()
    )
    metric_history = [metric_ratio]
    if rollout_weight > 0.0 or stability_weight > 0.0:
        current_residual = residual_after
        for rollout_step in range(int(rollout_steps)):
            current_residual, rollout_metric_ratio, _, _ = residual_terms(model, current_residual)
            metric_history.append(rollout_metric_ratio)
            if rollout_weight > 0.0:
                rollout_scale = float(rollout_decay) ** rollout_step
                loss = loss + rollout_weight * rollout_scale * (rollout_metric_ratio ** 2).mean()
    if stability_weight > 0.0:
        target = float(contraction_target)
        stability_terms = [(torch.relu(metric_entry - target) ** 2).mean() for metric_entry in metric_history]
        loss = loss + stability_weight * torch.stack(stability_terms).mean()
    return loss


def train_model(
    model,
    residuals,
    steps=150,
    lr=2e-3,
    weight_decay=0.0,
    scheduler_name="none",
    scheduler_step_size=0,
    scheduler_gamma=0.5,
    scheduler_eta_min=0.0,
    log_steps=0,
    physical_weight=0.2,
    correction_weight=1e-5,
    rollout_weight=0.0,
    rollout_steps=1,
    rollout_decay=1.0,
    stability_weight=0.0,
    contraction_target=0.98,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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

    loss_history = []
    lr_history = []
    total_steps = int(steps)
    for step_idx in range(total_steps):
        current_residuals = residuals() if callable(residuals) else residuals
        optimizer.zero_grad(set_to_none=True)
        loss = metric_loss(
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

        loss_value = float(loss.detach())
        lr_value = float(optimizer.param_groups[0]["lr"])
        loss_history.append(loss_value)
        lr_history.append(lr_value)

        if int(log_steps) > 0 and ((step_idx + 1) % int(log_steps) == 0 or step_idx == 0 or step_idx + 1 == total_steps):
            print(f"    step [{step_idx + 1:4d}/{total_steps}] loss={loss_value:.3e} lr={lr_value:.2e}")

    return {"loss": loss_history, "lr": lr_history}


def append_case_history(history_rows, case, args, residual_info, history):
    for step_idx, (loss_value, lr_value) in enumerate(zip(history["loss"], history["lr"]), start=1):
        history_rows.append(
            {
                "case": case,
                "model_type": args.model_type,
                "step": step_idx,
                "loss": loss_value,
                "lr": lr_value,
                "size": args.size,
                "n_angles": args.n_angles,
                "width": args.width,
                "residual_mode": args.residual_mode,
                "residuals_per_step": residual_info["residuals_per_step"],
                "rollout_bank_size": residual_info["rollout_bank_size"],
                "steps": args.steps,
                "scheduler": args.scheduler,
                "scheduler_step_size": args.scheduler_step_size,
                "scheduler_gamma": args.scheduler_gamma,
                "scheduler_eta_min": args.scheduler_eta_min,
            }
        )


def summarize_curve(history):
    losses = history["loss"]
    if not losses:
        return "empty"
    checkpoints = {
        "start": losses[0],
        "25%": losses[max(0, len(losses) // 4 - 1)],
        "50%": losses[max(0, len(losses) // 2 - 1)],
        "75%": losses[max(0, (3 * len(losses)) // 4 - 1)],
        "end": losses[-1],
    }
    return " ".join(f"{name}={value:.3e}" for name, value in checkpoints.items())


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


def evaluate(rte, model, source, maxiter, fgmres_maxiter, tol, learned_label="LEARNED"):
    rte.source = source.to(device=rte.sigma_t.device, dtype=rte.complex_dtype)

    def dsa1(residual):
        return inner_base_apply(rte, residual, depth=1)

    def dsa2(residual):
        return inner_base_apply(rte, residual, depth=2)

    def learned(residual):
        return model(residual)

    rows = []
    for label, precond, b_calls in (("DSA", dsa1, 1), ("DSA2", dsa2, 2), (learned_label, learned, 2)):
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
    parser.add_argument("--model-type", choices=["small_cnn", "rte_unet"], default="small_cnn")
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--n-angles", type=int, default=12)
    parser.add_argument("--eta-scale", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--delta-scale", type=float, default=0.1)
    parser.add_argument("--residual-mode", choices=["pool", "resample"], default="resample")
    parser.add_argument("--residuals-per-step", type=int, default=16)
    parser.add_argument("--rollout-samples-per-step", type=int, default=8)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "step", "cosine"], default="none")
    parser.add_argument("--scheduler-step-size", type=int, default=0)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-5)
    parser.add_argument("--log-steps", type=int, default=0)
    parser.add_argument("--n-rollout", type=int, default=5)
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--random-modes", type=int, default=5)
    parser.add_argument("--aux-sources", type=int, default=0)
    parser.add_argument("--physical-weight", type=float, default=0.2)
    parser.add_argument("--correction-weight", type=float, default=1e-5)
    parser.add_argument("--rollout-weight", type=float, default=0.0)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--rollout-decay", type=float, default=1.0)
    parser.add_argument("--stability-weight", type=float, default=0.0)
    parser.add_argument("--contraction-target", type=float, default=0.98)
    parser.add_argument("--unet-levels", type=int, default=4)
    parser.add_argument("--unet-gate-scale", type=float, default=0.1)
    parser.add_argument("--unet-init-output-scale", type=float, default=0.02)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--fgmres-maxiter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-csv", default="rte/defect_cnn_small_results.csv")
    parser.add_argument("--output-tex", default="rte/defect_cnn_small_results_table.tex")
    parser.add_argument("--history-csv", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float32
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    rows = []
    history_rows = []
    for case in args.cases:
        rte, _ = make_case(case, args.size, args.n_angles, dtype, args.eta_scale, device=device)
        train_source = rte.source.clone()
        training_sources = make_training_sources(
            case,
            args.size,
            args.n_angles,
            dtype,
            device=device,
            num_aux_sources=args.aux_sources,
        )
        heldout_source = make_source(
            args.size,
            args.n_angles,
            dtype,
            center=(0.68, 0.34),
            angular="beam" if case in ("anisotropic", "mixed") else "dipole",
        ).to(device=device)
        if args.residual_mode == "pool":
            residuals = collect_residuals(
                rte,
                "transport_dsa",
                sources=training_sources,
                n_rollout=args.n_rollout,
                n_random=args.n_random,
                random_modes=args.random_modes,
            )
            residual_info = {"rollout_bank_size": len(residuals), "residuals_per_step": len(residuals)}
        else:
            residuals, residual_info = make_residual_sampler(
                rte,
                "transport_dsa",
                sources=training_sources,
                n_rollout=args.n_rollout,
                residuals_per_step=args.residuals_per_step,
                rollout_samples_per_step=args.rollout_samples_per_step,
                random_modes=args.random_modes,
                seed=args.seed,
            )

        if args.model_type == "rte_unet":
            model = RTEUNetDefectPrecond(
                rte=rte,
                channels=args.width,
                num_levels=args.unet_levels,
                base_kind="transport_dsa",
                base_depth=2,
                gate_scale=args.unet_gate_scale,
                init_output_scale=args.unet_init_output_scale,
            ).to(device)
        else:
            model = CoeffConditionedDefectCNN(
                rte=rte,
                width=args.width,
                blocks=args.blocks,
                base_kind="transport_dsa",
                delta_scale=args.delta_scale,
            ).to(device)

        print(
            f"case={case}: training {args.model_type} on mode={args.residual_mode} "
            f"rollout_bank={residual_info['rollout_bank_size']} residuals_per_step={residual_info['residuals_per_step']}"
        )
        history = train_model(
            model,
            residuals,
            steps=args.steps,
            lr=args.lr,
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
        print(f"  loss {history['loss'][0]:.3e} -> {history['loss'][-1]:.3e}")
        print(f"  curve {summarize_curve(history)}")
        append_case_history(history_rows, case, args, residual_info, history)

        learned_label = "UNET-DSA2" if args.model_type == "rte_unet" else "CNN-DSA2"

        for split, source in (("train", train_source), ("heldout", heldout_source)):
            for result in evaluate(
                rte,
                model,
                source,
                args.maxiter,
                args.fgmres_maxiter,
                args.tol,
                learned_label=learned_label,
            ):
                row = {
                    "case": case,
                    "split": split,
                    "train_loss_initial": history["loss"][0],
                    "train_loss_final": history["loss"][-1],
                    "train_lr_initial": history["lr"][0],
                    "train_lr_final": history["lr"][-1],
                    "size": args.size,
                    "n_angles": args.n_angles,
                    "model_type": args.model_type,
                    "residual_mode": args.residual_mode,
                    "width": args.width,
                    "blocks": args.blocks,
                    "residuals_per_step": residual_info["residuals_per_step"],
                    "rollout_bank_size": residual_info["rollout_bank_size"],
                    "delta_scale": args.delta_scale,
                    "n_rollout": args.n_rollout,
                    "n_random": args.n_random,
                    "random_modes": args.random_modes,
                    "aux_sources": args.aux_sources,
                    "weight_decay": args.weight_decay,
                    "scheduler": args.scheduler,
                    "scheduler_step_size": args.scheduler_step_size,
                    "scheduler_gamma": args.scheduler_gamma,
                    "scheduler_eta_min": args.scheduler_eta_min,
                    "rollout_weight": args.rollout_weight,
                    "rollout_steps": args.rollout_steps,
                    "rollout_decay": args.rollout_decay,
                    "stability_weight": args.stability_weight,
                    "contraction_target": args.contraction_target,
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

    if args.history_csv:
        history_path = ROOT / args.history_csv
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)
        print(f"wrote {history_path}")


if __name__ == "__main__":
    main()
