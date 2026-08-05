"""Large-scale RTE experiments for the angular-spectral learned preconditioner.

The earlier RTE experiments mostly optimized one fixed coefficient medium.  This
driver adds a finite coefficient/source dataset so that sample count and
coefficient diversity can be separated from optimizer effort.

Examples on a remote GPU server:

    python -m rte.run_large_angular_spectral_experiments --plan priority --execute

    python -m rte.run_large_angular_spectral_experiments \
        --dataset-size 4096 --regimes all --epochs 2000 --steps-per-epoch 2 \
        --batch-size 64 --model-kind angular_spectral_defect2 --execute

An epoch is ``steps_per_epoch`` independent minibatch updates.  The script
records both the finite dataset size and the number of sample presentations so
that "thousands of epochs" is not confused with a single repeatedly sampled
residual batch.
"""

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from rte.models import PeriodicRTE2D
from rte.npbs_bottleneck_harness import objective_components, solve_fgmres
from rte.npbs_unet_ablation import (
    AngularSpectralDefect2Preconditioner,
    NPBSUNetPreconditioner,
    inner_base_apply,
)


REGIMES = ("diffusive", "heterogeneous", "anisotropic", "mixed")


@dataclass
class ExperimentConfig:
    name: str = "angular_ensemble"
    dataset_size: int = 4096
    eval_size: int = 32
    size: int = 64
    n_angles: int = 12
    regimes: str = "all"
    anisotropy_g: float = 0.65
    eta_scale: float = 0.1
    model_kind: str = "angular_spectral_defect2"
    width: int = 16
    levels: int = 3
    angular_mode_width: int = 4
    angular_mix_rank: int = 4
    angular_mix_scale: float = 0.25
    angular_mode_mix_scale: float = 0.05
    angular_residual_gate_scale: float = 0.25
    angular_gate_scale: float = 0.1
    defect_relaxation: float = 1.0
    batch_size: int = 64
    epochs: int = 2000
    steps_per_epoch: int = 2
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    scheduler: str = "cosine"
    scheduler_min_lr_ratio: float = 1.0e-2
    grad_clip: float = 1.0
    loss_mode: str = "solver_teacher"
    rollout_weight: float = 0.2
    rollout_steps: int = 2
    rollout_decay: float = 0.8
    teacher_depth: int = 4
    teacher_weight: float = 0.25
    teacher_action_weight: float = 1.0
    teacher_ramp_epochs: int = 200
    residual_teacher_depth: int = 4
    residual_teacher_relaxation: float = 1.0
    random_modes: int = 5
    random_probe_fraction: float = 0.5
    eval_every: int = 100
    eval_count: int = 8
    fgmres_maxiter: int = 20
    fgmres_mode: str = "real-stacked"
    tol: float = 1.0e-6
    seed: int = 20260804
    device: str = "cuda:0"
    output_root: str = "rte/large_angular_spectral_runs"
    resume: str = ""
    save_dataset: str = ""


def parse_regimes(value):
    values = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    if values == ["all"]:
        return REGIMES
    unknown = sorted(set(values) - set(REGIMES))
    if unknown:
        raise ValueError(f"unknown regimes: {unknown}; use all or a comma-separated subset of {REGIMES}")
    if not values:
        raise ValueError("at least one regime is required")
    return tuple(values)


def make_source_batch(size, n_angles, dtype, centers, angular_codes, phases, amplitudes):
    """Create a batch of localized RHS fields from compact source parameters."""
    device = centers.device
    x = torch.arange(size, dtype=dtype, device=device) / size
    y = torch.arange(size, dtype=dtype, device=device) / size
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    source_xy = torch.exp(
        -100.0 * ((xx.view(1, 1, size, size) - centers[:, 0].view(-1, 1, 1, 1)) ** 2
                   + (yy.view(1, 1, size, size) - centers[:, 1].view(-1, 1, 1, 1)) ** 2)
    )
    angles = torch.arange(n_angles, dtype=dtype, device=device).view(1, n_angles, 1, 1)
    angles = angles * (2.0 * math.pi / n_angles)
    phase = phases.view(-1, 1, 1, 1)
    beam = torch.exp(2.0 * torch.cos(angles - phase))
    beam = beam / beam.mean(dim=1, keepdim=True)
    dipole = 1.0 + 0.7 * torch.cos(angles - phase)
    isotropic = torch.ones_like(beam)
    code = angular_codes.view(-1, 1, 1, 1)
    angular = torch.where(code == 2, beam, torch.where(code == 1, dipole, isotropic))
    source = amplitudes.view(-1, 1, 1, 1) * angular * source_xy
    return torch.complex(source, torch.zeros_like(source))


def make_medium_dataset(cfg, count, seed, device="cpu"):
    """Generate coefficient fields and compact source parameters on CPU."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    dtype = torch.float32
    size = int(cfg.size)
    x = torch.arange(size, dtype=dtype).view(1, size, 1) / size
    y = torch.arange(size, dtype=dtype).view(1, 1, size) / size
    regimes = parse_regimes(cfg.regimes)
    regime_ids = torch.randint(len(regimes), (count,), generator=generator)
    centers = torch.rand((count, 2), generator=generator) * 0.72 + 0.14
    phases = torch.rand(count, generator=generator) * (2.0 * math.pi)
    amplitudes = 0.6 + 0.8 * torch.rand(count, generator=generator)
    angular_codes = torch.randint(0, 3, (count,), generator=generator)

    medium_centers_x = torch.rand(count, generator=generator) * 0.56 + 0.22
    medium_centers_y = torch.rand(count, generator=generator) * 0.56 + 0.22
    width = 0.025 + 0.11 * torch.rand(count, generator=generator)
    cx = medium_centers_x.view(-1, 1, 1)
    cy = medium_centers_y.view(-1, 1, 1)
    inclusion = torch.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / width.view(-1, 1, 1).square()))
    channel_phase = phases.view(-1, 1, 1)
    channel = (torch.sin(2.0 * math.pi * y + channel_phase) > (-0.3 + 0.6 * torch.rand(count, generator=generator).view(-1, 1, 1))).to(dtype)
    interface = x < (0.40 + 0.16 * torch.rand(count, generator=generator).view(-1, 1, 1))
    roughness = 0.75 + 0.5 * torch.rand((count, 1, 1), generator=generator)

    sigma_diff = (12.0 + 16.0 * torch.rand((count, 1, 1), generator=generator)) * (1.0 + 0.04 * (roughness - 1.0) * inclusion)
    albedo_diff = 0.965 + 0.03 * torch.rand((count, 1, 1), generator=generator)

    sigma_het = 0.8 + 2.0 * torch.rand((count, 1, 1), generator=generator) + (5.0 + 10.0 * torch.rand((count, 1, 1), generator=generator)) * inclusion + (1.0 + 3.0 * torch.rand((count, 1, 1), generator=generator)) * channel
    albedo_het = 0.76 + 0.21 * torch.rand((count, 1, 1), generator=generator)

    sigma_aniso = (2.0 + 7.0 * torch.rand((count, 1, 1), generator=generator)) * (1.0 + 0.25 * inclusion)
    albedo_aniso = 0.90 + 0.09 * torch.rand((count, 1, 1), generator=generator)

    thin = 0.20 + 0.95 * torch.rand((count, 1, 1), generator=generator)
    thick = 8.0 + 14.0 * torch.rand((count, 1, 1), generator=generator)
    sigma_mixed = torch.where(interface, thin, thick) + (0.5 + 2.5 * torch.rand((count, 1, 1), generator=generator)) * inclusion
    albedo_mixed = torch.where(
        interface,
        0.10 + 0.30 * torch.rand((count, 1, 1), generator=generator),
        0.94 + 0.05 * torch.rand((count, 1, 1), generator=generator),
    )
    albedo_diff = albedo_diff.expand(-1, size, size)
    albedo_het = albedo_het.expand(-1, size, size)
    albedo_aniso = albedo_aniso.expand(-1, size, size)
    albedo_mixed = albedo_mixed.expand(-1, size, size)

    sigma_stack = torch.stack([sigma_diff, sigma_het, sigma_aniso, sigma_mixed], dim=1)
    albedo_stack = torch.stack([albedo_diff, albedo_het, albedo_aniso, albedo_mixed], dim=1)
    selected = regime_ids.view(-1, 1, 1, 1).expand(-1, 1, size, size)
    sigma_t = torch.gather(sigma_stack, 1, selected).squeeze(1).unsqueeze(1)
    albedo = torch.gather(albedo_stack, 1, selected).squeeze(1).unsqueeze(1)
    sigma_s = torch.clamp(albedo * sigma_t, max=0.995 * sigma_t)

    # The operator currently has one angular phase matrix per run.  Keep g
    # fixed within an ensemble and vary sigma_t, albedo, geometry, and sources.
    # Store only compact source parameters; materializing all complex source
    # fields would add about 1.6 GB for 4096 size-64 samples.
    return {
        "sigma_t": sigma_t.contiguous(),
        "sigma_s": sigma_s.contiguous(),
        "centers": centers.contiguous(),
        "angular_codes": angular_codes.contiguous(),
        "phases": phases.contiguous(),
        "amplitudes": amplitudes.contiguous(),
        "regime_ids": regime_ids.contiguous(),
        "regimes": regimes,
    }


def save_or_load_dataset(cfg, count, seed):
    if not cfg.save_dataset:
        return make_medium_dataset(cfg, count, seed)
    path = Path(cfg.save_dataset)
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        if int(payload["sigma_t"].shape[0]) < int(count):
            raise ValueError(f"cached dataset has {payload['sigma_t'].shape[0]} samples, requested {count}")
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = make_medium_dataset(cfg, count, seed)
    torch.save(payload, path)
    return payload


def make_operator(cfg, sigma_t, sigma_s, source, device):
    rte = PeriodicRTE2D(
        shape=(cfg.size, cfg.size),
        dx=1.0 / cfg.size,
        n_angles=cfg.n_angles,
        anisotropy_g=cfg.anisotropy_g,
        eta_scale=cfg.eta_scale,
        fdtype=torch.float32,
        device=device,
    )
    rte.setup(
        sigma_t=sigma_t.to(device=device),
        sigma_s=sigma_s.to(device=device),
        source=source.to(device=device),
    )
    return rte


def choose_indices(count, batch_size, generator):
    return torch.randint(count, (batch_size,), generator=generator)


def make_training_residuals(rte, source, cfg, generator):
    """Sample residuals along a classical trajectory plus random probes."""
    with torch.no_grad():
        psi = torch.zeros_like(source)
        trajectory = []
        depth = max(1, int(cfg.residual_teacher_depth))
        for stage in range(depth + 1):
            residual = rte.source - rte.A_op(psi)
            trajectory.append(residual)
            if stage < depth:
                psi = psi + rte.apply_preconditioner(residual, kind="transport_dsa")
        stages = torch.randint(depth + 1, (source.shape[0],), generator=generator).to(device=source.device)
        batch_indices = torch.arange(source.shape[0], device=source.device)
        residual = torch.stack(trajectory, dim=0)[stages, batch_indices]
        random_mask = torch.rand(source.shape[0], generator=generator).to(device=source.device) < float(cfg.random_probe_fraction)
        if bool(random_mask.any()):
            probe = torch.randn_like(source)
            probe_hat = torch.fft.fftn(probe, dim=(-2, -1), norm="backward")
            kx = torch.fft.fftfreq(cfg.size, device=source.device).view(cfg.size, 1)
            ky = torch.fft.fftfreq(cfg.size, device=source.device).view(1, cfg.size)
            mask = ((kx.abs() * cfg.size <= cfg.random_modes) & (ky.abs() * cfg.size <= cfg.random_modes)).to(probe.dtype)
            probe = torch.fft.ifftn(probe_hat * mask.view(1, 1, cfg.size, cfg.size), dim=(-2, -1), norm="backward")
            residual = torch.where(random_mask.view(-1, 1, 1, 1), rte.A_op(probe), residual)
        return residual.detach()


def make_model(cfg, rte):
    if cfg.model_kind == "angular_spectral_defect2":
        return AngularSpectralDefect2Preconditioner(
            rte,
            base_kind="transport_dsa",
            width=cfg.width,
            levels=cfg.levels,
            mode_width=cfg.angular_mode_width,
            angular_mix_rank=cfg.angular_mix_rank,
            angular_mix_scale=cfg.angular_mix_scale,
            angular_mode_mix_scale=cfg.angular_mode_mix_scale,
            residual_gate_scale=cfg.angular_residual_gate_scale,
            gate_scale=cfg.angular_gate_scale,
            defect_relaxation=cfg.defect_relaxation,
            output_init="zero",
        )
    if cfg.model_kind == "unet_defect2":
        return NPBSUNetPreconditioner(
            rte,
            form="defect2",
            base_kind="transport_dsa",
            width=cfg.width,
            levels=cfg.levels,
            defect_relaxation=cfg.defect_relaxation,
            output_init="zero",
        )
    raise ValueError("model-kind must be angular_spectral_defect2 or unet_defect2")


def make_scheduler(optimizer, cfg, total_steps):
    if cfg.scheduler == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps),
        eta_min=cfg.lr * cfg.scheduler_min_lr_ratio,
    )


def source_from_data(cfg, data, indices, device):
    return make_source_batch(
        cfg.size,
        cfg.n_angles,
        torch.float32,
        data["centers"][indices].to(device=device),
        data["angular_codes"][indices].to(device=device),
        data["phases"][indices].to(device=device),
        data["amplitudes"][indices].to(device=device),
    )


def setup_batch(cfg, rte, data, indices, device):
    sigma_t = data["sigma_t"][indices].to(device=device)
    sigma_s = data["sigma_s"][indices].to(device=device)
    source = source_from_data(cfg, data, indices, device)
    rte.setup(sigma_t=sigma_t, sigma_s=sigma_s, source=source)
    return source


@torch.no_grad()
def evaluate(model, rte, data, cfg, device):
    count = min(int(cfg.eval_count), int(data["sigma_t"].shape[0]))
    one_step = []
    fgmres = []
    fgmres_iters = []
    for idx in range(count):
        source = setup_batch(cfg, rte, data, torch.tensor([idx]), device)
        residual = source
        correction = model(residual)
        next_residual = residual - rte.A_op(correction)
        one_step.append(float(torch.linalg.norm(next_residual) / torch.clamp(torch.linalg.norm(residual), min=1e-30)))
        iters, final, *_ = solve_fgmres(
            rte,
            source,
            model,
            maxiter=cfg.fgmres_maxiter,
            tol=cfg.tol,
            real_stacked=cfg.fgmres_mode == "real-stacked",
        )
        fgmres.append(float(final))
        fgmres_iters.append(int(iters))
    return {
        "eval_count": count,
        "heldout_one_step": sum(one_step) / max(1, len(one_step)),
        "heldout_fgmres": sum(fgmres) / max(1, len(fgmres)),
        "heldout_fgmres_iters": sum(fgmres_iters) / max(1, len(fgmres_iters)),
    }


def write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(cfg):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    output_dir = Path(cfg.output_root) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    train_data = save_or_load_dataset(cfg, cfg.dataset_size, cfg.seed)
    eval_cfg = replace(cfg, dataset_size=max(cfg.eval_size, 1), seed=cfg.seed + 100003)
    eval_data = make_medium_dataset(eval_cfg, cfg.eval_size, eval_cfg.seed)
    first = torch.arange(min(cfg.batch_size, cfg.dataset_size))
    first_source = source_from_data(cfg, train_data, first, device)
    rte = make_operator(cfg, train_data["sigma_t"][first], train_data["sigma_s"][first], first_source, device)
    model = make_model(cfg, rte).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = int(cfg.epochs) * int(cfg.steps_per_epoch)
    scheduler = make_scheduler(optimizer, cfg, total_steps)
    start_epoch = 0
    best_score = float("inf")
    checkpoint_path = output_dir / "checkpoint.pt"
    if cfg.resume:
        payload = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", 0))
        best_score = float(payload.get("best_score", best_score))

    index_generator = torch.Generator(device="cpu")
    index_generator.manual_seed(cfg.seed + 19)
    history = []
    start_time = time.perf_counter()
    for epoch in range(start_epoch, int(cfg.epochs)):
        losses = []
        raw_values = []
        teacher_values = []
        action_values = []
        for _ in range(int(cfg.steps_per_epoch)):
            indices = choose_indices(cfg.dataset_size, cfg.batch_size, index_generator)
            source = setup_batch(cfg, rte, train_data, indices, device)
            residual = make_training_residuals(rte, source, cfg, index_generator)
            teacher_weight = cfg.teacher_weight * min(
                1.0,
                float(epoch + 1) / max(1, int(cfg.teacher_ramp_epochs)),
            )

            def teacher_fn(current_residual):
                return inner_base_apply(
                    rte,
                    current_residual,
                    depth=cfg.teacher_depth,
                    kind="transport_dsa",
                    relaxation=cfg.residual_teacher_relaxation,
                )

            optimizer.zero_grad(set_to_none=True)
            loss, scalars = objective_components(
                model,
                residual,
                cfg.loss_mode,
                metric_map="base",
                physical_weight=0.0,
                rollout_weight=cfg.rollout_weight,
                rollout_steps=cfg.rollout_steps,
                rollout_decay=cfg.rollout_decay,
                correction_weight=1.0e-6,
                teacher_fn=teacher_fn,
                teacher_weight=teacher_weight,
                teacher_action_weight=cfg.teacher_action_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(float(loss.detach()))
            raw_values.append(scalars["raw_ratio"])
            teacher_values.append(scalars["teacher_ratio"])
            action_values.append(scalars["teacher_action_ratio"])

        row = {
            "epoch": epoch + 1,
            "updates": (epoch + 1) * cfg.steps_per_epoch,
            "sample_presentations": (epoch + 1) * cfg.steps_per_epoch * cfg.batch_size,
            "loss": sum(losses) / len(losses),
            "raw_ratio": sum(raw_values) / len(raw_values),
            "teacher_ratio": sum(teacher_values) / len(teacher_values),
            "teacher_action_ratio": sum(action_values) / len(action_values),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed": time.perf_counter() - start_time,
        }
        if (epoch + 1) % max(1, int(cfg.eval_every)) == 0 or epoch + 1 == int(cfg.epochs):
            metrics = evaluate(model, rte, eval_data, cfg, device)
            row.update(metrics)
            if metrics["heldout_fgmres"] < best_score:
                best_score = metrics["heldout_fgmres"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": None if scheduler is None else scheduler.state_dict(),
                        "epoch": epoch + 1,
                        "best_score": best_score,
                        "config": asdict(cfg),
                    },
                    checkpoint_path,
                )
        history.append(row)
        if (epoch + 1) % max(1, int(cfg.eval_every)) == 0 or epoch + 1 == int(cfg.epochs):
            print(
                f"[{cfg.name}] epoch={epoch + 1}/{cfg.epochs} "
                f"loss={row['loss']:.3e} raw={row['raw_ratio']:.3e} "
                f"heldout_fgmres={row.get('heldout_fgmres', float('nan')):.3e} "
                f"best={best_score:.3e}"
            )
        if (epoch + 1) % max(1, int(cfg.eval_every)) == 0 or epoch + 1 == int(cfg.epochs):
            write_history(output_dir / "history.csv", history)

    result = {
        "name": cfg.name,
        "dataset_size": cfg.dataset_size,
        "epochs": cfg.epochs,
        "steps_per_epoch": cfg.steps_per_epoch,
        "sample_presentations": cfg.epochs * cfg.steps_per_epoch * cfg.batch_size,
        "best_heldout_fgmres": best_score,
        "elapsed_seconds": time.perf_counter() - start_time,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def priority_plan(args):
    base = dict(
        size=args.size,
        n_angles=args.n_angles,
        eval_size=args.eval_size,
        device=args.device,
        output_root=args.output_root,
        seed=args.seed,
    )
    return [
        ExperimentConfig(
            name="fixed_mixed_sample_scale_4096_2000",
            dataset_size=4096,
            regimes="mixed",
            epochs=2000,
            steps_per_epoch=2,
            model_kind="angular_spectral_defect2",
            **base,
        ),
        ExperimentConfig(
            name="coefficient_ensemble_4096_2000",
            dataset_size=4096,
            regimes="all",
            epochs=2000,
            steps_per_epoch=2,
            model_kind="angular_spectral_defect2",
            **base,
        ),
        ExperimentConfig(
            name="coefficient_ensemble_unet_control_4096_2000",
            dataset_size=4096,
            regimes="all",
            epochs=2000,
            steps_per_epoch=2,
            model_kind="unet_defect2",
            width=16,
            **base,
        ),
        ExperimentConfig(
            name="coefficient_ensemble_capacity_8192_3000",
            dataset_size=8192,
            regimes="all",
            epochs=3000,
            steps_per_epoch=2,
            model_kind="angular_spectral_defect2",
            width=24,
            angular_mode_width=8,
            angular_mix_rank=4,
            batch_size=32,
            **base,
        ),
    ]


def command_line_config(args):
    values = vars(args).copy()
    for key in ("execute", "plan"):
        values.pop(key, None)
    values["name"] = args.name
    values["regimes"] = args.regimes
    return ExperimentConfig(**values)


def main():
    parser = argparse.ArgumentParser(description="Large coefficient/source dataset experiments for RTE learned preconditioners.")
    parser.add_argument("--plan", choices=["single", "priority"], default="single")
    parser.add_argument("--execute", action="store_true", help="Run the selected experiment(s); otherwise print the plan.")
    parser.add_argument("--name", default="angular_ensemble")
    parser.add_argument("--dataset-size", type=int, default=4096)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--n-angles", type=int, default=12)
    parser.add_argument("--regimes", default="all")
    parser.add_argument("--anisotropy-g", type=float, default=0.65)
    parser.add_argument("--eta-scale", type=float, default=0.1)
    parser.add_argument("--model-kind", choices=["angular_spectral_defect2", "unet_defect2"], default="angular_spectral_defect2")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--angular-mode-width", type=int, default=4)
    parser.add_argument("--angular-mix-rank", type=int, default=4)
    parser.add_argument("--angular-mix-scale", type=float, default=0.25)
    parser.add_argument("--angular-mode-mix-scale", type=float, default=0.05)
    parser.add_argument("--angular-residual-gate-scale", type=float, default=0.25)
    parser.add_argument("--angular-gate-scale", type=float, default=0.1)
    parser.add_argument("--defect-relaxation", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--steps-per-epoch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--scheduler-min-lr-ratio", type=float, default=1.0e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss-mode", choices=["raw_rollout_teacher", "solver_teacher"], default="solver_teacher")
    parser.add_argument("--rollout-weight", type=float, default=0.2)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--rollout-decay", type=float, default=0.8)
    parser.add_argument("--teacher-depth", type=int, default=4)
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    parser.add_argument("--teacher-action-weight", type=float, default=1.0)
    parser.add_argument("--teacher-ramp-epochs", type=int, default=200)
    parser.add_argument("--residual-teacher-depth", type=int, default=4)
    parser.add_argument("--residual-teacher-relaxation", type=float, default=1.0)
    parser.add_argument("--random-modes", type=int, default=5)
    parser.add_argument("--random-probe-fraction", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-count", type=int, default=8)
    parser.add_argument("--fgmres-maxiter", type=int, default=20)
    parser.add_argument("--fgmres-mode", choices=["complex", "real-stacked"], default="real-stacked")
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default="rte/large_angular_spectral_runs")
    parser.add_argument("--resume", default="")
    parser.add_argument("--save-dataset", default="")
    args = parser.parse_args()

    if args.plan == "priority":
        configs = priority_plan(args)
        print(json.dumps([asdict(item) for item in configs], indent=2, sort_keys=True))
        if not args.execute:
            return
        for config in configs:
            run_experiment(config)
        return

    config = command_line_config(args)
    print(json.dumps(asdict(config), indent=2, sort_keys=True))
    if args.execute:
        run_experiment(config)


if __name__ == "__main__":
    main()
