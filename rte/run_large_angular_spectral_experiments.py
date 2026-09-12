"""Large-scale RTE experiments for the angular-spectral learned preconditioner.

The earlier RTE experiments mostly optimized one fixed coefficient medium.  This
driver adds a finite coefficient/source dataset so that sample count and
coefficient diversity can be separated from optimizer effort.

Examples on a remote GPU server:

    python -m rte.run_large_angular_spectral_experiments --plan priority --execute

    python -m rte.run_large_angular_spectral_experiments \
        --dataset-size 4096 --regimes all --epochs 20 \
        --batch-size 64 --model-kind angular_spectral_defect2 --execute

By default an epoch is one shuffled pass, including the final partial batch.
An explicit positive steps_per_epoch caps that pass. Actual updates, sample
presentations, and effective passes are recorded. Validation selects the best
checkpoint, including the classical step-zero fallback; a separate test split
is evaluated only after selection. A failed training run or a selected step-zero
checkpoint alone does not establish a negative result about an operator class.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import tempfile
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
DATASET_VERSION = 3
CHECKPOINT_VERSION = 2


@dataclass
class ExperimentConfig:
    name: str = "angular_ensemble"
    dataset_size: int = 4096
    dataset_pool_size: int = 0
    eval_size: int = 32
    test_size: int = 32
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
    steps_per_epoch: int = 0
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
    if len(set(values)) != len(values):
        raise ValueError("regimes must not contain duplicates")
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
    if cfg.dataset_pool_size:
        if cfg.dataset_pool_size < count:
            raise ValueError("dataset_pool_size must cover the requested sample count")
        pool = make_medium_dataset(replace(cfg, dataset_pool_size=0), cfg.dataset_pool_size, seed)
        data = {key: value[:count].clone() if torch.is_tensor(value) else value for key, value in pool.items()}
        data["metadata"] = dataset_metadata(cfg, count, seed)
        return data
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    dtype = torch.float32
    size = int(cfg.size)
    x = torch.arange(size, dtype=dtype).view(1, size, 1) / size
    y = torch.arange(size, dtype=dtype).view(1, 1, size) / size
    regimes = parse_regimes(cfg.regimes)
    local_ids = torch.randint(len(regimes), (count,), generator=generator)
    regime_ids = torch.tensor([REGIMES.index(regime) for regime in regimes])[local_ids]
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
    channel_phase = (2.0 * math.pi * torch.rand(count, generator=generator)).view(-1, 1, 1)
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
        "metadata": dataset_metadata(cfg, count, seed),
    }


def dataset_metadata(cfg, count, seed):
    return {
        "version": DATASET_VERSION,
        "count": int(count),
        "size": int(cfg.size),
        "n_angles": int(cfg.n_angles),
        "seed": int(seed),
        "pool_size": int(cfg.dataset_pool_size or count),
        "regimes": list(parse_regimes(cfg.regimes)),
        "dtype": "float32",
    }


def validate_dataset(data, expected):
    if data.get("metadata") != expected:
        raise ValueError("dataset cache metadata mismatch; regenerate the cache")
    count, size = expected["count"], expected["size"]
    shapes = {"sigma_t": (count, 1, size, size), "sigma_s": (count, 1, size, size),
              "centers": (count, 2), "angular_codes": (count,), "phases": (count,),
              "amplitudes": (count,), "regime_ids": (count,)}
    for key, shape in shapes.items():
        value = data.get(key)
        dtype = torch.int64 if key in ("angular_codes", "regime_ids") else torch.float32
        if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"invalid cached {key}: expected {shape}, {dtype}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite cached {key}")
    if tuple(data.get("regimes", ())) != tuple(expected["regimes"]):
        raise ValueError("cached regime labels mismatch")
    allowed = torch.tensor([REGIMES.index(regime) for regime in expected["regimes"]])
    if not bool(torch.isin(data["regime_ids"], allowed).all()):
        raise ValueError("invalid cached global regime IDs")
    if not bool(((data["angular_codes"] >= 0) & (data["angular_codes"] <= 2)).all()):
        raise ValueError("invalid cached angular source codes")
    if not bool(((data["centers"] >= 0) & (data["centers"] <= 1)).all()) or not bool((data["amplitudes"] > 0).all()):
        raise ValueError("invalid cached source parameters")
    if not bool(((data["sigma_t"] > 0) & (data["sigma_s"] >= 0) & (data["sigma_s"] < data["sigma_t"])).all()):
        raise ValueError("invalid cached subcritical coefficients")


def dataset_fingerprint(data):
    digest = hashlib.sha256(json.dumps(data["metadata"], sort_keys=True).encode())
    for key in sorted(data):
        if torch.is_tensor(data[key]):
            digest.update(key.encode())
            digest.update(data[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_or_load_dataset(cfg, count, seed):
    if not cfg.save_dataset:
        return make_medium_dataset(cfg, count, seed)
    path = Path(cfg.save_dataset)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_dataset(payload, dataset_metadata(cfg, count, seed))
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


def epoch_batches(cfg, generator):
    permutation = torch.randperm(cfg.dataset_size, generator=generator)
    batches = permutation.split(cfg.batch_size)
    return batches[:cfg.steps_per_epoch] if cfg.steps_per_epoch else batches


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
                psi = psi + cfg.residual_teacher_relaxation * rte.apply_preconditioner(residual, kind="transport_dsa")
        stages = torch.randint(depth + 1, (source.shape[0],), generator=generator).to(device=source.device)
        batch_indices = torch.arange(source.shape[0], device=source.device)
        residual = torch.stack(trajectory, dim=0)[stages, batch_indices]
        random_mask = torch.rand(source.shape[0], generator=generator).to(device=source.device) < float(cfg.random_probe_fraction)
        if bool(random_mask.any()):
            probe = torch.randn(source.shape, dtype=source.dtype, generator=generator).to(source.device)
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
            extended_coeff_features=True,
            scale_aware_gates=True,
            gate_scale=cfg.angular_gate_scale,
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


def teacher_apply(rte, residual, cfg):
    return inner_base_apply(rte, residual, depth=cfg.teacher_depth, kind="transport_dsa",
                            relaxation=cfg.residual_teacher_relaxation)


class ClassicalStackPreconditioner(torch.nn.Module):
    """Parameter-free unit-relaxation DSA stack, without any network evaluation."""

    def __init__(self, rte, depth):
        super().__init__()
        self.rte = rte
        self.depth = int(depth)
        self.base_kind = "transport_dsa"
        self.base_calls_per_apply = self.depth

    def base(self, residual):
        return self.rte.apply_preconditioner(residual, kind=self.base_kind)

    def metric_base(self, residual):
        return self.base(residual)

    def forward(self, residual):
        return inner_base_apply(self.rte, residual, depth=self.depth, kind=self.base_kind)


def training_objective(model, residual, cfg, teacher_weight):
    return objective_components(
        model, residual, cfg.loss_mode, metric_map="base", physical_weight=0.0,
        rollout_weight=cfg.rollout_weight, rollout_steps=cfg.rollout_steps,
        rollout_decay=cfg.rollout_decay, correction_weight=1.0e-6,
        teacher_fn=lambda r: teacher_apply(model.rte, r, cfg),
        teacher_weight=teacher_weight, teacher_action_weight=cfg.teacher_action_weight,
    )


def evaluation_indices(data, count):
    """Fixed round-robin selection across the represented coefficient regimes."""
    generator = torch.Generator().manual_seed(data["metadata"]["seed"] + 71)
    groups = []
    for regime_id in sorted(data["regime_ids"].unique().tolist()):
        group = torch.where(data["regime_ids"] == regime_id)[0]
        groups.append(group[torch.randperm(len(group), generator=generator)].tolist())
    selected = []
    while len(selected) < min(count, len(data["regime_ids"])):
        for group in groups:
            if group and len(selected) < count:
                selected.append(group.pop())
    return selected


@torch.no_grad()
def evaluate(model, rte, data, cfg, device):
    """Full-weight objective on fixed probes; true solver residuals on sources.

    No training RNG is consumed. Cases are retained so aggregate results do not
    conceal regime failures. These are validation metrics, not an unbiased test
    once used to select a checkpoint.
    """
    was_training = model.training
    old_medium = (rte.sigma_t, rte.sigma_s, rte.source)
    model.eval()
    cases = []
    try:
        for idx in evaluation_indices(data, cfg.eval_count):
            source = setup_batch(cfg, rte, data, torch.tensor([idx]), device)
            generator = torch.Generator().manual_seed(data["metadata"]["seed"] + 107 + idx)
            residual = make_training_residuals(rte, source, cfg, generator)
            loss, scalars = training_objective(model, residual, cfg, cfg.teacher_weight)
            source_defect = source - rte.A_op(model(source))
            source_ratio = float(torch.linalg.norm(source_defect) / torch.linalg.norm(source).clamp_min(1e-30))
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            solver_result = solve_fgmres(
                rte, source, model, maxiter=cfg.fgmres_maxiter, tol=cfg.tol,
                real_stacked=cfg.fgmres_mode == "real-stacked",
            )
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            solve_seconds = time.perf_counter() - start
            iters, final = solver_result[:2]
            preconditioner_calls = int(solver_result[6][-1])
            case = {
                "index": idx, "regime": REGIMES[int(data["regime_ids"][idx])],
                "objective": float(loss), "physical_ratio": scalars["raw_ratio"],
                "source_one_step_ratio": source_ratio, "fgmres_ratio": float(final),
                "fgmres_iters": int(iters), "fgmres_success": float(final < cfg.tol),
                "solve_seconds": solve_seconds, "preconditioner_calls": preconditioner_calls,
                "base_calls": preconditioner_calls * model.base_calls_per_apply,
            }
            if not all(math.isfinite(value) for value in case.values() if isinstance(value, (int, float))):
                raise FloatingPointError(f"nonfinite evaluation on sample {idx}")
            cases.append(case)
    finally:
        rte.setup(sigma_t=old_medium[0], sigma_s=old_medium[1], source=old_medium[2])
        model.train(was_training)
    if not cases:
        raise ValueError("evaluation requires at least one sample")

    def mean(key, power=1):
        return sum(case[key] ** power for case in cases) / len(cases)

    metrics = {
        "count": len(cases), "objective": mean("objective"),
        "physical_mean_ratio": mean("physical_ratio"),
        "physical_mean_square": mean("physical_ratio", 2),
        "source_one_step_mean_ratio": mean("source_one_step_ratio"),
        "source_one_step_mean_square": mean("source_one_step_ratio", 2),
        "fgmres_mean_ratio": mean("fgmres_ratio"),
        "fgmres_mean_square": mean("fgmres_ratio", 2),
        "fgmres_mean_iters": mean("fgmres_iters"),
        "fgmres_success_rate": mean("fgmres_success"),
        "fgmres_worst_ratio": max(case["fgmres_ratio"] for case in cases),
        "solve_seconds_mean": mean("solve_seconds"),
        "preconditioner_calls_mean": mean("preconditioner_calls"),
        "base_calls_mean": mean("base_calls"),
    }
    return metrics, cases


def write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_config(cfg):
    for key in ("dataset_size", "eval_size", "test_size", "size", "n_angles", "width",
                "levels", "batch_size", "eval_every", "eval_count", "fgmres_maxiter",
                "teacher_depth", "residual_teacher_depth", "angular_mode_width", "angular_mix_rank"):
        if getattr(cfg, key) < 1:
            raise ValueError(f"{key} must be positive")
    if cfg.size < 2 or cfg.size < 2 ** (cfg.levels - 1):
        raise ValueError("size must be >= 2 and support the requested UNet levels")
    if cfg.epochs < 0 or cfg.steps_per_epoch < 0 or cfg.rollout_steps < 0 or cfg.teacher_ramp_epochs < 0:
        raise ValueError("epoch, rollout, and ramp counts must be nonnegative")
    if cfg.steps_per_epoch > math.ceil(cfg.dataset_size / cfg.batch_size):
        raise ValueError("steps_per_epoch is a cap and cannot exceed one full pass")
    if cfg.fgmres_maxiter > 60:
        raise ValueError("the supporting FGMRES harness permits at most 60 iterations (three 20-step cycles)")
    if cfg.dataset_pool_size < 0 or (cfg.dataset_pool_size and cfg.dataset_pool_size < cfg.dataset_size):
        raise ValueError("dataset_pool_size must be zero or cover dataset_size")
    if not 0 <= cfg.random_probe_fraction <= 1 or cfg.random_modes < 0:
        raise ValueError("invalid random probe settings")
    if not abs(cfg.anisotropy_g) < 1 or cfg.eta_scale < 0:
        raise ValueError("invalid transport parameters")
    for key in ("lr", "tol", "grad_clip", "residual_teacher_relaxation"):
        if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("weight_decay", "teacher_weight", "teacher_action_weight", "rollout_weight", "rollout_decay"):
        if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 <= cfg.scheduler_min_lr_ratio <= 1:
        raise ValueError("scheduler_min_lr_ratio must be in [0, 1]")
    if cfg.scheduler not in ("none", "cosine") or cfg.loss_mode not in ("raw_rollout", "raw_rollout_teacher", "solver_teacher"):
        raise ValueError("unsupported scheduler or loss mode")
    parse_regimes(cfg.regimes)


def resolve_device(value):
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested {value}, but CUDA is unavailable; explicitly use --device cpu")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"requested CUDA device {device.index} does not exist")
    elif device.type != "cpu":
        raise ValueError("only CPU and CUDA are supported by this runner")
    return device


def runtime_metadata(device):
    return {
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
        "num_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def capture_rng(generator, device):
    return {
        "python": random.getstate(), "torch_cpu": torch.get_rng_state(),
        "sampler": generator.get_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }


def restore_rng(state, generator, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["sampler"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_resume(payload, cfg, fingerprint, runtime):
    if payload.get("selection_only"):
        raise ValueError("best.pt is for inference; use last.pt for exact resume")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version; legacy checkpoints cannot resume exactly")
    ignored = {"resume", "output_root", "name", "save_dataset"}
    changed = [key for key, value in asdict(cfg).items()
               if key not in ignored and payload["config"].get(key) != value]
    if changed:
        raise ValueError(f"resume requires the same configuration and horizon; changed: {changed}")
    if payload["dataset_fingerprint"] != fingerprint:
        raise ValueError("resume dataset fingerprint mismatch")
    if payload["runtime"] != runtime:
        raise ValueError("resume runtime mismatch; exact continuation requires the same environment")


def run_experiment(cfg):
    validate_config(cfg)
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    runtime = runtime_metadata(device)
    output_dir = Path(cfg.output_root) / cfg.name
    if not cfg.resume and any((output_dir / name).exists() for name in ("config.json", "last.pt", "best.pt", "result.json")):
        raise FileExistsError(f"run already exists at {output_dir}; resume or choose a new name")
    train_data = save_or_load_dataset(cfg, cfg.dataset_size, cfg.seed)
    fingerprint = dataset_fingerprint(train_data)
    split_cfg = replace(cfg, dataset_pool_size=0)
    val_data = make_medium_dataset(split_cfg, cfg.eval_size, cfg.seed + 100003)
    first = torch.arange(min(cfg.batch_size, cfg.dataset_size))
    first_source = source_from_data(cfg, train_data, first, device)
    rte = make_operator(cfg, train_data["sigma_t"][first], train_data["sigma_s"][first], first_source, device)
    model = make_model(cfg, rte).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = cfg.steps_per_epoch or math.ceil(cfg.dataset_size / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    scheduler = make_scheduler(optimizer, cfg, total_steps)
    index_generator = torch.Generator().manual_seed(cfg.seed + 19)
    initial_model = cpu_tree(model.state_dict())
    start_epoch = 0
    updates = presentations = 0
    history = []
    elapsed_before = 0.0
    best = None
    baseline = None
    payload = None
    if cfg.resume:
        payload = torch.load(cfg.resume, map_location="cpu", weights_only=True)
        validate_resume(payload, cfg, fingerprint, runtime)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = payload["epoch"]
        updates, presentations = payload["updates"], payload["sample_presentations"]
        history, best, baseline = payload["history"], payload["best"], payload["baseline"]
        elapsed_before = payload["elapsed_seconds"]
        restore_rng(payload["rng"], index_generator, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps({"config": asdict(cfg), "runtime": runtime,
                                     "dataset_fingerprint": fingerprint}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    start_time = time.perf_counter()

    def elapsed():
        return elapsed_before + time.perf_counter() - start_time

    def assess(row):
        records = {}
        for split, data in (("train", train_data), ("val", val_data)):
            metrics, cases = evaluate(model, rte, data, cfg, device)
            row.update({f"{split}_{key}": value for key, value in metrics.items()})
            records[split] = cases
        return records

    def pack_checkpoint(epoch):
        return {
            "version": CHECKPOINT_VERSION, "config": asdict(cfg), "runtime": runtime,
            "dataset_fingerprint": fingerprint, "model": cpu_tree(model.state_dict()),
            "optimizer": cpu_tree(optimizer.state_dict()),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "epoch": epoch, "updates": updates, "sample_presentations": presentations,
            "rng": capture_rng(index_generator, device), "history": history,
            "best": best, "baseline": baseline, "elapsed_seconds": elapsed(),
        }

    def save_best():
        save_checkpoint(best_path, {"version": CHECKPOINT_VERSION, "config": asdict(cfg),
                                    "runtime": runtime, "dataset_fingerprint": fingerprint,
                                    "model": best["model"], "epoch": best["epoch"],
                                    "score": best["score"], "metrics": best["metrics"],
                                    "selection_only": True})

    if payload is None:
        row = {"epoch": 0, "updates": 0, "sample_presentations": 0, "effective_passes": 0.0}
        baseline = assess(row)
        best = {"epoch": 0, "score": row["val_fgmres_mean_ratio"],
                "model": initial_model, "metrics": dict(row), "cases": baseline}
        row["elapsed"] = elapsed()
        history.append(row)
        save_best()
        save_checkpoint(last_path, pack_checkpoint(0))
        write_history(output_dir / "history.csv", history)
    else:
        # The selected weights travel with last.pt, so resume may use a new directory.
        save_best()
        save_checkpoint(last_path, payload)
        write_history(output_dir / "history.csv", history)

    for epoch in range(start_epoch, int(cfg.epochs)):
        model.train()
        totals = {"batch_objective": 0.0, "batch_physical_mean_ratio": 0.0,
                  "batch_teacher_ratio": 0.0, "batch_teacher_action_ratio": 0.0}
        epoch_presentations = 0
        for indices in epoch_batches(cfg, index_generator):
            source = setup_batch(cfg, rte, train_data, indices, device)
            residual = make_training_residuals(rte, source, cfg, index_generator)
            teacher_weight = cfg.teacher_weight * min(
                1.0,
                float(updates + 1) / max(1, cfg.teacher_ramp_epochs * steps_per_epoch),
            )

            optimizer.zero_grad(set_to_none=True)
            loss, scalars = training_objective(model, residual, cfg, teacher_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch + 1}, update {updates + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            batch_count = len(indices)
            updates += 1
            presentations += batch_count
            epoch_presentations += batch_count
            for key, value in zip(totals, (float(loss.detach()), scalars["raw_ratio"],
                                          scalars["teacher_ratio"], scalars["teacher_action_ratio"])):
                totals[key] += batch_count * value

        row = {
            "epoch": epoch + 1, "updates": updates, "sample_presentations": presentations,
            "effective_passes": presentations / cfg.dataset_size,
            **{key: value / epoch_presentations for key, value in totals.items()},
            "teacher_weight": teacher_weight, "lr": optimizer.param_groups[0]["lr"],
        }
        due = (epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.epochs
        improved = False
        if due:
            records = assess(row)
            if row["val_fgmres_mean_ratio"] < best["score"]:
                improved = True
                best = {"epoch": epoch + 1, "score": row["val_fgmres_mean_ratio"],
                        "model": cpu_tree(model.state_dict()), "metrics": dict(row), "cases": records}
        row["elapsed"] = elapsed()
        history.append(row)
        checkpoint = pack_checkpoint(epoch + 1)
        if improved:
            save_best()
        save_checkpoint(last_path, checkpoint)
        write_history(output_dir / "history.csv", history)
        if due:
            print(f"[{cfg.name}] epoch={epoch + 1}/{cfg.epochs} updates={updates} "
                  f"passes={row['effective_passes']:.3f} train={row['train_physical_mean_square']:.3e} "
                  f"val={row['val_physical_mean_square']:.3e} best_fgmres={best['score']:.3e}")

    # The test split is never used by assess(), the scheduler, or checkpoint selection.
    test_data = make_medium_dataset(split_cfg, cfg.test_size, cfg.seed + 200003)
    test_cfg = replace(cfg, eval_count=cfg.test_size)
    model.load_state_dict(best["model"])
    test_metrics, test_cases = evaluate(model, rte, test_data, test_cfg, device)
    classical2 = ClassicalStackPreconditioner(rte, depth=2)
    classical4 = ClassicalStackPreconditioner(rte, depth=4)
    classical_metrics, classical_cases = evaluate(classical2, rte, test_data, test_cfg, device)
    classical4_metrics, classical4_cases = evaluate(classical4, rte, test_data, test_cfg, device)
    result = {
        "name": cfg.name, "dataset_size": cfg.dataset_size, "epochs": cfg.epochs,
        "steps_per_epoch": cfg.steps_per_epoch, "resolved_steps_per_epoch": steps_per_epoch,
        "updates": updates, "sample_presentations": presentations,
        "effective_passes": presentations / cfg.dataset_size,
        "best_epoch": best["epoch"], "best_validation_fgmres": best["score"],
        "test": test_metrics, "test_cases": test_cases,
        "classical_defect2_test": classical_metrics, "classical_defect2_test_cases": classical_cases,
        "classical_dsa4_test": classical4_metrics, "classical_dsa4_test_cases": classical4_cases,
        "baseline_train_val_cases": baseline, "selected_train_val_cases": best["cases"],
        "elapsed_seconds": elapsed(), "runtime": runtime,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "last_checkpoint": str(last_path), "best_checkpoint": str(best_path), "config": str(config_path),
        "interpretation": "Finite-sample validation selects a checkpoint, not a theorem or a negative result about model capacity. Test results require paired replicated controls before general claims.",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def priority_plan(args):
    """Exploratory ensemble controls, not a reinterpretation of fixed-medium trials.

    All arms use full passes and the same update horizon. The small-data arm
    sees a strict prefix of the large pool. Evaluation and teacher schedules
    are aligned in updates, not in differently sized epochs. Equal updates are
    not equal wall time or parameter count.
    """
    cfg = command_line_config(args)
    if cfg.resume or cfg.save_dataset:
        raise ValueError("resume/cache paths apply to a single run; use --plan single")
    if cfg.steps_per_epoch:
        raise ValueError("priority controls require full passes: --steps-per-epoch 0")
    if cfg.dataset_size % cfg.batch_size or cfg.dataset_size < 2 * cfg.batch_size:
        raise ValueError("priority dataset_size must be a batch-size multiple with at least two batches")
    large_steps = cfg.dataset_size // cfg.batch_size
    small_steps = max(step for step in range(1, min(4, large_steps // 2) + 1) if large_steps % step == 0)
    ratio = large_steps // small_steps
    cfg = replace(cfg, epochs=max(1000, cfg.epochs), steps_per_epoch=0, regimes="all",
                  model_kind="angular_spectral_defect2", loss_mode="solver_teacher",
                  dataset_pool_size=cfg.dataset_size)
    return [
        replace(cfg, name="ensemble_small_angular_teacher", dataset_size=small_steps * cfg.batch_size,
                epochs=cfg.epochs * ratio, teacher_ramp_epochs=cfg.teacher_ramp_epochs * ratio,
                eval_every=cfg.eval_every * ratio),
        replace(cfg, name="ensemble_large_angular_teacher"),
        replace(cfg, name="ensemble_large_old_unet_teacher", model_kind="unet_defect2"),
        replace(cfg, name="ensemble_large_angular_raw", loss_mode="raw_rollout", teacher_weight=0.0,
                rollout_weight=0.0),
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
    parser.add_argument("--dataset-pool-size", type=int, default=0, help="Generate this common pool and take a prefix; 0 uses dataset-size.")
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=32)
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
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0: full shuffled pass; positive: cap batches per pass.")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--scheduler-min-lr-ratio", type=float, default=1.0e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss-mode", choices=["raw_rollout", "raw_rollout_teacher", "solver_teacher"], default="solver_teacher")
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
    parser.add_argument("--resume", default="", help="Resume last.pt at an epoch boundary with an unchanged horizon.")
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
