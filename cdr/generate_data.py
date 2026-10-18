"""Generate coefficient/source samples without solving for target solutions.

The bicubic fields, finite-difference velocity and Gaussian blobs reproduce
the distribution construction of the original scripts. They are not forced
to match across periodic boundaries; see ``cdr/README.md`` for this limitation.
"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn import functional as F

from cdr.utils import select_device


def build_ill_config(ill_level):
    level = float(np.clip(ill_level, 0.0, 1.0))
    return {
        "kappa_grf_scale": 0.2 + 0.8 * level,
        "kappa_floor": 0.5 - 0.45 * level,
        "psi_scale": 0.01 + 0.08 * level,
        "num_sources_range": (1, 2) if level < 0.4 else (2, 4),
        "amplitude_range": (10.0 + 10.0 * level, 20.0 + 40.0 * level),
        "std_range": (max(0.005, 0.02 - 0.015 * level), max(0.015, 0.10 - 0.06 * level)),
    }


def finite_difference_derivative(field, dx, axis):
    """Central interior and one-sided boundary derivatives from the source code."""
    if field.ndim != 2 or min(field.shape) < 2 or dx <= 0 or axis not in (0, 1):
        raise ValueError("expect a 2D field of size >= 2, positive dx, and axis 0 or 1")
    moved = field.movedim(axis, 0)
    derivative = torch.empty_like(moved)
    derivative[1:-1] = (moved[2:] - moved[:-2]) / (2 * dx)
    derivative[0] = (moved[1] - moved[0]) / dx
    derivative[-1] = (moved[-1] - moved[-2]) / dx
    return derivative.movedim(0, axis)


def generate_grf(low_resolution, resolution, scale, device, generator):
    noise = torch.randn(1, 1, low_resolution, low_resolution, device=device, generator=generator)
    field = F.interpolate(noise, size=(resolution, resolution), mode="bicubic", align_corners=False)[0, 0]
    return scale * (field - field.mean()) / field.std().clamp_min(1e-8)


def generate_blob_sources(resolution, config, device, rng):
    grid = torch.linspace(0, 1, resolution, device=device)
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    field = torch.zeros((resolution, resolution), device=device)
    for _ in range(rng.integers(*config["num_sources_range"])):
        cx, cy = rng.random(2)
        amplitude = rng.uniform(*config["amplitude_range"])
        std = rng.uniform(*config["std_range"])
        field += amplitude * torch.exp(-((x - cx).square() + (y - cy).square()) / std**2)
    return field


def generate_dataset(save_path, num_samples, resolution=128, low_resolution=16,
                     ill_level=0.4, seed=42, device="cpu"):
    """Write a new dataset directory and return its path.

    Existing sample files are rejected to avoid mixing experiments. Seeds are
    local to this call; train and test generation should use different seeds.
    """
    if num_samples <= 0 or resolution < 4 or low_resolution < 2 or low_resolution > resolution:
        raise ValueError("need samples > 0 and 2 <= low_resolution <= resolution with resolution >= 4")
    if not 0 <= ill_level <= 1:
        raise ValueError("ill_level must lie in [0, 1]")
    destination = Path(save_path)
    if destination.exists() and any(destination.glob("sample_*.pt")):
        raise FileExistsError(f"{destination} already contains samples; choose a new output directory")
    destination.mkdir(parents=True, exist_ok=True)
    device = select_device(str(device))
    generator = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    for index in range(num_samples):
        config = build_ill_config(rng.uniform(0.8, 1.2) * ill_level)
        kappa = config["kappa_floor"] * (
            torch.exp(generate_grf(low_resolution, resolution, config["kappa_grf_scale"], device, generator))
            + 1.0
        )
        psi = generate_grf(low_resolution, resolution, config["psi_scale"], device, generator)
        # Preserve the research generator's channel ordering. This ordering
        # does not make v spectrally divergence-free under the PDE convention.
        velocity = torch.stack((
            finite_difference_derivative(psi, 1.0 / resolution, axis=0),
            -finite_difference_derivative(psi, 1.0 / resolution, axis=1),
        ), dim=-1)
        rhs = generate_blob_sources(resolution, config, device, rng)
        torch.save({"kappa": kappa.cpu().float(), "v": velocity.cpu().float(), "f": rhs.cpu().float()},
                   destination / f"sample_{index:04d}.pt")
    metadata = {
        "format_version": 1, "num_samples": num_samples, "resolution": resolution,
        "low_resolution": low_resolution, "ill_level": ill_level, "seed": seed,
        "reaction": 1.0, "generator": "legacy_bicubic_fd",
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new dataset root (creates train/ and test/)")
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--test-samples", type=int, default=400)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--low-resolution", type=int, default=16)
    parser.add_argument("--ill-level", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args(argv)
    for split, count, seed in (("train", args.train_samples, args.seed), ("test", args.test_samples, args.seed + 1)):
        path = generate_dataset(args.output / split, count, args.resolution, args.low_resolution,
                                args.ill_level, seed, args.device)
        print(f"Generated {count} {split} samples: {path}")


if __name__ == "__main__":
    main()
