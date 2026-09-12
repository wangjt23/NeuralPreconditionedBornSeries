"""Search for amplified residuals of the fixed-medium, real-linear controls.

Power iteration supplies a witnessed lower bound, not a certified operator
norm or a proof of divergent iteration. Only the legacy real-linear model
and fixed classical maps are included; no nonlinear residual gates are used.
"""

import argparse
import csv
from pathlib import Path

import torch

from rte.npbs_unet_ablation import NPBSUNetPreconditioner, inner_base_apply
from rte.residual_subspace import squared_norm
from rte.run_hard_case_experiments import make_case


def residual_power_vector(operator, method, template, seed, steps=30):
    """Return a detached difficult residual for a real-linear correction map."""
    if steps < 1:
        raise ValueError("steps must be positive")
    generator = torch.Generator(device=template.device).manual_seed(seed)
    residual = torch.randn(template.shape, dtype=template.dtype, device=template.device, generator=generator)
    residual = residual / residual.norm()
    for _ in range(steps):
        residual = residual.detach().requires_grad_(True)
        error = residual - operator(method(residual))
        adjoint_action, = torch.autograd.grad(squared_norm(error).sum(), residual)
        if not torch.isfinite(adjoint_action).all():
            raise FloatingPointError("nonfinite adjoint action")
        norm = adjoint_action.norm()
        if norm == 0:
            break
        residual = adjoint_action / norm
    return residual.detach()


def amplification_probe(operator, method, template, seed, steps=30):
    residual = residual_power_vector(operator, method, template, seed, steps)
    with torch.no_grad():
        error = residual - operator(method(residual))
        amplification = float(error.norm() / residual.norm())
        angular_energy = torch.fft.fft(residual, dim=1, norm="ortho").abs().square().sum((0, 2, 3))
        modes = torch.fft.fftfreq(residual.shape[1]) * residual.shape[1]
        spatial_energy = torch.fft.fftn(residual, dim=(-2, -1), norm="ortho").abs().square().sum((0, 1))
        kx = torch.fft.fftfreq(residual.shape[-2]) * residual.shape[-2]
        ky = torch.fft.fftfreq(residual.shape[-1]) * residual.shape[-1]
        high = (kx[:, None].abs() > 5) | (ky[None, :].abs() > 5)
        features = {"angular_above_m4_fraction": float(angular_energy[modes.abs() > 4].sum() / angular_energy.sum()),
                    "scalar_angular_fraction": float(angular_energy[0] / angular_energy.sum()),
                    "spatial_above_k5_fraction": float(spatial_energy[high].sum() / spatial_energy.sum())}
        start_norm = residual.norm()
        for _ in range(12):
            residual = residual - operator(method(residual))
        twelve_steps = float(residual.norm() / start_norm)
    return amplification, twelve_steps, features


def run(directory, steps, model_kind="legacy_raw"):
    rows = []
    torch.set_num_threads(1)
    for path in sorted(directory.glob(f"*_{model_kind}.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        args = state["args"]
        seed = int(path.stem.split("_")[-3])
        rte, _ = make_case(args["case"], args["size"], args["n_angles"], torch.float32, .1)
        model = NPBSUNetPreconditioner(rte, form="defect2", base_kind=args["base_kind"],
                                      width=16, levels=3, output_init="zero")
        model.load_state_dict(state["model"])
        model.eval()
        methods = {"base2": lambda r: inner_base_apply(rte, r, depth=2, kind=args["base_kind"]),
                   "base4": lambda r: inner_base_apply(rte, r, depth=4, kind=args["base_kind"]),
                   model_kind: model}
        for name, method in methods.items():
            witnessed, final, features = amplification_probe(rte.A_op, method, rte.source, seed + 7000, steps)
            row = {"seed": seed, "method": name, "power_steps": steps,
                   "witnessed_one_step_ratio": witnessed, "witnessed_stationary_12_ratio": final,
                   "selected_step": state["selected_step"] if name == model_kind else "", **features}
            rows.append(row)
            print(row, flush=True)
    if not rows:
        raise FileNotFoundError("No locally generated legacy checkpoints found")
    filename = "amplification.csv" if model_kind == "legacy_raw" else f"amplification_{model_kind}.csv"
    with (directory / filename).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("rte/results/residual_subspace_mixed"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--model-kind", choices=["legacy_raw", "legacy_replay"], default="legacy_raw")
    args = parser.parse_args()
    run(args.directory, args.steps, args.model_kind)
