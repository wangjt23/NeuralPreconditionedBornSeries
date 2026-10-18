"""Load a state-dict checkpoint and solve CDR samples by residual correction."""

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cdr.dataloader import ConvectionDiffusionDataset
from cdr.model import FNOSolver
from cdr.utils import select_device


def load_model(checkpoint_path, device="cpu", physics_dtype=torch.float64):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("expected a version-1 checkpoint created by cdr/train.py")
    model = FNOSolver(**checkpoint["model_config"], physics_dtype=physics_dtype).to(select_device(str(device)))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def run_inference(checkpoint, data_path, output, samples=400, iterations=600,
                  tolerance=1e-6, device="auto", physics_dtype=torch.float64):
    device = select_device(str(device))
    model = load_model(checkpoint, device, physics_dtype)
    dataset = ConvectionDiffusionDataset(data_path, samples, model.config["resolution"])
    output = Path(output)
    if output.exists() and (any(output.glob("sample_*.pt")) or (output / "summary.json").exists()):
        raise FileExistsError(f"{output} already contains inference results; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, coefficients in enumerate(dataset):
        coefficients = coefficients[None].to(device=device, dtype=physics_dtype)
        solution, residual, history = model.solve(coefficients, iterations, tolerance)
        status = "converged" if residual < tolerance else "iteration_limit"
        if not math.isfinite(residual):
            status = "nonfinite"
        elif residual > 100:
            status = "diverged"
        torch.save({
            "u": solution.cpu(), "relative_residual": residual,
            "residual_history": history, "status": status,
            "source_file": dataset.paths[index].name,
        }, output / f"sample_{index:04d}.pt")
        # JSON represents nonfinite failures as null, with status recorded.
        records.append({
            "sample": dataset.paths[index].name, "iterations": len(history) - 1,
            "relative_residual": residual if math.isfinite(residual) else None,
            "status": status,
        })
        print(f"{dataset.paths[index].name}: {status}, {len(history) - 1} iterations, residual={residual:.6g}")
    summary = {
        "checkpoint": str(checkpoint), "tolerance": tolerance,
        "physics_dtype": str(physics_dtype), "samples": len(records),
        "converged": sum(record["status"] == "converged" for record in records),
        "results": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Converged {summary['converged']}/{summary['samples']} samples; results: {output}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--physics-dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args(argv)
    return run_inference(args.checkpoint, args.data_path, args.output, args.samples,
                         args.iterations, args.tolerance, args.device,
                         getattr(torch, args.physics_dtype))


if __name__ == "__main__":
    main()
