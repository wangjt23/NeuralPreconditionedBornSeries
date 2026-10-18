"""Evaluate fixed-state inner solves or full Newton solves from a checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nonlinear.data import load_states
from nonlinear.model import load_checkpoint
from nonlinear.solvers import solve_inner, solve_newton


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", nargs="+", type=Path, required=True,
                        help="Trajectory files; select the same held-out slice from each")
    parser.add_argument("--mode", choices=["inner", "newton"], default="inner")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=5000, help="First held-out sample in each file")
    parser.add_argument("--count", type=int, default=500, help="Samples per file (two files give 1000 tests)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=100, help="Inner residual correction budget")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Inner relative residual tolerance")
    parser.add_argument("--newton-iterations", type=int, default=10)
    parser.add_argument("--newton-tolerance", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = load_checkpoint(args.checkpoint, args.device)
    if args.start < 0 or args.count < 1 or args.batch_size < 1:
        parser.error("start/count must select available samples; batch-size must be positive")
    state_sets = []
    for path in args.data:
        states = load_states(path, model.config.grid_size)
        if args.start + args.count > len(states):
            parser.error(f"{path}: need {args.start + args.count} states, found {len(states)}")
        state_sets.append((path, states[args.start:args.start + args.count]))
    solutions, records = [], []
    for path, states in state_sets:
        for offset in range(0, len(states), args.batch_size):
            batch = states[offset:offset + args.batch_size].to(args.device)
            if args.mode == "inner":
                result = solve_inner(model.operator, model, batch, max_iterations=args.max_iterations,
                                     tolerance=args.tolerance)
            else:
                result = solve_newton(model.operator, model, batch, args.newton_iterations,
                                      args.newton_tolerance, args.max_iterations, args.tolerance)
            solutions.append(result.solution.cpu())
            for index in range(len(batch)):
                records.append({"source_file": str(path), "sample": args.start + offset + index,
                                "iterations": result.iterations[index].item(),
                                "converged": result.converged[index].item(),
                                "relative_residual": result.final_residual[index].item(),
                                "residual_history": result.residual_history[:, index].cpu().tolist()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"solution": torch.cat(solutions), "mode": args.mode,
                "config": vars(model.config), "records": records}, args.output)
    records_path = args.output.with_suffix(".json")
    if records_path == args.output:
        records_path = args.output.with_suffix(".records.json")
    records_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(records), "converged": sum(r["converged"] for r in records),
                      "mean_relative_residual": sum(r["relative_residual"] for r in records) / len(records)}))


if __name__ == "__main__":
    main()
