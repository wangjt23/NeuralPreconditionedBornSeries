"""Generate a small tensor-only Newton dataset for testing the research workflow."""

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nonlinear.data import synthetic_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-scale", type=float, default=10)
    parser.add_argument("--source-grid", choices=["interior", "legacy-endpoints"], default="legacy-endpoints")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = synthetic_data(args.samples, args.grid_size, args.seed, args.source_scale, args.source_grid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved {args.samples} synthetic Newton systems to {args.output}")


if __name__ == "__main__":
    main()
