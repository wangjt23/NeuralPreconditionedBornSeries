"""Train baseline / NPBS-l2 / NPBS-weighted Fourier Newton inverse models."""

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nonlinear.data import load_states, split_states, synthetic_data
from nonlinear.model import ModelConfig, NeuralNewtonSolver, save_checkpoint


def train_model(model: NeuralNewtonSolver, train_states: torch.Tensor,
                test_states: torch.Tensor, epochs: int = 200, batch_size: int = 50,
                learning_rate: float = 1e-4, weight_decay: float = 1e-6,
                seed: int = 0, step_size: int = 50, gamma: float = 0.5) -> list[dict]:
    if epochs < 1 or batch_size < 1 or step_size < 1:
        raise ValueError("epochs, batch_size, and step_size must be positive")
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_states), batch_size=batch_size,
                        shuffle=True, generator=generator)
    test_loader = DataLoader(TensorDataset(test_states), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # Keep training RHS draws independent of data shuffling and validation.
    rhs_generator = torch.Generator(device=device).manual_seed(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (states,) in loader:
            states = states.to(device)
            rhs = torch.randn(states.shape, generator=rhs_generator, device=device, dtype=states.dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(states, model(states, rhs), rhs)
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss is non-finite")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(states)
        model.eval()
        validation, raw_validation = 0.0, 0.0
        with torch.no_grad():
            for (states,) in test_loader:
                states = states.to(device)
                rhs = model.operator.rhs(states)
                prediction = model(states, rhs)
                validation += model.loss(states, prediction, rhs).item() * len(states)
                raw_validation += model.loss(states, prediction, rhs, weighted=False).item() * len(states)
        record = {"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"],
                  "train_loss": total / len(train_states),
                  "validation_loss": validation / len(test_states),
                  "validation_raw_residual": raw_validation / len(test_states)}
        if not all(math.isfinite(record[key]) for key in
                   ("train_loss", "validation_loss", "validation_raw_residual")):
            raise FloatingPointError(f"non-finite epoch metrics at epoch {epoch + 1}")
        history.append(record)
        print(json.dumps(record), flush=True)
        scheduler.step()
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", nargs="+", type=Path, help="Tensor datasets; each is split before concatenation")
    source.add_argument("--synthetic-samples", type=int, help="Small smoke dataset, grid size <= 24")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["baseline", "npbs-l2", "npbs-weighted"], default="npbs-weighted")
    parser.add_argument("--grid-size", type=int, default=None, help="Explicitly resample loaded states to this grid")
    parser.add_argument("--width", type=int, default=25)
    parser.add_argument("--modes", nargs="+", type=int, default=[20, 20])
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--source-scale", type=float, default=1600)
    parser.add_argument("--source-grid", choices=["interior", "legacy-endpoints"], default="legacy-endpoints")
    parser.add_argument("--train-count", type=int, default=None,
                        help="Training samples per file (default: 5000 with --data, 80%% with --synthetic-samples)")
    parser.add_argument("--test-count", type=int, default=None,
                        help="Held-out samples after training (default: 500 with --data, remainder with --synthetic-samples)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.data:
        train_count = 5000 if args.train_count is None else args.train_count
        test_count = 500 if args.test_count is None else args.test_count
        splits = [split_states(load_states(path, args.grid_size), train_count, test_count)
                  for path in args.data]
    else:
        data = synthetic_data(args.synthetic_samples, args.grid_size or 12, args.seed,
                              args.source_scale, args.source_grid)
        splits = [split_states(data["u"], args.train_count, args.test_count)]
    train_states = torch.cat([pair[0] for pair in splits])
    test_states = torch.cat([pair[1] for pair in splits])
    config = ModelConfig(grid_size=train_states.shape[-1], width=args.width,
                         modes=tuple(args.modes), padding=args.padding, variant=args.variant,
                         source_scale=args.source_scale, source_grid=args.source_grid)
    model = NeuralNewtonSolver(config).to(args.device)
    history = train_model(model, train_states, test_states, args.epochs, args.batch_size,
                          args.lr, args.weight_decay, args.seed, args.step_size, args.gamma)
    metadata = {"seed": args.seed, "epochs": args.epochs, "train_samples": len(train_states),
                "test_samples": len(test_states), "history": history,
                "training": {"batch_size": args.batch_size, "learning_rate": args.lr,
                             "weight_decay": args.weight_decay, "step_size": args.step_size,
                             "gamma": args.gamma, "rhs": "standard-normal"}}
    save_checkpoint(args.output, model, metadata)
    args.output.with_suffix(".history.json").write_text(
        json.dumps({"config": asdict(config), **metadata}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
