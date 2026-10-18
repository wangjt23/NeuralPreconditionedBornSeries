"""Train the CDR inverse action from coefficients and random RHS probes."""

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from cdr.dataloader import ConvectionDiffusionDataset
from cdr.model import ABLATIONS, FNOSolver
from cdr.utils import select_device, set_seed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True, help="validation split for selecting the checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="new training run directory")
    parser.add_argument("--mode", choices=ABLATIONS, default="npbs")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--features", type=int, default=25)
    parser.add_argument("--modes", type=int, nargs="+", default=[32, 32, 32])
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler-gamma", type=float, default=0.8)
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--test-samples", type=int, default=400)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser


def run_training(args):
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.workers < 0:
        raise ValueError("epochs, batch size and learning rate must be positive; workers must be nonnegative")
    if not 0 < args.scheduler_gamma <= 1:
        raise ValueError("scheduler gamma must lie in (0, 1]")
    if (args.output / "best.pt").exists() or (args.output / "args.json").exists():
        raise FileExistsError(f"{args.output} already contains a run; choose a new output directory")
    set_seed(args.seed)
    device = select_device(args.device)
    train_data = ConvectionDiffusionDataset(args.train_path, args.train_samples, args.resolution)
    test_data = ConvectionDiffusionDataset(args.test_path, args.test_samples, args.resolution)
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, generator=shuffle_generator)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers)
    use_precond_res, use_precond_loss = ABLATIONS[args.mode]
    model = FNOSolver(args.resolution, args.features, args.modes, args.padding,
                      use_precond_res, use_precond_loss).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.epochs // 20), gamma=args.scheduler_gamma
    )
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / "args.json").write_text(json.dumps(config, indent=2) + "\n")
    best_loss = float("inf")
    history = []
    checkpoint_path = args.output / "best.pt"
    print(f"Training {args.mode} on {device}: {len(train_data)} train / {len(test_data)} validation samples")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for coefficients in train_loader:
            coefficients = coefficients.to(device)
            rhs = torch.randn_like(coefficients[:, :1])
            optimizer.zero_grad(set_to_none=True)
            prediction = model.train_forward(coefficients, rhs)
            loss = model.loss_fn(prediction, coefficients, rhs)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(coefficients)
        model.eval()
        test_loss = 0.0
        # Fixed validation probes make the checkpoint criterion comparable
        # across epochs without consuming the training RNG stream.
        validation_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        with torch.no_grad():
            for coefficients in test_loader:
                coefficients = coefficients.to(device)
                rhs = torch.randn(coefficients[:, :1].shape, device=device, generator=validation_generator)
                prediction = model.train_forward(coefficients, rhs)
                test_loss += model.loss_fn(prediction, coefficients, rhs).item() * len(coefficients)
        record = {
            "epoch": epoch, "train_loss": train_loss / len(train_data),
            "validation_loss": test_loss / len(test_data), "lr": optimizer.param_groups[0]["lr"],
        }
        if not math.isfinite(record["validation_loss"]):
            raise FloatingPointError(f"Nonfinite validation loss at epoch {epoch}")
        history.append(record)
        print(f"Epoch {epoch}/{args.epochs}: train={record['train_loss']:.6g}, val={record['validation_loss']:.6g}")
        if record["validation_loss"] < best_loss:
            best_loss = record["validation_loss"]
            torch.save({
                "format_version": 1, "model_config": model.config,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "epoch": epoch, "validation_loss": best_loss, "train_config": config,
            }, checkpoint_path)
        scheduler.step()
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Best checkpoint: {checkpoint_path}")
    return checkpoint_path


def main(argv=None):
    return run_training(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
