import argparse
import logging
import os
from pathlib import Path

import scipy.io
import torch
from torch.utils.data import DataLoader

try:
    from .RTE_dataloader import UnsuperviseDataset
    from .RTE_mgmodel import MG_precond
    from .rte_preconditioning import training_loss
    from .rte_config import (
        REPO_ROOT,
        dataset_filename,
        kernel_suffix,
        load_yaml_config,
        merge_cli_defaults,
        resolve_device,
        run_name,
        scheduler_subdir,
        set_seed,
    )
    from .utils import Timer, check_cuda_mem, greenprint, save_loss_history, save_model
except ImportError:
    from RTE_dataloader import UnsuperviseDataset
    from RTE_mgmodel import MG_precond
    from rte_preconditioning import training_loss
    from rte_config import (
        REPO_ROOT,
        dataset_filename,
        kernel_suffix,
        load_yaml_config,
        merge_cli_defaults,
        resolve_device,
        run_name,
        scheduler_subdir,
        set_seed,
    )
    from utils import Timer, check_cuda_mem, greenprint, save_loss_history, save_model


DEFAULTS = {
    "cuda_device": "cuda:0",
    "data_type": "diffusion",
    "num_coef": 1000,
    "num_data": 10000,
    "mesh_size": 16,
    "xl": 0.0,
    "xr": 1.0,
    "yl": 0.0,
    "yr": 1.0,
    "N": 1,
    "tol_exp": 5,
    "kernel_g": "0",
    "u_channels_list": [16, 32, 64, 128],
    "a_channels_list": [16, 32, 64, 128],
    "mg_levels": [2, 2, 2, 4],
    "lr": 0.001,
    "batch_size": 256,
    "num_epochs": 1000,
    "scheduler_name": "CosineAnnealingLR",
    "step_size": 10,
    "gamma": 0.5,
    "T_max": 100,
    "eta_min": 1e-6,
    "val_steps": 5,
    "seed": 1234,
    "save": True,
    "num_workers": 0,
    "loss_type": "raw",
    "preconditioner_form": "direct",
    "physical_loss_weight": 0.1,
    "rhs_mode": "random",
    "loss_normalization": "relative",
    "rhs_replay_path": None,
}


def build_parser(defaults):
    parser = argparse.ArgumentParser(description="Train MgNet RTE preconditioner.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file.")
    parser.add_argument("--cuda_device", type=str, default=defaults["cuda_device"])
    parser.add_argument("--data_type", type=str, default=defaults["data_type"])
    parser.add_argument("--num_coef", type=int, default=defaults["num_coef"])
    parser.add_argument("--num_data", type=int, default=defaults["num_data"])
    parser.add_argument("--mesh_size", type=int, default=defaults["mesh_size"])
    parser.add_argument("--xl", type=float, default=defaults["xl"])
    parser.add_argument("--xr", type=float, default=defaults["xr"])
    parser.add_argument("--yl", type=float, default=defaults["yl"])
    parser.add_argument("--yr", type=float, default=defaults["yr"])
    parser.add_argument("--N", type=int, default=defaults["N"])
    parser.add_argument("--tol_exp", type=float, default=defaults["tol_exp"])
    parser.add_argument("--kernel_g", type=str, default=defaults["kernel_g"])
    parser.add_argument("--u_channels_list", type=int, nargs="+", default=defaults["u_channels_list"])
    parser.add_argument("--a_channels_list", type=int, nargs="+", default=defaults["a_channels_list"])
    parser.add_argument("--mg_levels", type=int, nargs="+", default=defaults["mg_levels"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num_epochs", type=int, default=defaults["num_epochs"])
    parser.add_argument("--scheduler_name", type=str, default=defaults["scheduler_name"])
    parser.add_argument("--step_size", type=int, default=defaults["step_size"])
    parser.add_argument("--gamma", type=float, default=defaults["gamma"])
    parser.add_argument("--T_max", type=int, default=defaults["T_max"])
    parser.add_argument("--eta_min", type=float, default=defaults["eta_min"])
    parser.add_argument("--val_steps", type=int, default=defaults["val_steps"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--loss_type", choices=["raw", "born", "born_mixed"], default=defaults["loss_type"])
    parser.add_argument("--preconditioner_form", choices=["direct", "residual"], default=defaults["preconditioner_form"])
    parser.add_argument("--physical_loss_weight", type=float, default=defaults["physical_loss_weight"])
    parser.add_argument("--rhs_mode", choices=["random", "fixed", "replay"], default=defaults["rhs_mode"])
    parser.add_argument("--loss_normalization", choices=["relative", "mse"], default=defaults["loss_normalization"])
    parser.add_argument("--rhs_replay_path", type=str, default=defaults["rhs_replay_path"])
    parser.add_argument("--no_save", action="store_true", help="Do not save loss curves or model weights.")
    return parser


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    known, _ = bootstrap.parse_known_args(argv)
    defaults = merge_cli_defaults(DEFAULTS, load_yaml_config(known.config))
    parser = build_parser(defaults)
    args = parser.parse_args(argv)
    args.save = not args.no_save
    return args


def build_scheduler(args, optimizer):
    if args.scheduler_name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
    raise ValueError(f"scheduler {args.scheduler_name} not supported")


def load_discretization(args, device):
    quadrature_path = REPO_ROOT / "discretized_parameters" / f"quadrature2DN{args.N}.mat"
    kappa_path = REPO_ROOT / "discretized_parameters" / f"KappaN{args.N}{kernel_suffix(args.kernel_g)}.mat"
    if not quadrature_path.exists():
        raise FileNotFoundError(quadrature_path)
    if not kappa_path.exists():
        raise FileNotFoundError(kappa_path)

    quadrature = scipy.io.loadmat(quadrature_path)
    ct = torch.tensor(quadrature["ct"]).squeeze(-1).to(torch.float32).to(device)
    st = torch.tensor(quadrature["st"]).squeeze(-1).to(torch.float32).to(device)
    omega = torch.tensor(quadrature["omega"]).squeeze(-1).to(torch.float32).to(device)
    M = quadrature["M"].item()
    Kappa = torch.tensor(scipy.io.loadmat(kappa_path)["Kappa"]).to(torch.float32).to(device)
    return M, ct, st, omega, Kappa


def load_dataset(args, device):
    coef_path = REPO_ROOT / "discretized_parameters" / dataset_filename(
        args.data_type, args.N, args.mesh_size, args.num_coef, args.tol_exp
    )
    if not coef_path.exists():
        command = (
            "python RTE_datagenerator.py "
            f"--data_type {args.data_type} --num_coef {args.num_coef} "
            f"--mesh_size {args.mesh_size} --N {args.N} --tol_exp {args.tol_exp} "
            f"--kernel_g {args.kernel_g}"
        )
        raise FileNotFoundError(f"{coef_path}\nGenerate it first with:\n  {command}")
    return torch.load(coef_path, map_location="cpu", weights_only=True)


def load_replay_rhs(args):
    if args.rhs_mode != "replay":
        return None
    if not args.rhs_replay_path:
        raise ValueError("rhs_mode=replay requires --rhs_replay_path")

    loaded = torch.load(args.rhs_replay_path, map_location="cpu", weights_only=True)
    replay_rhs = loaded["rhs"] if isinstance(loaded, dict) else loaded
    if replay_rhs.ndim != 4:
        raise ValueError("Replay RHS tensor must have shape [num_rhs, channels, mesh_size, mesh_size]")
    return replay_rhs.to(torch.float32)


def make_loaders(args, data, device):
    replay_rhs = load_replay_rhs(args)
    dataset = UnsuperviseDataset(
        args.N,
        args.mesh_size,
        args.num_coef,
        args.num_data,
        data,
        device="cpu",
        seed_num=args.seed,
        store_rhs=args.rhs_mode == "fixed",
        replay_rhs=replay_rhs,
    )
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(int(args.seed))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader


def move_batch(batch, device):
    return [tensor.to(device) for tensor in batch]


def residual_loss(model, batch, M, args, device, loss_fn, return_metrics=False):
    return training_loss(
        model=model,
        batch=batch,
        M=M,
        mesh_size=args.mesh_size,
        device=device,
        loss_fn=loss_fn,
        loss_type=args.loss_type,
        preconditioner_form=args.preconditioner_form,
        physical_loss_weight=args.physical_loss_weight,
        rhs_mode=getattr(args, "rhs_mode", "fixed"),
        loss_normalization=getattr(args, "loss_normalization", "mse"),
        return_metrics=return_metrics,
    )


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s-%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.cuda_device)
    greenprint(f"Using device: {device}")

    M, *_ = load_discretization(args, device)
    model = MG_precond(M, args.u_channels_list, args.a_channels_list, args.mg_levels).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("parameters: total=%s trainable=%s", f"{total_params:,}", f"{trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = build_scheduler(args, optimizer)
    loss_fn = torch.nn.MSELoss().to(device)

    data = load_dataset(args, device)
    train_loader, val_loader = make_loaders(args, data, device)
    timer = Timer(device)
    train_losses = []
    val_losses = []

    for epoch in range(args.num_epochs):
        model.train()
        timer.reset()
        epoch_loss = 0.0
        epoch_physical = 0.0
        epoch_born = 0.0
        for batch in train_loader:
            loss, metrics = residual_loss(model, batch, M, args, device, loss_fn, return_metrics=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_physical += metrics["physical"].item()
            epoch_born += metrics.get("born", metrics["physical"]).item()
        epoch_loss /= max(len(train_loader), 1)
        epoch_physical /= max(len(train_loader), 1)
        epoch_born /= max(len(train_loader), 1)
        scheduler.step()
        check_cuda_mem(device)
        train_losses.append(epoch_loss)
        logging.info(
            "data_type=%s epoch=[%d/%d] train_loss=%.4e train_physical=%.4e train_born=%.4e time=%.2fs lr=%.2e",
            args.data_type,
            epoch + 1,
            args.num_epochs,
            epoch_loss,
            epoch_physical,
            epoch_born,
            timer.elapsed("s"),
            scheduler.get_last_lr()[0],
        )

        if (epoch + 1) % args.val_steps == 0:
            model.eval()
            timer.reset()
            val_loss = 0.0
            val_physical = 0.0
            val_born = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    loss, metrics = residual_loss(model, batch, M, args, device, loss_fn, return_metrics=True)
                    val_loss += loss.item()
                    val_physical += metrics["physical"].item()
                    val_born += metrics.get("born", metrics["physical"]).item()
            val_loss /= max(len(val_loader), 1)
            val_physical /= max(len(val_loader), 1)
            val_born /= max(len(val_loader), 1)
            val_losses.append(val_loss)
            logging.info(
                "epoch=[%d/%d] val_loss=%.4e val_physical=%.4e val_born=%.4e time=%.2fs",
                epoch + 1,
                args.num_epochs,
                val_loss,
                val_physical,
                val_born,
                timer.elapsed("s"),
            )

    if args.save:
        sub_dir = scheduler_subdir(args)
        train_loss_dir = REPO_ROOT / "train_loss" / sub_dir
        val_loss_dir = REPO_ROOT / "val_loss" / sub_dir
        model_dir = REPO_ROOT / "model_precond" / sub_dir
        name = run_name(args, args.num_epochs)
        save_loss_history(str(train_loss_dir), train_losses, name)
        save_loss_history(str(val_loss_dir), val_losses, name)
        save_model(str(model_dir), model, name)
        logging.info("saved run artifacts as %s", name)


if __name__ == "__main__":
    main()
