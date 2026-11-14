from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent


def load_yaml_config(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to use --config") from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flattened.update(flatten_config(value))
        else:
            flattened[key] = value
    return flattened


def merge_cli_defaults(defaults: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    merged.update(flatten_config(config))
    return merged


def number_token(value: int | float | str) -> str:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return ("%g" % float(value)).replace("-", "m").replace(".", "p")


def kernel_suffix(g: int | float | str) -> str:
    token = number_token(g)
    if token in {"0", "0p0"}:
        return "g0"
    if token == "0p2":
        return "g02"
    return f"g{token}"


def dataset_filename(data_type: str, N: int, mesh_size: int, num_coef: int, tol_exp: int | float | str) -> str:
    return f"{data_type}N{N}I{mesh_size}d{num_coef}tol{number_token(tol_exp)}.pt"


def run_name(args: Any, epoch: int) -> str:
    name = (
        f"{args.data_type}I{args.mesh_size}N{args.N}L{len(args.u_channels_list)}"
        f"batch{args.batch_size}epoch{epoch}d{args.num_coef}{args.num_data}"
        f"tol{number_token(args.tol_exp)}_multilevel"
    )
    loss_type = getattr(args, "loss_type", "raw")
    rhs_mode = getattr(args, "rhs_mode", "fixed")
    loss_normalization = getattr(args, "loss_normalization", "mse")
    form = getattr(args, "preconditioner_form", "direct")
    name = f"{name}_{loss_type}_{rhs_mode}_{loss_normalization}"
    if form != "direct":
        name = f"{name}_{form}"
    return name


def scheduler_subdir(args: Any) -> str:
    if args.scheduler_name == "StepLR":
        return f"StepLR_stepsize{args.step_size}_gamma{args.gamma}/lr{args.lr}/"
    if args.scheduler_name == "CosineAnnealingLR":
        return f"CosineAnnealingLR_T_max{args.T_max}/lr{args.lr}/"
    raise ValueError(f"scheduler {args.scheduler_name} not supported")


def resolve_device(cuda_device: str) -> torch.device:
    if torch.cuda.is_available() and cuda_device != "cpu":
        return torch.device(cuda_device)
    return torch.device("cpu")


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
