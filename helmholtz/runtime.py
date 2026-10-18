"""Shared command-line configuration for the Helmholtz experiment runners."""

import os
import random

import numpy as np
import torch

from .dataloader import Dataconfig


def add_runtime_arguments(parser, *, inference=False):
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--data-dir", help="Directory containing the selected dataset files")
    parser.add_argument("--output-dir", help="Override train.save_dir (and finetune.save_dir)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    if inference:
        parser.add_argument("--checkpoint", help="Explicit learned-model state_dict path")
        parser.add_argument("--no-plots", action="store_true", help="Skip field plots")


def configure_runtime(cfg, args):
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but PyTorch cannot access CUDA")
    if not (device == "cpu" or device.startswith("cuda")):
        raise ValueError("Supported devices are cpu, cuda, and cuda:N")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    if args.output_dir:
        for section in ("train", "finetune"):
            if section in cfg:
                cfg[section]["save_dir"] = args.output_dir
    return device


def configure_data(cfg, model_cfg, args):
    values = dict(cfg.get("data", {}))
    data_dir = args.data_dir or values.pop("dir_path", None)
    # Always consume dir_path even if the CLI took precedence.
    values.pop("dir_path", None)
    data = Dataconfig(model_cfg["data_type"], data_dir=data_dir, **values)
    if not os.path.exists(data.dir_path):
        raise FileNotFoundError(
            f"Velocity data not found: {data.dir_path}. Set --data-dir, "
            "data.dir_path in YAML, or NPBS_DATA_ROOT. See helmholtz/README.md."
        )
    return data
