import argparse
from pathlib import Path

import torch

try:
    from .RTE_ATFPS_func import funcDinv, func_inflow_Dirichlet
    from .rte_config import dataset_filename, load_yaml_config, merge_cli_defaults, resolve_device, set_seed
    from .train import DEFAULTS, load_discretization
except ImportError:
    from RTE_ATFPS_func import funcDinv, func_inflow_Dirichlet
    from rte_config import dataset_filename, load_yaml_config, merge_cli_defaults, resolve_device, set_seed
    from train import DEFAULTS, load_discretization


COLLECT_DEFAULTS = {
    **DEFAULTS,
    "num_rhs": 64,
    "num_steps": 4,
    "output": "discretized_parameters/replay_rhs.pt",
}


def build_parser(defaults):
    parser = argparse.ArgumentParser(description="Collect replay residuals from short block-Jacobi solves.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cuda_device", type=str, default=defaults["cuda_device"])
    parser.add_argument("--data_type", type=str, default=defaults["data_type"])
    parser.add_argument("--num_coef", type=int, default=defaults["num_coef"])
    parser.add_argument("--mesh_size", type=int, default=defaults["mesh_size"])
    parser.add_argument("--N", type=int, default=defaults["N"])
    parser.add_argument("--tol_exp", type=float, default=defaults["tol_exp"])
    parser.add_argument("--kernel_g", type=str, default=defaults["kernel_g"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--num_rhs", type=int, default=defaults["num_rhs"])
    parser.add_argument("--num_steps", type=int, default=defaults["num_steps"])
    parser.add_argument("--output", type=str, default=defaults["output"])
    return parser


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    known, _ = bootstrap.parse_known_args(argv)
    defaults = merge_cli_defaults(COLLECT_DEFAULTS, load_yaml_config(known.config))
    return build_parser(defaults).parse_args(argv)


def load_rte_data(args):
    path = Path(__file__).resolve().parent / "discretized_parameters" / dataset_filename(
        args.data_type, args.N, args.mesh_size, args.num_coef, args.tol_exp
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.cuda_device)
    M, *_ = load_discretization(args, device)
    data = load_rte_data(args)

    num = min(args.num_rhs, data["Coef"].shape[0])
    fsmLRBTC = data["fsmLRBTC"][:num].to(device)
    I2A = data["I2A"][:num].to(device)
    MLRBT = data["MLRBT"][:num].to(device)
    VecSize = data["VecSize"][:num].to(device)

    rhs0 = (2 * torch.rand(num, 8 * M, args.mesh_size, args.mesh_size, device=device) - 1).to(torch.float32)
    x = torch.zeros_like(rhs0)
    residual = rhs0.clone()
    replay_rhs = [residual.detach().cpu()]

    def operator(value):
        return func_inflow_Dirichlet(value, I2A, fsmLRBTC, MLRBT, VecSize, M, args.mesh_size, args.mesh_size, device)

    for _ in range(args.num_steps):
        x = x + funcDinv(residual, I2A, fsmLRBTC, MLRBT, VecSize, M, args.mesh_size, args.mesh_size, device)
        residual = rhs0 - operator(x)
        replay_rhs.append(residual.detach().cpu())

    payload = {
        "rhs": torch.cat(replay_rhs, dim=0),
        "data_type": args.data_type,
        "num_coef": args.num_coef,
        "mesh_size": args.mesh_size,
        "N": args.N,
        "tol_exp": args.tol_exp,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "source": "block_jacobi",
    }
    output = Path(__file__).resolve().parent / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"saved replay residuals to {output}")
    print(f"rhs shape: {tuple(payload['rhs'].shape)}")


if __name__ == "__main__":
    main()