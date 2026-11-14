import argparse
import json
from pathlib import Path

import torch

try:
    from .RTE_ATFPS_func import funcDinv, func_inflow_Dirichlet
    from .RTE_mgmodel import MG_precond
    from .gmres import fgmres
    from .rte_preconditioning import d_metric_relative_residual, mgnet_preconditioner
    from .rte_config import (
        REPO_ROOT,
        dataset_filename,
        load_yaml_config,
        merge_cli_defaults,
        resolve_device,
        scheduler_subdir,
        set_seed,
    )
    from .train import DEFAULTS, load_discretization
except ImportError:
    from RTE_ATFPS_func import funcDinv, func_inflow_Dirichlet
    from RTE_mgmodel import MG_precond
    from gmres import fgmres
    from rte_preconditioning import d_metric_relative_residual, mgnet_preconditioner
    from rte_config import (
        REPO_ROOT,
        dataset_filename,
        load_yaml_config,
        merge_cli_defaults,
        resolve_device,
        scheduler_subdir,
        set_seed,
    )
    from train import DEFAULTS, load_discretization


INFERENCE_DEFAULTS = {
    **DEFAULTS,
    "checkpoint": None,
    "num_test": 3,
    "tol": 1e-6,
    "max_iter": 50,
    "restart": 50,
    "output": "outputs/inference_summary.json",
    "preconditioners": ["block_jacobi", "mgnet"],
    "preconditioner_form": "direct",
}


def build_parser(defaults):
    parser = argparse.ArgumentParser(description="Evaluate RTE preconditioners with right-preconditioned GMRES.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=defaults["checkpoint"])
    parser.add_argument("--cuda_device", type=str, default=defaults["cuda_device"])
    parser.add_argument("--data_type", type=str, default=defaults["data_type"])
    parser.add_argument("--num_coef", type=int, default=defaults["num_coef"])
    parser.add_argument("--num_data", type=int, default=defaults["num_data"])
    parser.add_argument("--mesh_size", type=int, default=defaults["mesh_size"])
    parser.add_argument("--N", type=int, default=defaults["N"])
    parser.add_argument("--tol_exp", type=float, default=defaults["tol_exp"])
    parser.add_argument("--kernel_g", type=str, default=defaults["kernel_g"])
    parser.add_argument("--u_channels_list", type=int, nargs="+", default=defaults["u_channels_list"])
    parser.add_argument("--a_channels_list", type=int, nargs="+", default=defaults["a_channels_list"])
    parser.add_argument("--mg_levels", type=int, nargs="+", default=defaults["mg_levels"])
    parser.add_argument("--scheduler_name", type=str, default=defaults["scheduler_name"])
    parser.add_argument("--step_size", type=int, default=defaults["step_size"])
    parser.add_argument("--gamma", type=float, default=defaults["gamma"])
    parser.add_argument("--T_max", type=int, default=defaults["T_max"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num_test", type=int, default=defaults["num_test"])
    parser.add_argument("--tol", type=float, default=defaults["tol"])
    parser.add_argument("--max_iter", type=int, default=defaults["max_iter"])
    parser.add_argument("--restart", type=int, default=defaults["restart"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--output", type=str, default=defaults["output"])
    parser.add_argument("--preconditioners", nargs="+", default=defaults["preconditioners"])
    parser.add_argument("--preconditioner_form", choices=["direct", "residual"], default=defaults["preconditioner_form"])
    return parser


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    known, _ = bootstrap.parse_known_args(argv)
    defaults = merge_cli_defaults(INFERENCE_DEFAULTS, load_yaml_config(known.config))
    return build_parser(defaults).parse_args(argv)


def default_checkpoint(args):
    subdir = scheduler_subdir(args)
    pattern = (
        f"{args.data_type}I{args.mesh_size}N{args.N}L{len(args.u_channels_list)}"
        f"batch{args.batch_size}epoch*d{args.num_coef}{args.num_data}tol*_multilevel.pt"
    )
    candidates = sorted((REPO_ROOT / "model_precond" / subdir).glob(pattern))
    candidates.extend(sorted((REPO_ROOT / "model_precond" / subdir).glob(pattern.replace(".pt", "_*.pt"))))
    return candidates[-1] if candidates else None


def load_rte_data(args):
    path = REPO_ROOT / "discretized_parameters" / dataset_filename(
        args.data_type, args.N, args.mesh_size, args.num_coef, args.tol_exp
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def run_solver(name, preconditioner, operator, rhs, args, metric=None):
    solution, history = fgmres(
        A=operator,
        b=rhs,
        M=preconditioner,
        tol=args.tol,
        maxiter=args.max_iter,
        restart=args.restart,
        verbose=False,
    )
    result = {
        "name": name,
        "iterations": max(len(history) - 1, 0),
        "final_relative_residual": float(history[-1]),
        "history": [float(v) for v in history],
    }
    if metric is not None:
        result["final_d_metric_relative_residual"] = float(metric(rhs - operator(solution)))
    return result


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.cuda_device)
    M, *_ = load_discretization(args, device)
    data = load_rte_data(args)

    num = min(args.num_test, data["Coef"].shape[0])
    Coef = data["Coef"][:num].to(device)
    fsmLRBTC = data["fsmLRBTC"][:num].to(device)
    I2A = data["I2A"][:num].to(device)
    MLRBT = data["MLRBT"][:num].to(device)
    VecSize = data["VecSize"][:num].to(device)
    rhs = (2 * torch.rand(num, 8 * M, args.mesh_size, args.mesh_size, device=device) - 1).to(torch.float32)

    operator = lambda value: func_inflow_Dirichlet(
        value, I2A, fsmLRBTC, MLRBT, VecSize, M, args.mesh_size, args.mesh_size, device
    )
    metric = lambda residual: d_metric_relative_residual(
        residual, rhs, I2A, fsmLRBTC, MLRBT, VecSize, M, args.mesh_size, device
    )

    results = []
    if "identity" in args.preconditioners:
        results.append(run_solver("identity", lambda value: value, operator, rhs, args, metric=metric))
    if "block_jacobi" in args.preconditioners:
        jacobi = lambda value: funcDinv(value, I2A, fsmLRBTC, MLRBT, VecSize, M, args.mesh_size, args.mesh_size, device)
        results.append(run_solver("block_jacobi", jacobi, operator, rhs, args, metric=metric))

    requested_mgnet = [name for name in args.preconditioners if name in {"mgnet", "mgnet_raw", "mgnet_born"}]
    if requested_mgnet:
        checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args)
        if checkpoint is None:
            raise FileNotFoundError("No MgNet checkpoint found; pass --checkpoint or omit mgnet.")
        model = MG_precond(M, args.u_channels_list, args.a_channels_list, args.mg_levels).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        a_list, inv_a_list = model.setup(Coef)

        def make_mgnet(mode):
            def preconditioner(value):
                with torch.no_grad():
                    return mgnet_preconditioner(
                        value,
                        model,
                        a_list,
                        inv_a_list,
                        I2A,
                        fsmLRBTC,
                        MLRBT,
                        VecSize,
                        M,
                        args.mesh_size,
                        device,
                        mode=mode,
                        form=args.preconditioner_form,
                    )

            return preconditioner

        if "mgnet" in requested_mgnet or "mgnet_raw" in requested_mgnet:
            results.append(run_solver("mgnet_raw", make_mgnet("raw"), operator, rhs, args, metric=metric))
        if "mgnet_born" in requested_mgnet:
            results.append(run_solver("mgnet_born", make_mgnet("born"), operator, rhs, args, metric=metric))

    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_type": args.data_type,
        "num_test": num,
        "tol": args.tol,
        "max_iter": args.max_iter,
        "restart": args.restart,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
