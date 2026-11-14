import argparse
import csv
from pathlib import Path

import scipy.io
import torch

from RTE_ATFPS_func import func_inflow_Dirichlet
from RTE_datagenerator import RTECoef
from RTE_mgmodel import RTEUNetPrecond
from rte_config import REPO_ROOT, kernel_suffix
from rte_preconditioning import (
    born_base_preconditioner,
    born_two_step_terms,
    corrected_candidate,
    d_metric_relative_residual,
    mgnet_preconditioner,
)


def load_discretization(n, kernel_g, device):
    quadrature = scipy.io.loadmat(REPO_ROOT / "discretized_parameters" / f"quadrature2DN{n}.mat")
    kappa = scipy.io.loadmat(REPO_ROOT / "discretized_parameters" / f"KappaN{n}{kernel_suffix(kernel_g)}.mat")["Kappa"]
    return (
        int(quadrature["M"].item()),
        torch.tensor(quadrature["ct"]).squeeze(-1).to(torch.float32).to(device),
        torch.tensor(quadrature["st"]).squeeze(-1).to(torch.float32).to(device),
        torch.tensor(quadrature["omega"]).squeeze(-1).to(torch.float32).to(device),
        torch.tensor(kappa).to(torch.float32).to(device),
    )


def make_batch(data_type, num_coef, mesh_size, n, kernel_g, tol_exp, device):
    m, ct, st, omega, kappa = load_discretization(n, kernel_g, device)
    data = RTECoef(mesh_size, num_coef, data_type=data_type, device=device)
    coef, fsmLRBTC, I2A, MLRBT, vec_size = data.getinfo(m, ct, st, omega, kappa, 10.0 ** (-float(tol_exp)))
    return m, coef, fsmLRBTC, I2A, MLRBT, vec_size


def true_residual(candidate, rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, mesh_size, device):
    return func_inflow_Dirichlet(candidate, I2A, fsmLRBTC, MLRBT, vec_size, m, mesh_size, mesh_size, device) - rhs


def residual_metrics(candidate, rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, mesh_size, device):
    residual = true_residual(candidate, rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, mesh_size, device)
    rhs_norm = torch.clamp(torch.linalg.norm(rhs), min=1e-30)
    return {
        "true_rel": float(torch.linalg.norm(residual) / rhs_norm),
        "d_metric_rel": float(d_metric_relative_residual(residual, rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, mesh_size, device)),
    }


def run_case(args, data_type, device):
    m, coef, fsmLRBTC, I2A, MLRBT, vec_size = make_batch(
        data_type, args.num_coef, args.mesh_size, args.N, args.kernel_g, args.tol_exp, device
    )
    rhs = 2.0 * torch.rand(args.num_coef, 8 * m, args.mesh_size, args.mesh_size, device=device) - 1.0

    with torch.no_grad():
        born1 = born_base_preconditioner(rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, args.mesh_size, device)
        z0, z1 = born_two_step_terms(rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, args.mesh_size, device)
        born2 = corrected_candidate(z0 + z1, rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, args.mesh_size, device)

        residual_model = RTEUNetPrecond(
            m,
            channels=args.channels,
            num_levels=args.levels,
            input_channels=8 * m,
            output_channels=8 * m,
        ).to(device)
        residual_model.eval()
        a_res, inv_res = residual_model.setup(coef)
        unet_residual_zero = mgnet_preconditioner(
            rhs,
            residual_model,
            a_res,
            inv_res,
            I2A,
            fsmLRBTC,
            MLRBT,
            vec_size,
            m,
            args.mesh_size,
            device,
            mode="born",
            form="residual",
        )

        defect_model = RTEUNetPrecond(
            m,
            channels=args.channels,
            num_levels=args.levels,
            input_channels=16 * m,
            output_channels=8 * m,
        ).to(device)
        defect_model.eval()
        a_def, inv_def = defect_model.setup(coef)
        unet_defect2_zero = mgnet_preconditioner(
            rhs,
            defect_model,
            a_def,
            inv_def,
            I2A,
            fsmLRBTC,
            MLRBT,
            vec_size,
            m,
            args.mesh_size,
            device,
            mode="born",
            form="defect2",
        )

    methods = {
        "identity": rhs,
        "born1": born1,
        "born2": born2,
        "unet_residual_zero": unet_residual_zero,
        "unet_defect2_zero": unet_defect2_zero,
    }
    rows = []
    for name, candidate in methods.items():
        row = {
            "data_type": data_type,
            "method": name,
            "mesh_size": args.mesh_size,
            "num_coef": args.num_coef,
        }
        row.update(residual_metrics(candidate, rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, args.mesh_size, device))
        if name == "unet_residual_zero":
            row["fallback_error"] = float(torch.linalg.norm(candidate - born1) / torch.clamp(torch.linalg.norm(born1), min=1e-30))
        elif name == "unet_defect2_zero":
            row["fallback_error"] = float(torch.linalg.norm(candidate - born2) / torch.clamp(torch.linalg.norm(born2), min=1e-30))
        else:
            row["fallback_error"] = 0.0
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Ablate Born residual/defect2 preconditioner revisions on tiny ATFPS RTE cases.")
    parser.add_argument("--data-types", nargs="+", default=["diffusion", "transport", "interface"])
    parser.add_argument("--num-coef", type=int, default=2)
    parser.add_argument("--mesh-size", type=int, default=4)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--kernel-g", default="0")
    parser.add_argument("--tol-exp", type=float, default=5)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--output", default="outputs/defect_revision_ablation.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    rows = []
    for data_type in args.data_types:
        rows.extend(run_case(args, data_type, device))

    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'case':12s} {'method':20s} {'true_rel':>12s} {'d_metric':>12s} {'fallback':>12s}")
    for row in rows:
        print(
            f"{row['data_type']:12s} {row['method']:20s} "
            f"{row['true_rel']:12.4e} {row['d_metric_rel']:12.4e} {row['fallback_error']:12.4e}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
