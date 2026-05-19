import argparse
import csv
from pathlib import Path

import scipy.io
import torch

from RTE_ATFPS_func import funcDinv, func_inflow_Dirichlet
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


def candidate_from_model(
    model,
    form,
    rhs,
    coeff_features,
    inv_features,
    I2A,
    fsmLRBTC,
    MLRBT,
    vec_size,
    m,
    mesh_size,
    device,
    defect_relaxation,
):
    if form == "residual":
        reference = funcDinv(rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, mesh_size, mesh_size, device)
        update = model(reference, coeff_features, inv_features)
        return corrected_candidate(reference + update, rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, mesh_size, device)
    if form == "defect2":
        z0, z1 = born_two_step_terms(rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, mesh_size, device)
        update = model(torch.cat([z0, z1], dim=1), coeff_features, inv_features)
        return corrected_candidate(
            z0 + float(defect_relaxation) * z1 + update,
            rhs,
            I2A,
            fsmLRBTC,
            MLRBT,
            vec_size,
            m,
            mesh_size,
            device,
        )
    raise ValueError(f"unknown form {form}")


def train_unet(model, form, batch, m, mesh_size, device, steps, lr, defect_relaxation):
    if steps <= 0:
        return None, None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    coef, fsmLRBTC, I2A, MLRBT, vec_size, rhs = batch
    rhs_norm = torch.clamp(torch.linalg.norm(rhs), min=1e-30)
    losses = []
    model.train()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        coeff_features, inv_features = model.setup(coef)
        candidate = candidate_from_model(
            model,
            form,
            rhs,
            coeff_features,
            inv_features,
            I2A,
            fsmLRBTC,
            MLRBT,
            vec_size,
            m,
            mesh_size,
            device,
            defect_relaxation,
        )
        residual = true_residual(candidate, rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, mesh_size, device)
        loss = (torch.linalg.norm(residual) / rhs_norm) ** 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return losses[0], losses[-1]


def run_case(args, data_type, device):
    m, coef, fsmLRBTC, I2A, MLRBT, vec_size = make_batch(
        data_type, args.num_coef, args.mesh_size, args.N, args.kernel_g, args.tol_exp, device
    )
    rhs = 2.0 * torch.rand(args.num_coef, 8 * m, args.mesh_size, args.mesh_size, device=device) - 1.0
    batch = (coef, fsmLRBTC, I2A, MLRBT, vec_size, rhs)

    with torch.no_grad():
        born1 = born_base_preconditioner(rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, args.mesh_size, device)
        z0, z1 = born_two_step_terms(rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, args.mesh_size, device)
        born2 = corrected_candidate(
            z0 + float(args.defect_relaxation) * z1,
            rhs,
            I2A,
            fsmLRBTC,
            MLRBT,
            vec_size,
            m,
            args.mesh_size,
            device,
        )

        residual_model = RTEUNetPrecond(
            m,
            channels=args.channels,
            num_levels=args.levels,
            input_channels=8 * m,
            output_channels=8 * m,
        ).to(device)
        residual_model.eval()
        a_res, inv_res = residual_model.setup(coef)
        unet_residual_zero = candidate_from_model(
            residual_model,
            "residual",
            rhs,
            a_res,
            inv_res,
            I2A,
            fsmLRBTC,
            MLRBT,
            vec_size,
            m,
            args.mesh_size,
            device,
            args.defect_relaxation,
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
        unet_defect2_zero = candidate_from_model(
            defect_model,
            "defect2",
            rhs,
            a_def,
            inv_def,
            I2A,
            fsmLRBTC,
            MLRBT,
            vec_size,
            m,
            args.mesh_size,
            device,
            args.defect_relaxation,
        )

    residual_trained = RTEUNetPrecond(
        m,
        channels=args.channels,
        num_levels=args.levels,
        input_channels=8 * m,
        output_channels=8 * m,
    ).to(device)
    residual_loss0, residual_loss1 = train_unet(
        residual_trained, "residual", batch, m, args.mesh_size, device, args.train_steps, args.lr, args.defect_relaxation
    )

    defect_trained = RTEUNetPrecond(
        m,
        channels=args.channels,
        num_levels=args.levels,
        input_channels=16 * m,
        output_channels=8 * m,
    ).to(device)
    defect_loss0, defect_loss1 = train_unet(
        defect_trained, "defect2", batch, m, args.mesh_size, device, args.train_steps, args.lr, args.defect_relaxation
    )

    with torch.no_grad():
        a_res_t, inv_res_t = residual_trained.setup(coef)
        unet_residual_trained = candidate_from_model(
            residual_trained,
            "residual",
            rhs,
            a_res_t,
            inv_res_t,
            I2A,
            fsmLRBTC,
            MLRBT,
            vec_size,
            m,
            args.mesh_size,
            device,
            args.defect_relaxation,
        )
        a_def_t, inv_def_t = defect_trained.setup(coef)
        unet_defect2_trained = candidate_from_model(
            defect_trained,
            "defect2",
            rhs,
            a_def_t,
            inv_def_t,
            I2A,
            fsmLRBTC,
            MLRBT,
            vec_size,
            m,
            args.mesh_size,
            device,
            args.defect_relaxation,
        )

    methods = {
        "identity": rhs,
        "born1": born1,
        "born2": born2,
        "unet_residual_zero": unet_residual_zero,
        "unet_defect2_zero": unet_defect2_zero,
        "unet_residual_trained": unet_residual_trained,
        "unet_defect2_trained": unet_defect2_trained,
    }
    rows = []
    for name, candidate in methods.items():
        row = {
            "data_type": data_type,
            "method": name,
            "mesh_size": args.mesh_size,
            "num_coef": args.num_coef,
            "train_steps": args.train_steps,
            "train_loss_initial": "",
            "train_loss_final": "",
            "defect_relaxation": args.defect_relaxation,
        }
        row.update(residual_metrics(candidate, rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, args.mesh_size, device))
        if name == "unet_residual_zero":
            row["fallback_error"] = float(torch.linalg.norm(candidate - born1) / torch.clamp(torch.linalg.norm(born1), min=1e-30))
        elif name == "unet_defect2_zero":
            row["fallback_error"] = float(torch.linalg.norm(candidate - born2) / torch.clamp(torch.linalg.norm(born2), min=1e-30))
        elif name == "unet_residual_trained":
            row["fallback_error"] = float(torch.linalg.norm(candidate - born1) / torch.clamp(torch.linalg.norm(born1), min=1e-30))
            row["train_loss_initial"] = residual_loss0
            row["train_loss_final"] = residual_loss1
        elif name == "unet_defect2_trained":
            row["fallback_error"] = float(torch.linalg.norm(candidate - born2) / torch.clamp(torch.linalg.norm(born2), min=1e-30))
            row["train_loss_initial"] = defect_loss0
            row["train_loss_final"] = defect_loss1
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
    parser.add_argument("--train-steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--defect-relaxation", type=float, default=1.0)
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

    print(
        f"{'case':12s} {'method':22s} {'true_rel':>12s} {'d_metric':>12s} "
        f"{'fallback':>12s} {'loss0':>12s} {'loss1':>12s}"
    )
    for row in rows:
        print(
            f"{row['data_type']:12s} {row['method']:22s} "
            f"{row['true_rel']:12.4e} {row['d_metric_rel']:12.4e} {row['fallback_error']:12.4e} "
            f"{row['train_loss_initial'] if row['train_loss_initial'] != '' else '':>12} "
            f"{row['train_loss_final'] if row['train_loss_final'] != '' else '':>12}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
