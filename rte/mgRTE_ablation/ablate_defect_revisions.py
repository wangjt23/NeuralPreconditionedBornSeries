import argparse
import csv
from pathlib import Path

import scipy.io
import torch
import torch.nn as nn

from RTE_ATFPS_func import funcDinv, func_inflow_Dirichlet
from RTE_datagenerator import RTECoef
from RTE_mgmodel import RTEUNetPrecond
from rte_config import REPO_ROOT, kernel_suffix
from rte_preconditioning import (
    born_base_preconditioner,
    born_two_step_terms,
    corrected_candidate,
    d_metric_relative_residual,
    maybe_resample_rhs,
    mgnet_preconditioner,
    training_loss,
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


def batch_l2_norm(x):
    dims = tuple(range(1, x.ndim))
    if torch.is_complex(x):
        sq = x.real.square() + x.imag.square()
    else:
        sq = x.square()
    return torch.sqrt(torch.clamp(torch.sum(sq, dim=dims), min=1e-30))


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


def build_scheduler(optimizer, scheduler_name, steps, step_size, gamma, eta_min):
    if scheduler_name == "none":
        return None
    if scheduler_name == "step":
        actual_step = int(step_size) if int(step_size) > 0 else max(1, int(steps) // 3)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=actual_step, gamma=float(gamma))
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(steps)),
            eta_min=float(eta_min),
        )
    raise ValueError(f"unsupported scheduler {scheduler_name}")


def apply_model_and_residual(
    model,
    form,
    residual,
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
    candidate = candidate_from_model(
        model,
        form,
        residual,
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
    residual_after = residual - func_inflow_Dirichlet(
        candidate,
        I2A,
        fsmLRBTC,
        MLRBT,
        vec_size,
        m,
        mesh_size,
        mesh_size,
        device,
    )
    ratio = batch_l2_norm(residual_after) / torch.clamp(batch_l2_norm(residual), min=1e-30)
    return candidate, residual_after, ratio


def true_residual_objective(
    model,
    form,
    batch,
    m,
    mesh_size,
    device,
    defect_relaxation,
    rhs_mode,
):
    coef, fsmLRBTC, I2A, MLRBT, vec_size, rhs = [tensor.to(device) for tensor in batch]
    coeff_features, inv_features = model.setup(coef)
    rhs = maybe_resample_rhs(rhs, rhs_mode)
    _, residual_after, ratio = apply_model_and_residual(
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
    return (ratio.square()).mean(), {
        "objective": ratio.mean().detach(),
        "physical": ratio.mean().detach(),
    }


def rollout_true_objective(
    model,
    form,
    batch,
    m,
    mesh_size,
    device,
    defect_relaxation,
    rollout_steps,
    rollout_weight,
    rollout_decay,
    stability_weight,
    contraction_target,
    rhs_mode,
):
    coef, fsmLRBTC, I2A, MLRBT, vec_size, rhs = [tensor.to(device) for tensor in batch]
    coeff_features, inv_features = model.setup(coef)
    rhs = maybe_resample_rhs(rhs, rhs_mode)
    _, current_residual, ratio = apply_model_and_residual(
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
    loss = (ratio.square()).mean()
    ratios = [ratio]

    for rollout_idx in range(int(rollout_steps)):
        _, current_residual, next_ratio = apply_model_and_residual(
            model,
            form,
            current_residual,
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
        ratios.append(next_ratio)
        if float(rollout_weight) > 0.0:
            rollout_scale = float(rollout_decay) ** rollout_idx
            loss = loss + float(rollout_weight) * rollout_scale * (next_ratio.square()).mean()

    if float(stability_weight) > 0.0:
        target = float(contraction_target)
        stability_terms = [(torch.relu(r - target).square()).mean() for r in ratios]
        loss = loss + float(stability_weight) * torch.stack(stability_terms).mean()

    return loss, {
        "objective": ratios[0].mean().detach(),
        "physical": ratios[0].mean().detach(),
        "rollout_last": ratios[-1].mean().detach(),
    }


def train_unet(
    model,
    form,
    batch,
    m,
    mesh_size,
    device,
    steps,
    lr,
    defect_relaxation,
    objective,
    weight_decay,
    scheduler_name,
    scheduler_step_size,
    scheduler_gamma,
    scheduler_eta_min,
    physical_loss_weight,
    rhs_mode,
    rollout_steps,
    rollout_weight,
    rollout_decay,
    stability_weight,
    contraction_target,
    log_steps,
):
    if steps <= 0:
        return {"loss": [], "lr": []}, {}

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, scheduler_name, steps, scheduler_step_size, scheduler_gamma, scheduler_eta_min)
    losses = []
    lrs = []
    model.train()
    loss_fn = nn.MSELoss()
    last_metrics = {}
    total_steps = int(steps)

    for step_idx in range(total_steps):
        optimizer.zero_grad(set_to_none=True)
        if objective == "true":
            loss, last_metrics = true_residual_objective(
                model,
                form,
                batch,
                m,
                mesh_size,
                device,
                defect_relaxation,
                rhs_mode,
            )
        elif objective == "rollout_true":
            loss, last_metrics = rollout_true_objective(
                model,
                form,
                batch,
                m,
                mesh_size,
                device,
                defect_relaxation,
                rollout_steps,
                rollout_weight,
                rollout_decay,
                stability_weight,
                contraction_target,
                rhs_mode,
            )
        elif objective in {"born", "born_mixed"}:
            loss, last_metrics = training_loss(
                model,
                batch,
                m,
                mesh_size,
                device,
                loss_fn,
                loss_type=objective,
                preconditioner_form=form,
                physical_loss_weight=physical_loss_weight,
                rhs_mode=rhs_mode,
                loss_normalization="relative",
                return_metrics=True,
            )
        else:
            raise ValueError(f"unknown objective {objective}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_value = float(loss.detach())
        losses.append(loss_value)
        lrs.append(float(optimizer.param_groups[0]["lr"]))
        if int(log_steps) > 0 and ((step_idx + 1) % int(log_steps) == 0 or step_idx == 0 or step_idx + 1 == total_steps):
            print(
                f"    step [{step_idx + 1:4d}/{total_steps}] loss={loss_value:.3e} "
                f"lr={lrs[-1]:.2e} objective={float(last_metrics['objective']):.3e}"
            )

    model.eval()
    return {"loss": losses, "lr": lrs}, last_metrics


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
    residual_history, residual_train_metrics = train_unet(
        residual_trained,
        "residual",
        batch,
        m,
        args.mesh_size,
        device,
        args.train_steps,
        args.lr,
        args.defect_relaxation,
        args.objective,
        args.weight_decay,
        args.scheduler,
        args.scheduler_step_size,
        args.scheduler_gamma,
        args.scheduler_eta_min,
        args.physical_loss_weight,
        args.rhs_mode,
        args.rollout_steps,
        args.rollout_weight,
        args.rollout_decay,
        args.stability_weight,
        args.contraction_target,
        args.log_steps,
    )

    defect_trained = RTEUNetPrecond(
        m,
        channels=args.channels,
        num_levels=args.levels,
        input_channels=16 * m,
        output_channels=8 * m,
    ).to(device)
    defect_history, defect_train_metrics = train_unet(
        defect_trained,
        "defect2",
        batch,
        m,
        args.mesh_size,
        device,
        args.train_steps,
        args.lr,
        args.defect_relaxation,
        args.objective,
        args.weight_decay,
        args.scheduler,
        args.scheduler_step_size,
        args.scheduler_gamma,
        args.scheduler_eta_min,
        args.physical_loss_weight,
        args.rhs_mode,
        args.rollout_steps,
        args.rollout_weight,
        args.rollout_decay,
        args.stability_weight,
        args.contraction_target,
        args.log_steps,
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
            "objective": args.objective,
            "train_steps": args.train_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "scheduler_step_size": args.scheduler_step_size,
            "scheduler_gamma": args.scheduler_gamma,
            "scheduler_eta_min": args.scheduler_eta_min,
            "rhs_mode": args.rhs_mode,
            "rollout_steps": args.rollout_steps,
            "rollout_weight": args.rollout_weight,
            "rollout_decay": args.rollout_decay,
            "stability_weight": args.stability_weight,
            "contraction_target": args.contraction_target,
            "train_loss_initial": "",
            "train_loss_final": "",
            "train_lr_initial": "",
            "train_lr_final": "",
            "train_objective_final": "",
            "defect_relaxation": args.defect_relaxation,
        }
        row.update(residual_metrics(candidate, rhs, I2A, fsmLRBTC, MLRBT, vec_size, m, args.mesh_size, device))
        if name == "unet_residual_zero":
            row["fallback_error"] = float(torch.linalg.norm(candidate - born1) / torch.clamp(torch.linalg.norm(born1), min=1e-30))
        elif name == "unet_defect2_zero":
            row["fallback_error"] = float(torch.linalg.norm(candidate - born2) / torch.clamp(torch.linalg.norm(born2), min=1e-30))
        elif name == "unet_residual_trained":
            row["fallback_error"] = float(torch.linalg.norm(candidate - born1) / torch.clamp(torch.linalg.norm(born1), min=1e-30))
            row["train_loss_initial"] = residual_history["loss"][0] if residual_history["loss"] else ""
            row["train_loss_final"] = residual_history["loss"][-1] if residual_history["loss"] else ""
            row["train_lr_initial"] = residual_history["lr"][0] if residual_history["lr"] else ""
            row["train_lr_final"] = residual_history["lr"][-1] if residual_history["lr"] else ""
            row["train_objective_final"] = float(residual_train_metrics.get("objective", torch.tensor(float("nan"))))
        elif name == "unet_defect2_trained":
            row["fallback_error"] = float(torch.linalg.norm(candidate - born2) / torch.clamp(torch.linalg.norm(born2), min=1e-30))
            row["train_loss_initial"] = defect_history["loss"][0] if defect_history["loss"] else ""
            row["train_loss_final"] = defect_history["loss"][-1] if defect_history["loss"] else ""
            row["train_lr_initial"] = defect_history["lr"][0] if defect_history["lr"] else ""
            row["train_lr_final"] = defect_history["lr"][-1] if defect_history["lr"] else ""
            row["train_objective_final"] = float(defect_train_metrics.get("objective", torch.tensor(float("nan"))))
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
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "step", "cosine"], default="none")
    parser.add_argument("--scheduler-step-size", type=int, default=0)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-5)
    parser.add_argument("--objective", choices=["true", "born", "born_mixed", "rollout_true"], default="true")
    parser.add_argument("--physical-loss-weight", type=float, default=0.1)
    parser.add_argument("--rhs-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--rollout-weight", type=float, default=0.25)
    parser.add_argument("--rollout-decay", type=float, default=0.8)
    parser.add_argument("--stability-weight", type=float, default=0.5)
    parser.add_argument("--contraction-target", type=float, default=0.95)
    parser.add_argument("--log-steps", type=int, default=0)
    parser.add_argument("--defect-relaxation", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/defect_revision_ablation.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
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
