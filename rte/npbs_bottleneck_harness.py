import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rte.npbs_unet_ablation import (  # noqa: E402
    CoeffMixedMomentTSAPreconditioner,
    CoeffWeightedMomentTSAPreconditioner,
    AngularSpectralDefect2Preconditioner,
    LearnedRPMomentTSAPreconditioner,
    MomentNPBSUNetPreconditioner,
    NPBSUNetPreconditioner,
    ResidualSubspaceMomentTSAPreconditioner,
    batch_l2_norm,
    collect_residuals_from_sources,
    effective_tsa_harmonics,
    ensure_residual_batch,
    inner_base_apply,
    make_residual_sampler,
    make_source,
    make_training_sources,
    one_step_quality,
    solve_fgmres,
    solve_stationary,
)
from rte.run_hard_case_experiments import make_case  # noqa: E402


LOSS_MODES = {
    "metric",
    "raw",
    "metric_raw",
    "metric_rollout",
    "raw_rollout",
    "raw_rollout_teacher",
    "solver_teacher",
    "metric_raw_rollout",
    "teacher",
    "raw_teacher",
    "metric_raw_teacher",
}

CHECKPOINT_METRICS = {
    "none": None,
    "heldout_fgmres": ("heldout", "fgmres_final"),
    "heldout_stationary": ("heldout", "stationary_final"),
    "heldout_one_step_true": ("heldout", "one_step_true"),
    "heldout_composite": "heldout_composite",
}


class FixedStackPreconditioner(nn.Module):
    def __init__(self, rte, kind, depth=1, tsa_harmonics=None, relaxation=1.0):
        super().__init__()
        self.rte = rte
        self.kind = kind
        self.depth = int(depth)
        self.tsa_harmonics = tsa_harmonics
        self.relaxation = float(relaxation)
        self.base_calls_per_apply = self.depth

    def forward(self, residual):
        return inner_base_apply(
            self.rte,
            residual,
            depth=self.depth,
            kind=self.kind,
            tsa_harmonics=self.tsa_harmonics,
            relaxation=self.relaxation,
        )


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def damped_moment_reference_apply(model, residual):
    """Fixed damped TSA-MK map matching the structured moment initializations."""
    rte = model.rte
    transport = rte.transport_preconditioner(residual)
    remaining = residual - rte.A_op(transport)
    synthetic = rte.reference_moment_transport_solve(
        remaining,
        harmonics=model.moment_harmonics,
    )
    return float(model.transport_relaxation) * transport + float(model.synthetic_relaxation) * synthetic


def metric_apply(model, residual, metric_map):
    if metric_map == "base":
        return model.base(residual)
    if metric_map == "forward":
        return model(residual)
    if metric_map == "damped_moment":
        if not hasattr(model, "moment_harmonics"):
            raise ValueError("--metric-map damped_moment only applies to moment TSA models")
        return damped_moment_reference_apply(model, residual)
    raise ValueError(f"unknown metric map: {metric_map}")


def l2_ratio(numerator, denominator):
    return batch_l2_norm(numerator) / torch.clamp(batch_l2_norm(denominator), min=1e-30)


def loss_weights(loss_mode, physical_weight, rollout_weight, teacher_weight):
    if loss_mode == "metric":
        return 1.0, 0.0, 0.0, 0.0, 0.0
    if loss_mode == "raw":
        return 0.0, 1.0, 0.0, 0.0, 0.0
    if loss_mode == "metric_raw":
        return 1.0, float(physical_weight), 0.0, 0.0, 0.0
    if loss_mode == "metric_rollout":
        return 1.0, 0.0, float(rollout_weight), 0.0, 0.0
    if loss_mode == "raw_rollout":
        return 0.0, 1.0, 0.0, float(rollout_weight), 0.0
    if loss_mode in ("raw_rollout_teacher", "solver_teacher"):
        return 0.0, 1.0, 0.0, float(rollout_weight), float(teacher_weight)
    if loss_mode == "metric_raw_rollout":
        return 1.0, float(physical_weight), float(rollout_weight), float(rollout_weight), 0.0
    if loss_mode == "teacher":
        return 0.0, 0.0, 0.0, 0.0, float(teacher_weight)
    if loss_mode == "raw_teacher":
        return 0.0, 1.0, 0.0, 0.0, float(teacher_weight)
    if loss_mode == "metric_raw_teacher":
        return 1.0, float(physical_weight), 0.0, 0.0, float(teacher_weight)
    raise ValueError(f"unknown loss mode: {loss_mode}")


def eval_row(eval_rows, split):
    return next(item for item in eval_rows if item["split"] == split)


def normalized_ratio(value, baseline, eps=1e-30):
    return float(value) / max(float(baseline), float(eps))


def checkpoint_score(
    eval_rows,
    selection,
    baseline_rows=None,
    composite_phys_weight=1.0,
    composite_stationary_weight=0.0,
):
    if selection == "none":
        return None
    if selection == "heldout_composite":
        row = eval_row(eval_rows, "heldout")
        baseline = eval_row(baseline_rows if baseline_rows is not None else eval_rows, "heldout")
        physical = float(row["one_step_true"]) ** 2
        baseline_physical = float(baseline["one_step_true"]) ** 2
        return (
            normalized_ratio(row["fgmres_final"], baseline["fgmres_final"])
            + float(composite_phys_weight) * normalized_ratio(physical, baseline_physical)
            + float(composite_stationary_weight) * normalized_ratio(
                row["stationary_final"],
                baseline["stationary_final"],
            )
        )
    split, field = CHECKPOINT_METRICS[selection]
    row = eval_row(eval_rows, split)
    return float(row[field])


def resolve_scheduler_eta_min(lr, scheduler_eta_min, scheduler_min_lr_ratio):
    if scheduler_eta_min is not None:
        return float(scheduler_eta_min)
    return float(lr) * float(scheduler_min_lr_ratio)


def build_scheduler(
    optimizer,
    scheduler_name,
    steps,
    lr,
    scheduler_step_size,
    scheduler_gamma,
    scheduler_eta_min,
    scheduler_min_lr_ratio,
    scheduler_pct_start,
):
    if scheduler_name == "none":
        return None
    if scheduler_name == "step":
        step_size = scheduler_step_size if scheduler_step_size > 0 else max(1, steps // 3)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=scheduler_gamma)
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, steps),
            eta_min=resolve_scheduler_eta_min(lr, scheduler_eta_min, scheduler_min_lr_ratio),
        )
    if scheduler_name == "onecycle":
        if steps < 5:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, steps),
                eta_min=resolve_scheduler_eta_min(lr, scheduler_eta_min, scheduler_min_lr_ratio),
            )
        min_lr_ratio = max(float(scheduler_min_lr_ratio), 1e-8)
        div_factor = max(1.0, min_lr_ratio ** -0.5)
        pct_start = min(max(float(scheduler_pct_start), 1.0 / steps), 1.0 - 1.0 / steps)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(lr),
            total_steps=steps,
            pct_start=pct_start,
            anneal_strategy="cos",
            div_factor=div_factor,
            final_div_factor=div_factor,
        )
    raise ValueError(f"unknown scheduler: {scheduler_name}")


def probe_growth_penalty(model, residuals, probe_steps, contraction_target):
    current = ensure_residual_batch(residuals)
    initial_norm = torch.clamp(batch_l2_norm(current.detach()), min=1e-30)
    ratios = []
    for _ in range(int(probe_steps)):
        correction = model(current)
        current = current - model.rte.A_op(correction)
        ratios.append(batch_l2_norm(current) / initial_norm)
    if not ratios:
        zero = torch.zeros((), dtype=torch.float32, device=current.device)
        return zero, {"probe_ratio": 0.0, "probe_peak_ratio": 0.0, "probe_penalty": 0.0}
    ratio_stack = torch.stack(ratios, dim=0)
    penalty = torch.relu(ratio_stack - float(contraction_target)).square().mean()
    return penalty, {
        "probe_ratio": float(ratio_stack[-1].detach().mean()),
        "probe_peak_ratio": float(ratio_stack.detach().max(dim=0).values.mean()),
        "probe_penalty": float(penalty.detach()),
    }


def update_identity_weight(current_weight, eval_rows, step, args):
    if args.identity_schedule == "none" or current_weight <= 0.0:
        return current_weight
    if args.identity_schedule != "safe_decay":
        raise ValueError(f"unknown identity schedule: {args.identity_schedule}")
    if step < int(args.identity_decay_start):
        return current_weight
    if float(args.identity_safety_threshold) <= 0.0:
        return current_weight
    heldout_fgmres = float(eval_row(eval_rows, "heldout")["fgmres_final"])
    if heldout_fgmres > float(args.identity_safety_threshold):
        return current_weight
    next_weight = max(float(args.identity_weight_min), current_weight * float(args.identity_decay_factor))
    return min(current_weight, next_weight)


def objective_components(
    model,
    residuals,
    loss_mode,
    metric_map="base",
    physical_weight=0.5,
    rollout_weight=0.2,
    rollout_steps=3,
    rollout_decay=0.8,
    stability_weight=0.0,
    contraction_target=0.95,
    correction_weight=1e-6,
    teacher_fn=None,
    teacher_weight=1.0,
    teacher_action_weight=1.0,
    identity_weight=0.0,
    probe_residuals=None,
    probe_weight=0.0,
    probe_steps=0,
    probe_contraction_target=0.95,
):
    residual = ensure_residual_batch(residuals)
    metric_w, raw_w, rollout_metric_w, rollout_raw_w, teacher_w = loss_weights(
        loss_mode,
        physical_weight=physical_weight,
        rollout_weight=rollout_weight,
        teacher_weight=teacher_weight,
    )

    correction = model(residual)
    next_residual = residual - model.rte.A_op(correction)
    metric_next = metric_apply(model, next_residual, metric_map=metric_map)
    metric_ref = metric_apply(model, residual, metric_map=metric_map).detach()

    metric_ratio = l2_ratio(metric_next, metric_ref)
    physical_ratio = l2_ratio(next_residual, residual.detach())
    correction_ratio = l2_ratio(correction, metric_ref)

    loss = torch.zeros((), dtype=torch.float32, device=residual.device)
    if metric_w:
        loss = loss + metric_w * metric_ratio.square().mean()
    if raw_w:
        loss = loss + raw_w * physical_ratio.square().mean()
    if correction_weight:
        loss = loss + float(correction_weight) * correction_ratio.square().mean()
    if identity_weight > 0.0 and hasattr(model, "identity_penalty"):
        loss = loss + float(identity_weight) * model.identity_penalty(residual.shape[0], residual.device)

    rollout_metric_values = []
    rollout_raw_values = []
    metric_history = [metric_ratio]
    current_residual = next_residual
    rollout_steps = int(rollout_steps)
    if rollout_steps > 0 and (rollout_metric_w > 0.0 or rollout_raw_w > 0.0 or stability_weight > 0.0):
        for rollout_idx in range(rollout_steps):
            rollout_correction = model(current_residual)
            rollout_next = current_residual - model.rte.A_op(rollout_correction)
            rollout_metric = l2_ratio(
                metric_apply(model, rollout_next, metric_map=metric_map),
                metric_apply(model, current_residual, metric_map=metric_map).detach(),
            )
            rollout_raw = l2_ratio(rollout_next, current_residual.detach())
            rollout_metric_values.append(rollout_metric)
            rollout_raw_values.append(rollout_raw)
            metric_history.append(rollout_metric)
            scale = float(rollout_decay) ** rollout_idx
            if rollout_metric_w:
                loss = loss + rollout_metric_w * scale * rollout_metric.square().mean()
            if rollout_raw_w:
                loss = loss + rollout_raw_w * scale * rollout_raw.square().mean()
            current_residual = rollout_next

    if stability_weight > 0.0:
        target = float(contraction_target)
        stability = torch.stack([(torch.relu(entry - target).square()).mean() for entry in metric_history]).mean()
        loss = loss + float(stability_weight) * stability
    else:
        stability = torch.zeros((), dtype=torch.float32, device=residual.device)

    if teacher_w:
        if teacher_fn is None:
            raise ValueError(f"loss mode {loss_mode} requires a teacher function")
        with torch.no_grad():
            teacher = teacher_fn(residual)
        teacher_ratio = l2_ratio(correction - teacher, teacher)
        teacher_next = residual - model.rte.A_op(teacher)
        teacher_action_ratio = l2_ratio(next_residual - teacher_next, residual.detach())
        if loss_mode == "solver_teacher":
            loss = loss + teacher_w * float(teacher_action_weight) * teacher_action_ratio.square().mean()
        else:
            loss = loss + teacher_w * teacher_ratio.square().mean()
    else:
        teacher_ratio = torch.zeros_like(metric_ratio)
        teacher_action_ratio = torch.zeros_like(metric_ratio)

    if probe_weight > 0.0:
        probe_source = residual if probe_residuals is None else probe_residuals
        probe_penalty, probe_scalars = probe_growth_penalty(
            model,
            probe_source,
            probe_steps=probe_steps,
            contraction_target=probe_contraction_target,
        )
        loss = loss + float(probe_weight) * probe_penalty
    else:
        probe_scalars = {"probe_ratio": 0.0, "probe_peak_ratio": 0.0, "probe_penalty": 0.0}

    scalars = {
        "loss": float(loss.detach()),
        "metric_ratio": float(metric_ratio.detach().mean()),
        "raw_ratio": float(physical_ratio.detach().mean()),
        "correction_ratio": float(correction_ratio.detach().mean()),
        "teacher_ratio": float(teacher_ratio.detach().mean()),
        "teacher_action_ratio": float(teacher_action_ratio.detach().mean()),
        "stability": float(stability.detach()),
        **probe_scalars,
    }
    if rollout_metric_values:
        scalars["rollout_metric_ratio"] = float(torch.cat([x.detach() for x in rollout_metric_values]).mean())
    else:
        scalars["rollout_metric_ratio"] = 0.0
    if rollout_raw_values:
        scalars["rollout_raw_ratio"] = float(torch.cat([x.detach() for x in rollout_raw_values]).mean())
    else:
        scalars["rollout_raw_ratio"] = 0.0
    return loss, scalars


def make_model(args, rte, base_kind, tsa_harmonics):
    if args.model_kind == "angular_spectral_defect2":
        return AngularSpectralDefect2Preconditioner(
            rte,
            base_kind=base_kind,
            tsa_harmonics=tsa_harmonics,
            width=args.width,
            levels=args.levels,
            mode_width=args.angular_mode_width,
            angular_mix_rank=args.angular_mix_rank,
            angular_mix_scale=args.angular_mix_scale,
            angular_mode_mix_scale=args.angular_mode_mix_scale,
            residual_gate_scale=args.angular_residual_gate_scale,
            gate_scale=args.angular_gate_scale,
            defect_relaxation=args.defect_relaxation,
            output_init=args.output_init,
        )
    if args.model_kind == "unet_residual":
        return NPBSUNetPreconditioner(
            rte,
            form="residual",
            base_kind=base_kind,
            tsa_harmonics=tsa_harmonics,
            width=args.width,
            levels=args.levels,
            output_init=args.output_init,
        )
    if args.model_kind == "unet_defect2":
        return NPBSUNetPreconditioner(
            rte,
            form="defect2",
            base_kind=base_kind,
            tsa_harmonics=tsa_harmonics,
            width=args.width,
            levels=args.levels,
            defect_relaxation=args.defect_relaxation,
            output_init=args.output_init,
        )
    if args.model_kind == "moment_unet_defect2":
        return MomentNPBSUNetPreconditioner(
            rte,
            base_kind=base_kind,
            tsa_harmonics=tsa_harmonics,
            moment_harmonics=args.moment_harmonics,
            width=args.width,
            levels=args.levels,
            defect_relaxation=args.defect_relaxation,
            output_init=args.output_init,
        )
    if args.model_kind == "coeff_weighted_moment_tsa":
        return CoeffWeightedMomentTSAPreconditioner(
            rte,
            moment_harmonics=args.moment_harmonics,
            width=args.width,
            blocks=max(1, args.levels),
            weight_scale=args.moment_weight_scale,
            transport_relaxation=args.moment_transport_relaxation,
            synthetic_relaxation=args.moment_synthetic_relaxation,
            freeze_scalar_weight=not args.learn_scalar_moment,
        )
    if args.model_kind == "coeff_mixed_moment_tsa":
        return CoeffMixedMomentTSAPreconditioner(
            rte,
            moment_harmonics=args.moment_harmonics,
            width=args.width,
            blocks=max(1, args.levels),
            weight_scale=args.moment_weight_scale,
            transport_relaxation=args.moment_transport_relaxation,
            synthetic_relaxation=args.moment_synthetic_relaxation,
            freeze_scalar_weight=not args.learn_scalar_moment,
        )
    if args.model_kind == "learned_rp_moment_tsa":
        return LearnedRPMomentTSAPreconditioner(
            rte,
            moment_harmonics=args.moment_harmonics,
            width=args.width,
            blocks=max(1, args.levels),
            weight_scale=args.moment_weight_scale,
            transport_relaxation=args.moment_transport_relaxation,
            synthetic_relaxation=args.moment_synthetic_relaxation,
            freeze_scalar_weight=not args.learn_scalar_moment,
        )
    if args.model_kind == "residual_subspace_moment_tsa":
        return ResidualSubspaceMomentTSAPreconditioner(
            rte,
            moment_harmonics=args.moment_harmonics,
            width=args.width,
            blocks=max(1, args.levels),
            weight_scale=args.moment_weight_scale,
            transport_relaxation=args.moment_transport_relaxation,
            synthetic_relaxation=args.moment_synthetic_relaxation,
            freeze_scalar_weight=not args.learn_scalar_moment,
        )
    raise ValueError(f"unknown model kind: {args.model_kind}")


def train_harness(model, residuals, args, teacher_fn=None, evaluator=None, probe_residuals=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        args.scheduler,
        int(args.steps),
        args.lr,
        int(args.scheduler_step_size),
        float(args.scheduler_gamma),
        args.scheduler_eta_min,
        args.scheduler_min_lr_ratio,
        args.scheduler_pct_start,
    )

    history = []
    def teacher_weight_at(step):
        ramp_steps = int(getattr(args, "teacher_ramp_steps", 0))
        if ramp_steps <= 0:
            return float(args.teacher_weight)
        return float(args.teacher_weight) * min(1.0, float(step) / float(ramp_steps))

    current_identity_weight = float(args.identity_weight)
    initial_batch = residuals() if callable(residuals) else residuals
    initial_probe_batch = probe_residuals() if callable(probe_residuals) else probe_residuals
    with torch.no_grad():
        _, initial = objective_components(model, initial_batch, args.loss_mode, args.metric_map, args.physical_weight,
                                          args.rollout_weight, args.rollout_steps, args.rollout_decay,
                                          args.stability_weight, args.contraction_target, args.correction_weight,
                                          teacher_fn=teacher_fn,
                                          teacher_weight=teacher_weight_at(0),
                                          teacher_action_weight=args.teacher_action_weight,
                                          identity_weight=current_identity_weight,
                                          probe_residuals=initial_probe_batch,
                                          probe_weight=args.probe_weight,
                                          probe_steps=args.probe_steps,
                                          probe_contraction_target=args.probe_contraction_target)
    initial["step"] = 0
    initial["lr"] = optimizer.param_groups[0]["lr"]
    initial["identity_weight"] = current_identity_weight
    history.append(initial)

    checkpoint = None
    checkpoint_baseline = None
    if evaluator is not None and args.select_checkpoint != "none":
        initial_eval = evaluator(model)
        checkpoint_baseline = initial_eval
        initial_score = checkpoint_score(
            initial_eval,
            args.select_checkpoint,
            baseline_rows=checkpoint_baseline,
            composite_phys_weight=args.checkpoint_phys_weight,
            composite_stationary_weight=args.checkpoint_stationary_weight,
        )
        checkpoint = {
            "metric": args.select_checkpoint,
            "step": 0,
            "score": initial_score,
            "rows": initial_eval,
            "identity_weight": current_identity_weight,
        }
        if args.verbose:
            print(
                f"eval {0:5d}/{args.steps} {args.select_checkpoint}={initial_score:.3e} "
                f"best_step=0 best={initial_score:.3e}"
            )

    start = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        current = residuals() if callable(residuals) else residuals
        current_probe = probe_residuals() if callable(probe_residuals) else probe_residuals
        optimizer.zero_grad(set_to_none=True)
        loss, scalars = objective_components(
            model,
            current,
            args.loss_mode,
            metric_map=args.metric_map,
            physical_weight=args.physical_weight,
            rollout_weight=args.rollout_weight,
            rollout_steps=args.rollout_steps,
            rollout_decay=args.rollout_decay,
            stability_weight=args.stability_weight,
            contraction_target=args.contraction_target,
            correction_weight=args.correction_weight,
            teacher_fn=teacher_fn,
            teacher_weight=teacher_weight_at(step),
            teacher_action_weight=args.teacher_action_weight,
            identity_weight=current_identity_weight,
            probe_residuals=current_probe,
            probe_weight=args.probe_weight,
            probe_steps=args.probe_steps,
            probe_contraction_target=args.probe_contraction_target,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if args.history_every > 0 and (step == 1 or step == args.steps or step % args.history_every == 0):
            scalars["step"] = step
            scalars["lr"] = optimizer.param_groups[0]["lr"]
            scalars["identity_weight"] = current_identity_weight
            history.append(scalars)
            if args.verbose:
                print(
                    f"step {step:5d}/{args.steps} loss={scalars['loss']:.3e} "
                    f"metric={scalars['metric_ratio']:.3e} raw={scalars['raw_ratio']:.3e} "
                    f"teacher={scalars['teacher_ratio']:.3e}"
                )
        if evaluator is not None and args.select_checkpoint != "none" and args.eval_every > 0:
            if step == 1 or step == args.steps or step % args.eval_every == 0:
                eval_rows = evaluator(model)
                eval_score = checkpoint_score(
                    eval_rows,
                    args.select_checkpoint,
                    baseline_rows=checkpoint_baseline,
                    composite_phys_weight=args.checkpoint_phys_weight,
                    composite_stationary_weight=args.checkpoint_stationary_weight,
                )
                if checkpoint is None or eval_score < checkpoint["score"]:
                    checkpoint = {
                        "metric": args.select_checkpoint,
                        "step": step,
                        "score": eval_score,
                        "rows": eval_rows,
                        "identity_weight": current_identity_weight,
                    }
                if args.verbose:
                    print(
                        f"eval {step:5d}/{args.steps} {args.select_checkpoint}={eval_score:.3e} "
                        f"best_step={checkpoint['step']} best={checkpoint['score']:.3e}"
                    )
                next_identity_weight = update_identity_weight(current_identity_weight, eval_rows, step, args)
                if args.verbose and next_identity_weight < current_identity_weight:
                    print(
                        f"identity step={step:5d}/{args.steps} weight={current_identity_weight:.3e} "
                        f"-> {next_identity_weight:.3e}"
                    )
                current_identity_weight = next_identity_weight

    elapsed = time.perf_counter() - start
    if not history or history[-1].get("step") != args.steps:
        current = residuals() if callable(residuals) else residuals
        current_probe = probe_residuals() if callable(probe_residuals) else probe_residuals
        with torch.no_grad():
            _, final = objective_components(model, current, args.loss_mode, args.metric_map, args.physical_weight,
                                            args.rollout_weight, args.rollout_steps, args.rollout_decay,
                                            args.stability_weight, args.contraction_target, args.correction_weight,
                                            teacher_fn=teacher_fn,
                                            teacher_weight=teacher_weight_at(args.steps),
                                            teacher_action_weight=args.teacher_action_weight,
                                            identity_weight=current_identity_weight,
                                            probe_residuals=current_probe,
                                            probe_weight=args.probe_weight,
                                            probe_steps=args.probe_steps,
                                            probe_contraction_target=args.probe_contraction_target)
        final["step"] = args.steps
        final["lr"] = optimizer.param_groups[0]["lr"]
        final["identity_weight"] = current_identity_weight
        history.append(final)
    return history, elapsed, checkpoint, current_identity_weight


def evaluate_model(rte, model, train_source, heldout_source, base_kind, tsa_harmonics, args):
    rows = []
    for split, source in (("train", train_source), ("heldout", heldout_source)):
        one_true, one_born, one_base = one_step_quality(
            rte,
            model,
            source.to(device=rte.sigma_t.device, dtype=rte.complex_dtype),
            base_kind=base_kind,
            tsa_harmonics=tsa_harmonics,
        )
        stat_iters, stat_final, stat_factor, stat_time, _, _, _ = solve_stationary(
            rte,
            source,
            model,
            maxiter=args.maxiter,
            tol=args.tol,
        )
        fgmres_iters, fgmres_final, fgmres_factor, fgmres_time, _, _, _ = solve_fgmres(
            rte,
            source,
            model,
            maxiter=args.fgmres_maxiter,
            tol=args.tol,
            real_stacked=args.fgmres_mode == "real-stacked",
        )
        rows.append(
            {
                "split": split,
                "one_step_true": one_true,
                "one_step_born_metric": one_born,
                "one_step_base_metric": one_base,
                "stationary_iters": stat_iters,
                "stationary_final": stat_final,
                "stationary_factor": stat_factor,
                "stationary_time": stat_time,
                "fgmres_iters": fgmres_iters,
                "fgmres_final": fgmres_final,
                "fgmres_factor": fgmres_factor,
                "fgmres_time": fgmres_time,
            }
        )
    return rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    torch.manual_seed(args.seed)
    if args.select_checkpoint != "none" and args.eval_every <= 0:
        raise ValueError("--eval-every must be positive when --select-checkpoint is enabled")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.benchmark = True

    rte, _ = make_case(args.case, args.size, args.n_angles, dtype, args.eta_scale, device=device)
    base_kind = args.base_kind
    tsa_harmonics = effective_tsa_harmonics(base_kind, args.tsa_harmonics, args.moment_harmonics)
    residual_rollout_kind = args.residual_teacher_kind or base_kind
    residual_rollout_harmonics = effective_tsa_harmonics(
        residual_rollout_kind,
        args.residual_teacher_harmonics,
        args.moment_harmonics,
    )
    residual_rollout_depth = max(1, int(args.residual_teacher_depth))
    train_source = rte.source.clone()
    training_sources = make_training_sources(
        args.case,
        args.size,
        args.n_angles,
        dtype,
        device=device,
        num_aux_sources=args.aux_sources,
    )
    training_sources[0] = train_source.clone()
    heldout_source = make_source(
        args.size,
        args.n_angles,
        dtype,
        center=(0.68, 0.34),
        angular="beam" if args.case in ("anisotropic", "mixed") else "dipole",
    ).to(device=device)

    if args.residual_mode == "pool":
        residuals = collect_residuals_from_sources(
            rte,
            base_kind,
            tsa_harmonics=tsa_harmonics,
            sources=training_sources,
            n_rollout=args.n_rollout,
            n_random=args.n_random,
            random_modes=args.random_modes,
            rollout_kind=residual_rollout_kind,
            rollout_harmonics=residual_rollout_harmonics,
            rollout_depth=residual_rollout_depth,
            rollout_relaxation=args.residual_teacher_relaxation,
        )
    else:
        residuals, _ = make_residual_sampler(
            rte,
            base_kind,
            tsa_harmonics=tsa_harmonics,
            sources=training_sources,
            n_rollout=args.n_rollout,
            residuals_per_step=args.residuals_per_step,
            rollout_samples_per_step=args.rollout_samples_per_step,
            random_modes=args.random_modes,
            seed=args.seed,
            rollout_kind=residual_rollout_kind,
            rollout_harmonics=residual_rollout_harmonics,
            rollout_depth=residual_rollout_depth,
            rollout_relaxation=args.residual_teacher_relaxation,
        )

    model = make_model(args, rte, base_kind, tsa_harmonics).to(device)
    teacher_fn = None
    if args.teacher_depth > 0:
        teacher_kind = args.teacher_kind or base_kind
        teacher_harmonics = effective_tsa_harmonics(teacher_kind, args.teacher_harmonics, args.moment_harmonics)

        def teacher_fn(residual):
            return inner_base_apply(
                rte,
                residual,
                depth=args.teacher_depth,
                kind=teacher_kind,
                tsa_harmonics=teacher_harmonics,
                relaxation=args.teacher_relaxation,
            )

    evaluator = None
    if args.select_checkpoint != "none":
        def evaluator(current_model):
            with torch.no_grad():
                return evaluate_model(rte, current_model, train_source, heldout_source, base_kind, tsa_harmonics, args)

    probe_residuals = None
    if args.probe_weight > 0.0 and (args.probe_residuals_per_step > 0 or args.probe_rollout_samples_per_step > 0):
        probe_rollout = args.probe_rollout_samples_per_step
        if probe_rollout <= 0:
            probe_rollout = min(args.rollout_samples_per_step, args.probe_residuals_per_step)
        probe_residuals, _ = make_residual_sampler(
            rte,
            base_kind,
            tsa_harmonics=tsa_harmonics,
            sources=training_sources,
            n_rollout=args.n_rollout,
            residuals_per_step=max(1, args.probe_residuals_per_step),
            rollout_samples_per_step=max(0, probe_rollout),
            random_modes=args.random_modes,
            seed=args.seed + 7919,
        )

    history, train_time, checkpoint, final_identity_weight = train_harness(
        model,
        residuals,
        args,
        teacher_fn=teacher_fn,
        evaluator=evaluator,
        probe_residuals=probe_residuals,
    )
    eval_rows = checkpoint["rows"] if checkpoint is not None else evaluate_model(
        rte,
        model,
        train_source,
        heldout_source,
        base_kind,
        tsa_harmonics,
        args,
    )
    selected_step = checkpoint["step"] if checkpoint is not None else args.steps
    selected_score = checkpoint["score"] if checkpoint is not None else ""
    selected_identity_weight = checkpoint["identity_weight"] if checkpoint is not None else args.identity_weight

    summary_rows = []
    initial = history[0]
    final = history[-1]
    for row in eval_rows:
        summary = {
            "case": args.case,
            "size": args.size,
            "n_angles": args.n_angles,
            "base_kind": base_kind,
            "tsa_harmonics": "" if tsa_harmonics is None else tsa_harmonics,
            "model_kind": args.model_kind,
            "loss_mode": args.loss_mode,
            "metric_map": args.metric_map,
            "width": args.width,
            "levels": args.levels,
            "params": count_parameters(model),
            "steps": args.steps,
            "train_time": train_time,
            "scheduler": args.scheduler,
            "scheduler_step_size": args.scheduler_step_size,
            "scheduler_gamma": args.scheduler_gamma,
            "scheduler_eta_min": args.scheduler_eta_min,
            "scheduler_min_lr_ratio": args.scheduler_min_lr_ratio,
            "scheduler_pct_start": args.scheduler_pct_start,
            "teacher_depth": args.teacher_depth,
            "teacher_kind": args.teacher_kind or base_kind,
            "teacher_harmonics": args.teacher_harmonics,
            "teacher_relaxation": args.teacher_relaxation,
            "teacher_weight": args.teacher_weight,
            "teacher_action_weight": args.teacher_action_weight,
            "teacher_ramp_steps": args.teacher_ramp_steps,
            "residual_teacher_depth": args.residual_teacher_depth,
            "residual_teacher_kind": args.residual_teacher_kind or base_kind,
            "residual_teacher_harmonics": args.residual_teacher_harmonics,
            "residual_teacher_relaxation": args.residual_teacher_relaxation,
            "identity_weight": args.identity_weight,
            "identity_schedule": args.identity_schedule,
            "identity_weight_min": args.identity_weight_min,
            "identity_decay_factor": args.identity_decay_factor,
            "identity_decay_start": args.identity_decay_start,
            "identity_safety_threshold": args.identity_safety_threshold,
            "selected_identity_weight": selected_identity_weight,
            "final_identity_weight": final_identity_weight,
            "rollout_weight": args.rollout_weight,
            "rollout_steps": args.rollout_steps,
            "rollout_decay": args.rollout_decay,
            "stability_weight": args.stability_weight,
            "contraction_target": args.contraction_target,
            "probe_weight": args.probe_weight,
            "probe_steps": args.probe_steps,
            "probe_contraction_target": args.probe_contraction_target,
            "probe_residuals_per_step": args.probe_residuals_per_step,
            "probe_rollout_samples_per_step": args.probe_rollout_samples_per_step,
            "moment_harmonics": args.moment_harmonics,
            "moment_weight_scale": args.moment_weight_scale,
            "moment_transport_relaxation": args.moment_transport_relaxation,
            "moment_synthetic_relaxation": args.moment_synthetic_relaxation,
            "learn_scalar_moment": args.learn_scalar_moment,
            "angular_mode_width": args.angular_mode_width,
            "angular_mix_rank": args.angular_mix_rank,
            "angular_mix_scale": args.angular_mix_scale,
            "angular_mode_mix_scale": args.angular_mode_mix_scale,
            "angular_residual_gate_scale": args.angular_residual_gate_scale,
            "angular_gate_scale": args.angular_gate_scale,
            "residual_mode": args.residual_mode,
            "residuals_per_step": args.residuals_per_step,
            "rollout_samples_per_step": args.rollout_samples_per_step,
            "n_rollout": args.n_rollout,
            "n_random": args.n_random,
            "random_modes": args.random_modes,
            "aux_sources": args.aux_sources,
            "checkpoint_metric": args.select_checkpoint,
            "checkpoint_phys_weight": args.checkpoint_phys_weight,
            "checkpoint_stationary_weight": args.checkpoint_stationary_weight,
            "checkpoint_eval_every": args.eval_every,
            "selected_step": selected_step,
            "selected_score": selected_score,
            "loss_initial": initial["loss"],
            "loss_final": final["loss"],
            "metric_initial": initial["metric_ratio"],
            "metric_final": final["metric_ratio"],
            "raw_initial": initial["raw_ratio"],
            "raw_final": final["raw_ratio"],
            "teacher_initial": initial["teacher_ratio"],
            "teacher_final": final["teacher_ratio"],
            "teacher_action_initial": initial["teacher_action_ratio"],
            "teacher_action_final": final["teacher_action_ratio"],
            **row,
        }
        summary_rows.append(summary)

    write_rows(args.output_csv, summary_rows)
    if args.history_csv:
        hist_rows = []
        for row in history:
            hist_rows.append(
                {
                    "case": args.case,
                    "size": args.size,
                    "model_kind": args.model_kind,
                    "loss_mode": args.loss_mode,
                    "metric_map": args.metric_map,
                    **row,
                }
            )
        write_rows(args.history_csv, hist_rows)

    if args.verbose:
        print(f"wrote {args.output_csv}")
        if args.history_csv:
            print(f"wrote {args.history_csv}")
    return summary_rows, history


def main():
    parser = argparse.ArgumentParser(
        description="Targeted NPBS/RTE bottleneck harness for loss, teacher, and mode-expressivity diagnostics."
    )
    parser.add_argument("--case", choices=["diffusive", "heterogeneous", "anisotropic", "mixed"], default="mixed")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--n-angles", type=int, default=12)
    parser.add_argument("--base-kind", default="transport_tsa")
    parser.add_argument("--tsa-harmonics", type=int, default=None)
    parser.add_argument("--moment-harmonics", type=int, default=4)
    parser.add_argument(
        "--model-kind",
        choices=[
            "unet_residual",
            "unet_defect2",
            "angular_spectral_defect2",
            "moment_unet_defect2",
            "coeff_weighted_moment_tsa",
            "coeff_mixed_moment_tsa",
            "learned_rp_moment_tsa",
            "residual_subspace_moment_tsa",
        ],
        default="unet_defect2",
    )
    parser.add_argument("--loss-mode", choices=sorted(LOSS_MODES), default="metric_raw_rollout")
    parser.add_argument("--metric-map", choices=["base", "forward", "damped_moment"], default="base")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--angular-mode-width", type=int, default=4)
    parser.add_argument("--angular-mix-rank", type=int, default=2)
    parser.add_argument("--angular-mix-scale", type=float, default=0.25)
    parser.add_argument("--angular-mode-mix-scale", type=float, default=0.05)
    parser.add_argument("--angular-residual-gate-scale", type=float, default=0.25)
    parser.add_argument("--angular-gate-scale", type=float, default=0.1)
    parser.add_argument("--defect-relaxation", type=float, default=1.0)
    parser.add_argument("--output-init", choices=["zero", "small"], default="zero")
    parser.add_argument("--moment-weight-scale", type=float, default=0.5)
    parser.add_argument("--moment-transport-relaxation", type=float, default=0.4)
    parser.add_argument("--moment-synthetic-relaxation", type=float, default=0.4)
    parser.add_argument("--learn-scalar-moment", action="store_true")
    parser.add_argument("--eta-scale", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["none", "step", "cosine", "onecycle"], default="cosine")
    parser.add_argument("--scheduler-step-size", type=int, default=0)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-eta-min", type=float, default=None)
    parser.add_argument("--scheduler-min-lr-ratio", type=float, default=1e-2)
    parser.add_argument("--scheduler-pct-start", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--physical-weight", type=float, default=0.5)
    parser.add_argument("--correction-weight", type=float, default=1e-6)
    parser.add_argument("--rollout-weight", type=float, default=0.2)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--rollout-decay", type=float, default=0.8)
    parser.add_argument("--stability-weight", type=float, default=0.0)
    parser.add_argument("--contraction-target", type=float, default=0.95)
    parser.add_argument("--teacher-depth", type=int, default=0)
    parser.add_argument("--teacher-kind", default="")
    parser.add_argument("--teacher-harmonics", type=int, default=None)
    parser.add_argument("--teacher-relaxation", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--teacher-action-weight", type=float, default=1.0)
    parser.add_argument("--teacher-ramp-steps", type=int, default=0)
    parser.add_argument("--identity-weight", type=float, default=0.0)
    parser.add_argument("--residual-teacher-depth", type=int, default=0)
    parser.add_argument("--residual-teacher-kind", default="")
    parser.add_argument("--residual-teacher-harmonics", type=int, default=None)
    parser.add_argument("--residual-teacher-relaxation", type=float, default=1.0)
    parser.add_argument("--identity-schedule", choices=["none", "safe_decay"], default="none")
    parser.add_argument("--identity-weight-min", type=float, default=0.0)
    parser.add_argument("--identity-decay-factor", type=float, default=0.5)
    parser.add_argument("--identity-decay-start", type=int, default=1)
    parser.add_argument("--identity-safety-threshold", type=float, default=0.0)
    parser.add_argument("--residual-mode", choices=["pool", "resample"], default="resample")
    parser.add_argument("--residuals-per-step", type=int, default=64)
    parser.add_argument("--rollout-samples-per-step", type=int, default=32)
    parser.add_argument("--n-rollout", type=int, default=8)
    parser.add_argument("--n-random", type=int, default=6)
    parser.add_argument("--random-modes", type=int, default=5)
    parser.add_argument("--aux-sources", type=int, default=2)
    parser.add_argument("--probe-weight", type=float, default=0.0)
    parser.add_argument("--probe-steps", type=int, default=3)
    parser.add_argument("--probe-contraction-target", type=float, default=0.95)
    parser.add_argument("--probe-residuals-per-step", type=int, default=0)
    parser.add_argument("--probe-rollout-samples-per-step", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--fgmres-maxiter", type=int, default=20)
    parser.add_argument("--fgmres-mode", choices=["complex", "real-stacked"], default="real-stacked")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--history-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--select-checkpoint", choices=sorted(CHECKPOINT_METRICS), default="none")
    parser.add_argument("--checkpoint-phys-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-stationary-weight", type=float, default=0.0)
    parser.add_argument("--output-csv", default="rte/npbs_bottleneck_harness.csv")
    parser.add_argument("--history-csv", default="rte/npbs_bottleneck_harness_history.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
