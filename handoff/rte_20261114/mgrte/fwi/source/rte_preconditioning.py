import torch

try:
    from .RTE_ATFPS_func import correction, funcDinv, func_inflow_Dirichlet, restrict_basis, squeeze_basis
except ImportError:
    from RTE_ATFPS_func import correction, funcDinv, func_inflow_Dirichlet, restrict_basis, squeeze_basis


def atfps_operator(value, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device):
    return func_inflow_Dirichlet(value, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, mesh_size, device)


def selected_loss(value, target, VecSize, loss_fn):
    value_selected, _ = squeeze_basis(value, VecSize)
    target_selected, _ = squeeze_basis(target, VecSize)
    return loss_fn(value_selected, target_selected)


def selected_relative_loss(value, target, VecSize, eps=1e-30):
    value_selected, _ = squeeze_basis(value, VecSize)
    target_selected, _ = squeeze_basis(target, VecSize)
    numerator = torch.linalg.norm(value_selected - target_selected)
    denominator = torch.clamp(torch.linalg.norm(target_selected), min=eps)
    return numerator / denominator


def selected_objective(value, target, VecSize, loss_fn, normalization="relative"):
    if normalization == "relative":
        return selected_relative_loss(value, target, VecSize)
    if normalization == "mse":
        return selected_loss(value, target, VecSize, loss_fn)
    raise ValueError(f"loss_normalization {normalization} not supported")


def maybe_resample_rhs(rhs, mode):
    if mode in {"fixed", "replay"}:
        return rhs
    if mode == "random":
        return 2 * torch.rand(rhs.shape, dtype=rhs.dtype, device=rhs.device) - 1
    raise ValueError(f"rhs_mode {mode} not supported")


def corrected_candidate(candidate, residual, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device):
    _, residual_unselected = restrict_basis(residual, VecSize, M, mesh_size, mesh_size, device)
    _, candidate_corrected, _ = correction(
        candidate,
        residual_unselected,
        I2A,
        fsmLRBTC,
        MLRBT,
        VecSize,
        M,
        mesh_size,
        mesh_size,
        device,
    )
    return candidate_corrected


def apply_preconditioner_form(reference, update, form):
    if form == "direct":
        return update
    if form == "residual":
        return reference + update
    raise ValueError(f"preconditioner_form {form} not supported")


def mgnet_preconditioner(
    residual,
    model,
    a_list,
    inv_a_list,
    I2A,
    fsmLRBTC,
    MLRBT,
    VecSize,
    M,
    mesh_size,
    device,
    mode="raw",
    form="direct",
):
    if mode == "raw":
        reference = residual
    elif mode == "born":
        reference = funcDinv(residual, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, mesh_size, device)
    else:
        raise ValueError(f"preconditioner mode {mode} not supported")

    update = model(reference, a_list, inv_a_list)
    candidate = apply_preconditioner_form(reference, update, form)
    return corrected_candidate(candidate, residual, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device)


def training_loss(
    model,
    batch,
    M,
    mesh_size,
    device,
    loss_fn,
    loss_type="raw",
    preconditioner_form="direct",
    physical_loss_weight=0.1,
    rhs_mode="random",
    loss_normalization="relative",
    return_metrics=False,
):
    Coef, fsmLRBTC, I2A, MLRBT, VecSize, rhs = [tensor.to(device) for tensor in batch]
    rhs = maybe_resample_rhs(rhs, rhs_mode)
    a_list, inv_a_list = model.setup(Coef)

    if loss_type == "raw":
        pred = model(rhs, a_list, inv_a_list)
        Apred = atfps_operator(pred, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device)
        loss = selected_objective(Apred, rhs, VecSize, loss_fn, loss_normalization)
        if return_metrics:
            return loss, {"objective": loss.detach(), "physical": loss.detach()}
        return loss

    if loss_type not in {"born", "born_mixed"}:
        raise ValueError(f"loss_type {loss_type} not supported")

    q = funcDinv(rhs, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, mesh_size, device)
    update = model(q, a_list, inv_a_list)
    pred = apply_preconditioner_form(q, update, preconditioner_form)
    pred = corrected_candidate(pred, rhs, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device)

    Apred = atfps_operator(pred, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device)
    q_pred = funcDinv(Apred, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, mesh_size, device)
    born_loss = selected_objective(q_pred, q, VecSize, loss_fn, loss_normalization)

    if loss_type == "born":
        if return_metrics:
            physical_loss = selected_objective(Apred, rhs, VecSize, loss_fn, loss_normalization)
            return born_loss, {
                "objective": born_loss.detach(),
                "born": born_loss.detach(),
                "physical": physical_loss.detach(),
            }
        return born_loss

    physical_loss = selected_objective(Apred, rhs, VecSize, loss_fn, loss_normalization)
    loss = born_loss + physical_loss_weight * physical_loss
    if return_metrics:
        return loss, {
            "objective": loss.detach(),
            "born": born_loss.detach(),
            "physical": physical_loss.detach(),
        }
    return loss


def d_metric_relative_residual(residual, rhs, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, device):
    pre_residual = funcDinv(residual, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, mesh_size, device)
    pre_rhs = funcDinv(rhs, I2A, fsmLRBTC, MLRBT, VecSize, M, mesh_size, mesh_size, device)
    denom = torch.clamp(torch.linalg.norm(pre_rhs), min=1e-30)
    return (torch.linalg.norm(pre_residual) / denom).item()
