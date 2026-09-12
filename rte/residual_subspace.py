"""Physical residual minimization over classical and learned correction directions.

This is a small real least-squares problem for each sample, using the real
inner product on complex fields. It is nonlinear in the residual and should
be used with flexible GMRES. A one-step safeguard is not a Krylov convergence
theorem. See docs/research/PROJECT_AUDIT_2026_09_12.md for the proof boundary.
"""

import torch
from torch import nn


def squared_norm(x):
    return x.abs().square().flatten(1).sum(1)


def physical_residual_loss(operator, residual, correction):
    """Mean squared per-sample relative physical residual, not mean ratio."""
    denominator = squared_norm(residual).clamp_min(torch.finfo(residual.real.dtype).tiny)
    return (squared_norm(residual - operator(correction)) / denominator).mean()


def residual_minimizing_correction(operator, residual, baseline, directions, ridge=1e-6):
    """Minimize ||r-A(z_B+Q c)|| in a small affine correction space.

    Columns of A Q are normalized and the real Gram system is solved in
    float64 with positive ridge regularization (works with duplicate/zero
    columns). True residuals select among the fitted step, baseline, zero,
    and each baseline+direction candidate. The latter check protects against
    roundoff and never makes the individual one-step residual worse than any
    of these finite candidates. It does not compare future Krylov spaces.
    """
    if ridge <= 0:
        raise ValueError("ridge must be strictly positive")
    if residual.ndim < 2 or baseline.shape != residual.shape:
        raise ValueError("expected matching batched residual and baseline")
    if any(q.shape != residual.shape for q in directions):
        raise ValueError("all correction directions must match residual shape")
    baseline_error = residual - operator(baseline)
    candidates = [baseline, torch.zeros_like(baseline)]
    if directions:
        actions = torch.stack([operator(q) for q in directions], dim=-1)
        flat = actions.reshape(residual.shape[0], -1, len(directions))
        # Normalize by detached action norms. The floor leaves a finite
        # derivative at zero output; pure projection still has a zero-output
        # stationary point, so projected_training_loss also trains the raw map.
        # With positive ridge this stop-gradient normalization gives a surrogate
        # derivative, not the exact derivative of the displayed forward loss.
        norm = flat.abs().square().sum(1).sqrt().detach()
        floor = squared_norm(residual).sqrt().detach().unsqueeze(1) * 1e-12
        scale = torch.maximum(norm, floor).clamp_min(1e-30)
        flat = flat / scale.unsqueeze(1)
        work = flat.to(torch.complex128 if flat.is_complex() else torch.float64)
        target = baseline_error.flatten(1).to(work.dtype)
        gram = (work.mH @ work).real
        rhs = (work.mH @ target.unsqueeze(-1)).real
        eye = torch.eye(len(directions), dtype=gram.dtype, device=gram.device)
        weights = torch.linalg.solve(gram + float(ridge) * eye, rhs).squeeze(-1)
        weights = (weights / scale).to(residual.real.dtype)
        fitted = baseline
        for j, direction in enumerate(directions):
            fitted = fitted + weights[:, j].reshape(-1, *([1] * (residual.ndim - 1))) * direction
        candidates.append(fitted)
        candidates.extend(baseline + q for q in directions)
    errors = [baseline_error, residual]
    errors.extend(residual - operator(candidate) for candidate in candidates[2:])
    scores = torch.stack([squared_norm(e) for e in errors], dim=1)
    scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, float("inf")))
    selected = scores.argmin(1)
    stacked = torch.stack(candidates, dim=1)
    return stacked[torch.arange(residual.shape[0], device=residual.device), selected]


class ResidualMinimizedPreconditioner(nn.Module):
    """Use a model's explicit classical baseline and learned correction once."""

    def __init__(self, model, ridge=1e-6, rescale_baseline=True):
        super().__init__()
        if not hasattr(model, "correction_parts"):
            raise TypeError("model must expose correction_parts(residual)")
        self.model = model
        self.ridge = float(ridge)
        self.rescale_baseline = bool(rescale_baseline)
        self.base_calls_per_apply = model.base_calls_per_apply

    @property
    def rte(self):
        return self.model.rte

    def forward(self, residual):
        baseline, delta = self.model.correction_parts(residual)
        directions = [delta, baseline] if self.rescale_baseline else [delta]
        return residual_minimizing_correction(self.rte.A_op, residual, baseline, directions, self.ridge)


def projected_training_loss(model, residual, raw_weight=0.25, ridge=1e-6):
    """Projected physical loss plus native physical loss to escape zero output.

    The native term can supply a nonzero first derivative at zero learned
    output; tangent orthogonality or an exact baseline can still make it zero.
    Detached action normalization gives a surrogate gradient. No solution labels
    are required. raw_weight is explicit and must be strictly positive.
    """
    if raw_weight <= 0:
        raise ValueError("raw_weight must be positive to train from zero output")
    baseline, delta = model.correction_parts(residual)
    projected = residual_minimizing_correction(model.rte.A_op, residual, baseline, [delta, baseline], ridge)
    native_loss = physical_residual_loss(model.rte.A_op, residual, baseline + delta)
    projected_loss = physical_residual_loss(model.rte.A_op, residual, projected)
    return projected_loss + raw_weight * native_loss, {
        "physical_loss": native_loss.detach(),
        "projected_physical_loss": projected_loss.detach(),
    }
