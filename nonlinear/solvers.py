"""Residual-correction inner iteration and undamped outer Newton iteration."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from .operators import NewtonOperator, relative_norm


@dataclass
class SolveResult:
    solution: Tensor
    residual_history: Tensor  # (number of updates + 1, batch)
    iterations: Tensor  # per sample, excluding updates after convergence
    converged: Tensor

    @property
    def final_residual(self) -> Tensor:
        return self.residual_history[-1]


@torch.no_grad()
def solve_inner(operator: NewtonOperator, correction_map: Callable[[Tensor, Tensor], Tensor],
                state: Tensor, rhs: Tensor | None = None, max_iterations: int = 100,
                tolerance: float = 1e-4) -> SolveResult:
    """Solve J(state) delta = rhs using delta += correction_map(state, residual).

    The reported residual belongs to the returned iterate, including when the
    iteration budget is zero. Converged samples are frozen independently.
    """
    if max_iterations < 0 or tolerance <= 0:
        raise ValueError("max_iterations must be nonnegative and tolerance positive")
    rhs = operator.rhs(state) if rhs is None else rhs
    if rhs.shape != state.shape:
        raise ValueError("rhs and state must have identical shapes")
    delta = torch.zeros_like(state)
    residual = rhs.clone()
    history = [relative_norm(residual, rhs)]
    if not torch.isfinite(state).all() or not torch.isfinite(history[0]).all():
        raise ValueError("state and rhs must be finite")
    iterations = torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
    for _ in range(max_iterations):
        active = history[-1] > tolerance
        if not active.any():
            break
        update = correction_map(state, residual)
        if update.shape != delta.shape:
            raise ValueError("correction map returned an incompatible shape")
        delta = delta + torch.where(active[:, None, None, None], update, 0)
        iterations += active
        residual = rhs - operator(state, delta)
        norm = relative_norm(residual, rhs)
        if not torch.isfinite(norm).all():
            raise FloatingPointError("inner iteration produced a non-finite residual")
        history.append(norm)
    values = torch.stack(history)
    return SolveResult(delta, values, iterations, values[-1] <= tolerance)


@torch.no_grad()
def solve_newton(operator: NewtonOperator, correction_map: Callable[[Tensor, Tensor], Tensor],
                 initial: Tensor, max_iterations: int = 10, tolerance: float = 1e-4,
                 inner_iterations: int = 100, inner_tolerance: float = 1e-4) -> SolveResult:
    """Undamped Newton updates; convergence is measured against the source norm.

    Inner iteration exhaustion does not imply convergence. The actual nonlinear
    residual remains the outer stopping criterion, as in the original solver.
    """
    if max_iterations < 0 or tolerance <= 0 or inner_iterations < 0 or inner_tolerance <= 0:
        raise ValueError("iteration budgets must be nonnegative and tolerances positive")
    state = initial.clone()
    reference = operator.source.to(state).expand_as(state)
    history = [relative_norm(operator.residual(state), reference)]
    if not torch.isfinite(history[0]).all():
        raise ValueError("initial nonlinear residual must be finite")
    iterations = torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
    for _ in range(max_iterations):
        active = history[-1] > tolerance
        if not active.any():
            break
        inner = solve_inner(operator, correction_map, state, max_iterations=inner_iterations,
                            tolerance=inner_tolerance)
        state += torch.where(active[:, None, None, None], inner.solution, 0)
        iterations += active
        norm = relative_norm(operator.residual(state), reference)
        if not torch.isfinite(norm).all():
            raise FloatingPointError("Newton iteration produced a non-finite residual")
        history.append(norm)
    values = torch.stack(history)
    return SolveResult(state, values, iterations, values[-1] <= tolerance)
