"""Coefficient-conditioned Fourier preconditioner and Richardson iteration.

Adapted from ``cdr/baseline/fnosolver.py`` in the research code. In particular,
the learned Fourier multiplier intentionally retains separate real/imaginary
weights and componentwise multiplication; it is not complex multiplication.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .operators import SpectralPDELoss


ABLATIONS = {
    "npbs": (True, True),
    "preconditioned_l2": (True, False),
    "plain_l2": (False, False),
}


class FourierConv2d(nn.Module):
    """Original channel mixing of real and imaginary Fourier components."""

    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        shape = (in_channels, out_channels, modes_x, modes_y, 2)
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(shape))
        self.weights2 = nn.Parameter(scale * torch.rand(shape))

    def forward(self, field):
        transform = torch.view_as_real(torch.fft.rfft2(field))
        result = field.new_zeros(
            field.shape[0], self.out_channels, field.shape[-2], field.shape[-1] // 2 + 1, 2
        )
        mx, my = self.modes_x, self.modes_y
        result[:, :, :mx, :my] = torch.einsum(
            "bixyz,ioxyz->boxyz", transform[:, :, :mx, :my], self.weights1
        )
        result[:, :, -mx:, :my] = torch.einsum(
            "bixyz,ioxyz->boxyz", transform[:, :, -mx:, :my], self.weights2
        )
        return torch.fft.irfft2(torch.view_as_complex(result), s=field.shape[-2:])


class FourierLayer(nn.Module):
    def __init__(self, features, modes, nonlinear=False):
        super().__init__()
        self.local = nn.Conv2d(features, features, 1, bias=nonlinear)
        self.spectral = FourierConv2d(features, features, *modes)
        self.activation = nn.GELU() if nonlinear else nn.Identity()

    def forward(self, field):
        return self.activation(self.local(field) + self.spectral(field))


class FourierPreconditioner(nn.Module):
    """Nonlinear in coefficients, linear in the supplied residual/RHS."""

    def __init__(self, resolution=128, features=25, modes=(32, 32, 32), padding=8):
        super().__init__()
        padded = resolution + padding
        if resolution < 4 or padding < 0 or padded % 2:
            raise ValueError("resolution >= 4 and padding >= 0 must give an even padded grid")
        if features <= 0 or not modes or any(m <= 0 or m > padded // 2 for m in modes):
            raise ValueError("features/modes must be positive, with modes <= (resolution + padding) / 2")
        self.resolution = resolution
        self.padding = padding
        self.normalize_coefficients = nn.LayerNorm([3, padded, padded])
        self.lift_coefficients = nn.Conv2d(3, features, 3, padding=1)
        self.lift_rhs = nn.Conv2d(1, features, 1, bias=False)
        self.project = nn.Conv2d(features, 1, 1, bias=False)
        self.coefficient_layers = nn.Sequential(*[
            FourierLayer(features, (mode, mode), nonlinear=i < len(modes) - 1)
            for i, mode in enumerate(modes)
        ])
        full_modes = (padded // 2, padded // 2 + 1)
        self.rhs_layer = FourierLayer(features, full_modes)
        self.output_layer = FourierLayer(features, full_modes)

    def forward(self, coefficients, rhs):
        p = self.padding
        coefficients = F.pad(coefficients, (0, p, 0, p))
        rhs = F.pad(rhs, (0, p, 0, p))
        coefficients = self.lift_coefficients(self.normalize_coefficients(coefficients))
        rhs = self.lift_rhs(rhs)
        output = self.output_layer(self.coefficient_layers(coefficients) * self.rhs_layer(rhs))
        return self.project(output)[..., :self.resolution, :self.resolution].contiguous()


class FNOSolver(nn.Module):
    """Learn an inverse action and apply it as a stationary residual correction."""

    def __init__(self, resolution=128, features=25, modes=(32, 32, 32), padding=8,
                 use_precond_res=True, use_precond_loss=True, physics_dtype=torch.float32):
        super().__init__()
        self.config = {
            "resolution": resolution, "features": features, "modes": list(modes),
            "padding": padding, "use_precond_res": use_precond_res,
            "use_precond_loss": use_precond_loss,
        }
        self.use_precond_res = use_precond_res
        self.use_precond_loss = use_precond_loss
        self.precond = FourierPreconditioner(resolution, features, modes, padding).float()
        self.pdeloss = SpectralPDELoss(resolution, dtype=physics_dtype)

    def train_forward(self, coefficients, rhs):
        """Apply the learned inverse action to a batch of random RHS probes."""
        if self.use_precond_res:
            rhs = self.pdeloss.apply_G_op_outer(rhs, coefficients)
        return self.precond(coefficients[:, :3].float(), rhs.float())

    def loss_fn(self, solution, coefficients, rhs):
        return self.pdeloss(solution, coefficients, rhs, use_pre=self.use_precond_loss)

    def forward(self, solution, coefficients):
        residual = coefficients[:, 3:4] - self.pdeloss.apply_A_op(solution, coefficients)
        return solution + self.train_forward(coefficients, residual).to(solution.dtype)

    @torch.no_grad()
    def solve(self, coefficients, iter_num=600, tol=1e-6, divergence_threshold=100.0):
        """Return solution, maximum relative residual, and residual history.

        History includes the initial zero-guess residual and is recomputed
        after every update; its final value therefore belongs to the returned
        solution. Stopping is based on the worst sample in a batch.
        """
        if iter_num < 0 or tol <= 0 or divergence_threshold <= 0:
            raise ValueError("iter_num must be nonnegative and tolerances positive")
        if coefficients.ndim != 4 or coefficients.shape[1] != 4:
            raise ValueError("solve expects (batch, 4, N, N) coefficients including the RHS")
        coefficients = coefficients.to(dtype=self.pdeloss.xi_x.dtype)
        rhs = coefficients[:, 3:4]
        solution = torch.zeros_like(rhs)
        norms = torch.linalg.vector_norm(rhs.flatten(1), dim=1)
        denominator = norms.clamp_min(torch.finfo(rhs.dtype).tiny)

        def relative_residual():
            residual = rhs - self.pdeloss.apply_A_op(solution, coefficients)
            return (torch.linalg.vector_norm(residual.flatten(1), dim=1) / denominator).max().item()

        history = [relative_residual()]
        for _ in range(iter_num):
            if history[-1] < tol or not math.isfinite(history[-1]) or history[-1] > divergence_threshold:
                break
            solution = self(solution, coefficients)
            history.append(relative_residual())
        return solution, history[-1], history
