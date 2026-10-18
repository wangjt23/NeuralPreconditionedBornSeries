"""FFT collocation operators used by the original CDR experiments.

The equation is ``-kappa * Δu - ∇kappa · ∇u + v · ∇u + u = f``
on a periodic square. Spatial axis 0 is x and axis 1 is y. Products are
evaluated pointwise without dealiasing, matching the research implementation.
"""

import torch
from torch import nn


class SpectralPDELoss(nn.Module):
    """Physical operator and per-sample constant-coefficient Green operator.

    ``coefficients`` has shape ``(batch, >=3, N, N)`` with channels
    ``kappa, vx, vy[, f]``. The reaction coefficient and damping are both one.
    Fourier grids are buffers so normal ``.to(device)`` transfers work.
    """

    def __init__(self, resolution=128, dx=None, dtype=torch.float32):
        super().__init__()
        if resolution < 4:
            raise ValueError("resolution must be at least 4")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("FFT physics requires float32 or float64")
        self.resolution = int(resolution)
        self.dx = 1.0 / resolution if dx is None else float(dx)
        if self.dx <= 0:
            raise ValueError("dx must be positive")
        frequency = 2 * torch.pi * torch.fft.fftfreq(resolution, d=self.dx, dtype=dtype)
        self.register_buffer("xi_x", frequency.view(1, 1, -1, 1), persistent=False)
        self.register_buffer("xi_y", frequency.view(1, 1, 1, -1), persistent=False)
        self.register_buffer("xi2", self.xi_x.square() + self.xi_y.square(), persistent=False)

    def _check_shape(self, field, coefficients):
        shape = (self.resolution, self.resolution)
        if field.ndim != 4 or field.shape[1] != 1 or field.shape[-2:] != shape:
            raise ValueError(f"field must have shape (batch, 1, {shape[0]}, {shape[1]})")
        if (coefficients.ndim != 4 or coefficients.shape[1] < 3
                or coefficients.shape[0] != field.shape[0] or coefficients.shape[-2:] != shape):
            raise ValueError("coefficients must have matching batch/grid and at least three channels")

    def gradient(self, field):
        """Return first derivatives along the two spatial axes."""
        transform = torch.fft.fft2(field)
        return (
            torch.fft.ifft2(1j * self.xi_x * transform).real,
            torch.fft.ifft2(1j * self.xi_y * transform).real,
        )

    def apply_A_op(self, field, coefficients):
        """Apply the variable-coefficient CDR operator with reaction = 1."""
        self._check_shape(field, coefficients)
        kappa, vx, vy = (coefficients[:, i:i + 1] for i in range(3))
        ux, uy = self.gradient(field)
        kx, ky = self.gradient(kappa)
        laplacian = torch.fft.ifft2(-self.xi2 * torch.fft.fft2(field)).real
        diffusion = -kappa * laplacian
        coefficient_gradient = -(kx * ux + ky * uy)
        convection = vx * ux + vy * uy
        return diffusion + coefficient_gradient + convection + field

    def green_symbol(self, coefficients):
        """Return 1 / (mean(kappa)|xi|² + i mean(v)·xi + 2) per sample."""
        mean = coefficients[:, :3].mean(dim=(-2, -1), keepdim=True)
        return 1.0 / (
            mean[:, 0:1] * self.xi2
            + 1j * (mean[:, 1:2] * self.xi_x + mean[:, 2:3] * self.xi_y)
            + 2.0
        )

    @staticmethod
    def apply_G_op(field, symbol):
        """Apply a previously computed Green symbol, returning a real field."""
        return torch.fft.ifft2(symbol * torch.fft.fft2(field)).real

    def apply_G_op_outer(self, field, coefficients):
        self._check_shape(field, coefficients)
        return self.apply_G_op(field, self.green_symbol(coefficients))

    def forward(self, solution, coefficients, rhs=None, use_pre=True):
        """Mean relative physical or left-preconditioned residual norm.

        ``rhs`` is explicit during label-free training, so the input dataset is
        never modified. If omitted, the fourth coefficient channel supplies it.
        """
        if rhs is None:
            if coefficients.shape[1] < 4:
                raise ValueError("supply rhs or a fourth coefficient channel")
            rhs = coefficients[:, 3:4]
        residual = self.apply_A_op(solution, coefficients) - rhs
        if use_pre:
            symbol = self.green_symbol(coefficients)
            residual = self.apply_G_op(residual, symbol)
            rhs = self.apply_G_op(rhs, symbol)
        numerator = torch.linalg.vector_norm(residual.flatten(1), dim=1)
        denominator = torch.linalg.vector_norm(rhs.flatten(1), dim=1)
        return (numerator / (denominator + 1e-8)).mean()
