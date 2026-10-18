"""Five-point Newton Jacobian and differentiable DST-I Poisson inverse."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def dst1(x: Tensor, dim: int = -1) -> Tensor:
    """Unnormalized sine transform: sum_j x_j sin(pi j k / (n + 1))."""
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    zero = torch.zeros_like(x[..., :1])
    extension = torch.cat((zero, x, zero, -x.flip(-1)), dim=-1)
    transformed = -0.5 * torch.fft.rfft(extension, dim=-1).imag[..., 1:n + 1]
    return transformed.movedim(-1, dim)


class DirichletPoissonInverse(nn.Module):
    """Exact inverse of the five-point negative Laplacian on an n x n grid."""

    def __init__(self, grid_size: int):
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        self.grid_size = grid_size
        # Retain double precision until the module is explicitly cast by its caller.
        modes = torch.arange(1, grid_size + 1, dtype=torch.float64)
        eigenvalues_1d = 4 * (grid_size + 1)**2 * torch.sin(
            math.pi * modes / (2 * (grid_size + 1))
        ).square()
        self.register_buffer("eigenvalues", eigenvalues_1d[:, None] + eigenvalues_1d[None, :])

    def forward(self, rhs: Tensor) -> Tensor:
        if rhs.shape[-2:] != (self.grid_size, self.grid_size):
            raise ValueError("rhs spatial shape does not match the Poisson grid")
        coefficients = dst1(dst1(rhs, -1), -2)
        coefficients = coefficients / self.eigenvalues.to(rhs)
        return dst1(dst1(coefficients, -1), -2) * (2 / (self.grid_size + 1))**2


class NewtonOperator(nn.Module):
    """F(u) = (-Delta_h)u - u**2 - f; J(u)v = (-Delta_h)v - 2uv.

    Values have shape (batch, 1, n, n); the boundary outside this interior grid
    is zero. ``legacy-endpoints`` reproduces the source sampling convention
    in the original experiments. ``interior`` samples f at the actual grid.
    """

    def __init__(self, grid_size: int = 64, source_scale: float = 1600,
                 source_grid: str = "legacy-endpoints"):
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if source_grid not in {"interior", "legacy-endpoints"}:
            raise ValueError("source_grid must be 'interior' or 'legacy-endpoints'")
        self.grid_size = grid_size
        self.source_scale = source_scale
        self.source_grid = source_grid
        self.h = 1 / (grid_size + 1)
        kernel = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]],
                              dtype=torch.float64).reshape(1, 1, 3, 3) / self.h**2
        self.register_buffer("kernel", kernel)
        if source_grid == "interior":
            points = torch.arange(1, grid_size + 1, dtype=torch.float64) * self.h
        else:
            points = torch.linspace(0, 1, grid_size, dtype=torch.float64)
        wave = torch.sin(math.pi * points)
        self.register_buffer("source", -source_scale * wave[:, None] * wave[None, :])

    def _check_shape(self, value: Tensor) -> None:
        if value.ndim != 4 or value.shape[1:] != (1, self.grid_size, self.grid_size):
            raise ValueError(f"expected (batch, 1, {self.grid_size}, {self.grid_size})")

    def negative_laplacian(self, value: Tensor) -> Tensor:
        self._check_shape(value)
        return F.conv2d(value, self.kernel.to(value), padding=1)

    def residual(self, state: Tensor) -> Tensor:
        return self.negative_laplacian(state) - state.square() - self.source.to(state)

    def rhs(self, state: Tensor) -> Tensor:
        """Right-hand side -F(u) of the Newton correction equation."""
        return -self.residual(state)

    def forward(self, state: Tensor, correction: Tensor) -> Tensor:
        self._check_shape(state)
        if state.shape != correction.shape:
            raise ValueError("state and correction must have identical shapes")
        return self.negative_laplacian(correction) - 2 * state * correction


def relative_norm(error: Tensor, reference: Tensor) -> Tensor:
    """Per-sample relative norm, using the absolute norm for a zero reference."""
    numerator = torch.linalg.vector_norm(error.flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(reference.flatten(1), dim=1)
    return numerator / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
