"""State-conditioned Fourier inverse with the three original loss variants."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .operators import DirichletPoissonInverse, NewtonOperator


@dataclass(frozen=True)
class ModelConfig:
    grid_size: int = 64
    width: int = 25
    modes: tuple[int, ...] = (20, 20)
    padding: int = 8
    variant: str = "npbs-weighted"
    source_scale: float = 1600.0
    source_grid: str = "legacy-endpoints"

    def __post_init__(self):
        if self.grid_size < 2 or self.width < 1 or self.padding < 0:
            raise ValueError("invalid grid_size, width, or padding")
        if self.variant not in {"baseline", "npbs-l2", "npbs-weighted"}:
            raise ValueError("variant must be baseline, npbs-l2, or npbs-weighted")
        if not self.modes or min(self.modes) < 1:
            raise ValueError("modes must contain positive integers")
        if max(self.modes) > (self.grid_size + self.padding) // 2:
            raise ValueError("Fourier modes exceed half the padded grid")


class SpectralLayer(nn.Module):
    """The original Fourier layer with separate real/imaginary multipliers.

    This deliberately preserves the author's ``fourier_conv_2d`` convention;
    it is not the complex multiplication used in standard FNO layers.
    """

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / width**2
        shape = (width, width, modes1, modes2, 2)
        self.positive = nn.Parameter(scale * torch.rand(shape))
        self.negative = nn.Parameter(scale * torch.rand(shape))

    def forward(self, value: Tensor) -> Tensor:
        transformed = torch.view_as_real(torch.fft.rfft2(value))
        output = torch.zeros_like(transformed)
        m, k = self.modes1, self.modes2
        output[:, :, :m, :k] = torch.einsum(
            "bixyz,ioxyz->boxyz", transformed[:, :, :m, :k], self.positive
        )
        output[:, :, -m:, :k] = torch.einsum(
            "bixyz,ioxyz->boxyz", transformed[:, :, -m:, :k], self.negative
        )
        return torch.fft.irfft2(torch.view_as_complex(output), s=value.shape[-2:])


class FourierBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int, nonlinear: bool):
        super().__init__()
        self.local = nn.Conv2d(width, width, 1, bias=nonlinear)
        self.spectral = SpectralLayer(width, modes1, modes2)
        self.nonlinear = nonlinear

    def forward(self, value: Tensor) -> Tensor:
        value = self.local(value) + self.spectral(value)
        return F.gelu(value) if self.nonlinear else value


class FourierInverse(nn.Module):
    """Nonlinear in state, linear in the residual for each fixed state."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n = config.grid_size + config.padding
        w = config.width
        self.state_norm = nn.LayerNorm([1, n, n])
        self.state_lift = nn.Conv2d(1, w, 3, padding=1)
        self.residual_lift = nn.Conv2d(1, w, 1, bias=False)
        self.state_blocks = nn.Sequential(*(
            FourierBlock(w, modes, modes, nonlinear=i < len(config.modes) - 1)
            for i, modes in enumerate(config.modes)
        ))
        self.residual_block = FourierBlock(w, n // 2, n // 2 + 1, nonlinear=False)
        self.output_block = FourierBlock(w, n // 2, n // 2 + 1, nonlinear=False)
        self.project = nn.Conv2d(w, 1, 1, bias=False)

    def forward(self, state: Tensor, residual: Tensor) -> Tensor:
        n, p = self.config.grid_size, self.config.padding
        state = F.pad(state, (0, p, 0, p))
        residual = F.pad(residual, (0, p, 0, p))
        state_features = self.state_blocks(self.state_lift(self.state_norm(state)))
        residual_features = self.residual_block(self.residual_lift(residual))
        result = self.project(self.output_block(state_features * residual_features))
        return result[..., :n, :n].contiguous()


class NeuralNewtonSolver(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.inverse = FourierInverse(self.config)
        self.operator = NewtonOperator(self.config.grid_size, self.config.source_scale,
                                       self.config.source_grid)
        self.poisson_inverse = DirichletPoissonInverse(self.config.grid_size)

    def forward(self, state: Tensor, rhs: Tensor) -> Tensor:
        self.operator._check_shape(state)
        if rhs.shape != state.shape:
            raise ValueError("state and rhs must have identical shapes")
        if self.config.variant != "baseline":
            rhs = self.poisson_inverse(rhs)
        return self.inverse(state, rhs)

    def loss(self, state: Tensor, correction: Tensor, rhs: Tensor,
             weighted: bool | None = None) -> Tensor:
        """Original batch Frobenius-norm ratio (not a squared energy norm)."""
        error = self.operator(state, correction) - rhs
        if weighted is None:
            weighted = self.config.variant == "npbs-weighted"
        if weighted:
            error, rhs = self.poisson_inverse(error), self.poisson_inverse(rhs)
        denominator = torch.linalg.vector_norm(rhs)
        denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        return torch.linalg.vector_norm(error) / denominator


def save_checkpoint(path: str | Path, model: NeuralNewtonSolver,
                    metadata: dict[str, Any] | None = None) -> None:
    """Write tensors and primitive metadata; no pickled model instances."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "config": asdict(model.config),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metadata": metadata or {}}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> NeuralNewtonSolver:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("expected a version 1 NPBS nonlinear state_dict checkpoint")
    model = NeuralNewtonSolver(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()
