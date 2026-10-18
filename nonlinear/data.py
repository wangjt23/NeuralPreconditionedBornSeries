"""Explicit data loading and small, self-contained synthetic Newton systems."""

from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .operators import NewtonOperator


def as_fields(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("states must be a torch Tensor")
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[1] != 1 or value.shape[2] != value.shape[3]:
        raise ValueError("expected states with shape (samples, 1, n, n) or (samples, n, n)")
    if len(value) == 0 or value.shape[-1] < 2 or not torch.isfinite(value).all():
        raise ValueError("states must be nonempty, finite, and have grid size >= 2")
    return value.float().contiguous()


def load_states(path: str | Path, grid_size: int | None = None) -> Tensor:
    """Load a tensor dictionary or a legacy tensor-only (u, du[, solution]) tuple.

    All ordinary loads use weights_only=True. No model instances or arbitrary
    pickle globals are accepted. Only u is needed for unsupervised training.
    """
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        states = data["u"]
    elif isinstance(data, (tuple, list)) and data:
        states = data[0]
    else:
        raise ValueError("expected {'u': tensor, ...} or a tuple beginning with u")
    states = as_fields(states)
    if grid_size is not None and states.shape[-1] != grid_size:
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        states = F.interpolate(states, (grid_size, grid_size), mode="bilinear", align_corners=False)
    return states


def split_states(states: Tensor, train_count: int | None = None,
                 test_count: int | None = None) -> tuple[Tensor, Tensor]:
    """Use disjoint consecutive slices of each input file before concatenation."""
    train_count = int(0.8 * len(states)) if train_count is None else train_count
    test_count = len(states) - train_count if test_count is None else test_count
    if train_count < 1 or test_count < 1 or train_count + test_count > len(states):
        raise ValueError("each data file needs nonempty disjoint train and test slices")
    return states[:train_count], states[train_count:train_count + test_count]


def dense_laplacian(grid_size: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Small-grid reference matrix, used only in synthetic data and tests."""
    identity = torch.eye(grid_size, dtype=dtype)
    off = torch.ones(grid_size - 1, dtype=dtype)
    tridiagonal = 2 * identity - torch.diag(off, 1) - torch.diag(off, -1)
    return (torch.kron(tridiagonal, identity) + torch.kron(identity, tridiagonal)) * (grid_size + 1)**2


def synthetic_data(samples: int = 32, grid_size: int = 12, seed: int = 0,
                   source_scale: float = 10, source_grid: str = "legacy-endpoints") -> dict:
    """Smooth random states and exact dense Newton corrections, for smoke checks.

    This is a new small synthetic distribution, not the original paper dataset.
    Dense reference solves are intentionally restricted to grids at most 24 x 24.
    """
    if samples < 1 or not 2 <= grid_size <= 24:
        raise ValueError("synthetic data requires samples >= 1 and 2 <= grid_size <= 24")
    generator = torch.Generator().manual_seed(seed)
    coordinate = torch.arange(1, grid_size + 1, dtype=torch.float64) / (grid_size + 1)
    modes = torch.arange(1, 4, dtype=torch.float64)
    waves = torch.sin(torch.pi * modes[:, None] * coordinate[None, :])
    coefficients = torch.randn(samples, 3, 3, dtype=torch.float64, generator=generator) * 0.2
    states = torch.einsum("bkl,ki,lj->bij", coefficients, waves, waves).unsqueeze(1)
    operator = NewtonOperator(grid_size, source_scale, source_grid)
    matrix = dense_laplacian(grid_size)
    rhs = operator.rhs(states)
    corrections = torch.stack([
        torch.linalg.solve(matrix - torch.diag(2 * state.flatten()), value.flatten())
        for state, value in zip(states, rhs)
    ]).reshape_as(states)
    return {"u": states.float(), "du": corrections.float(),
            "metadata": {"kind": "synthetic-smoke", "seed": seed,
                         "source_scale": source_scale, "source_grid": source_grid}}
