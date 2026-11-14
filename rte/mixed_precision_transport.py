"""FP64 physical/Krylov arithmetic with unchanged FP32 preconditioner kernels."""

import torch
from torch import nn

from rte.models import PeriodicRTE2D


def high_precision_copy(low):
    high = PeriodicRTE2D(shape=low.shape, dx=low.dx, n_angles=low.n_angles,
        anisotropy_g=low.anisotropy_g, eta_scale=low.eta_scale,
        fdtype=torch.float64, device=low.directions.device)
    # Preserve the discrete matrix geometry, not a freshly rounded quadrature.
    for name in ('directions', 'weights', 'phase_matrix', 'kx', 'ky'):
        setattr(high, name, getattr(low, name).double())
    high.setup(low.sigma_t.double(), low.sigma_s.double(), low.source.to(torch.complex128))
    return high


class MixedPrecisionPreconditioner(nn.Module):
    def __init__(self, high, low, model):
        super().__init__()
        self.rte, self.low_operator, self.model = high, low, model
        self.depth = model.depth
        self._generation = None

    def synchronize_material(self):
        if self._generation != self.rte.setup_generation:
            self.low_operator.setup(self.rte.sigma_t.float(), self.rte.sigma_s.float(), self.rte.source.to(torch.complex64))
            self._generation = self.rte.setup_generation

    @torch.no_grad()
    def prepare(self):
        self.eval()
        self.synchronize_material()
        if hasattr(self.model, 'prepare'):
            self.model.prepare()
        return self

    def forward(self, residual):
        self.synchronize_material()
        return self.model(residual.to(torch.complex64)).to(torch.complex128)
