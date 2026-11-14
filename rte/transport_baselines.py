"""Classical transport controls with explicit shift, depth and relaxation.

The reference's artificial shift never changes the physical A used in defect
updates. eta_scale=0 also sets eta_floor=0, so a homogeneous full-angle reference
is the physical inverse up to floating-point error. These are periodic Fourier
controls, not production boundary-consistent DSA or upwind transport sweeps.
"""

import torch
from torch import nn

from rte.models import PeriodicRTE2D


class ClassicalTransportStack(nn.Module):
    KINDS = ("transport", "reference", "tsa", "dsa", "dsa_transport")

    def __init__(self, rte, kind="tsa", depth=1, eta_scale=0., relaxation=1.):
        super().__init__()
        if kind not in self.KINDS or depth < 1 or eta_scale < 0 or relaxation <= 0:
            raise ValueError("invalid classical transport settings")
        self.rte = rte
        self.kind, self.depth = kind, int(depth)
        self.eta_scale, self.relaxation = float(eta_scale), float(relaxation)
        self.reference = PeriodicRTE2D(
            shape=rte.shape, dx=rte.dx, n_angles=rte.n_angles, anisotropy_g=rte.anisotropy_g,
            fdtype=rte.float_dtype, device=rte.sigma_t.device, eta_scale=eta_scale,
            eta_floor=0. if eta_scale == 0 else 1e-6)
        self._grid = (rte.shape, rte.dx, rte.n_angles, rte.anisotropy_g, rte.float_dtype)
        self._signature = None
        self.prepare()

    @torch.no_grad()
    def prepare(self):
        if self._grid != (self.rte.shape, self.rte.dx, self.rte.n_angles, self.rte.anisotropy_g, self.rte.float_dtype):
            raise RuntimeError("grid/phase changed; reconstruct the classical reference")
        signature = (getattr(self.rte, "setup_generation", 0),
                     tuple((x.data_ptr(), x._version, x.dtype, str(x.device), x.shape)
                           for x in (self.rte.sigma_t, self.rte.sigma_s)))
        if self._signature != signature:
            self.reference.setup(self.rte.sigma_t, self.rte.sigma_s, self.rte.source)
            if self.kind == "dsa_transport":
                # P1 diffusion coefficient for this circular quadrature, not g^1.
                p1 = torch.fft.fft(self.rte.phase_matrix[0] * self.rte.weights)[1].real
                transport_rate = (self.rte.sigma_t - p1 * self.rte.sigma_s).mean((-2, -1), keepdim=True)
                self.reference.diffusion_coeff = 1. / (2 * transport_rate)
            self._signature = signature
        if self.kind in ("reference", "tsa"):
            self.reference.reference_transport_inverse
        return self

    def single(self, residual):
        if self.kind == "reference":
            return self.reference.reference_transport_solve(residual)
        z = self.reference.G(residual)
        if self.kind == "transport":
            return z
        defect = residual - self.rte.A_op(z)
        if self.kind == "tsa":
            return z + self.reference.reference_transport_solve(defect)
        scalar = self.rte.scalar_flux(defect)
        return z + self.rte.lift_scalar_flux(self.reference.diffusion_solve(scalar))

    def forward(self, residual):
        self.prepare()
        z = torch.zeros_like(residual)
        for stage in range(self.depth):
            defect = residual if stage == 0 else residual - self.rte.A_op(z)
            z = z + self.relaxation * self.single(defect)
        return z
