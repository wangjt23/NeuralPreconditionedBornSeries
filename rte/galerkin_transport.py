"""Strict-band Fourier Galerkin RTE action with dealiased coefficient products.

For coarse size m, material Fourier coefficients up to |k_i|<m suffice for
R C P on the |k_i|<m/2 band. Evaluate products on a 2m grid and restrict them;
frequencies that alias cannot land in the retained band. The excluded coarse
Nyquist complement receives a positive diagonal extension, not a null operator.
"""

import torch

from rte.models import PeriodicRTE2D
from rte.multilevel_neural_transport import prolong_field, restrict_field


class GalerkinTransport:
    def __init__(self, fine, coarse_size=32):
        if coarse_size*2 > fine.shape[0] or fine.shape[0] != fine.shape[1]:
            raise ValueError('square fine grid at least twice the coarse size required')
        self.fine, self.size = fine, coarse_size
        self.product_size = 2*coarse_size
        self.product = PeriodicRTE2D(shape=(self.product_size, self.product_size),
            dx=fine.dx*fine.shape[0]/self.product_size, n_angles=fine.n_angles,
            anisotropy_g=fine.anisotropy_g, eta_scale=0., fdtype=fine.float_dtype,
            device=fine.sigma_t.device)
        # Preserve exactly the fine discretization's stored quadrature and
        # wavenumbers (especially when fine is a promoted FP32 geometry).
        self.product.directions = fine.directions
        indices = torch.cat([torch.arange(self.product_size//2, device=fine.sigma_t.device),
            torch.arange(fine.shape[0]-self.product_size//2, fine.shape[0], device=fine.sigma_t.device)])
        self.product.kx = fine.kx.index_select(-2, indices)
        self.product.ky = fine.ky.index_select(-1, indices)
        # These band-limited coefficient fields can exhibit Gibbs undershoot.
        # Do not clamp them or call the positivity-enforcing material setup:
        # they are convolution representations, not independent physical media.
        self.t = restrict_field(fine.sigma_t.to(fine.complex_dtype), self.product_size)
        self.s = restrict_field(fine.sigma_s.to(fine.complex_dtype), self.product_size)
        self.phase = (fine.phase_matrix*fine.weights[None]).to(fine.complex_dtype)
        self.mean_t = fine.sigma_t.mean((-2, -1), keepdim=True)

    def collision(self, lifted):
        return self.t*lifted-self.s*torch.einsum('lm,bmxy->blxy', self.phase, lifted)

    def __call__(self, value):
        band = restrict_field(value, self.size)
        lifted = prolong_field(band, self.product_size)
        collision = restrict_field(self.collision(lifted), self.size)
        # Streaming is diagonal in k and commutes with the strict-band transfer.
        streaming = restrict_field(self.product.streaming_op(lifted), self.size)
        return collision+streaming+self.mean_t*(value-band)

    def direct(self, value):
        band = restrict_field(value, self.size)
        return restrict_field(self.fine.A_op(prolong_field(band, self.fine.shape[0])), self.size)+self.mean_t*(value-band)
