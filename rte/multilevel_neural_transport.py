"""Full-angle two-level RTE correction; rediscretized, not Galerkin.

Fourier transfers retain all angles and the strict coarse spatial band. Coarse
Nyquist rows/columns are omitted consistently. Positive cell-averaged materials
define the coarse physical operator. Fine HSS corrections retain full bandwidth.
"""

import torch
from torch import nn
from torch.nn import functional as F

from rte.models import PeriodicRTE2D
from rte.compiled_hss import CompiledHSS


def restrict_field(value, size):
    n = value.shape[-1]
    if value.shape[-2] != n or size % 2 or n % 2 or not 2 <= size <= n:
        raise ValueError('even square grids and a smaller positive coarse size required')
    spectrum = torch.fft.fftshift(torch.fft.fftn(value, dim=(-2, -1), norm='forward'), dim=(-2, -1))
    start = (n-size)//2
    small = spectrum[..., start:start+size, start:start+size].clone()
    small[..., 0, :] = 0
    small[..., :, 0] = 0
    return torch.fft.ifftn(torch.fft.ifftshift(small, dim=(-2, -1)), dim=(-2, -1), norm='forward')


def prolong_field(value, size):
    m = value.shape[-1]
    if value.shape[-2] != m or m % 2 or size % 2 or size < m:
        raise ValueError('even square grids and a larger fine size required')
    small = torch.fft.fftshift(torch.fft.fftn(value, dim=(-2, -1), norm='forward'), dim=(-2, -1)).clone()
    small[..., 0, :] = 0
    small[..., :, 0] = 0
    pad = (size-m)//2
    large = F.pad(small, (pad, pad, pad, pad))
    return torch.fft.ifftn(torch.fft.ifftshift(large, dim=(-2, -1)), dim=(-2, -1), norm='forward')


class MultilevelNeuralTransport(nn.Module):
    def __init__(self, rte, coarse_factory, coarse_size=32, pre=0, post=2, shift_factor=1.):
        super().__init__()
        n = rte.shape[0]
        if rte.shape != (n, n) or n % coarse_size or coarse_size < 2 or coarse_size % 2:
            raise ValueError('fine grid must be an integer multiple of an even coarse grid')
        if min(pre, post) < 0 or pre+post < 1:
            raise ValueError('at least one full-band fine correction required')
        self.rte, self.coarse_size, self.pre, self.post = rte, coarse_size, pre, post
        self.depth = pre+post
        self.coarse_operator = PeriodicRTE2D(
            shape=(coarse_size, coarse_size), dx=rte.dx*n/coarse_size,
            n_angles=rte.n_angles, anisotropy_g=rte.anisotropy_g,
            eta_scale=0., fdtype=rte.float_dtype, device=rte.directions.device)
        self.coarse_model = coarse_factory(self.coarse_operator)
        self.before = CompiledHSS(rte, pre, shift_factor) if pre else None
        self.after = CompiledHSS(rte, post, shift_factor) if post else None
        self._setup_key = self._setup_inputs = None

    def setup_coarse(self):
        op = self.rte
        inputs = (op.sigma_t, op.sigma_s)
        key = (op.setup_generation, tuple((x.data_ptr(), x._version, x.dtype, str(x.device), tuple(x.shape)) for x in inputs))
        if key != self._setup_key:
            ratio = op.shape[0]//self.coarse_size
            with torch.no_grad():
                t, s = [F.avg_pool2d(x, ratio) for x in inputs]
                source = torch.zeros(len(t), op.n_angles, self.coarse_size, self.coarse_size,
                                     dtype=op.complex_dtype, device=t.device)
                self.coarse_operator.setup(t, s, source)
            self._setup_key, self._setup_inputs = key, inputs

    @torch.no_grad()
    def prepare(self):
        self.eval()
        self.setup_coarse()
        self.coarse_model.prepare()
        for model in (self.before, self.after):
            if model is not None:
                model.prepare()
        return self

    def forward(self, residual):
        self.setup_coarse()
        if self.before is None:
            z, defect = torch.zeros_like(residual), residual
        else:
            z = self.before(residual)
            defect = residual-self.rte.A_op(z)
        coarse = self.coarse_model(restrict_field(defect, self.coarse_size))
        z = z+prolong_field(coarse, self.rte.shape[0])
        if self.after is not None:
            z = z+self.after(residual-self.rte.A_op(z))
        return z
