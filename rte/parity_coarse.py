"""Positive, exact strict-band coarse control for a learned Schur hierarchy.

Galerkin products can be assembled on a 2m product grid, not the fine grid.
The dense coarse factor is practical only for small m. No learning is used.
"""

import torch

from rte.multilevel_neural_transport import prolong_field, restrict_field


class GalerkinSchurCorrection:
    def __init__(self, parity, size=8, chunk=32):
        if size > parity.rte.shape[0] or size % 2:
            raise ValueError('even coarse size no larger than the fine grid required')
        self.parity, self.size = parity, size
        self.channels = parity.half
        self.extension = parity.rte.sigma_t.mean()
        self.product_size = 2*size
        self.compressed = self.product_size <= parity.rte.shape[0]
        if self.compressed:
            h, n, p = self.channels, parity.rte.shape[0], self.product_size
            def compress(matrix):
                fields = matrix.permute(0, 3, 4, 1, 2).reshape(-1, h*h, n, n)
                return restrict_field(fields, p).reshape(-1, h, h, p, p).permute(0, 3, 4, 1, 2)
            self.ce = compress(parity.ce)
            self.co_inverse = compress(parity.co_inverse)
            indices = torch.cat((torch.arange(p//2, device=parity.stream.device),
                                 torch.arange(n-p//2, n, device=parity.stream.device)))
            self.stream = parity.stream.index_select(-2, indices).index_select(-1, indices)
        dim = self.channels*size*size
        columns = []
        for start in range(0, dim, chunk):
            end = min(start+chunk, dim)
            probes = torch.zeros(end-start, dim, dtype=parity.rte.complex_dtype,
                                 device=parity.rte.sigma_t.device)
            probes[torch.arange(end-start), torch.arange(start, end)] = 1
            columns.append(self.action(probes.reshape(-1, self.channels, size, size)).flatten(1).T)
        matrix = torch.cat(columns, dim=1)
        self.symmetry_defect = float(torch.linalg.vector_norm(matrix-matrix.mH)/torch.linalg.vector_norm(matrix))
        self.matrix = (matrix+matrix.mH)/2
        self.factor = torch.linalg.cholesky(self.matrix)

    def direct_action(self, value):
        band = restrict_field(value, self.size)
        lifted = prolong_field(band, self.parity.rte.shape[0])
        return restrict_field(self.parity.schur(lifted), self.size)+self.extension*(value-band)

    def action(self, value):
        if not self.compressed:
            return self.direct_action(value)
        band = restrict_field(value, self.size)
        lifted = prolong_field(band, self.product_size)
        derivative = lambda v: torch.fft.ifftn(self.stream*torch.fft.fftn(v, dim=(-2, -1)), dim=(-2, -1))
        image = self.parity.local(self.ce, lifted)-derivative(self.parity.local(self.co_inverse, derivative(lifted)))
        return restrict_field(image, self.size)+self.extension*(value-band)

    def __call__(self, rhs):
        coarse = restrict_field(rhs, self.size)
        value = torch.cholesky_solve(coarse.flatten(1).T, self.factor).T.reshape_as(coarse)
        return prolong_field(restrict_field(value, self.size), self.parity.rte.shape[0])


class SchurMultilevelControl:
    def __init__(self, parity, coarse=None, equilibrate=False):
        self.parity, self.coarse = parity, coarse
        self.scale = (parity.rte.sigma_t/parity.rte.sigma_t.mean((-2, -1), keepdim=True)).sqrt() if equilibrate else 1.

    def __call__(self, rhs):
        fine = self.scale*self.parity.reference(self.scale*rhs)
        return fine if self.coarse is None else fine+self.coarse(rhs)


class CharacteristicSchurCorrection:
    """Exact Galerkin matrix on a small direction-aligned Fourier dictionary.

    Matrix entries use coefficient convolution directly, avoiding k fine-grid
    applications. This analytic dictionary is a classical control, not AI.
    """

    def __init__(self, parity, per_angle=49, transverse_weight=.01):
        if len(parity.rte.sigma_t) != 1:
            raise ValueError('dictionary construction currently supports one medium at a time')
        self.parity = parity
        self.half, self.size = parity.half, parity.rte.shape[0]
        if not 1 <= per_angle <= self.size*self.size or transverse_weight <= 0:
            raise ValueError('invalid dictionary size or transverse weight')
        wave = parity.stream.imag[0]
        radius = parity.rte.kx.square()+parity.rte.ky.square()
        score = wave.square()+transverse_weight*radius[0]
        spatial = score.flatten(1).argsort(dim=1, stable=True)[:, :per_angle]
        angular = torch.arange(self.half, device=wave.device)[:, None].expand_as(spatial)
        self.indices = (angular*self.size*self.size+spatial).flatten()
        a, x, y = angular.flatten(), (spatial//self.size).flatten(), (spatial % self.size).flatten()
        dx, dy = (x[:, None]-x[None, :]) % self.size, (y[:, None]-y[None, :]) % self.size
        def coefficients(matrix):
            fields = matrix[0].permute(2, 3, 0, 1)
            return torch.fft.fftn(fields, dim=(-2, -1), norm='forward')
        ce, ci = coefficients(parity.ce), coefficients(parity.co_inverse)
        k = wave[a, x, y]
        matrix = ce[a[:, None], a[None, :], dx, dy]+k[:, None]*k[None, :]*ci[a[:, None], a[None, :], dx, dy]
        self.symmetry_defect = float(torch.linalg.vector_norm(matrix-matrix.mH)/torch.linalg.vector_norm(matrix))
        self.matrix = (matrix+matrix.mH)/2
        self.factor = torch.linalg.cholesky(self.matrix)

    def restrict(self, value):
        return torch.fft.fftn(value, dim=(-2, -1), norm='ortho').flatten(1)[:, self.indices]

    def prolong(self, value):
        fields = value.new_zeros(len(value), self.half*self.size*self.size)
        fields[:, self.indices] = value
        return torch.fft.ifftn(fields.reshape(-1, self.half, self.size, self.size), dim=(-2, -1), norm='ortho')

    def direct_action(self, value):
        return self.restrict(self.parity.schur(self.prolong(value)))

    def action(self, value):
        return (self.matrix @ value.T).T

    def __call__(self, rhs):
        reduced = self.restrict(rhs)
        return self.prolong(torch.cholesky_solve(reduced.T, self.factor).T)
