"""Full-angle neural operator sandwiches for the periodic Fourier RTE.

Nonlinear material encoders generate cached, residual-linear operators. Two
stages are the default cost hypothesis, not a convergence or speed guarantee.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class NeuralTransportOperator(nn.Module):
    def __init__(self, rte, depth=2, width=24, mixing=True, spectral=True,
                 mixing_bound=.5, spectral_bound=.7, base='reference'):
        super().__init__()
        if depth < 1 or width < 1 or not 0 <= mixing_bound < 1 or spectral_bound < 0:
            raise ValueError('invalid neural operator dimensions or bounds')
        self.rte = rte
        if base not in ('reference', 'hss'):
            raise ValueError('base must be reference or hss')
        self.base_kind = base
        self.depth, self.width = int(depth), int(width)
        self.mixing, self.spectral = bool(mixing), bool(spectral)
        self.mixing_bound, self.spectral_bound = float(mixing_bound), float(spectral_bound)
        self.n_angles = rte.n_angles
        self.factor_stages_per_apply = self.depth
        self.stem = nn.Conv2d(8, width, 3, padding=1, padding_mode='circular')
        self.context = nn.ModuleList([
            nn.Conv2d(width, width, 3, padding=d, dilation=d, padding_mode='circular')
            for d in (1, 2, 4)])
        self.matrix_head = nn.Conv2d(width, 2 * depth * rte.n_angles**2, 1)
        self.symbol_net = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 32),
                                        nn.SiLU(), nn.Linear(32, 2 * depth * rte.n_angles))
        self.shift_net = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, depth)) if base == 'hss' else None
        for head in (self.matrix_head, self.symbol_net[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.shift_net is not None:
            nn.init.zeros_(self.shift_net[-1].weight)
            nn.init.zeros_(self.shift_net[-1].bias)
        self._prepared = self._prepared_signature = self._prepared_inputs = None
        self._base = self._base_signature = self._base_inputs = None

    def _inputs(self, parameters=True):
        op = self.rte
        tensors = (op.sigma_t, op.sigma_s, op.phase_matrix, op.weights, op.kx, op.ky, op.directions)
        return tensors + tuple(self.parameters()) if parameters else tensors

    def _signature(self, parameters=True):
        op = self.rte
        if op.n_angles != self.n_angles:
            raise ValueError('angle count changed; reconstruct the neural operator')
        return ((op.setup_generation, op.dx, op.shape, self.depth, self.mixing, self.spectral,
                 self.mixing_bound, self.spectral_bound, self.base_kind),
                tuple((x.data_ptr(), x._version, x.dtype, str(x.device), tuple(x.shape))
                      for x in self._inputs(parameters)))

    def _reference(self):
        signature = self._signature(False)
        if signature != self._base_signature:
            # Coefficient gradients are not part of this preconditioner training.
            # Cache only this parameter-independent inverse during backprop too.
            with torch.no_grad():
                op = self.rte
                b = len(op.sigma_t)
                t = op.sigma_t.mean((-2, -1)).reshape(b, 1, 1, 1)
                s = op.sigma_s.mean((-2, -1)).reshape(b, 1, 1, 1, 1)
                stream = op._stream_symbol(b, op.complex_dtype, op.sigma_t.device).permute(0, 2, 3, 1)
                eye = torch.eye(op.n_angles, dtype=op.complex_dtype, device=stream.device)
                phase = (op.phase_matrix * op.weights[None]).to(op.complex_dtype)
                matrix = (stream + t).unsqueeze(-1) * eye - s * phase
                self._base = torch.linalg.inv(matrix)
                self._base_signature, self._base_inputs = signature, self._inputs(False)
        return self._base

    def _features(self):
        op = self.rte
        t, s = op.sigma_t, op.sigma_s
        a, albedo = t - s, s / t
        logt = t.log()
        centered = logt - logt.mean((-2, -1), keepdim=True)
        local = F.avg_pool2d(F.pad(centered, (2, 2, 2, 2), mode='circular'), 5, stride=1)
        gx = .5 * (torch.roll(logt, -1, -2) - torch.roll(logt, 1, -2))
        gy = .5 * (torch.roll(logt, -1, -1) - torch.roll(logt, 1, -1))
        spatial = torch.cat([centered, albedo, (a/t).log(), torch.log1p(t*op.dx),
                             local, torch.full_like(t, op.anisotropy_g), gx, gy], 1)
        dims = (1, 2, 3)
        summary = torch.stack([t.mean(dims).log(), a.mean(dims).log(), albedo.mean(dims),
                               logt.std(dims, unbiased=False), albedo.std(dims, unbiased=False),
                               a.amin(dims).log(), torch.full_like(t.mean(dims), op.anisotropy_g),
                               torch.full_like(t.mean(dims), math.log(op.dx))], -1)
        return spatial, summary

    def _fields(self):
        op = self.rte
        compile_real = not self.training and not torch.is_grad_enabled()
        b, h, w = len(op.sigma_t), *op.shape
        spatial, summary = self._features()
        dtype = self.stem.weight.dtype
        hidden = F.silu(self.stem(spatial.to(dtype)))
        for layer in self.context:
            hidden = hidden + F.silu(layer(hidden)) / math.sqrt(3)
        raw = self.matrix_head(hidden).reshape(b, self.depth, 2, self.n_angles, self.n_angles, h, w)
        raw = raw.permute(0, 1, 2, 5, 6, 3, 4)
        # Q Delta Q with Q=I-11^T/L protects both scalar input and angular mean.
        delta = self.mixing_bound * raw.tanh() / self.n_angles
        delta = delta - delta.mean(-1, keepdim=True) - delta.mean(-2, keepdim=True) + delta.mean((-2, -1), keepdim=True)
        eye = torch.eye(self.n_angles, dtype=delta.dtype, device=delta.device)
        maps = eye + delta * float(self.mixing)

        kx = op.kx.reshape(h, 1).expand(h, w)
        ky = op.ky.reshape(1, w).expand(h, w)
        magnitude = (kx*kx + ky*ky).sqrt()[None]
        scale = op.sigma_t.mean((1, 2, 3))[:, None, None]
        frequency = torch.stack([kx/(scale+magnitude), ky/(scale+magnitude),
                                 torch.log1p(magnitude/scale),
                                 torch.log1p(magnitude*op.dx).expand(b, h, w)], -1)
        inputs = torch.cat([frequency, summary[:, None, None].expand(b, h, w, 8)], -1).to(dtype)
        raw_symbol = self.symbol_net(inputs).reshape(b, h, w, self.depth, self.n_angles, 2)
        raw_symbol = raw_symbol.permute(0, 3, 1, 2, 4, 5).tanh()
        nonzero = (magnitude != 0)[:, None, :, :, None]
        exponent = self.spectral_bound * float(self.spectral) * nonzero * torch.complex(raw_symbol[..., 0], raw_symbol[..., 1])
        gain = exponent.exp().to(op.complex_dtype)
        maps = maps.to(op.float_dtype if compile_real else op.complex_dtype)
        if self.base_kind == 'reference':
            # Full-angle reference coupling with learned complex modulation.
            symbols = gain.unsqueeze(-1) * self._reference()[:, None]
        else:
            phase_eigenvalues = torch.fft.fft(op.phase_matrix[0] * op.weights).real
            rates = op.sigma_t - op.sigma_s * phase_eigenvalues[None, :, None, None]
            alpha0 = (rates.amin((1, 2, 3)) * rates.amax((1, 2, 3))).sqrt()
            shifts = alpha0[:, None] * (math.log(8) * self.shift_net(summary.to(dtype)).tanh()).exp().to(alpha0.dtype)
            alpha = shifts[:, :, None, None, None]
            kernel = torch.fft.ifft(1 / (alpha + rates[:, None]), dim=2).real.permute(0, 1, 3, 4, 2)
            indices = torch.arange(self.n_angles, device=rates.device)
            difference = (indices[:, None]-indices[None, :]) % self.n_angles
            collision = kernel[..., difference].to(maps.dtype)
            # Compile collision inverse and learned input map into one matrix.
            input_map = collision @ maps[:, :, 0]
            maps = torch.stack([input_map, maps[:, :, 1]], 2)
            stream = op._stream_symbol(b, op.complex_dtype, rates.device).permute(0, 2, 3, 1)[:, None]
            spectral_alpha = shifts[:, :, None, None, None]
            symbols = gain * (2*spectral_alpha / (spectral_alpha+stream))
        return maps, symbols

    def fields(self):
        if self.training or torch.is_grad_enabled():
            return self._fields()
        signature = self._signature()
        if signature != self._prepared_signature:
            self._prepared = self._fields()
            self._prepared_signature, self._prepared_inputs = signature, self._inputs()
        return self._prepared

    @torch.no_grad()
    def prepare(self):
        self.eval()
        return self.fields()

    @staticmethod
    def local_map(matrix, residual):
        if not matrix.is_complex():
            # R/P and the local collision inverse are real. Treat real/imag as
            # two RHS columns: one real batched GEMM, half the cached storage.
            rhs = torch.view_as_real(residual).permute(0, 2, 3, 1, 4)
            result = matrix @ rhs
            return torch.view_as_complex(result.contiguous()).permute(0, 3, 1, 2)
        return torch.einsum('bxyij,bjxy->bixy', matrix, residual)

    def forward(self, residual):
        maps, symbols = self.fields()
        z, defect = torch.zeros_like(residual), residual
        for stage in range(self.depth):
            restricted = self.local_map(maps[:, stage, 0], defect)
            rhs = torch.fft.fftn(restricted, dim=(-2, -1))
            if self.base_kind == 'reference':
                solution = torch.einsum('bxyij,bjxy->bixy', symbols[:, stage], rhs)
            else:
                solution = symbols[:, stage].permute(0, 3, 1, 2) * rhs
            correction = torch.fft.ifftn(solution, dim=(-2, -1))
            z = z + self.local_map(maps[:, stage, 1], correction)
            if stage + 1 < self.depth:
                defect = residual - self.rte.A_op(z)
        return z
