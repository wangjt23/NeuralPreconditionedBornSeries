"""Classical HSS with the same cached local-matrix representation as NTO.

This gives the classical control the angular-FFT-to-matrix preparation
optimization too. There are no learned weights or identity-map overheads.
"""

import torch
from torch import nn

from rte.neural_transport_operator import NeuralTransportOperator


class CompiledHSS(nn.Module):
    def __init__(self, rte, depth=4, shift_factor=1.):
        super().__init__()
        if depth < 1 or shift_factor <= 0:
            raise ValueError('positive depth and shift factor required')
        self.rte, self.depth, self.shift_factor = rte, depth, shift_factor
        self.factor_stages_per_apply = depth
        self._key = self._inputs = self._collision = self._stream = None

    @torch.no_grad()
    def prepare(self):
        op = self.rte
        inputs = (op.sigma_t, op.sigma_s, op.phase_matrix, op.weights, op.kx, op.ky, op.directions)
        key = ((op.setup_generation, op.dx, op.shape, self.depth, self.shift_factor),
               tuple((x.data_ptr(), x._version, x.dtype, str(x.device), tuple(x.shape)) for x in inputs))
        if key != self._key:
            phase = torch.fft.fft(op.phase_matrix[0]*op.weights).real
            rates = op.sigma_t-op.sigma_s*phase[None, :, None, None]
            alpha = self.shift_factor*(rates.amin((1, 2, 3))*rates.amax((1, 2, 3))).sqrt()[:, None, None, None]
            kernel = torch.fft.ifft(1/(alpha+rates), dim=1).real.permute(0, 2, 3, 1)
            indices = torch.arange(op.n_angles, device=rates.device)
            self._collision = kernel[..., (indices[:, None]-indices[None, :]) % op.n_angles]
            self._stream = 2*alpha/(alpha+op._stream_symbol(len(rates), op.complex_dtype, rates.device))
            self._key, self._inputs = key, inputs
        return self

    def forward(self, residual):
        self.prepare()
        z, defect = torch.zeros_like(residual), residual
        for stage in range(self.depth):
            local = NeuralTransportOperator.local_map(self._collision, defect)
            update = torch.fft.ifftn(self._stream*torch.fft.fftn(local, dim=(-2, -1)), dim=(-2, -1))
            z = z+update
            if stage+1 < self.depth:
                defect = residual-self.rte.A_op(z)
        return z
