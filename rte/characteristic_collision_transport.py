"""Direction-shared neural transport symbols in characteristic coordinates.

The shared network sees Omega.k and its transverse coordinate, rather than
learning separate Cartesian-output heads for each angle. Its amplitude is even
under k reversal and phase odd; the DC amplitude is shared across angles.
"""

import torch
from torch import nn

from rte.neural_transport_operator import NeuralTransportOperator
from rte.spectral_collision_transport import SpectralCollisionTransport


class CharacteristicCollisionTransport(SpectralCollisionTransport):
    def __init__(self, rte, depth=4, spectral_width=48, spectral_bound=.35):
        super().__init__(rte, depth, spectral_width, spectral_bound)
        self.symbol_net = nn.Sequential(nn.Linear(12, spectral_width), nn.SiLU(),
            nn.Linear(spectral_width, spectral_width), nn.SiLU(), nn.Linear(spectral_width, 2*depth))
        nn.init.zeros_(self.symbol_net[-1].weight)
        nn.init.zeros_(self.symbol_net[-1].bias)

    def modulation(self, kx, ky, summary, scale):
        op = self.rte
        directions = op.directions
        along = (directions[:, 0, None, None]*kx+directions[:, 1, None, None]*ky)[None]/scale
        across = (-directions[:, 1, None, None]*kx+directions[:, 0, None, None]*ky)[None]/scale
        even = torch.stack([torch.log1p(along.square()), torch.log1p(across.square()),
            along*across/(1+along.square()+across.square()), 1/(1+along.square())], -1)
        b, angles, h, w = along.shape
        inputs = torch.cat([even, summary[:, None, None, None].expand(b, angles, h, w, 8)], -1)
        raw = self.symbol_net(inputs.to(self.symbol_net[0].weight.dtype)).reshape(b, angles, h, w, self.depth, 2).tanh()
        amplitude = raw[..., 0]
        phase = raw[..., 1]*(along/(1+along.square()).sqrt())[..., None]
        return (self.spectral_bound*torch.complex(amplitude, phase)).exp().permute(0, 4, 1, 2, 3)

    def _fields(self):
        op = self.rte
        rates, shifts, gains, _ = self.backbone._fields()
        _, summary = NeuralTransportOperator._features(self)
        kx, ky = op.kx.reshape(op.shape[0], 1), op.ky.reshape(1, op.shape[1])
        modulation = self.modulation(kx, ky, summary, op.sigma_t.mean((1, 2, 3))[:, None, None, None])
        alpha = shifts[:, :, None, None, None]
        stream = op._stream_symbol(len(rates), op.complex_dtype, rates.device)[:, None]
        return gains/(alpha+rates[:, None]), modulation*(2*alpha/(alpha+stream))
