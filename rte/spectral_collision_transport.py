"""Learn frequency/angle transport symbols without adding residual-path kernels.

Retains CT's fast angular FFT factors. A material/frequency MLP is compiled into
the existing streaming multiplier once per material; no dense local L-by-L maps.
"""

import torch
from torch import nn

from rte.collision_transport import CollisionTransportPreconditioner
from rte.neural_transport_operator import NeuralTransportOperator


class SpectralCollisionTransport(nn.Module):
    def __init__(self, rte, depth=4, spectral_width=48, spectral_bound=.35):
        super().__init__()
        self.rte, self.depth, self.spectral_bound = rte, depth, spectral_bound
        self.factor_stages_per_apply = depth
        self.backbone = CollisionTransportPreconditioner(rte, depth=depth, learned=True, width=12)
        self.symbol_net = nn.Sequential(nn.Linear(12, spectral_width), nn.SiLU(),
            nn.Linear(spectral_width, spectral_width), nn.SiLU(), nn.Linear(spectral_width, 2*depth*rte.n_angles))
        nn.init.zeros_(self.symbol_net[-1].weight)
        nn.init.zeros_(self.symbol_net[-1].bias)
        self._prepared = self._key = self._inputs = None

    def _signature(self):
        inputs = (*self.backbone._cache_inputs(), *self.symbol_net.parameters())
        return ((self.backbone._signature(), self.spectral_bound),
                tuple((x.data_ptr(), x._version, str(x.device), x.dtype, tuple(x.shape)) for x in inputs)), inputs

    def _fields(self):
        op = self.rte
        rates, shifts, gains, _ = self.backbone._fields()
        # Reuse the documented material statistics of NTO, not its dense maps.
        _, summary = NeuralTransportOperator._features(self)
        b, h, w = len(rates), *op.shape
        kx, ky = op.kx.reshape(h, 1).expand(h, w), op.ky.reshape(1, w).expand(h, w)
        magnitude = (kx*kx+ky*ky).sqrt()[None]
        scale = op.sigma_t.mean((1, 2, 3))[:, None, None]
        coordinates = torch.stack([kx/(scale+magnitude), ky/(scale+magnitude),
            torch.log1p(magnitude/scale), torch.log1p(magnitude*op.dx).expand(b, h, w)], -1)
        inputs = torch.cat([coordinates, summary[:, None, None].expand(b, h, w, 8)], -1)
        raw = self.symbol_net(inputs.to(self.symbol_net[0].weight.dtype)).reshape(b, h, w, self.depth, op.n_angles, 2).tanh()
        raw = raw.permute(0, 3, 4, 1, 2, 5)
        exponent = self.spectral_bound*torch.complex(raw[..., 0], raw[..., 1])
        modulation = (exponent*(magnitude != 0)[:, None, None]).exp()
        alpha = shifts[:, :, None, None, None]
        stream = op._stream_symbol(b, op.complex_dtype, rates.device)[:, None]
        return gains/(alpha+rates[:, None]), modulation*(2*alpha/(alpha+stream))

    def fields(self):
        if self.training or torch.is_grad_enabled():
            return self._fields()
        key, inputs = self._signature()
        if key != self._key:
            self._prepared, self._key, self._inputs = self._fields(), key, inputs
        return self._prepared

    @torch.no_grad()
    def prepare(self):
        self.eval()
        return self.fields()

    def forward(self, residual):
        collision, streaming = self.fields()
        z, defect = torch.zeros_like(residual), residual
        for stage in range(self.depth):
            local = torch.fft.ifft(torch.fft.fft(defect, dim=1, norm='ortho')*collision[:, stage], dim=1, norm='ortho')
            update = torch.fft.ifftn(torch.fft.fftn(local, dim=(-2, -1))*streaming[:, stage], dim=(-2, -1))
            z = z+update
            if stage+1 < self.depth:
                defect = residual-self.rte.A_op(z)
        return z


def make_resolution_model(args, op):
    if args.model == 'cct':
        from rte.characteristic_collision_transport import CharacteristicCollisionTransport
        return CharacteristicCollisionTransport(op, args.depth, getattr(args, 'spectral_width', 48), getattr(args, 'spectral_bound', .35))
    if args.model == 'sct':
        return SpectralCollisionTransport(op, args.depth, getattr(args, 'spectral_width', 48), getattr(args, 'spectral_bound', .35))
    from rte.train_neural_transport import make_model
    return make_model(args, op)
