"""Source-conditioned forcing plus fine-grid characteristic transport decoding.

The neural source fields depend nonlinearly on material, illumination and
global context. The transport filters are known inexpensive structure, not
an AI advantage by themselves. All filters are stable directional averages;
they preserve transverse fine frequencies omitted by a radial low-pass.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from rte.global_source_predictor import GlobalSourcePredictor


class CharacteristicDecoder(nn.Module):
    def __init__(self, width, angles=12, scales=3):
        super().__init__()
        self.angles, self.scales = angles, scales
        self.forcing = nn.Conv2d(width, 2*angles*scales, 1)
        self.combine = nn.Conv2d(2*angles*scales, 2*angles, 1)
        self.log_rates = nn.Parameter(torch.linspace(math.log(.25), math.log(4.), scales))
        nn.init.zeros_(self.combine.weight)
        nn.init.zeros_(self.combine.bias)

    def filter(self, fields, sigma_t):
        """fields [B,S,L,N,N]; rate/(rate+i Omega.k) has modulus <= 1."""
        n, m = fields.shape[-2:]
        dtype, device = fields.real.dtype, fields.device
        angle = torch.arange(self.angles, dtype=dtype, device=device)*(2*math.pi/self.angles)
        kx = torch.fft.fftfreq(n, d=1/n, dtype=dtype, device=device)*2*math.pi
        ky = torch.fft.fftfreq(m, d=1/m, dtype=dtype, device=device)*2*math.pi
        wave = angle.cos()[:, None, None]*kx[None, :, None]+angle.sin()[:, None, None]*ky[None, None, :]
        mean = sigma_t.mean((-2, -1), keepdim=True)[:, None]
        rates = mean*self.log_rates.clamp(-6, 6).exp()[None, :, None, None, None]
        symbol = rates/(rates+1j*wave[None, None])
        return torch.fft.ifft2(symbol*torch.fft.fft2(fields))

    def forward(self, hidden, sigma_t):
        raw = self.forcing(hidden)
        real, imaginary = raw.chunk(2, 1)
        fields = torch.complex(real, imaginary).reshape(len(hidden), self.scales, self.angles, *hidden.shape[-2:])
        response = self.filter(fields, sigma_t).flatten(1, 2)
        return self.combine(torch.cat((response.real, response.imag), 1))


class CharacteristicSourcePredictor(GlobalSourcePredictor):
    def __init__(self, angles=12, width=32, blocks=4, modes=6):
        super().__init__(angles, width, blocks, modes)
        self.characteristic = CharacteristicDecoder(width, angles)

    def forward(self, t, s, source, reference):
        # Identical base features and decoder, with an additional fine-grid
        # directional path. Kept here to preserve the first pilot's source.
        absorption = (t-s).clamp_min(1e-8)
        scale = self.intensity_scale(t, s, source).clamp_min(1e-8)
        qscale = source.abs().square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
        logt, loga = t.log(), absorption.log()
        gx = (torch.roll(logt, -1, -2)-torch.roll(logt, 1, -2))/2
        gy = (torch.roll(logt, -1, -1)-torch.roll(logt, 1, -1))/2
        material = torch.cat((logt/4, loga/4, s/t, logt-logt.mean((-2, -1), keepdim=True),
            gx, gy, torch.log1p(t/t.shape[-1]), torch.full_like(t, math.log(t.shape[-1])/5)), 1)
        q, ref = source/qscale, reference/scale
        first = F.gelu(self.stem(torch.cat((material, q.real, q.imag, ref.real, ref.imag), 1)))
        second = F.gelu(self.down1(F.avg_pool2d(first, 2)))
        hidden = F.gelu(self.down2(F.avg_pool2d(second, 2)))
        hidden = self.global_blocks(hidden)
        hidden = F.gelu(self.up1(torch.cat((F.interpolate(hidden, size=second.shape[-2:], mode='nearest'), second), 1)))
        hidden = F.gelu(self.up2(torch.cat((F.interpolate(hidden, size=first.shape[-2:], mode='nearest'), first), 1)))
        real, imaginary = (self.output(hidden)+self.characteristic(hidden, t)).chunk(2, 1)
        return reference+scale*torch.complex(real, imaginary)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not any(key.startswith('characteristic.') for key in state_dict):
            # A documented base-to-directional upgrade starts as exactly the
            # base predictor. Only newly initialized branch keys may be absent.
            result = super().load_state_dict(state_dict, strict=False, assign=assign)
            if result.unexpected_keys or any(not key.startswith('characteristic.') for key in result.missing_keys):
                raise RuntimeError('incompatible base checkpoint for directional upgrade')
            return result
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
