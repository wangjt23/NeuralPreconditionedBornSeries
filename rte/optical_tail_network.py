"""Material-conditioned axial attention biased by periodic optical distance.

Global coupling occurs on the half-resolution grid, not only a small radial
Fourier band. Queries/keys depend on material; values carry residual response.
Known line-integrated attenuation supplies a bias, not a learned inverse.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from rte.tail_response_network import TailResponseNetwork


class OpticalAxisAttention(nn.Module):
    def __init__(self, width, axis, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by attention heads')
        self.width, self.heads, self.axis = width, heads, axis
        self.query_key = nn.Conv2d(4, 2*width, 1)
        self.value = nn.Conv2d(width, width, 1)
        self.output = nn.Conv2d(width, width, 1)
        self.log_rates = nn.Parameter(torch.linspace(math.log(.005), math.log(.5), heads))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden, geometry, sigma_t):
        if self.axis == 1:
            hidden, geometry, sigma_t = [x.transpose(-2, -1) for x in (hidden, geometry, sigma_t)]
        b, c, n, lines = hidden.shape
        def tokens(x):
            return x.permute(0, 3, 2, 1).reshape(b*lines, n, self.heads, c//self.heads).transpose(1, 2)
        query, key = self.query_key(geometry).chunk(2, 1)
        query, key, value = [tokens(x) for x in (query, key, self.value(hidden))]
        # Cell-centred line integrals on the physical unit torus. Both possible
        # periodic paths are available; the shorter optical path sets the bias.
        density = sigma_t[:, 0].permute(0, 2, 1).reshape(b*lines, n)
        coordinate = (density.cumsum(-1)-.5*density)/n
        total = density.sum(-1)/n
        distance = (coordinate[:, :, None]-coordinate[:, None, :]).abs()
        distance = torch.minimum(distance, (total[:, None, None]-distance).clamp_min(0))
        bias = -distance[:, None]*self.log_rates.clamp(-9, 4).exp()[None, :, None, None]
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=bias)
        attended = attended.transpose(1, 2).reshape(b, lines, n, c).permute(0, 3, 2, 1)
        result = hidden+self.output(attended)
        return result.transpose(-2, -1) if self.axis == 1 else result


class OpticalTailNetwork(TailResponseNetwork):
    def __init__(self, angles=12, width=32, blocks=4, modes=6):
        super().__init__(angles, width, blocks, modes)
        self.optical_x = OpticalAxisAttention(width, 0)
        self.optical_y = OpticalAxisAttention(width, 1)

    def forward(self, t, s, residual, reference):
        rms = lambda x: x.abs().square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-30)
        rscale, scale = rms(residual), rms(reference)
        q, ref = residual/rscale, reference/scale
        logt, loga = t.log(), (t-s).clamp_min(1e-8).log()
        gx = (torch.roll(logt, -1, -2)-torch.roll(logt, 1, -2))/2
        gy = (torch.roll(logt, -1, -1)-torch.roll(logt, 1, -1))/2
        material = torch.cat((logt/4, loga/4, s/t, logt-logt.mean((-2, -1), keepdim=True),
            gx, gy, torch.log1p(t/t.shape[-1]), torch.full_like(t, math.log(t.shape[-1])/5)), 1)
        first = F.gelu(self.stem(torch.cat((material, q.real, q.imag, ref.real, ref.imag), 1)))
        second = F.gelu(self.down1(F.avg_pool2d(first, 2)))
        hidden = self.global_blocks(F.gelu(self.down2(F.avg_pool2d(second, 2))))
        hidden = F.gelu(self.up1(torch.cat((F.interpolate(hidden, size=second.shape[-2:], mode='nearest'), second), 1)))
        geometry, density = F.avg_pool2d(material[:, :4], 2), F.avg_pool2d(t, 2)
        hidden = self.optical_x(hidden, geometry, density)
        hidden = self.optical_y(hidden, geometry, density)
        hidden = F.gelu(self.up2(torch.cat((F.interpolate(hidden, size=first.shape[-2:], mode='nearest'), first), 1)))
        real, imaginary = (self.output(hidden)+self.characteristic(hidden, t)).chunk(2, 1)
        result = reference+scale*torch.complex(real, imaginary)
        return torch.where(residual.abs().flatten(1).sum(1)[:, None, None, None] > 0, result, torch.zeros_like(result))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not any(key.startswith('optical_') for key in state_dict):
            result = super().load_state_dict(state_dict, strict=False, assign=assign)
            if result.unexpected_keys or any(not key.startswith('optical_') for key in result.missing_keys):
                raise RuntimeError('incompatible checkpoint for optical upgrade')
            return result
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
