"""One-shot full-angle source predictor with global Fourier and local fine paths.

This is a nonlinear solution initializer, NOT an arbitrary-residual inverse.
Known homogeneous transport is retained as an input and an exact skip. The
network runs once, and ordinary FGMRES enforces the final physical tolerance.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GlobalFourierBlock(nn.Module):
    def __init__(self, width, modes=6):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(2, width, width, modes, modes, dtype=torch.complex64)/width)
        self.local = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(4, width)

    def forward(self, x):
        n, m = x.shape[-2:]
        k = min(self.modes, n//2, m//2)
        spectrum = torch.fft.rfft2(x, norm='ortho')
        result = torch.zeros_like(spectrum)
        result[..., :k, :k] = torch.einsum('bixy,ioxy->boxy', spectrum[..., :k, :k], self.weight[0, :, :, :k, :k])
        result[..., -k:, :k] = torch.einsum('bixy,ioxy->boxy', spectrum[..., -k:, :k], self.weight[1, :, :, :k, :k])
        global_value = torch.fft.irfft2(result, s=(n, m), norm='ortho')
        return x+F.gelu(self.norm(global_value+self.local(x)))/2


class GlobalSourcePredictor(nn.Module):
    def __init__(self, angles=12, width=32, blocks=4, modes=6):
        super().__init__()
        self.angles, self.width = angles, width
        conv = lambda a, b: nn.Conv2d(a, b, 3, padding=1, padding_mode='circular')
        # Eight material/scale features plus complex source and reference field.
        self.stem = conv(8+4*angles, width)
        self.down1 = conv(width, width)
        self.down2 = conv(width, width)
        self.global_blocks = nn.Sequential(*[GlobalFourierBlock(width, modes) for _ in range(blocks)])
        self.up1 = conv(2*width, width)
        self.up2 = conv(2*width, width)
        self.output = nn.Conv2d(width, 2*angles, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def intensity_scale(t, s, source):
        return source.abs().mean((1, 2, 3), keepdim=True)/(t-s).mean((1, 2, 3), keepdim=True).clamp_min(1e-8)

    def forward(self, t, s, source, reference):
        absorption = (t-s).clamp_min(1e-8)
        scale = self.intensity_scale(t, s, source).clamp_min(1e-8)
        qscale = source.abs().square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
        logt, loga = t.log(), absorption.log()
        gx = (torch.roll(logt, -1, -2)-torch.roll(logt, 1, -2))/2
        gy = (torch.roll(logt, -1, -1)-torch.roll(logt, 1, -1))/2
        material = torch.cat((logt/4, loga/4, s/t, logt-logt.mean((-2, -1), keepdim=True),
            gx, gy, torch.log1p(t/t.shape[-1]), torch.full_like(t, math.log(t.shape[-1])/5)), 1)
        q, ref = source/qscale, reference/scale
        inputs = torch.cat((material, q.real, q.imag, ref.real, ref.imag), 1)
        first = F.gelu(self.stem(inputs))
        second = F.gelu(self.down1(F.avg_pool2d(first, 2)))
        hidden = F.gelu(self.down2(F.avg_pool2d(second, 2)))
        hidden = self.global_blocks(hidden)
        hidden = F.gelu(self.up1(torch.cat((F.interpolate(hidden, size=second.shape[-2:], mode='nearest'), second), 1)))
        hidden = F.gelu(self.up2(torch.cat((F.interpolate(hidden, size=first.shape[-2:], mode='nearest'), first), 1)))
        real, imaginary = self.output(hidden).chunk(2, 1)
        return reference+scale*torch.complex(real, imaginary)


def balance_integrated_source(operator, value, source):
    """Exact integrated collision balance for the STORED phase/quadrature.

    Periodic Fourier streaming sums to zero, including the stored Nyquist
    convention. No assumption of exactly normalized rounded phase is made.
    """
    phase_columns = (operator.phase_matrix*operator.weights[None]).sum(0)
    row = operator.sigma_t-operator.sigma_s*phase_columns[None, :, None, None]
    numerator = source.sum((1, 2, 3), keepdim=True)-(row*value).sum((1, 2, 3), keepdim=True)
    denominator = row.sum((1, 2, 3), keepdim=True)
    return value+numerator/denominator


class FrozenTSA:
    """Same TSA arithmetic with precomputed parameter-independent reference data."""
    def __init__(self, operator, inverse, streaming_inverse, depth=8):
        self.operator, self.inverse, self.streaming_inverse, self.depth = operator, inverse, streaming_inverse, depth

    def __call__(self, rhs):
        result = torch.zeros_like(rhs)
        for step in range(self.depth):
            defect = rhs if step == 0 else rhs-self.operator.A_op(result)
            transported = torch.fft.ifft2(self.streaming_inverse*torch.fft.fft2(defect))
            remaining = defect-self.operator.A_op(transported)
            transformed = torch.einsum('bxyij,bjxy->bixy', self.inverse, torch.fft.fft2(remaining))
            result = result+transported+torch.fft.ifft2(transformed)
        return result
