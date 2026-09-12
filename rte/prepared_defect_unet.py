"""Frozen coefficient-cache control for the original, simple defect2 U-Net.

This is an inference snapshot, not a trainable model. Recreate it after changes
to network parameters or coefficients. It deliberately supports only the
baseline gate family used in the collision/transport experiment.
"""

import copy

import torch
from torch import nn


class PreparedDefectUNet(nn.Module):
    @torch.no_grad()
    def __init__(self, model):
        super().__init__()
        if model.form != "defect2" or model.scale_aware_gates or model.rom_head is not None or model.base_module is not None:
            raise ValueError("only the standard defect2 coefficient-gated U-Net is supported")
        self.rte = model.rte
        self.base_kind = model.base_kind
        self.tsa_harmonics = model.tsa_harmonics
        self.delta_scale = model.delta_scale
        self.defect_relaxation = model.defect_relaxation
        self.base_calls_per_apply = 2
        self.unet = copy.deepcopy(model.unet).eval().requires_grad_(False)
        features = model.coeff_encoder(model.coeff_features(1))
        for level, projector in enumerate(model.unet.enc_gate):
            self.register_buffer(f"encoder_gate_{level}", 1 + model.unet.gate_scale * projector(features[level]).tanh())
        for step, projector in enumerate(model.unet.dec_gate):
            level = model.unet.levels - 2 - step
            self.register_buffer(f"decoder_gate_{step}", 1 + model.unet.gate_scale * projector(features[level]).tanh())
        self._coeff_signature = self._signature()
        self.eval()

    def _signature(self):
        return (self.rte.dx, self.rte.anisotropy_g,
                tuple((t.data_ptr(), t._version, t.device, t.dtype, t.shape)
                      for t in (self.rte.sigma_t, self.rte.sigma_s)))

    def forward(self, residual):
        if self._signature() != self._coeff_signature:
            raise RuntimeError("medium changed; recreate the prepared U-Net")
        if self.rte.sigma_t.shape[0] != 1:
            raise ValueError("prepared U-Net requires a single coefficient field")
        base = lambda r: self.rte.apply_preconditioner(r, kind=self.base_kind, tsa_harmonics=self.tsa_harmonics)
        z0 = base(residual)
        z1 = base(residual - self.rte.A_op(z0))
        features = torch.cat([z0.real, z0.imag, z1.real, z1.imag], 1).to(self.unet.lift.conv.weight.dtype)
        x, skips = self.unet.lift(features), []
        for level in range(self.unet.levels):
            x = self.unet.enc_mix[level](x) * getattr(self, f"encoder_gate_{level}")
            skips.append(x)
            if level < self.unet.levels - 1:
                x = self.unet.down[level](x)
        for step in range(self.unet.levels - 1):
            level = self.unet.levels - 2 - step
            x = self.unet.resize_like(self.unet.up[step](x), skips[level])
            x = self.unet.dec_mix1[step](torch.cat([skips[level], x], 1))
            x = self.unet.dec_mix2[step](x * getattr(self, f"decoder_gate_{step}"))
        real, imag = self.unet.proj(x).chunk(2, dim=1)
        return z0 + self.defect_relaxation * z1 + self.delta_scale * torch.complex(real, imag).to(residual.dtype)
