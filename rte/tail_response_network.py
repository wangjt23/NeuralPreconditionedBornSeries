"""Signed/complex residual-response network with reference-RMS output scaling.

Reuses the tested global/local characteristic architecture, but does not use
the positive-source intensity scale as the output scale of a Krylov tail.
Runs at most once per solve. FGMRES determines its useful coefficient.
"""

import torch

from rte.characteristic_source_predictor import CharacteristicSourcePredictor
from rte.global_source_predictor import GlobalSourcePredictor


class TailResponseNetwork(CharacteristicSourcePredictor):
    def forward(self, t, s, residual, reference):
        rms = lambda value: value.abs().square().mean((1, 2, 3), keepdim=True).sqrt()
        source_scale = rms(residual).clamp_min(1e-30)
        reference_scale = rms(reference).clamp_min(1e-30)
        normalized_source = residual/source_scale
        # Adapter keeps all inherited normalization floors away from tiny
        # physical tails while exposing unit-RMS residual/reference features.
        internal_scale = GlobalSourcePredictor.intensity_scale(t, s, normalized_source).clamp_min(1e-12)
        internal_reference = reference/reference_scale*internal_scale
        value = super().forward(t, s, normalized_source, internal_reference)
        result = value*(reference_scale/internal_scale)
        return torch.where(rms(residual) > 0, result, torch.zeros_like(result))
