"""Coefficient-conditioned collision/streaming factorization for periodic RTE.

The unmodified layer is the classical HSS correction
2*alpha*(alpha+S)^(-1)*(alpha+C)^(-1), with A=S+C. Collision is
diagonal in angular Fourier coordinates at each spatial point; streaming is
diagonal in spatial Fourier coordinates for each ordinate. All modes survive.
Only coefficient-dependent parameters are learned: the residual map remains
complex-linear. Learned gains/shifts do not inherit the fixed-HSS convergence
theorem automatically. Intended deployment is as a Krylov preconditioner.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CollisionTransportPreconditioner(nn.Module):
    def __init__(self, rte, depth=2, learned=False, width=12, shift_factor=1.0,
                 gain_scale=0.7, protect_scalar=True, learn_gains=True, macro_harmonics=None,
                 stage_factors=None, spatial_gains=True):
        super().__init__()
        if depth < 1 or width < 1 or shift_factor <= 0 or gain_scale < 0:
            raise ValueError("depth/width/shift_factor must be positive; gain_scale nonnegative")
        self.rte = rte
        self.depth = int(depth)
        self.learned = bool(learned)
        self.shift_factor = float(shift_factor)
        self.stage_factors = tuple(stage_factors) if stage_factors is not None else (1.,) * self.depth
        if len(self.stage_factors) != self.depth or any(x <= 0 for x in self.stage_factors):
            raise ValueError("stage_factors must contain one positive factor per stage")
        self.gain_scale = float(gain_scale)
        self.protect_scalar = bool(protect_scalar)
        self.learn_gains = bool(learn_gains)
        self.spatial_gains = bool(spatial_gains)
        self.macro_harmonics = macro_harmonics
        if macro_harmonics is not None and not 1 <= macro_harmonics < (rte.n_angles + 1) // 2:
            raise ValueError("macro_harmonics must retain a proper symmetric angular space")
        self.n_abs_modes = rte.n_angles // 2 + 1
        self.base_calls_per_apply = 0  # No transport_dsa/transport_tsa calls.
        self.factor_stages_per_apply = self.depth
        self._prepared = None
        self._prepared_signature = None
        self._prepared_filters = None
        self._prepared_inputs = None
        if self.learned:
            self.shift_net = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, depth))
            nn.init.zeros_(self.shift_net[-1].weight)
            nn.init.zeros_(self.shift_net[-1].bias)
            if self.learn_gains:
                self.gain_net = nn.Sequential(
                    nn.Conv2d(6, width, 3, padding=1, padding_mode="circular"), nn.SiLU(),
                    nn.Conv2d(width, width, 1), nn.SiLU(),
                    nn.Conv2d(width, depth * self.n_abs_modes + int(macro_harmonics is not None), 1),
                )
                nn.init.zeros_(self.gain_net[-1].weight)
                nn.init.zeros_(self.gain_net[-1].bias)

    def collision_spectrum(self):
        kernel = self.rte.phase_matrix[0] * self.rte.weights
        phase_eigenvalues = torch.fft.fft(kernel).real
        return (self.rte.sigma_t - self.rte.sigma_s * phase_eigenvalues.view(1, -1, 1, 1))

    def _cache_inputs(self):
        return (self.rte.sigma_t, self.rte.sigma_s, self.rte.phase_matrix,
                  self.rte.weights, self.rte.kx, self.rte.ky, self.rte.directions, *self.parameters())

    def _signature(self):
        values = self._cache_inputs()
        return ((self.rte.dx, self.rte.anisotropy_g, self.shift_factor, self.gain_scale,
                 self.protect_scalar, self.macro_harmonics, self.stage_factors, self.spatial_gains,
                 getattr(self.rte, "setup_generation", 0)),
                tuple((x.data_ptr(), x._version, str(x.device), x.dtype, tuple(x.shape)) for x in values))

    def _fields(self):
        eigenvalues = self.collision_spectrum()
        low = eigenvalues.amin((1, 2, 3))
        high = eigenvalues.amax((1, 2, 3))
        if not bool((low > 0).all()):
            raise ValueError("collision spectrum must be strictly positive")
        alpha0 = self.shift_factor * (low * high).sqrt()
        shifts = alpha0[:, None] * alpha0.new_tensor(self.stage_factors)[None]
        gains = torch.ones(eigenvalues.shape[0], self.depth, *eigenvalues.shape[1:],
                           dtype=eigenvalues.dtype, device=eigenvalues.device)
        macro_gain = torch.ones_like(self.rte.sigma_t)
        if self.learned:
            st, ss = self.rte.sigma_t, self.rte.sigma_s
            absorption = st - ss
            albedo = ss / st
            logst = st.log()
            mean = logst.mean((-2, -1), keepdim=True)
            local = F.avg_pool2d(F.pad(logst, (2, 2, 2, 2), mode="circular"), 5, stride=1)
            optical = torch.log1p(st * self.rte.dx)
            coeff = torch.cat([logst - mean, albedo, (absorption / st).log(),
                               optical, local - mean, torch.full_like(st, self.rte.anisotropy_g)], 1)
            summary = torch.stack([low.log(), high.log(), st.mean((1, 2, 3)).log(),
                                   absorption.mean((1, 2, 3)).log(),
                                   albedo.mean((1, 2, 3)), albedo.std((1, 2, 3)),
                                   optical.mean((1, 2, 3)),
                                   torch.full_like(low, self.rte.anisotropy_g)], 1)
            network_dtype = self.shift_net[0].weight.dtype
            # Smooth positive shifts, within a factor eight of the analytic scale.
            shifts = shifts * torch.exp(math.log(8) * torch.tanh(self.shift_net(summary.to(network_dtype)))).to(shifts.dtype)
            if self.learn_gains:
                output = self.gain_net(coeff.to(network_dtype))
                raw = output[:, :self.depth * self.n_abs_modes].reshape(
                    eigenvalues.shape[0], self.depth, self.n_abs_modes, *eigenvalues.shape[-2:])
                index = torch.arange(self.rte.n_angles, device=st.device)
                index = torch.minimum(index, self.rte.n_angles - index)
                raw = raw[:, :, index].to(eigenvalues.dtype)
                if self.protect_scalar:
                    mask = (index != 0).view(1, 1, -1, 1, 1)
                    raw = raw * mask
                gains = torch.exp(self.gain_scale * torch.tanh(raw))
                if not self.spatial_gains:
                    gains = gains.mean((-2, -1), keepdim=True).expand_as(gains)
                if self.macro_harmonics is not None:
                    macro_gain = torch.exp(self.gain_scale * torch.tanh(output[:, -1:])).to(eigenvalues.dtype)
        return eigenvalues, shifts, gains, macro_gain

    def _macro_inverse(self, eigenvalues, stream):
        indices = [0]
        for m in range(1, self.macro_harmonics + 1):
            indices.extend([m, self.rte.n_angles - m])
        indices = torch.tensor(indices, device=eigenvalues.device)
        basis = torch.fft.ifft(torch.eye(self.rte.n_angles, dtype=stream.dtype,
                                         device=stream.device), dim=0, norm="ortho")[:, indices]
        projected_stream = torch.einsum("lm,blxy,ln->bxymn", basis.conj(), stream, basis)
        average_collision = eigenvalues[:, indices].mean((-2, -1))
        matrix = projected_stream + torch.diag_embed(average_collision)[:, None, None]
        return indices, torch.linalg.inv(matrix)

    def fields(self):
        if self.training or torch.is_grad_enabled():
            fields = self._fields()
            macro = None if self.macro_harmonics is None else self._macro_inverse(
                fields[0], self.rte._stream_symbol(fields[0].shape[0], self.rte.complex_dtype, fields[0].device))
            return fields, macro
        signature = self._signature()
        if self._prepared is None or signature != self._prepared_signature:
            fields = tuple(value.detach() for value in self._fields())
            eigenvalues, shifts, gains, _ = fields
            alpha = shifts[:, :, None, None, None]
            stream = self.rte._stream_symbol(eigenvalues.shape[0], self.rte.complex_dtype, eigenvalues.device)
            # All coefficients are fixed during a solve. Cache the multipliers,
            # not just their ingredients, and give classical HSS the same cache.
            self._prepared_filters = (gains / (alpha + eigenvalues[:, None]),
                                      2 * alpha / (alpha + stream[:, None]))
            macro = None if self.macro_harmonics is None else self._macro_inverse(
                fields[0], self.rte._stream_symbol(fields[0].shape[0], self.rte.complex_dtype, fields[0].device))
            self._prepared = fields, macro
            self._prepared_signature = signature
            # Keep source tensors alive until refresh: this also prevents
            # address reuse for direct replacements outside rte.setup().
            self._prepared_inputs = self._cache_inputs()
        return self._prepared

    @torch.no_grad()
    def prepare(self):
        """Build coefficient-dependent fields once; subsequent no-grad calls reuse them."""
        self.eval()
        return self.fields()

    def forward(self, residual):
        (eigenvalues, shifts, gains, macro_gain), macro = self.fields()
        if eigenvalues.shape[0] not in (1, residual.shape[0]):
            raise ValueError("coefficient and residual batches do not broadcast")
        cached = not self.training and not torch.is_grad_enabled()
        stream = None if cached else self.rte._stream_symbol(residual.shape[0], residual.dtype, residual.device)
        correction = torch.zeros_like(residual)
        defect = residual
        for stage in range(self.depth):
            alpha = shifts[:, stage].view(-1, 1, 1, 1)
            angular = torch.fft.fft(defect, dim=1, norm="ortho")
            if cached:
                collision_filter, streaming_filter = self._prepared_filters
                local = torch.fft.ifft(angular * collision_filter[:, stage], dim=1, norm="ortho")
                spatial = torch.fft.fftn(local, dim=(-2, -1))
                update = torch.fft.ifftn(spatial * streaming_filter[:, stage], dim=(-2, -1))
            else:
                local = torch.fft.ifft(angular * gains[:, stage] / (alpha + eigenvalues), dim=1, norm="ortho")
                spatial = torch.fft.fftn(local, dim=(-2, -1))
                update = 2 * alpha * torch.fft.ifftn(spatial / (alpha + stream), dim=(-2, -1))
            correction = correction + update
            if stage + 1 < self.depth:
                defect = residual - self.rte.A_op(correction)
        if macro is not None:
            indices, inverse = macro
            defect = residual - self.rte.A_op(correction)
            moments = torch.fft.fft(defect, dim=1, norm="ortho")[:, indices]
            rhs = torch.fft.fftn(moments, dim=(-2, -1))
            solution = torch.einsum("bxymn,bnxy->bmxy", inverse, rhs)
            angular = torch.zeros_like(defect)
            angular[:, indices] = torch.fft.ifftn(solution, dim=(-2, -1))
            correction = correction + macro_gain * torch.fft.ifft(angular, dim=1, norm="ortho")
        return correction
