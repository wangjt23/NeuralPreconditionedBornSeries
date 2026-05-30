import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "helmholtz" / "models"))

from rte.run_hard_case_experiments import contraction, make_case
from rte.train_defect_cnn_small import make_source, smooth_random_like


def circular_pad_conv(in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False):
    return _CircularConv2d(in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=bias)


class _CircularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=False):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        if self.padding:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="circular")
        return self.conv(x)


def norm_groups(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


class CoeffEncoder(nn.Module):
    def __init__(self, coeff_channels, width, levels):
        super().__init__()
        groups = norm_groups(width)
        self.lift = nn.Sequential(
            circular_pad_conv(coeff_channels, width, bias=True),
            nn.GroupNorm(groups, width),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    circular_pad_conv(width, width, bias=True),
                    nn.GroupNorm(groups, width),
                    nn.SiLU(),
                    circular_pad_conv(width, width, bias=True),
                    nn.GroupNorm(groups, width),
                    nn.SiLU(),
                )
                for _ in range(levels)
            ]
        )
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    circular_pad_conv(width, width, kernel_size=5, stride=2, bias=True),
                    nn.GroupNorm(groups, width),
                    nn.SiLU(),
                )
                for _ in range(levels - 1)
            ]
        )

    def forward(self, coeff):
        x = self.lift(coeff)
        features = []
        for level, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            if level < len(self.down):
                x = self.down[level](x)
        return features


class ScaleAwareCoeffGate(nn.Module):
    """
    Per-level coefficient gate for multilevel U-Net corrections.

    The local branch sees the coefficient feature map at the current U-Net
    resolution.  The summary branch sees mean/std/min/max statistics of that
    same scale plus a small level code, so coarse and fine gates can learn
    different coefficient dependence.
    """

    def __init__(
        self,
        channels,
        coeff_channels,
        level_idx,
        levels,
        gate_scale=0.1,
        channel_mix=False,
        stage=0.0,
    ):
        super().__init__()
        self.gate_scale = float(gate_scale)
        self.level_idx = int(level_idx)
        self.levels = max(int(levels), 1)
        self.channel_mix = bool(channel_mix)
        self.stage = float(stage)
        hidden = max(int(channels), 8)
        self.local = circular_pad_conv(coeff_channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.local.conv.weight)
        nn.init.zeros_(self.local.conv.bias)
        self.summary = nn.Sequential(
            nn.Linear(4 * coeff_channels + 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
        )
        nn.init.zeros_(self.summary[-1].weight)
        nn.init.zeros_(self.summary[-1].bias)
        self.gain_logit = nn.Parameter(torch.zeros(()))
        if self.channel_mix:
            self.mix = nn.Sequential(
                nn.Linear(4 * coeff_channels + 4, hidden),
                nn.SiLU(),
                nn.Linear(hidden, channels * channels),
            )
            nn.init.zeros_(self.mix[-1].weight)
            nn.init.zeros_(self.mix[-1].bias)
            self.mix_gain_logit = nn.Parameter(torch.zeros(()))

    def level_code(self, batch_size, device, dtype):
        denom = max(self.levels - 1, 1)
        level_fraction = float(self.level_idx) / float(denom)
        scale = 2.0 ** float(self.level_idx)
        values = torch.tensor(
            [level_fraction, math.log2(scale), 1.0 / scale, self.stage],
            dtype=dtype,
            device=device,
        )
        return values.view(1, 4).expand(batch_size, -1)

    def forward(self, x, coeff):
        if coeff.shape[-2:] != x.shape[-2:]:
            coeff = F.interpolate(coeff, size=x.shape[-2:], mode="bilinear", align_corners=False)
        summary = torch.cat(
            [
                spatial_summary(coeff),
                self.level_code(coeff.shape[0], coeff.device, coeff.dtype),
            ],
            dim=1,
        )
        global_bias = self.summary(summary).view(coeff.shape[0], x.shape[1], 1, 1)
        gate_logits = self.local(coeff) + global_bias
        bounded_gain = self.gate_scale * (1.0 + 0.5 * torch.tanh(self.gain_logit))
        out = x * (1.0 + bounded_gain * torch.tanh(gate_logits))
        if self.channel_mix:
            matrix = torch.tanh(self.mix(summary)).view(coeff.shape[0], x.shape[1], x.shape[1])
            mixed = torch.einsum("bcd,bdhw->bchw", matrix.to(dtype=x.dtype), x) / math.sqrt(x.shape[1])
            mix_gain = 0.5 * self.gate_scale * (1.0 + 0.5 * torch.tanh(self.mix_gain_logit))
            out = out + mix_gain * mixed
        return out


class LinearGatedUNet(nn.Module):
    """
    U-Net correction that is exactly linear in residual features for fixed
    coefficient features.  All residual-path convolutions are bias-free and
    activation-free; nonlinearity enters only through coefficient gates.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        width=16,
        levels=3,
        gate_scale=0.1,
        output_init="zero",
        scale_aware_gates=False,
        scale_gate_channel_mix=False,
        gate_coeff_channels=None,
    ):
        super().__init__()
        self.levels = int(levels)
        self.gate_scale = float(gate_scale)
        self.scale_aware_gates = bool(scale_aware_gates)
        self.scale_gate_channel_mix = bool(scale_gate_channel_mix)
        gate_coeff_channels = int(gate_coeff_channels or width)
        self.lift = circular_pad_conv(in_channels, width, kernel_size=1, bias=False)
        self.enc_mix = nn.ModuleList([circular_pad_conv(width, width, bias=False) for _ in range(levels)])
        if self.scale_aware_gates:
            self.enc_gate = nn.ModuleList(
                [
                    ScaleAwareCoeffGate(
                        width,
                        gate_coeff_channels,
                        level_idx=level,
                        levels=levels,
                        gate_scale=gate_scale,
                        stage=-1.0,
                    )
                    if not self.scale_gate_channel_mix
                    else ScaleAwareCoeffGate(
                        width,
                        gate_coeff_channels,
                        level_idx=level,
                        levels=levels,
                        gate_scale=gate_scale,
                        channel_mix=True,
                        stage=-1.0,
                    )
                    for level in range(levels)
                ]
            )
        else:
            self.enc_gate = nn.ModuleList([circular_pad_conv(width, width, kernel_size=1, bias=True) for _ in range(levels)])
        self.down = nn.ModuleList([circular_pad_conv(width, width, kernel_size=5, stride=2, bias=False) for _ in range(levels - 1)])
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(width, width, kernel_size=4, stride=2, padding=1, bias=False) for _ in range(levels - 1)]
        )
        self.dec_mix1 = nn.ModuleList([circular_pad_conv(2 * width, width, bias=False) for _ in range(levels - 1)])
        if self.scale_aware_gates:
            self.dec_gate = nn.ModuleList(
                [
                    ScaleAwareCoeffGate(
                        width,
                        gate_coeff_channels,
                        level_idx=levels - 2 - step,
                        levels=levels,
                        gate_scale=gate_scale,
                        stage=1.0,
                    )
                    if not self.scale_gate_channel_mix
                    else ScaleAwareCoeffGate(
                        width,
                        gate_coeff_channels,
                        level_idx=levels - 2 - step,
                        levels=levels,
                        gate_scale=gate_scale,
                        channel_mix=True,
                        stage=1.0,
                    )
                    for step in range(levels - 1)
                ]
            )
        else:
            self.dec_gate = nn.ModuleList([circular_pad_conv(width, width, kernel_size=1, bias=True) for _ in range(levels - 1)])
        self.dec_mix2 = nn.ModuleList([circular_pad_conv(width, width, bias=False) for _ in range(levels - 1)])
        self.proj = circular_pad_conv(width, out_channels, kernel_size=1, bias=False)
        if output_init == "zero":
            nn.init.zeros_(self.proj.conv.weight)
        elif output_init == "small":
            nn.init.normal_(self.proj.conv.weight, mean=0.0, std=1e-3)
        else:
            raise ValueError("output_init must be 'zero' or 'small'")

    @staticmethod
    def resize_like(x, reference):
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        diff_h = reference.size(-2) - x.size(-2)
        diff_w = reference.size(-1) - x.size(-1)
        if diff_h >= 0 and diff_w >= 0:
            return F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        top = max((-diff_h) // 2, 0)
        left = max((-diff_w) // 2, 0)
        return x[..., top : top + reference.size(-2), left : left + reference.size(-1)]

    def gate(self, x, coeff, projector):
        if isinstance(projector, ScaleAwareCoeffGate):
            return projector(x, coeff)
        if coeff.shape[-2:] != x.shape[-2:]:
            coeff = F.interpolate(coeff, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * (1.0 + self.gate_scale * torch.tanh(projector(coeff)))

    def forward(self, residual_features, coeff_features):
        x = self.lift(residual_features)
        skips = []
        for level in range(self.levels):
            x = self.enc_mix[level](x)
            x = self.gate(x, coeff_features[level], self.enc_gate[level])
            skips.append(x)
            if level < self.levels - 1:
                x = self.down[level](x)
        for step in range(self.levels - 1):
            level = self.levels - 2 - step
            x = self.resize_like(self.up[step](x), skips[level])
            x = torch.cat([skips[level], x], dim=1)
            x = self.dec_mix1[step](x)
            x = self.gate(x, coeff_features[level], self.dec_gate[step])
            x = self.dec_mix2[step](x)
        return self.proj(x)


def rte_coeff_feature_map(rte, batch_size, extended=False):
    sigma_t = rte.sigma_t.to(dtype=rte.float_dtype)
    sigma_s = rte.sigma_s.to(dtype=rte.float_dtype)
    albedo = sigma_s / torch.clamp(sigma_t, min=1e-12)
    st_mean = torch.mean(sigma_t, dim=(-2, -1), keepdim=True)
    st_std = torch.clamp(torch.std(sigma_t, dim=(-2, -1), keepdim=True), min=1e-6)
    st_norm = (sigma_t - st_mean) / st_std
    dx = torch.roll(sigma_t, shifts=-1, dims=-1) - torch.roll(sigma_t, shifts=1, dims=-1)
    dy = torch.roll(sigma_t, shifts=-1, dims=-2) - torch.roll(sigma_t, shifts=1, dims=-2)
    grad = torch.sqrt(dx * dx + dy * dy) / torch.clamp(st_mean, min=1e-12)
    g = torch.full_like(sigma_t, float(rte.anisotropy_g))
    features = [st_norm, albedo, grad, g]
    if extended:
        sigma_a = torch.clamp(sigma_t - sigma_s, min=0.0)
        sigma_tr = torch.clamp(sigma_t - float(rte.anisotropy_g) * sigma_s, min=1e-12)
        optical = float(rte.dx) * sigma_t
        sigma_a_rel = sigma_a / torch.clamp(sigma_t, min=1e-12)
        sigma_tr_rel = sigma_tr / torch.clamp(sigma_t, min=1e-12)
        features.extend([optical, sigma_a_rel, sigma_tr_rel, sigma_tr])
    coeff = torch.cat(features, dim=1)
    if coeff.shape[0] == 1 and batch_size > 1:
        coeff = coeff.expand(batch_size, -1, -1, -1)
    return coeff


def raw_gate_feature_pyramid(raw_coeff, reference_features):
    """
    Build raw physical coefficient features at each U-Net level.

    Channel convention for extended raw_coeff is
    [st_norm, albedo, grad, g, h*sigma_t, sigma_a/sigma_t,
    sigma_tr/sigma_t, sigma_tr].  The optical-thickness channel is rescaled by
    the level coarsening factor so coarse gates see coarse-cell optical depth.
    """
    pyramid = []
    for level, reference in enumerate(reference_features):
        raw_level = raw_coeff
        if raw_level.shape[-2:] != reference.shape[-2:]:
            raw_level = F.interpolate(raw_level, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        if raw_level.shape[1] >= 5:
            raw_level = raw_level.clone()
            raw_level[:, 4:5] = raw_level[:, 4:5] * (2.0 ** float(level))
        pyramid.append(raw_level)
    return pyramid


def spatial_summary(features):
    mean = features.mean(dim=(-2, -1))
    std = torch.sqrt(torch.clamp(features.var(dim=(-2, -1), unbiased=False), min=1e-12))
    amin = features.amin(dim=(-2, -1))
    amax = features.amax(dim=(-2, -1))
    return torch.cat([mean, std, amin, amax], dim=1)


class LearnedBaseRouter(nn.Module):
    """
    Coefficient/residual-conditioned soft router over classical RTE bases.

    The router learns which base correction to use and learns bounded
    transport/synthetic relaxation factors for DSA/TSA-style decompositions.
    It is initialized close to a selected classical base so the starting model
    remains interpretable.
    """

    CANDIDATE_ALIASES = {
        "transport": "transport",
        "green": "transport",
        "g_eta": "transport",
        "dsa": "dsa",
        "transport_dsa": "dsa",
        "tsa": "tsa",
        "transport_tsa": "tsa",
        "tsa_moment": "tsa_moment",
        "moment_tsa": "tsa_moment",
        "transport_tsa_moment": "tsa_moment",
    }

    def __init__(
        self,
        rte,
        candidates=("transport", "dsa", "tsa", "tsa_moment"),
        initial_candidate="tsa",
        tsa_harmonics=None,
        moment_harmonics=2,
        width=32,
        blocks=2,
        temperature=1.0,
        route_scale=4.0,
        relaxation_scale=0.5,
        use_residual_summary=False,
    ):
        super().__init__()
        self.rte = rte
        self.candidates = tuple(self.normalize_candidate(candidate) for candidate in candidates)
        if not self.candidates:
            raise ValueError("learned base router needs at least one candidate")
        self.tsa_harmonics = tsa_harmonics
        self.moment_harmonics = int(moment_harmonics)
        if "tsa_moment" in self.candidates and rte.angular_mode_indices(self.moment_harmonics) is None:
            raise ValueError("tsa_moment router candidate requires a valid moment_harmonics value")
        self.temperature = max(float(temperature), 1e-3)
        self.route_scale = float(route_scale)
        self.relaxation_scale = float(relaxation_scale)
        self.use_residual_summary = bool(use_residual_summary)
        self.base_calls_per_apply = len(self.candidates)
        self.router_candidates = ",".join(self.candidates)

        coeff_dim = 8 * 4
        residual_dim = 4 if self.use_residual_summary else 0
        hidden = max(int(width), 8)
        layers = [nn.Linear(coeff_dim + residual_dim, hidden), nn.SiLU()]
        for _ in range(max(int(blocks) - 1, 0)):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, 3 * len(self.candidates)))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        initial = self.normalize_candidate(initial_candidate)
        if initial not in self.candidates:
            initial = self.candidates[0]
        logits = torch.full((len(self.candidates),), -float(route_scale), dtype=rte.float_dtype)
        logits[self.candidates.index(initial)] = float(route_scale)
        self.register_buffer("initial_route_logits", logits, persistent=False)

    @classmethod
    def normalize_candidate(cls, candidate):
        key = str(candidate).lower()
        if key not in cls.CANDIDATE_ALIASES:
            raise ValueError(
                "learned base candidate must be one of transport, dsa, tsa, or tsa_moment "
                f"(got {candidate!r})"
            )
        return cls.CANDIDATE_ALIASES[key]

    def coeff_summary(self, batch_size, device):
        coeff = rte_coeff_feature_map(self.rte, batch_size, extended=True).to(device=device, dtype=torch.float32)
        return spatial_summary(coeff)

    def residual_summary(self, residual):
        with torch.no_grad():
            energy = batch_l2_norm(residual)
            flux = self.rte.scalar_flux(residual)
            flux_ratio = batch_l2_norm(flux) / torch.clamp(energy, min=1e-30)
            angular_fft = torch.fft.fft(residual, dim=1, norm="backward")
            n_angles = angular_fft.shape[1]
            low_indices = [0]
            if n_angles > 2:
                low_indices.extend([1, n_angles - 1])
            if n_angles > 4:
                low_indices.extend([2, n_angles - 2])
            low_modes = angular_fft[:, low_indices, :, :]
            low_ratio = batch_l2_norm(low_modes) / torch.clamp(batch_l2_norm(angular_fft), min=1e-30)
            per_angle = torch.sqrt(
                torch.clamp((residual.real.square() + residual.imag.square()).mean(dim=(-2, -1)), min=1e-30)
            )
            anisotropy_ratio = per_angle.amax(dim=1) / torch.clamp(per_angle.mean(dim=1), min=1e-30)
            log_energy = torch.log1p(energy).to(torch.float32)
            return torch.stack(
                [
                    log_energy,
                    flux_ratio.to(torch.float32),
                    low_ratio.to(torch.float32),
                    anisotropy_ratio.to(torch.float32),
                ],
                dim=1,
            )

    def router_outputs(self, residual):
        features = [self.coeff_summary(residual.shape[0], residual.device)]
        if self.use_residual_summary:
            features.append(self.residual_summary(residual).to(device=residual.device))
        raw = self.net(torch.cat(features, dim=1))
        route_raw, transport_raw, synthetic_raw = torch.chunk(raw, 3, dim=1)
        logits = self.initial_route_logits.to(device=residual.device, dtype=route_raw.dtype).view(1, -1)
        logits = logits + self.route_scale * torch.tanh(route_raw)
        weights = torch.softmax(logits / self.temperature, dim=1)
        transport_relax = 1.0 + self.relaxation_scale * torch.tanh(transport_raw)
        synthetic_relax = 1.0 + self.relaxation_scale * torch.tanh(synthetic_raw)
        return weights, transport_relax, synthetic_relax

    @staticmethod
    def expand_batch_weight(weight, like):
        return weight.to(dtype=like.dtype, device=like.device).view(weight.shape[0], 1, 1, 1)

    def candidate_correction(self, residual, candidate, transport_relax, synthetic_relax):
        transport = self.rte.transport_preconditioner(residual)
        t_weight = self.expand_batch_weight(transport_relax, residual)
        if candidate == "transport":
            return t_weight * transport
        remaining = residual - self.rte.A_op(transport)
        s_weight = self.expand_batch_weight(synthetic_relax, residual)
        if candidate == "dsa":
            scalar = self.rte.scalar_flux(remaining)
            synthetic = self.rte.lift_scalar_flux(self.rte.diffusion_solve(scalar), mode="isotropic")
        elif candidate == "tsa":
            synthetic = self.rte.reference_transport_solve(remaining, harmonics=self.tsa_harmonics)
        elif candidate == "tsa_moment":
            synthetic = self.rte.reference_moment_transport_solve(remaining, harmonics=self.moment_harmonics)
        else:
            raise ValueError(f"unknown router candidate: {candidate}")
        return t_weight * transport + s_weight * synthetic

    def forward(self, residual):
        route_weights, transport_relax, synthetic_relax = self.router_outputs(residual)
        correction = torch.zeros_like(residual)
        for idx, candidate in enumerate(self.candidates):
            candidate_update = self.candidate_correction(
                residual,
                candidate,
                transport_relax[:, idx],
                synthetic_relax[:, idx],
            )
            weight = self.expand_batch_weight(route_weights[:, idx], residual)
            correction = correction + weight * candidate_update
        return correction


class NPBSUNetPreconditioner(nn.Module):
    def __init__(
        self,
        rte,
        form="residual",
        base_kind="transport_dsa",
        tsa_harmonics=None,
        width=16,
        levels=3,
        gate_scale=0.1,
        delta_scale=1.0,
        defect_relaxation=1.0,
        output_init="zero",
        base_module=None,
        scale_aware_gates=False,
        scale_gate_channel_mix=False,
        extended_coeff_features=False,
    ):
        super().__init__()
        if form not in {"residual", "defect2"}:
            raise ValueError("form must be 'residual' or 'defect2'")
        self.rte = rte
        self.form = form
        self.base_kind = base_kind
        self.tsa_harmonics = tsa_harmonics
        self.n_angles = rte.n_angles
        self.delta_scale = float(delta_scale)
        self.defect_relaxation = float(defect_relaxation)
        self.base_module = base_module
        self.extended_coeff_features = bool(extended_coeff_features)
        self.scale_aware_gates = bool(scale_aware_gates)
        single_base_calls = getattr(base_module, "base_calls_per_apply", 1)
        self.base_calls_per_apply = single_base_calls if form == "residual" else 2 * single_base_calls
        in_channels = 2 * self.n_angles if form == "residual" else 4 * self.n_angles
        coeff_channels = 8 if self.extended_coeff_features else 4
        gate_coeff_channels = width + (8 if self.scale_aware_gates else 0)
        self.coeff_encoder = CoeffEncoder(coeff_channels=coeff_channels, width=width, levels=levels)
        self.unet = LinearGatedUNet(
            in_channels,
            2 * self.n_angles,
            width=width,
            levels=levels,
            gate_scale=gate_scale,
            output_init=output_init,
            scale_aware_gates=scale_aware_gates,
            scale_gate_channel_mix=scale_gate_channel_mix,
            gate_coeff_channels=gate_coeff_channels,
        )

    def base(self, residual):
        if self.base_module is not None:
            return self.base_module(residual)
        return self.rte.apply_preconditioner(residual, kind=self.base_kind, tsa_harmonics=self.tsa_harmonics)

    def metric_base(self, residual):
        return self.rte.apply_preconditioner(residual, kind=self.base_kind, tsa_harmonics=self.tsa_harmonics)

    def coeff_features(self, batch_size):
        return rte_coeff_feature_map(self.rte, batch_size, extended=self.extended_coeff_features)

    @staticmethod
    def ri(x):
        return torch.cat([x.real, x.imag], dim=1).to(torch.float32)

    def correction_from_features(self, features, batch_size, dtype, device):
        coeff = self.coeff_features(batch_size).to(device=device, dtype=features.dtype)
        coeff_features = self.coeff_encoder(coeff)
        gate_features = coeff_features
        if self.scale_aware_gates:
            raw_coeff = rte_coeff_feature_map(self.rte, batch_size, extended=True).to(device=device, dtype=features.dtype)
            raw_pyramid = raw_gate_feature_pyramid(raw_coeff, coeff_features)
            gate_features = [
                torch.cat([encoded, raw], dim=1)
                for encoded, raw in zip(coeff_features, raw_pyramid)
            ]
        delta_ri = self.unet(features, gate_features)
        real, imag = torch.chunk(delta_ri, 2, dim=1)
        return torch.complex(real, imag).to(dtype=dtype)

    def forward(self, residual):
        z0 = self.base(residual)
        if self.form == "residual":
            features = self.ri(z0)
            delta = self.correction_from_features(features, residual.shape[0], residual.dtype, residual.device)
            return z0 + self.delta_scale * delta
        defect = residual - self.rte.A_op(z0)
        z1 = self.base(defect)
        features = torch.cat([self.ri(z0), self.ri(z1)], dim=1)
        delta = self.correction_from_features(features, residual.shape[0], residual.dtype, residual.device)
        return z0 + self.defect_relaxation * z1 + self.delta_scale * delta


class MomentNPBSUNetPreconditioner(nn.Module):
    """
    Defect2 learned preconditioner whose trainable correction is restricted to
    low angular Fourier modes.
    """

    def __init__(
        self,
        rte,
        base_kind="transport_tsa_moment",
        tsa_harmonics=None,
        moment_harmonics=2,
        width=16,
        levels=3,
        gate_scale=0.1,
        delta_scale=1.0,
        defect_relaxation=1.0,
        output_init="zero",
        base_module=None,
        scale_aware_gates=False,
        scale_gate_channel_mix=False,
        extended_coeff_features=False,
    ):
        super().__init__()
        self.rte = rte
        self.form = "moment_defect2"
        self.base_kind = base_kind
        self.tsa_harmonics = tsa_harmonics
        self.moment_harmonics = int(moment_harmonics)
        modes = rte.angular_mode_indices(self.moment_harmonics)
        if modes is None:
            raise ValueError("moment_harmonics must select a proper low angular Fourier space")
        self.n_moments = len(modes)
        self.delta_scale = float(delta_scale)
        self.defect_relaxation = float(defect_relaxation)
        self.base_module = base_module
        self.extended_coeff_features = bool(extended_coeff_features)
        self.scale_aware_gates = bool(scale_aware_gates)
        self.base_calls_per_apply = 2 * getattr(base_module, "base_calls_per_apply", 1)
        coeff_channels = 8 if self.extended_coeff_features else 4
        gate_coeff_channels = width + (8 if self.scale_aware_gates else 0)
        self.coeff_encoder = CoeffEncoder(coeff_channels=coeff_channels, width=width, levels=levels)
        self.unet = LinearGatedUNet(
            4 * self.n_moments,
            2 * self.n_moments,
            width=width,
            levels=levels,
            gate_scale=gate_scale,
            output_init=output_init,
            scale_aware_gates=scale_aware_gates,
            scale_gate_channel_mix=scale_gate_channel_mix,
            gate_coeff_channels=gate_coeff_channels,
        )

    def base(self, residual):
        if self.base_module is not None:
            return self.base_module(residual)
        return self.rte.apply_preconditioner(residual, kind=self.base_kind, tsa_harmonics=self.tsa_harmonics)

    def metric_base(self, residual):
        return self.rte.apply_preconditioner(residual, kind=self.base_kind, tsa_harmonics=self.tsa_harmonics)

    def coeff_features(self, batch_size):
        return rte_coeff_feature_map(self.rte, batch_size, extended=self.extended_coeff_features)

    def restrict_moments(self, field):
        basis = self.rte.angular_fourier_basis(self.moment_harmonics, dtype=field.dtype, device=field.device)
        return torch.einsum("nm,bnhw->bmhw", basis.conj(), field)

    def prolong_moments(self, coeff):
        basis = self.rte.angular_fourier_basis(self.moment_harmonics, dtype=coeff.dtype, device=coeff.device)
        return torch.einsum("nm,bmhw->bnhw", basis, coeff)

    @staticmethod
    def ri(x):
        return torch.cat([x.real, x.imag], dim=1).to(torch.float32)

    def correction_from_moment_features(self, features, batch_size, dtype, device):
        coeff = self.coeff_features(batch_size).to(device=device, dtype=features.dtype)
        coeff_features = self.coeff_encoder(coeff)
        gate_features = coeff_features
        if self.scale_aware_gates:
            raw_coeff = rte_coeff_feature_map(self.rte, batch_size, extended=True).to(device=device, dtype=features.dtype)
            raw_pyramid = raw_gate_feature_pyramid(raw_coeff, coeff_features)
            gate_features = [
                torch.cat([encoded, raw], dim=1)
                for encoded, raw in zip(coeff_features, raw_pyramid)
            ]
        delta_ri = self.unet(features, gate_features)
        real, imag = torch.chunk(delta_ri, 2, dim=1)
        delta_moment = torch.complex(real, imag).to(dtype=dtype)
        return self.prolong_moments(delta_moment)

    def forward(self, residual):
        z0 = self.base(residual)
        defect = residual - self.rte.A_op(z0)
        z1 = self.base(defect)
        z0_m = self.restrict_moments(z0)
        z1_m = self.restrict_moments(z1)
        features = torch.cat([self.ri(z0_m), self.ri(z1_m)], dim=1)
        delta = self.correction_from_moment_features(features, residual.shape[0], residual.dtype, residual.device)
        return z0 + self.defect_relaxation * z1 + self.delta_scale * delta


class ModeWeightedMomentTSAPreconditioner(nn.Module):
    """
    Tiny learned structural preconditioner that starts exactly as closed
    TSA-Moment and learns bounded per-|m| weights on the synthetic correction.
    """

    def __init__(self, rte, moment_harmonics=2, weight_scale=0.5):
        super().__init__()
        self.rte = rte
        self.form = "weighted_moment_tsa"
        self.base_kind = "transport_tsa_moment"
        self.tsa_harmonics = int(moment_harmonics)
        self.moment_harmonics = int(moment_harmonics)
        modes = rte.angular_mode_indices(self.moment_harmonics)
        if modes is None:
            raise ValueError("moment_harmonics must select a proper low angular Fourier space")
        self.register_buffer("mode_abs", torch.tensor([abs(int(mode)) for mode in modes], dtype=torch.long), persistent=False)
        self.logits = nn.Parameter(torch.zeros(self.moment_harmonics + 1, dtype=rte.float_dtype))
        self.weight_scale = float(weight_scale)
        self.transport_relaxation = 1.0
        self.base_calls_per_apply = 1

    def mode_weights(self):
        weights_by_abs = 1.0 + self.weight_scale * torch.tanh(self.logits)
        return weights_by_abs[self.mode_abs]

    def base(self, residual):
        return self.rte.apply_preconditioner(
            residual,
            kind="transport_tsa_moment",
            tsa_harmonics=self.moment_harmonics,
        )

    def forward(self, residual):
        return self.rte.transport_moment_tsa_preconditioner(
            residual,
            harmonics=self.moment_harmonics,
            mode_weights=self.mode_weights(),
        )


class CoeffWeightedMomentTSAPreconditioner(nn.Module):
    """
    Coefficient-conditioned nonlinear-in-coefficients moment TSA.

    The closed moment solve is kept fixed. A small coefficient network predicts
    bounded physical-space weights for each absolute angular mode. The final
    projection is zero-initialized, so the initial map is exactly closed TSA-MK.
    For fixed coefficients the preconditioner remains linear in the residual.
    """

    def __init__(
        self,
        rte,
        moment_harmonics=2,
        width=16,
        blocks=2,
        weight_scale=0.5,
        transport_relaxation=1.0,
        synthetic_relaxation=1.0,
        freeze_scalar_weight=True,
    ):
        super().__init__()
        self.rte = rte
        self.form = "coeff_weighted_moment_tsa"
        self.base_kind = "transport_tsa_moment"
        self.tsa_harmonics = int(moment_harmonics)
        self.moment_harmonics = int(moment_harmonics)
        modes = rte.angular_mode_indices(self.moment_harmonics)
        if modes is None:
            raise ValueError("moment_harmonics must select a proper low angular Fourier space")
        self.n_moments = len(modes)
        self.register_buffer("mode_abs", torch.tensor([abs(int(mode)) for mode in modes], dtype=torch.long), persistent=False)
        self.weight_scale = float(weight_scale)
        self.transport_relaxation = float(transport_relaxation)
        self.synthetic_relaxation = float(synthetic_relaxation)
        self.freeze_scalar_weight = bool(freeze_scalar_weight)
        self.base_calls_per_apply = 1

        layers = [circular_pad_conv(8, width, bias=True), nn.SiLU()]
        for _ in range(int(blocks)):
            layers.extend([circular_pad_conv(width, width, bias=True), nn.SiLU()])
        layers.append(circular_pad_conv(width, self.moment_harmonics + 1, kernel_size=1, bias=True))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].conv.weight)
        nn.init.zeros_(self.net[-1].conv.bias)

    def coeff_features(self, batch_size):
        sigma_t = self.rte.sigma_t.to(dtype=self.rte.float_dtype)
        sigma_s = self.rte.sigma_s.to(dtype=self.rte.float_dtype)
        sigma_a = torch.clamp(sigma_t - sigma_s, min=0.0)
        sigma_tr = torch.clamp(sigma_t - float(self.rte.anisotropy_g) * sigma_s, min=1e-12)
        albedo = sigma_s / torch.clamp(sigma_t, min=1e-12)
        st_mean = torch.mean(sigma_t, dim=(-2, -1), keepdim=True)
        st_std = torch.clamp(torch.std(sigma_t, dim=(-2, -1), keepdim=True), min=1e-6)
        st_norm = (sigma_t - st_mean) / st_std
        dx = torch.roll(sigma_t, shifts=-1, dims=-1) - torch.roll(sigma_t, shifts=1, dims=-1)
        dy = torch.roll(sigma_t, shifts=-1, dims=-2) - torch.roll(sigma_t, shifts=1, dims=-2)
        grad = torch.sqrt(dx * dx + dy * dy) / torch.clamp(st_mean, min=1e-12)
        g = torch.full_like(sigma_t, float(self.rte.anisotropy_g))
        optical = float(self.rte.dx) * sigma_t
        sigma_a_rel = sigma_a / torch.clamp(sigma_t, min=1e-12)
        sigma_tr_rel = sigma_tr / torch.clamp(sigma_t, min=1e-12)
        coeff = torch.cat([st_norm, albedo, grad, g, optical, sigma_a_rel, sigma_tr_rel, sigma_tr], dim=1)
        if coeff.shape[0] == 1 and batch_size > 1:
            coeff = coeff.expand(batch_size, -1, -1, -1)
        return coeff

    def restrict_moments(self, field):
        basis = self.rte.angular_fourier_basis(self.moment_harmonics, dtype=field.dtype, device=field.device)
        return torch.einsum("nm,bnhw->bmhw", basis.conj(), field)

    def prolong_moments(self, coeff):
        basis = self.rte.angular_fourier_basis(self.moment_harmonics, dtype=coeff.dtype, device=coeff.device)
        return torch.einsum("nm,bmhw->bnhw", basis, coeff)

    def mode_weights(self, batch_size, dtype, device):
        coeff = self.coeff_features(batch_size).to(device=device, dtype=torch.float32)
        weights_by_abs = 1.0 + self.weight_scale * torch.tanh(self.net(coeff))
        if self.freeze_scalar_weight:
            scalar = torch.ones_like(weights_by_abs[:, 0:1])
            weights_by_abs = torch.cat([scalar, weights_by_abs[:, 1:]], dim=1)
        weights = weights_by_abs[:, self.mode_abs.to(device=device), :, :]
        return self.synthetic_relaxation * weights.to(dtype=dtype)

    def base(self, residual):
        return self.rte.apply_preconditioner(
            residual,
            kind="transport_tsa_moment",
            tsa_harmonics=self.moment_harmonics,
        )

    def forward(self, residual):
        transport_correction = self.rte.transport_preconditioner(residual)
        remaining_residual = residual - self.rte.A_op(transport_correction)
        synthetic_correction = self.rte.reference_moment_transport_solve(
            remaining_residual,
            harmonics=self.moment_harmonics,
        )
        synthetic_moments = self.restrict_moments(synthetic_correction)
        weights = self.mode_weights(residual.shape[0], residual.real.dtype, residual.device)
        weighted_synthetic = self.prolong_moments(synthetic_moments * weights.to(dtype=synthetic_moments.dtype))
        return self.transport_relaxation * transport_correction + weighted_synthetic


class CoeffMixedMomentTSAPreconditioner(nn.Module):
    """
    Coefficient-conditioned local linear mixing on retained moment coefficients.

    The model starts exactly at damped closed TSA-MK and learns a bounded real
    mixing matrix in the retained moment space. For fixed coefficients the map
    remains linear in the residual. The scalar moment can be preserved exactly.
    """

    def __init__(
        self,
        rte,
        moment_harmonics=2,
        width=16,
        blocks=2,
        weight_scale=0.5,
        transport_relaxation=1.0,
        synthetic_relaxation=1.0,
        freeze_scalar_weight=True,
    ):
        super().__init__()
        self.rte = rte
        self.form = "coeff_mixed_moment_tsa"
        self.base_kind = "transport_tsa_moment"
        self.tsa_harmonics = int(moment_harmonics)
        self.moment_harmonics = int(moment_harmonics)
        modes = rte.angular_mode_indices(self.moment_harmonics)
        if modes is None:
            raise ValueError("moment_harmonics must select a proper low angular Fourier space")
        self.n_moments = len(modes)
        self.weight_scale = float(weight_scale)
        self.transport_relaxation = float(transport_relaxation)
        self.synthetic_relaxation = float(synthetic_relaxation)
        self.freeze_scalar_weight = bool(freeze_scalar_weight)
        self.base_calls_per_apply = 1

        layers = [circular_pad_conv(8, width, bias=True), nn.SiLU()]
        for _ in range(int(blocks)):
            layers.extend([circular_pad_conv(width, width, bias=True), nn.SiLU()])
        layers.append(circular_pad_conv(width, self.n_moments * self.n_moments, kernel_size=1, bias=True))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].conv.weight)
        nn.init.zeros_(self.net[-1].conv.bias)

    def coeff_features(self, batch_size):
        sigma_t = self.rte.sigma_t.to(dtype=self.rte.float_dtype)
        sigma_s = self.rte.sigma_s.to(dtype=self.rte.float_dtype)
        sigma_a = torch.clamp(sigma_t - sigma_s, min=0.0)
        sigma_tr = torch.clamp(sigma_t - float(self.rte.anisotropy_g) * sigma_s, min=1e-12)
        albedo = sigma_s / torch.clamp(sigma_t, min=1e-12)
        st_mean = torch.mean(sigma_t, dim=(-2, -1), keepdim=True)
        st_std = torch.clamp(torch.std(sigma_t, dim=(-2, -1), keepdim=True), min=1e-6)
        st_norm = (sigma_t - st_mean) / st_std
        dx = torch.roll(sigma_t, shifts=-1, dims=-1) - torch.roll(sigma_t, shifts=1, dims=-1)
        dy = torch.roll(sigma_t, shifts=-1, dims=-2) - torch.roll(sigma_t, shifts=1, dims=-2)
        grad = torch.sqrt(dx * dx + dy * dy) / torch.clamp(st_mean, min=1e-12)
        g = torch.full_like(sigma_t, float(self.rte.anisotropy_g))
        optical = float(self.rte.dx) * sigma_t
        sigma_a_rel = sigma_a / torch.clamp(sigma_t, min=1e-12)
        sigma_tr_rel = sigma_tr / torch.clamp(sigma_t, min=1e-12)
        coeff = torch.cat([st_norm, albedo, grad, g, optical, sigma_a_rel, sigma_tr_rel, sigma_tr], dim=1)
        if coeff.shape[0] == 1 and batch_size > 1:
            coeff = coeff.expand(batch_size, -1, -1, -1)
        return coeff

    def restrict_moments(self, field):
        basis = self.rte.angular_fourier_basis(self.moment_harmonics, dtype=field.dtype, device=field.device)
        return torch.einsum("nm,bnhw->bmhw", basis.conj(), field)

    def prolong_moments(self, coeff):
        basis = self.rte.angular_fourier_basis(self.moment_harmonics, dtype=coeff.dtype, device=coeff.device)
        return torch.einsum("nm,bmhw->bnhw", basis, coeff)

    def mixing_matrix(self, batch_size, device):
        coeff = self.coeff_features(batch_size).to(device=device, dtype=torch.float32)
        delta = self.weight_scale * torch.tanh(self.net(coeff))
        matrix = delta.view(batch_size, self.n_moments, self.n_moments, *delta.shape[-2:])
        eye = torch.eye(self.n_moments, dtype=matrix.dtype, device=device).view(1, self.n_moments, self.n_moments, 1, 1)
        matrix = eye + matrix
        if self.freeze_scalar_weight:
            matrix[:, 0, :, :, :] = 0.0
            matrix[:, 0, 0, :, :] = 1.0
            matrix[:, 1:, 0, :, :] = 0.0
        return self.synthetic_relaxation * matrix

    def base(self, residual):
        return self.rte.apply_preconditioner(
            residual,
            kind="transport_tsa_moment",
            tsa_harmonics=self.moment_harmonics,
        )

    def forward(self, residual):
        transport_correction = self.rte.transport_preconditioner(residual)
        remaining_residual = residual - self.rte.A_op(transport_correction)
        synthetic_correction = self.rte.reference_moment_transport_solve(
            remaining_residual,
            harmonics=self.moment_harmonics,
        )
        synthetic_moments = self.restrict_moments(synthetic_correction)
        matrix = self.mixing_matrix(residual.shape[0], residual.device)
        mixed_moments = torch.einsum("bmnxy,bnxy->bmxy", matrix.to(dtype=synthetic_moments.dtype), synthetic_moments)
        mixed_synthetic = self.prolong_moments(mixed_moments)
        return self.transport_relaxation * transport_correction + mixed_synthetic


class LearnedRPMomentTSAPreconditioner(nn.Module):
    """
    Coefficient-conditioned learned restriction/prolongation around the fixed
    closed moment inverse.

    The retained Fourier basis is still used to define the low-order solve, but
    coefficient-conditioned local linear maps modify the retained moments before
    and after the closed inverse. Identity initialization recovers damped
    closed TSA-MK exactly.
    """

    def __init__(
        self,
        rte,
        moment_harmonics=2,
        width=16,
        blocks=2,
        weight_scale=0.5,
        transport_relaxation=1.0,
        synthetic_relaxation=1.0,
        freeze_scalar_weight=True,
    ):
        super().__init__()
        self.rte = rte
        self.form = "learned_rp_moment_tsa"
        self.base_kind = "transport_tsa_moment"
        self.tsa_harmonics = int(moment_harmonics)
        self.moment_harmonics = int(moment_harmonics)
        modes = rte.angular_mode_indices(self.moment_harmonics)
        if modes is None:
            raise ValueError("moment_harmonics must select a proper low angular Fourier space")
        self.n_moments = len(modes)
        self.weight_scale = float(weight_scale)
        self.transport_relaxation = float(transport_relaxation)
        self.synthetic_relaxation = float(synthetic_relaxation)
        self.freeze_scalar_weight = bool(freeze_scalar_weight)
        self.base_calls_per_apply = 1

        layers = [circular_pad_conv(8, width, bias=True), nn.SiLU()]
        for _ in range(int(blocks)):
            layers.extend([circular_pad_conv(width, width, bias=True), nn.SiLU()])
        layers.append(circular_pad_conv(width, 2 * self.n_moments * self.n_moments, kernel_size=1, bias=True))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].conv.weight)
        nn.init.zeros_(self.net[-1].conv.bias)

    def coeff_features(self, batch_size):
        sigma_t = self.rte.sigma_t.to(dtype=self.rte.float_dtype)
        sigma_s = self.rte.sigma_s.to(dtype=self.rte.float_dtype)
        sigma_a = torch.clamp(sigma_t - sigma_s, min=0.0)
        sigma_tr = torch.clamp(sigma_t - float(self.rte.anisotropy_g) * sigma_s, min=1e-12)
        albedo = sigma_s / torch.clamp(sigma_t, min=1e-12)
        st_mean = torch.mean(sigma_t, dim=(-2, -1), keepdim=True)
        st_std = torch.clamp(torch.std(sigma_t, dim=(-2, -1), keepdim=True), min=1e-6)
        st_norm = (sigma_t - st_mean) / st_std
        dx = torch.roll(sigma_t, shifts=-1, dims=-1) - torch.roll(sigma_t, shifts=1, dims=-1)
        dy = torch.roll(sigma_t, shifts=-1, dims=-2) - torch.roll(sigma_t, shifts=1, dims=-2)
        grad = torch.sqrt(dx * dx + dy * dy) / torch.clamp(st_mean, min=1e-12)
        g = torch.full_like(sigma_t, float(self.rte.anisotropy_g))
        optical = float(self.rte.dx) * sigma_t
        sigma_a_rel = sigma_a / torch.clamp(sigma_t, min=1e-12)
        sigma_tr_rel = sigma_tr / torch.clamp(sigma_t, min=1e-12)
        coeff = torch.cat([st_norm, albedo, grad, g, optical, sigma_a_rel, sigma_tr_rel, sigma_tr], dim=1)
        if coeff.shape[0] == 1 and batch_size > 1:
            coeff = coeff.expand(batch_size, -1, -1, -1)
        return coeff

    def restrict_moments(self, field):
        basis = self.rte.angular_fourier_basis(self.moment_harmonics, dtype=field.dtype, device=field.device)
        return torch.einsum("nm,bnhw->bmhw", basis.conj(), field)

    def prolong_moments(self, coeff):
        basis = self.rte.angular_fourier_basis(self.moment_harmonics, dtype=coeff.dtype, device=coeff.device)
        return torch.einsum("nm,bmhw->bnhw", basis, coeff)

    def rp_matrices(self, batch_size, device):
        coeff = self.coeff_features(batch_size).to(device=device, dtype=torch.float32)
        delta = self.weight_scale * torch.tanh(self.net(coeff))
        restrict_delta, prolong_delta = torch.chunk(delta, 2, dim=1)
        restrict_matrix = restrict_delta.view(batch_size, self.n_moments, self.n_moments, *delta.shape[-2:])
        prolong_matrix = prolong_delta.view(batch_size, self.n_moments, self.n_moments, *delta.shape[-2:])
        eye = torch.eye(self.n_moments, dtype=restrict_matrix.dtype, device=device).view(1, self.n_moments, self.n_moments, 1, 1)
        restrict_matrix = eye + restrict_matrix
        prolong_matrix = eye + prolong_matrix
        if self.freeze_scalar_weight:
            for matrix in (restrict_matrix, prolong_matrix):
                matrix[:, 0, :, :, :] = 0.0
                matrix[:, 0, 0, :, :] = 1.0
                matrix[:, 1:, 0, :, :] = 0.0
        return restrict_matrix, prolong_matrix

    def identity_penalty(self, batch_size, device):
        restrict_matrix, prolong_matrix = self.rp_matrices(batch_size, device)
        eye = torch.eye(self.n_moments, dtype=restrict_matrix.dtype, device=device).view(1, self.n_moments, self.n_moments, 1, 1)
        restrict_penalty = (restrict_matrix - eye).square().mean()
        prolong_penalty = (prolong_matrix - eye).square().mean()
        return 0.5 * (restrict_penalty + prolong_penalty)

    def base(self, residual):
        return self.rte.apply_preconditioner(
            residual,
            kind="transport_tsa_moment",
            tsa_harmonics=self.moment_harmonics,
        )

    def forward(self, residual):
        transport_correction = self.rte.transport_preconditioner(residual)
        remaining_residual = residual - self.rte.A_op(transport_correction)
        residual_moments = self.restrict_moments(remaining_residual)
        restrict_matrix, prolong_matrix = self.rp_matrices(residual.shape[0], residual.device)
        restricted_rhs = torch.einsum(
            "bmnxy,bnxy->bmxy",
            restrict_matrix.to(dtype=residual_moments.dtype),
            residual_moments,
        )
        synthetic_correction = self.rte.reference_moment_transport_solve(
            self.prolong_moments(restricted_rhs),
            harmonics=self.moment_harmonics,
        )
        synthetic_moments = self.restrict_moments(synthetic_correction)
        prolonged_moments = torch.einsum(
            "bmnxy,bnxy->bmxy",
            prolong_matrix.to(dtype=synthetic_moments.dtype),
            synthetic_moments,
        )
        learned_synthetic = self.prolong_moments(prolonged_moments)
        return self.transport_relaxation * transport_correction + self.synthetic_relaxation * learned_synthetic


class ResidualSubspaceMomentTSAPreconditioner(nn.Module):
    """
    Residual-dependent angular-subspace selector in the retained moment space.

    The current post-transport residual chooses a bounded global mixing matrix
    on the retained moment RHS before the fixed closed moment inverse is
    applied. This makes the preconditioner nonlinear in the residual and is a
    direct ablation of residual-dependent angular subspace selection.
    """

    def __init__(
        self,
        rte,
        moment_harmonics=2,
        width=16,
        blocks=2,
        weight_scale=0.5,
        transport_relaxation=1.0,
        synthetic_relaxation=1.0,
        freeze_scalar_weight=True,
    ):
        super().__init__()
        self.rte = rte
        self.form = "residual_subspace_moment_tsa"
        self.base_kind = "transport_tsa_moment"
        self.tsa_harmonics = int(moment_harmonics)
        self.moment_harmonics = int(moment_harmonics)
        modes = rte.angular_mode_indices(self.moment_harmonics)
        if modes is None:
            raise ValueError("moment_harmonics must select a proper low angular Fourier space")
        self.n_moments = len(modes)
        self.weight_scale = float(weight_scale)
        self.transport_relaxation = float(transport_relaxation)
        self.synthetic_relaxation = float(synthetic_relaxation)
        self.freeze_scalar_weight = bool(freeze_scalar_weight)
        self.base_calls_per_apply = 1

        hidden = max(int(width), 4 * self.n_moments)
        layers = [nn.Linear(3 * self.n_moments, hidden), nn.SiLU()]
        for _ in range(max(int(blocks) - 1, 0)):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, self.n_moments * self.n_moments))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def restrict_moments(self, field):
        basis = self.rte.angular_fourier_basis(self.moment_harmonics, dtype=field.dtype, device=field.device)
        return torch.einsum("nm,bnhw->bmhw", basis.conj(), field)

    def prolong_moments(self, coeff):
        basis = self.rte.angular_fourier_basis(self.moment_harmonics, dtype=coeff.dtype, device=coeff.device)
        return torch.einsum("nm,bmhw->bnhw", basis, coeff)

    def residual_summary(self, moment_field):
        energy = torch.sqrt(
            torch.clamp((moment_field.real.square() + moment_field.imag.square()).mean(dim=(-2, -1)), min=1e-12)
        )
        scale = torch.clamp(energy.mean(dim=1, keepdim=True), min=1e-6)
        real_mean = moment_field.real.mean(dim=(-2, -1)) / scale
        imag_mean = moment_field.imag.mean(dim=(-2, -1)) / scale
        norm_energy = energy / scale
        return torch.cat([real_mean, imag_mean, norm_energy], dim=1).to(torch.float32)

    def subspace_matrix(self, residual_moments):
        summary = self.residual_summary(residual_moments)
        delta = self.weight_scale * torch.tanh(self.net(summary))
        matrix = delta.view(residual_moments.shape[0], self.n_moments, self.n_moments)
        eye = torch.eye(self.n_moments, dtype=matrix.dtype, device=residual_moments.device).view(1, self.n_moments, self.n_moments)
        matrix = eye + matrix
        if self.freeze_scalar_weight:
            matrix[:, 0, :] = 0.0
            matrix[:, 0, 0] = 1.0
            matrix[:, 1:, 0] = 0.0
        return matrix

    def base(self, residual):
        return self.rte.apply_preconditioner(
            residual,
            kind="transport_tsa_moment",
            tsa_harmonics=self.moment_harmonics,
        )

    def forward(self, residual):
        transport_correction = self.rte.transport_preconditioner(residual)
        remaining_residual = residual - self.rte.A_op(transport_correction)
        residual_moments = self.restrict_moments(remaining_residual)
        subspace = self.subspace_matrix(residual_moments)
        adapted_rhs = torch.einsum("bmn,bnhw->bmhw", subspace.to(dtype=residual_moments.dtype), residual_moments)
        learned_synthetic = self.rte.reference_moment_transport_solve(
            self.prolong_moments(adapted_rhs),
            harmonics=self.moment_harmonics,
        )
        return self.transport_relaxation * transport_correction + self.synthetic_relaxation * learned_synthetic


def collect_residuals(rte, base_kind, tsa_harmonics=None, n_rollout=6, n_random=6):
    residuals = []
    psi = torch.zeros_like(rte.source)
    for _ in range(n_rollout):
        residual = (rte.source - rte.A_op(psi)).detach()
        residuals.append(residual)
        correction = rte.apply_preconditioner(residual, kind=base_kind, tsa_harmonics=tsa_harmonics)
        psi = psi + correction
    for _ in range(n_random):
        exact = smooth_random_like(rte.source, modes=5)
        residuals.append(rte.A_op(exact).detach())
    return residuals


def ensure_residual_batch(residuals):
    if torch.is_tensor(residuals):
        return residuals
    if not residuals:
        raise ValueError("residual batch is empty")
    return torch.cat(list(residuals), dim=0)


def batch_l2_norm(x):
    dims = tuple(range(1, x.ndim))
    if torch.is_complex(x):
        sq = x.real.square() + x.imag.square()
    else:
        sq = x.square()
    return torch.sqrt(torch.clamp(torch.sum(sq, dim=dims), min=1e-30))


def make_training_sources(case, size, n_angles, dtype, device, num_aux_sources=0):
    angular = "beam" if case in ("anisotropic", "mixed") else "isotropic"
    sources = [make_source(size, n_angles, dtype, center=(0.32, 0.47), angular=angular).to(device=device)]
    candidate_specs = [
        ((0.20, 0.72), "dipole"),
        ((0.70, 0.28), "beam" if case in ("anisotropic", "mixed") else "dipole"),
        ((0.52, 0.18), "isotropic"),
        ((0.80, 0.62), "beam"),
    ]
    for idx in range(int(num_aux_sources)):
        center, aux_angular = candidate_specs[idx % len(candidate_specs)]
        sources.append(make_source(size, n_angles, dtype, center=center, angular=aux_angular).to(device=device))
    return sources


def collect_residuals_from_sources(
    rte,
    base_kind,
    tsa_harmonics=None,
    sources=None,
    n_rollout=6,
    n_random=6,
    random_modes=5,
):
    residuals = []
    original_source = rte.source.clone()
    sources = [original_source] if sources is None else list(sources)
    try:
        with torch.no_grad():
            for source in sources:
                rte.source = source.to(device=original_source.device, dtype=original_source.dtype)
                psi = torch.zeros_like(rte.source)
                for _ in range(int(n_rollout)):
                    residual = (rte.source - rte.A_op(psi)).detach()
                    residuals.append(residual)
                    correction = rte.apply_preconditioner(residual, kind=base_kind, tsa_harmonics=tsa_harmonics)
                    psi = psi + correction
            if int(n_random) > 0:
                exact = smooth_random_like(original_source.expand(int(n_random), -1, -1, -1), modes=random_modes)
                residuals.extend(torch.unbind(rte.A_op(exact).detach(), dim=0))
    finally:
        rte.source = original_source
    return [residual.unsqueeze(0) if residual.ndim == 3 else residual for residual in residuals]


def make_residual_sampler(
    rte,
    base_kind,
    tsa_harmonics=None,
    sources=None,
    n_rollout=6,
    residuals_per_step=16,
    rollout_samples_per_step=8,
    random_modes=5,
    seed=0,
):
    rollout_bank = collect_residuals_from_sources(
        rte,
        base_kind,
        tsa_harmonics=tsa_harmonics,
        sources=sources,
        n_rollout=n_rollout,
        n_random=0,
        random_modes=random_modes,
    )
    original_source = rte.source.clone()
    sources = [original_source] if sources is None else list(sources)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    def sample():
        batches = []
        with torch.no_grad():
            if rollout_bank:
                count = min(int(rollout_samples_per_step), len(rollout_bank), int(residuals_per_step))
                if count > 0:
                    indices = torch.randperm(len(rollout_bank), generator=generator)[:count].tolist()
                    batches.append(torch.cat([rollout_bank[idx] for idx in indices], dim=0))

            remaining = int(residuals_per_step) - sum(batch.shape[0] for batch in batches)
            if remaining > 0:
                source_indices = torch.randint(len(sources), (remaining,), generator=generator)
                source_batch = torch.cat([sources[int(idx)] for idx in source_indices.tolist()], dim=0).to(
                    device=original_source.device,
                    dtype=original_source.dtype,
                )
                probe = smooth_random_like(source_batch, modes=random_modes)
                batches.append(rte.A_op(probe).detach())

        rte.source = original_source
        return torch.cat(batches, dim=0)

    return sample, {"rollout_bank_size": len(rollout_bank), "residuals_per_step": int(residuals_per_step)}


def inner_base_apply(rte, residual, depth=1, kind="transport_dsa", tsa_harmonics=None, relaxation=1.0):
    out = torch.zeros_like(residual)
    defect = residual
    for step_idx in range(depth):
        correction = rte.apply_preconditioner(defect, kind=kind, tsa_harmonics=tsa_harmonics)
        out = out + float(relaxation) * correction
        if step_idx + 1 < depth:
            defect = defect - rte.A_op(float(relaxation) * correction)
    return out


def tol_key(tol):
    return f"{float(tol):.0e}".replace("+", "")


def tolerance_snapshot(history, elapsed_history, base_calls_history, tol):
    target = float(tol)
    for idx, value in enumerate(history):
        if float(value) <= target:
            return {
                "iters_to_tol": int(idx),
                "time_to_tol": float(elapsed_history[idx]),
                "base_calls_to_tol": float(base_calls_history[idx]),
                "hit_tol": True,
            }
    return {
        "iters_to_tol": len(history) - 1,
        "time_to_tol": float("nan"),
        "base_calls_to_tol": float("nan"),
        "hit_tol": False,
    }


MOMENT_BASE_KINDS = {"transport_tsa_moment", "tsa_moment", "moment_tsa", "transport_moment_tsa"}
MOMENT_FORMS = {
    "moment_defect2",
    "weighted_moment_tsa",
    "coeff_weighted_moment_tsa",
    "coeff_mixed_moment_tsa",
    "learned_rp_moment_tsa",
    "residual_subspace_moment_tsa",
}


def normalize_moment_base_kind(kind):
    return str(kind).lower() in MOMENT_BASE_KINDS


def effective_tsa_harmonics(base_kind, requested, moment_harmonics):
    if normalize_moment_base_kind(base_kind):
        if requested is None:
            return int(moment_harmonics)
        return int(requested)
    return requested


def pack_real_imag(x):
    return torch.cat([x.real, x.imag], dim=1).to(torch.float32)


def unpack_real_imag(x, dtype):
    real, imag = torch.chunk(x, 2, dim=1)
    return torch.complex(real, imag).to(dtype=dtype)


def fgmres_with_trace_real(
    A,
    b,
    M,
    tol=1e-6,
    maxiter=100,
    restart=None,
    max_restarts=None,
):
    """Real-stacked FGMRES for learned or classical preconditioners."""
    eps = 1e-30
    if maxiter < 1:
        raise ValueError("maxiter must be >= 1")
    if restart is not None:
        restart = min(max(1, int(restart)), int(maxiter))
    if max_restarts is not None:
        max_restarts = int(max_restarts)
        if max_restarts < 0:
            raise ValueError("max_restarts must be >= 0")

    b_real = pack_real_imag(b)
    x_real = torch.zeros_like(b_real)
    b_norm = torch.clamp(torch.linalg.norm(b_real), min=eps)
    start = time.perf_counter()
    precond_calls = 0

    def elapsed():
        return float(time.perf_counter() - start)

    def A_real(xr):
        xc = unpack_real_imag(xr, dtype=b.dtype)
        yc = A(xc)
        return pack_real_imag(yc)

    def M_real(rr):
        nonlocal precond_calls
        rc = unpack_real_imag(rr, dtype=b.dtype)
        with torch.no_grad():
            zc = M(rc)
        precond_calls += 1
        return pack_real_imag(zc)

    def run_cycle(x0, inner_maxiter):
        r0 = b_real - A_real(x0)
        beta = torch.linalg.norm(r0)
        rel0 = float(beta / b_norm)
        cycle_history = [rel0]
        cycle_elapsed = [elapsed()]
        cycle_calls = [precond_calls]
        if rel0 < tol:
            return x0, cycle_history, cycle_elapsed, cycle_calls, True, False

        V = [r0 / torch.clamp(beta, min=eps)]
        Z = []
        H = torch.zeros((inner_maxiter + 1, inner_maxiter), dtype=torch.float32, device=b.device)
        x_cur = x0
        converged = False
        breakdown = False
        breakdown_tol = 1e-14
        for j in range(inner_maxiter):
            z_j = M_real(V[j])
            Z.append(z_j)
            w = A_real(z_j)
            for i in range(j + 1):
                H[i, j] = torch.sum(w * V[i])
                w = w - H[i, j] * V[i]
            H[j + 1, j] = torch.linalg.norm(w)
            h_next = float(torch.abs(H[j + 1, j]))
            if h_next > breakdown_tol:
                V.append(w / H[j + 1, j])
            H_j = H[: j + 2, : j + 1]
            e1 = torch.zeros(j + 2, dtype=torch.float32, device=b.device)
            e1[0] = beta
            y = torch.linalg.lstsq(H_j, e1).solution
            Z_tensor = torch.stack(Z, dim=0)
            y_shape = [j + 1] + [1] * (Z_tensor.ndim - 1)
            x_cur = x0 + torch.sum(Z_tensor * y[: j + 1].reshape(y_shape), dim=0)
            true_relres = float(torch.linalg.norm(b_real - A_real(x_cur)) / b_norm)
            cycle_history.append(true_relres)
            cycle_elapsed.append(elapsed())
            cycle_calls.append(precond_calls)
            if true_relres < tol:
                converged = True
                break
            if h_next <= breakdown_tol:
                breakdown = True
                break
        return x_cur, cycle_history, cycle_elapsed, cycle_calls, converged, breakdown

    if restart is None:
        x_real, history, elapsed_history, call_history, _, _ = run_cycle(x_real, int(maxiter))
        return x_real, history, elapsed_history, call_history

    history = []
    elapsed_history = []
    call_history = []
    total_iters = 0
    restarts_used = 0
    while total_iters < int(maxiter):
        inner_maxiter = min(int(restart), int(maxiter) - total_iters)
        x_real, cyc_h, cyc_t, cyc_c, converged, breakdown = run_cycle(x_real, inner_maxiter)
        if not history:
            history.extend(cyc_h)
            elapsed_history.extend(cyc_t)
            call_history.extend(cyc_c)
        else:
            history.extend(cyc_h[1:])
            elapsed_history.extend(cyc_t[1:])
            call_history.extend(cyc_c[1:])
        cycle_iters = len(cyc_h) - 1
        total_iters += cycle_iters
        if converged or breakdown or cycle_iters == 0:
            break
        if max_restarts is not None and restarts_used >= max_restarts:
            break
        restarts_used += 1

    return x_real, history, elapsed_history, call_history


def fgmres_with_trace(
    A,
    b,
    M,
    tol=1e-6,
    maxiter=100,
    restart=None,
    max_restarts=None,
):
    """
    Flexible GMRES variant that returns true residual, elapsed-time, and
    preconditioner-call traces for residual-vs-time and residual-vs-work curves.
    """
    eps = 1e-30
    if maxiter < 1:
        raise ValueError("maxiter must be >= 1")

    if restart is not None:
        restart = min(max(1, int(restart)), int(maxiter))
    if max_restarts is not None:
        max_restarts = int(max_restarts)
        if max_restarts < 0:
            raise ValueError("max_restarts must be >= 0")

    x = torch.zeros_like(b)
    b_norm = torch.clamp(torch.norm(b), min=eps)
    start = time.perf_counter()
    precond_calls = 0

    def elapsed():
        return float(time.perf_counter() - start)

    def run_cycle(x0, inner_maxiter):
        nonlocal precond_calls
        r0 = b - A(x0)
        beta = torch.norm(r0)
        rel0 = float(beta / b_norm)
        cycle_history = [rel0]
        cycle_elapsed = [elapsed()]
        cycle_calls = [precond_calls]
        if rel0 < tol:
            return x0, cycle_history, cycle_elapsed, cycle_calls, True, False

        V = [r0 / torch.clamp(beta, min=eps)]
        Z = []
        H = torch.zeros((inner_maxiter + 1, inner_maxiter), dtype=b.dtype, device=b.device)
        x_cur = x0
        converged = False
        breakdown = False
        breakdown_tol = 1e-14
        for j in range(inner_maxiter):
            z_j = M(V[j])
            precond_calls += 1
            Z.append(z_j)
            w = A(z_j)
            for i in range(j + 1):
                H[i, j] = torch.sum(w * V[i].conj())
                w = w - H[i, j] * V[i]
            H[j + 1, j] = torch.norm(w)
            h_next = torch.abs(H[j + 1, j]).item()
            if h_next > breakdown_tol:
                V.append(w / H[j + 1, j])
            H_j = H[: j + 2, : j + 1]
            e1 = torch.zeros(j + 2, dtype=b.dtype, device=b.device)
            e1[0] = beta
            y = torch.linalg.lstsq(H_j, e1).solution
            Z_tensor = torch.stack(Z, dim=0)
            y_shape = [j + 1] + [1] * (Z_tensor.ndim - 1)
            x_cur = x0 + torch.sum(Z_tensor * y[: j + 1].reshape(y_shape), dim=0)
            true_relres = float(torch.norm(b - A(x_cur)) / b_norm)
            cycle_history.append(true_relres)
            cycle_elapsed.append(elapsed())
            cycle_calls.append(precond_calls)
            if true_relres < tol:
                converged = True
                break
            if h_next <= breakdown_tol:
                breakdown = True
                break
        return x_cur, cycle_history, cycle_elapsed, cycle_calls, converged, breakdown

    if restart is None:
        x, history, elapsed_history, call_history, _, _ = run_cycle(x, int(maxiter))
        return x, history, elapsed_history, call_history

    history = []
    elapsed_history = []
    call_history = []
    total_iters = 0
    restarts_used = 0
    while total_iters < int(maxiter):
        inner_maxiter = min(int(restart), int(maxiter) - total_iters)
        x, cyc_h, cyc_t, cyc_c, converged, breakdown = run_cycle(x, inner_maxiter)
        if not history:
            history.extend(cyc_h)
            elapsed_history.extend(cyc_t)
            call_history.extend(cyc_c)
        else:
            history.extend(cyc_h[1:])
            elapsed_history.extend(cyc_t[1:])
            call_history.extend(cyc_c[1:])
        cycle_iters = len(cyc_h) - 1
        total_iters += cycle_iters
        if converged or breakdown or cycle_iters == 0:
            break
        if max_restarts is not None and restarts_used >= max_restarts:
            break
        restarts_used += 1

    return x, history, elapsed_history, call_history


def complex_to_real_channels(x):
    return torch.cat([x.real, x.imag], dim=1).to(x.real.dtype)


def real_channels_to_complex(x, n_angles, dtype):
    return torch.complex(x[:, :n_angles], x[:, n_angles : 2 * n_angles]).to(dtype=dtype)


def model_metric_base(model, residual):
    if hasattr(model, "metric_base"):
        return model.metric_base(residual)
    return model.base(residual)


def fgmres_real_stacked_with_trace(A, b, M, tol=1e-6, maxiter=100, restart=None, max_restarts=None):
    n_angles = b.shape[1]
    complex_dtype = b.dtype
    b_real = complex_to_real_channels(b)

    def A_real(x_real):
        return complex_to_real_channels(A(real_channels_to_complex(x_real, n_angles, complex_dtype)))

    def M_real(r_real):
        return complex_to_real_channels(M(real_channels_to_complex(r_real, n_angles, complex_dtype)))

    x_real, history, elapsed_history, call_history = fgmres_with_trace(
        A=A_real,
        b=b_real,
        M=M_real,
        tol=tol,
        maxiter=maxiter,
        restart=restart,
        max_restarts=max_restarts,
    )
    return real_channels_to_complex(x_real, n_angles, complex_dtype), history, elapsed_history, call_history


def metric_loss(model, residuals, physical_weight=0.5, correction_weight=1e-6):
    residual_batch = ensure_residual_batch(residuals)
    correction = model(residual_batch)
    next_residual = residual_batch - model.rte.A_op(correction)
    metric_error = model_metric_base(model, next_residual)
    denom_metric = torch.clamp(batch_l2_norm(model_metric_base(model, residual_batch)), min=1e-30)
    denom_phys = torch.clamp(batch_l2_norm(residual_batch), min=1e-30)
    correction_norm = batch_l2_norm(correction)
    metric_ratio = batch_l2_norm(metric_error) / denom_metric
    physical_ratio = batch_l2_norm(next_residual) / denom_phys
    correction_ratio = correction_norm / denom_metric
    return metric_ratio, physical_ratio, correction_ratio, next_residual


def select_objective_ratio(space, metric_ratio, physical_ratio):
    if str(space).lower() == "metric":
        return metric_ratio
    if str(space).lower() == "physical":
        return physical_ratio
    raise ValueError(f"unsupported objective space: {space}")


def objective_loss(
    model,
    residuals,
    metric_weight=1.0,
    physical_weight=0.5,
    correction_weight=1e-6,
    rollout_weight=0.0,
    rollout_steps=1,
    rollout_decay=1.0,
    rollout_space="metric",
    stability_weight=0.0,
    contraction_target=0.98,
    stability_space="metric",
    teacher_fn=None,
    teacher_weight=0.0,
):
    metric_ratio, physical_ratio, correction_ratio, residual_after = metric_loss(
        model,
        residuals,
        physical_weight=physical_weight,
        correction_weight=correction_weight,
    )
    loss = (
        float(metric_weight) * (metric_ratio.square()).mean()
        + physical_weight * (physical_ratio.square()).mean()
        + correction_weight * (correction_ratio.square()).mean()
    )
    rollout_ratio = select_objective_ratio(rollout_space, metric_ratio, physical_ratio)
    stability_history = [select_objective_ratio(stability_space, metric_ratio, physical_ratio)]
    if rollout_weight > 0.0 or stability_weight > 0.0:
        current_residual = residual_after
        for rollout_idx in range(int(rollout_steps)):
            rollout_metric_ratio, rollout_physical_ratio, _, current_residual = metric_loss(
                model,
                current_residual,
                physical_weight=physical_weight,
                correction_weight=correction_weight,
            )
            rollout_ratio = select_objective_ratio(rollout_space, rollout_metric_ratio, rollout_physical_ratio)
            stability_history.append(select_objective_ratio(stability_space, rollout_metric_ratio, rollout_physical_ratio))
            if rollout_weight > 0.0:
                rollout_scale = float(rollout_decay) ** rollout_idx
                loss = loss + rollout_weight * rollout_scale * (rollout_ratio.square()).mean()
    if stability_weight > 0.0:
        target = float(contraction_target)
        stability_terms = [(torch.relu(entry - target).square()).mean() for entry in stability_history]
        loss = loss + stability_weight * torch.stack(stability_terms).mean()
    if teacher_fn is not None and teacher_weight > 0.0:
        residual_batch = ensure_residual_batch(residuals)
        with torch.no_grad():
            teacher_correction = teacher_fn(residual_batch)
        student_correction = model(residual_batch)
        denom = torch.clamp(batch_l2_norm(teacher_correction), min=1e-30)
        imitation = batch_l2_norm(student_correction - teacher_correction) / denom
        loss = loss + float(teacher_weight) * (imitation.square()).mean()
    return loss


def resolve_scheduler_eta_min(lr, scheduler_eta_min, scheduler_min_lr_ratio):
    if scheduler_eta_min is not None:
        return float(scheduler_eta_min)
    return float(lr) * float(scheduler_min_lr_ratio)


def build_scheduler(
    optimizer,
    scheduler_name,
    steps,
    lr,
    scheduler_step_size,
    scheduler_gamma,
    scheduler_eta_min,
    scheduler_min_lr_ratio,
    scheduler_pct_start,
):
    if scheduler_name == "none":
        return None
    if scheduler_name == "step":
        step_size = int(scheduler_step_size) if int(scheduler_step_size) > 0 else max(1, int(steps) // 5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=float(scheduler_gamma))
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(steps)),
            eta_min=resolve_scheduler_eta_min(lr, scheduler_eta_min, scheduler_min_lr_ratio),
        )
    if scheduler_name == "onecycle":
        total_steps = int(steps)
        if total_steps < 3:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps),
                eta_min=resolve_scheduler_eta_min(lr, scheduler_eta_min, scheduler_min_lr_ratio),
            )
        min_lr_ratio = max(float(scheduler_min_lr_ratio), 1e-8)
        div_factor = max(1.0, min_lr_ratio ** -0.5)
        pct_start = min(max(float(scheduler_pct_start), 1.0 / total_steps), 1.0 - 1.0 / total_steps)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(lr),
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
            div_factor=div_factor,
            final_div_factor=div_factor,
        )
    raise ValueError(f"unsupported scheduler: {scheduler_name}")


def train_model(
    model,
    residuals,
    steps,
    lr,
    weight_decay=0.0,
    scheduler_name="none",
    scheduler_step_size=0,
    scheduler_gamma=0.5,
    scheduler_eta_min=None,
    scheduler_min_lr_ratio=1e-2,
    scheduler_pct_start=0.1,
    log_steps=0,
    metric_weight=1.0,
    physical_weight=0.5,
    correction_weight=1e-6,
    rollout_weight=0.0,
    rollout_steps=1,
    rollout_decay=1.0,
    rollout_space="metric",
    stability_weight=0.0,
    contraction_target=0.98,
    stability_space="metric",
    teacher_fn=None,
    teacher_weight=0.0,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(
        optimizer,
        scheduler_name,
        steps,
        lr,
        scheduler_step_size,
        scheduler_gamma,
        scheduler_eta_min,
        scheduler_min_lr_ratio,
        scheduler_pct_start,
    )

    history = {"loss": [], "lr": []}
    total_steps = int(steps)
    for step_idx in range(total_steps):
        current_residuals = residuals() if callable(residuals) else residuals
        optimizer.zero_grad(set_to_none=True)
        loss = objective_loss(
            model,
            current_residuals,
            metric_weight=metric_weight,
            physical_weight=physical_weight,
            correction_weight=correction_weight,
            rollout_weight=rollout_weight,
            rollout_steps=rollout_steps,
            rollout_decay=rollout_decay,
            rollout_space=rollout_space,
            stability_weight=stability_weight,
            contraction_target=contraction_target,
            stability_space=stability_space,
            teacher_fn=teacher_fn,
            teacher_weight=teacher_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        history["loss"].append(float(loss.detach()))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        if int(log_steps) > 0 and ((step_idx + 1) % int(log_steps) == 0 or step_idx == 0 or step_idx + 1 == total_steps):
            print(
                f"    step [{step_idx + 1:4d}/{total_steps}] loss={history['loss'][-1]:.3e} "
                f"lr={history['lr'][-1]:.2e}"
            )
    return history


def one_step_quality(rte, preconditioner, residual, base_kind, tsa_harmonics=None):
    with torch.no_grad():
        correction = preconditioner(residual)
        next_residual = residual - rte.A_op(correction)
        true_rel = float(torch.linalg.norm(next_residual) / torch.clamp(torch.linalg.norm(residual), min=1e-30))
        born_metric_error = rte.apply_preconditioner(next_residual, kind="transport")
        born_metric_ref = rte.apply_preconditioner(residual, kind="transport")
        born_metric_rel = float(
            torch.linalg.norm(born_metric_error) / torch.clamp(torch.linalg.norm(born_metric_ref), min=1e-30)
        )
        base_metric_error = rte.apply_preconditioner(next_residual, kind=base_kind, tsa_harmonics=tsa_harmonics)
        base_metric_ref = rte.apply_preconditioner(residual, kind=base_kind, tsa_harmonics=tsa_harmonics)
        base_metric_rel = float(
            torch.linalg.norm(base_metric_error) / torch.clamp(torch.linalg.norm(base_metric_ref), min=1e-30)
        )
    return true_rel, born_metric_rel, base_metric_rel


def solve_stationary(rte, source, preconditioner, maxiter, tol):
    old_source = rte.source
    try:
        rte.source = source.to(device=rte.sigma_t.device, dtype=rte.complex_dtype)
        psi = torch.zeros_like(rte.source)
        source_norm = torch.clamp(torch.linalg.norm(rte.source), min=1e-30)
        history = []
        elapsed_history = []
        preconditioner_calls = []
        start = time.perf_counter()
        with torch.no_grad():
            calls = 0
            converged = False
            for _ in range(maxiter):
                residual = rte.source - rte.A_op(psi)
                relres = float(torch.linalg.norm(residual) / source_norm)
                history.append(relres)
                elapsed_history.append(float(time.perf_counter() - start))
                preconditioner_calls.append(calls)
                if relres < tol:
                    converged = True
                    break
                psi = psi + preconditioner(residual)
                calls += 1
            if not converged:
                residual = rte.source - rte.A_op(psi)
                history.append(float(torch.linalg.norm(residual) / source_norm))
                elapsed_history.append(float(time.perf_counter() - start))
                preconditioner_calls.append(calls)
    finally:
        rte.source = old_source
    return len(history) - 1, history[-1], contraction(history), time.perf_counter() - start, history, elapsed_history, preconditioner_calls


def solve_fgmres(rte, source, preconditioner, maxiter, tol, real_stacked=False):
    old_source = rte.source
    try:
        rte.source = source.to(device=rte.sigma_t.device, dtype=rte.complex_dtype)

        def A(x):
            return rte.A_op(x)

        def M(r):
            with torch.no_grad():
                return preconditioner(r)

        with torch.no_grad():
            fgmres_fn = fgmres_real_stacked_with_trace if real_stacked else fgmres_with_trace
            _, history, elapsed_history, preconditioner_calls = fgmres_fn(
                A=A,
                b=rte.source,
                M=M,
                tol=tol,
                maxiter=maxiter,
                restart=min(20, maxiter),
                max_restarts=2,
            )
    finally:
        rte.source = old_source
    return (
        len(history) - 1,
        history[-1],
        contraction(history),
        elapsed_history[-1],
        history,
        elapsed_history,
        preconditioner_calls,
    )


def write_report(rows, path, settings):
    lines = [
        "# NPBS U-Net Preconditioner Ablation",
        "",
        "This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.",
        "",
        "## Settings",
        "",
    ]
    for key, value in settings.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Terminology",
            "",
            "- `transport` is the shifted/reference transport inverse `G_eta`.",
            "- `transport_dsa` applies one reference transport correction followed by one scalar diffusion synthetic correction.",
            "- `transport_tsa` applies one reference transport correction followed by one reference transport-scattering synthetic correction. `tsa_harmonics=None` keeps all angular Fourier modes; an integer such as `2` keeps only the lowest angular harmonics.",
            "- `transport_tsa_moment` applies one reference transport correction followed by a closed Galerkin angular Fourier synthetic correction `Q_K (Q_K^* A_ref Q_K)^{-1} Q_K^*` on the post-transport residual. This differs from `transport_tsa` with `tsa_harmonics=K`, which only projects the RHS before applying the full angular reference inverse.",
            "- `dsaK` and `tsaK` mean `K` residual-correction applications of the selected base preconditioner. Starting with `z=0` and residual `e_0=r`, each stage applies `z <- z + omega B_h e_j` and `e_{j+1} <- e_j - A_h(omega B_h e_j)`. A suffix such as `-w0.5` uses relaxation `omega=0.5`; no suffix means `omega=1`.",
            "- `UNet-defect2-w1` uses `z0=B_h r`, `z1=B_h(r-A_h z0)`, and returns `z0 + z1 + C_theta(z0,z1,coeff)`. It costs two base applications plus the U-Net evaluation per preconditioner call.",
            "- `--scale-aware-gates` replaces the simple per-level coefficient gate by a gate that uses encoded coefficient features, raw physical coefficient features, mean/std/min/max summaries at that U-Net scale, a level/stage code, and a learnable bounded gain. The raw features include coarse-cell optical thickness, absorption ratio, transport cross-section, albedo, gradients, and anisotropy. This is intended to let coarse levels and fine levels react differently to the coefficient field.",
            "- `--scale-gate-channel-mix` adds a bounded coefficient-conditioned channel-mixing update inside each scale-aware gate, so the gate can rotate/couple residual feature channels instead of only reweighting them diagonally. For fixed coefficients the residual path remains linear.",
            "- `RoutedUNet-*` replaces the fixed `B_h` inside the U-Net by a trainable soft router over classical candidates such as `transport`, `dsa`, `tsa`, and `tsa_moment`. The router is conditioned on coefficient summaries, optionally residual summaries, and learns bounded transport/synthetic relaxation factors. The metric loss still uses the fixed classical base so that a learned metric cannot be gamed.",
            "- `MomentUNet-K` uses the same defect2 structure but restricts the learned correction to angular Fourier modes `0,+/-1,...,+/-K` before prolonging back to ordinates.",
            "- `WeightedTSA-MK` is a tiny learned structural model initialized as closed TSA-MK; it learns bounded real weights for each absolute angular mode in the synthetic correction.",
            "- `CoeffWeightedTSA-MK` keeps the closed TSA-MK solve fixed, then uses a coefficient-conditioned nonlinear network to learn bounded physical-space weights for each absolute angular mode. It is nonlinear in coefficients but linear in residuals for fixed coefficients.",
            "- `CoeffMixedTSA-MK` keeps the closed TSA-MK solve fixed, then uses a coefficient-conditioned local linear mixing matrix on retained moments. It is initialized to the identity map and can preserve the scalar moment exactly.",
            "- `LearnedRP-TSA-MK` learns coefficient-conditioned restriction/prolongation maps around the fixed closed moment inverse. It is initialized to the identity and can preserve the scalar moment exactly.",
            "- `ResidualSubspaceTSA-MK` uses the current post-transport residual to choose a bounded global mixing matrix on retained moments before the fixed closed inverse. This is a residual-dependent angular-subspace ablation and is nonlinear in the residual.",
            "- By default `CoeffWeightedTSA-MK` freezes the scalar `m=0` synthetic weight to preserve the scalar diffusion moment; pass `--learn-scalar-moment` to ablate this constraint. `--moment-transport-relaxation` and `--moment-synthetic-relaxation` allow stable damped variants.",
            "- `--rollout-space` and `--stability-space` choose whether rollout and stability penalties measure contraction in the base metric space or in the physical residual norm.",
            "- `base-call equal` means equal counts of selected base-preconditioner applications inside one preconditioner call, excluding diagnostic metric evaluations, `A_h` applications, scalar reductions, and U-Net overhead. `UNet-defect2` is base-call equal to `dsa2`/`tsa2`; `dsa3`/`dsa4` and `tsa3`/`tsa4` are deeper classical work-budget baselines, not strict equal-cost matches.",
            "",
            "## Results",
            "",
        ]
    )
    lines.append(
        "| Size | Base | K | FGMRES | Case | Split | Method | one-step true | one-step base metric | stat final | FGMRES final | stat ttol | FGMRES ttol | loss0 | loss1 |"
    )
    lines.append("|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {size} | {base_kind} | {tsa_harmonics} | {fgmres_mode} | {case} | {split} | {method} | "
            "{one_step_true:.3e} | {one_step_base_metric:.3e} | {stationary_final:.3e} | {fgmres_final:.3e} | "
            "{stationary_time_to_tol} | {fgmres_time_to_tol} | {train_loss_initial} | {train_loss_final} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="NPBS-native matrix-free RTE U-Net preconditioner ablation.")
    parser.add_argument("--cases", nargs="+", default=["diffusive", "heterogeneous", "anisotropic", "mixed"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[24])
    parser.add_argument("--n-angles", type=int, default=12)
    parser.add_argument("--base-kinds", nargs="+", default=["transport_dsa"])
    parser.add_argument("--tsa-harmonics", type=int, default=None)
    parser.add_argument("--forms", nargs="+", default=["residual", "defect2"])
    parser.add_argument("--moment-harmonics", type=int, default=2)
    parser.add_argument("--moment-weight-scale", type=float, default=0.5)
    parser.add_argument("--moment-transport-relaxation", type=float, default=1.0)
    parser.add_argument("--moment-synthetic-relaxation", type=float, default=1.0)
    parser.add_argument("--learn-scalar-moment", action="store_true")
    parser.add_argument("--learned-base-router", action="store_true")
    parser.add_argument(
        "--learned-base-candidates",
        nargs="+",
        default=["transport", "dsa", "tsa", "tsa_moment"],
        help="Candidate classical bases for a learned soft router used inside residual/defect U-Net forms.",
    )
    parser.add_argument("--learned-base-width", type=int, default=32)
    parser.add_argument("--learned-base-blocks", type=int, default=2)
    parser.add_argument("--learned-base-temperature", type=float, default=1.0)
    parser.add_argument("--learned-base-route-scale", type=float, default=4.0)
    parser.add_argument("--learned-base-relaxation-scale", type=float, default=0.5)
    parser.add_argument(
        "--learned-base-use-residual",
        action="store_true",
        help="Let the router depend on residual summaries as well as coefficient summaries.",
    )
    parser.add_argument(
        "--learned-base-tsa-harmonics",
        type=int,
        default=None,
        help="Angular harmonics for the routed full-TSA candidate; defaults to --tsa-harmonics.",
    )
    parser.add_argument("--defect-relaxations", nargs="+", type=float, default=[1.0])
    parser.add_argument("--baseline-depths", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--baseline-relaxations", nargs="+", type=float, default=[1.0])
    parser.add_argument("--eta-scale", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--gate-scale", type=float, default=0.1)
    parser.add_argument(
        "--scale-aware-gates",
        action="store_true",
        help="Use per-level coefficient gates with scale summaries and level codes in U-Net forms.",
    )
    parser.add_argument(
        "--scale-gate-channel-mix",
        action="store_true",
        help="Let scale-aware gates apply bounded coefficient-conditioned channel mixing, not only diagonal reweighting.",
    )
    parser.add_argument(
        "--extended-coeff-features",
        action="store_true",
        help="Use optical thickness, absorption, and transport cross-section features in U-Net coefficient gates.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "step", "cosine", "onecycle"], default="cosine")
    parser.add_argument("--scheduler-step-size", type=int, default=0)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-eta-min", type=float, default=None)
    parser.add_argument("--scheduler-min-lr-ratio", type=float, default=1e-2)
    parser.add_argument("--scheduler-pct-start", type=float, default=0.1)
    parser.add_argument("--log-steps", type=int, default=0)
    parser.add_argument("--residual-mode", choices=["pool", "resample"], default="pool")
    parser.add_argument("--residuals-per-step", type=int, default=16)
    parser.add_argument("--rollout-samples-per-step", type=int, default=8)
    parser.add_argument("--n-rollout", type=int, default=6)
    parser.add_argument("--n-random", type=int, default=6)
    parser.add_argument("--random-modes", type=int, default=5)
    parser.add_argument("--aux-sources", type=int, default=0)
    parser.add_argument("--metric-weight", type=float, default=1.0)
    parser.add_argument("--physical-weight", type=float, default=0.5)
    parser.add_argument("--correction-weight", type=float, default=1e-6)
    parser.add_argument("--rollout-weight", type=float, default=0.0)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--rollout-decay", type=float, default=1.0)
    parser.add_argument("--rollout-space", choices=["metric", "physical"], default="metric")
    parser.add_argument("--stability-weight", type=float, default=0.0)
    parser.add_argument("--contraction-target", type=float, default=0.98)
    parser.add_argument("--stability-space", choices=["metric", "physical"], default="metric")
    parser.add_argument("--teacher-depth", type=int, default=0)
    parser.add_argument("--teacher-relaxation", type=float, default=1.0)
    parser.add_argument("--teacher-kind", default="")
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--fgmres-maxiter", type=int, default=25)
    parser.add_argument("--fgmres-mode", choices=["complex", "real-stacked"], default="complex")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--tol-grid", nargs="+", type=float, default=[])
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--output-init", choices=["zero", "small"], default="zero")
    parser.add_argument("--output-csv", default="rte/npbs_unet_ablation_results.csv")
    parser.add_argument("--trace-csv", default="")
    parser.add_argument("--output-md", default="rte/NPBS_UNET_ABLATION_REPORT.md")
    args = parser.parse_args()
    if args.scale_gate_channel_mix:
        args.scale_aware_gates = True

    torch.manual_seed(args.seed)
    dtype = torch.float32
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.benchmark = True
        if args.allow_tf32:
            torch.set_float32_matmul_precision("high")
    tol_grid = list(dict.fromkeys([float(args.tol)] + [float(t) for t in args.tol_grid]))
    rows = []
    trace_rows = []
    for size in args.sizes:
        for case in args.cases:
            rte, _ = make_case(case, size, args.n_angles, dtype, args.eta_scale, device=device)
            train_source = rte.source.clone()
            training_sources = make_training_sources(
                case,
                size,
                args.n_angles,
                dtype,
                device=device,
                num_aux_sources=args.aux_sources,
            )
            training_sources[0] = train_source.clone()
            heldout_source = make_source(
                size,
                args.n_angles,
                dtype,
                center=(0.68, 0.34),
                angular="beam" if case in ("anisotropic", "mixed") else "dipole",
            ).to(device=device)
            for base_kind in args.base_kinds:
                tsa_harmonics = effective_tsa_harmonics(base_kind, args.tsa_harmonics, args.moment_harmonics)
                if normalize_moment_base_kind(base_kind) and args.tsa_harmonics not in (None, args.moment_harmonics):
                    raise ValueError(
                        "For moment-TSA ablations, --tsa-harmonics must be omitted or match --moment-harmonics"
                    )
                if args.residual_mode == "pool":
                    residuals = collect_residuals_from_sources(
                        rte,
                        base_kind,
                        tsa_harmonics=tsa_harmonics,
                        sources=training_sources,
                        n_rollout=args.n_rollout,
                        n_random=args.n_random,
                        random_modes=args.random_modes,
                    )
                    residual_info = {"rollout_bank_size": len(residuals), "residuals_per_step": len(residuals)}
                else:
                    residuals, residual_info = make_residual_sampler(
                        rte,
                        base_kind,
                        tsa_harmonics=tsa_harmonics,
                        sources=training_sources,
                        n_rollout=args.n_rollout,
                        residuals_per_step=args.residuals_per_step,
                        rollout_samples_per_step=args.rollout_samples_per_step,
                        random_modes=args.random_modes,
                        seed=args.seed,
                    )
                base_label = base_kind.replace("transport_", "")
                baseline_methods = []
                for depth in args.baseline_depths:
                    for relaxation in args.baseline_relaxations:
                        if depth < 1 or relaxation <= 0.0:
                            continue

                        def base_apply(residual, d=depth, w=relaxation, kind=base_kind, tsa_h=tsa_harmonics):
                            return inner_base_apply(rte, residual, depth=d, kind=kind, tsa_harmonics=tsa_h, relaxation=w)

                        if abs(float(relaxation) - 1.0) < 1e-12:
                            label = f"{base_label}{int(depth)}"
                        else:
                            label = f"{base_label}{int(depth)}-w{float(relaxation):g}"
                        baseline_methods.append((label, base_apply, int(depth), "", "", "", "", "", ""))

                trained_models = []
                for form in args.forms:
                    relaxations = args.defect_relaxations if form == "defect2" else [1.0]
                    if form == "moment_defect2":
                        relaxations = args.defect_relaxations
                    if form in (
                        "weighted_moment_tsa",
                        "coeff_weighted_moment_tsa",
                        "coeff_mixed_moment_tsa",
                        "learned_rp_moment_tsa",
                        "residual_subspace_moment_tsa",
                    ):
                        relaxations = [1.0]
                    if form in MOMENT_FORMS and not normalize_moment_base_kind(base_kind):
                        raise ValueError(f"{form} requires --base-kinds transport_tsa_moment")
                    for relaxation in relaxations:
                        teacher_kind = args.teacher_kind if args.teacher_kind else base_kind

                        def teacher_fn(residual_batch, kind=teacher_kind):
                            return inner_base_apply(
                                rte,
                                residual_batch,
                                depth=args.teacher_depth,
                                kind=kind,
                                tsa_harmonics=tsa_harmonics,
                                relaxation=args.teacher_relaxation,
                            )

                        base_module = None
                        if args.learned_base_router:
                            if form not in ("residual", "defect2", "moment_defect2"):
                                raise ValueError(
                                    "--learned-base-router is currently supported for residual, defect2, "
                                    "and moment_defect2 U-Net forms"
                                )
                            routed_tsa_harmonics = (
                                args.learned_base_tsa_harmonics
                                if args.learned_base_tsa_harmonics is not None
                                else tsa_harmonics
                            )
                            base_module = LearnedBaseRouter(
                                rte,
                                candidates=args.learned_base_candidates,
                                initial_candidate=base_kind,
                                tsa_harmonics=routed_tsa_harmonics,
                                moment_harmonics=args.moment_harmonics,
                                width=args.learned_base_width,
                                blocks=args.learned_base_blocks,
                                temperature=args.learned_base_temperature,
                                route_scale=args.learned_base_route_scale,
                                relaxation_scale=args.learned_base_relaxation_scale,
                                use_residual_summary=args.learned_base_use_residual,
                            ).to(device)

                        if form == "moment_defect2":
                            model = MomentNPBSUNetPreconditioner(
                                rte,
                                base_kind=base_kind,
                                tsa_harmonics=tsa_harmonics,
                                moment_harmonics=args.moment_harmonics,
                                width=args.width,
                                levels=args.levels,
                                gate_scale=args.gate_scale,
                                defect_relaxation=relaxation,
                                output_init=args.output_init,
                                base_module=base_module,
                                scale_aware_gates=args.scale_aware_gates,
                                scale_gate_channel_mix=args.scale_gate_channel_mix,
                                extended_coeff_features=args.extended_coeff_features or args.scale_aware_gates,
                            ).to(device)
                        elif form == "weighted_moment_tsa":
                            if str(base_kind).lower() not in (
                                "transport_tsa_moment",
                                "tsa_moment",
                                "moment_tsa",
                                "transport_moment_tsa",
                            ):
                                raise ValueError("weighted_moment_tsa requires --base-kinds transport_tsa_moment")
                            model = ModeWeightedMomentTSAPreconditioner(
                                rte,
                                moment_harmonics=args.moment_harmonics,
                                weight_scale=args.moment_weight_scale,
                            ).to(device)
                        elif form == "coeff_weighted_moment_tsa":
                            if str(base_kind).lower() not in (
                                "transport_tsa_moment",
                                "tsa_moment",
                                "moment_tsa",
                                "transport_moment_tsa",
                            ):
                                raise ValueError("coeff_weighted_moment_tsa requires --base-kinds transport_tsa_moment")
                            model = CoeffWeightedMomentTSAPreconditioner(
                                rte,
                                moment_harmonics=args.moment_harmonics,
                                width=args.width,
                                blocks=max(1, args.levels),
                                weight_scale=args.moment_weight_scale,
                                transport_relaxation=args.moment_transport_relaxation,
                                synthetic_relaxation=args.moment_synthetic_relaxation,
                                freeze_scalar_weight=not args.learn_scalar_moment,
                            ).to(device)
                        elif form == "coeff_mixed_moment_tsa":
                            if str(base_kind).lower() not in (
                                "transport_tsa_moment",
                                "tsa_moment",
                                "moment_tsa",
                                "transport_moment_tsa",
                            ):
                                raise ValueError("coeff_mixed_moment_tsa requires --base-kinds transport_tsa_moment")
                            model = CoeffMixedMomentTSAPreconditioner(
                                rte,
                                moment_harmonics=args.moment_harmonics,
                                width=args.width,
                                blocks=max(1, args.levels),
                                weight_scale=args.moment_weight_scale,
                                transport_relaxation=args.moment_transport_relaxation,
                                synthetic_relaxation=args.moment_synthetic_relaxation,
                                freeze_scalar_weight=not args.learn_scalar_moment,
                            ).to(device)
                        elif form == "learned_rp_moment_tsa":
                            if str(base_kind).lower() not in (
                                "transport_tsa_moment",
                                "tsa_moment",
                                "moment_tsa",
                                "transport_moment_tsa",
                            ):
                                raise ValueError("learned_rp_moment_tsa requires --base-kinds transport_tsa_moment")
                            model = LearnedRPMomentTSAPreconditioner(
                                rte,
                                moment_harmonics=args.moment_harmonics,
                                width=args.width,
                                blocks=max(1, args.levels),
                                weight_scale=args.moment_weight_scale,
                                transport_relaxation=args.moment_transport_relaxation,
                                synthetic_relaxation=args.moment_synthetic_relaxation,
                                freeze_scalar_weight=not args.learn_scalar_moment,
                            ).to(device)
                        elif form == "residual_subspace_moment_tsa":
                            if str(base_kind).lower() not in (
                                "transport_tsa_moment",
                                "tsa_moment",
                                "moment_tsa",
                                "transport_moment_tsa",
                            ):
                                raise ValueError("residual_subspace_moment_tsa requires --base-kinds transport_tsa_moment")
                            model = ResidualSubspaceMomentTSAPreconditioner(
                                rte,
                                moment_harmonics=args.moment_harmonics,
                                width=args.width,
                                blocks=max(1, args.levels),
                                weight_scale=args.moment_weight_scale,
                                transport_relaxation=args.moment_transport_relaxation,
                                synthetic_relaxation=args.moment_synthetic_relaxation,
                                freeze_scalar_weight=not args.learn_scalar_moment,
                            ).to(device)
                        else:
                            model = NPBSUNetPreconditioner(
                                rte,
                                form=form,
                                base_kind=base_kind,
                                tsa_harmonics=tsa_harmonics,
                                width=args.width,
                                levels=args.levels,
                                gate_scale=args.gate_scale,
                                defect_relaxation=relaxation,
                                output_init=args.output_init,
                                base_module=base_module,
                                scale_aware_gates=args.scale_aware_gates,
                                scale_gate_channel_mix=args.scale_gate_channel_mix,
                                extended_coeff_features=args.extended_coeff_features or args.scale_aware_gates,
                            ).to(device)
                        history = train_model(
                            model,
                            residuals,
                            args.steps,
                            args.lr,
                            weight_decay=args.weight_decay,
                            scheduler_name=args.scheduler,
                            scheduler_step_size=args.scheduler_step_size,
                            scheduler_gamma=args.scheduler_gamma,
                            scheduler_eta_min=args.scheduler_eta_min,
                            scheduler_min_lr_ratio=args.scheduler_min_lr_ratio,
                            scheduler_pct_start=args.scheduler_pct_start,
                            log_steps=args.log_steps,
                            metric_weight=args.metric_weight,
                            physical_weight=args.physical_weight,
                            correction_weight=args.correction_weight,
                            rollout_weight=args.rollout_weight,
                            rollout_steps=args.rollout_steps,
                            rollout_decay=args.rollout_decay,
                            rollout_space=args.rollout_space,
                            stability_weight=args.stability_weight,
                            contraction_target=args.contraction_target,
                            stability_space=args.stability_space,
                            teacher_fn=teacher_fn if args.teacher_depth > 0 else None,
                            teacher_weight=args.teacher_weight,
                        )
                        model_label = (
                            f"MomentUNet-K{args.moment_harmonics}-w{relaxation:g}"
                            if form == "moment_defect2"
                            else f"WeightedTSA-M{args.moment_harmonics}"
                            if form == "weighted_moment_tsa"
                            else f"CoeffWeightedTSA-M{args.moment_harmonics}"
                            if form == "coeff_weighted_moment_tsa"
                            else f"CoeffMixedTSA-M{args.moment_harmonics}"
                            if form == "coeff_mixed_moment_tsa"
                            else f"LearnedRP-TSA-M{args.moment_harmonics}"
                            if form == "learned_rp_moment_tsa"
                            else f"ResidualSubspaceTSA-M{args.moment_harmonics}"
                            if form == "residual_subspace_moment_tsa"
                            else f"UNet-{form}-w{relaxation:g}"
                        )
                        if args.learned_base_router and form in ("residual", "defect2", "moment_defect2"):
                            model_label = "Routed" + model_label
                        if args.scale_aware_gates and form in ("residual", "defect2", "moment_defect2"):
                            model_label = "ScaleGated" + model_label
                        trained_models.append((model_label, model, history, residual_info))
                        print(
                            f"size={size} case={case} base={base_kind} form={form} mode={args.residual_mode} "
                            f"omega={relaxation:g} bank={residual_info['rollout_bank_size']} "
                            f"loss={history['loss'][0]:.3e}->{history['loss'][-1]:.3e}"
                        )

                methods = []
                for label, model, history, info in trained_models:
                    methods.append(
                        (
                            label,
                            model,
                            getattr(model, "base_calls_per_apply", 1 if model.form == "residual" else 2),
                            history['loss'][0],
                            history['loss'][-1],
                            history['lr'][0],
                            history['lr'][-1],
                            info['residuals_per_step'],
                            info['rollout_bank_size'],
                        )
                    )

                for split, source in (("train", train_source), ("heldout", heldout_source)):
                    residual = source.to(dtype=rte.complex_dtype) - rte.A_op(torch.zeros_like(rte.source))
                    for label, precond, calls, loss0, loss1, lr0, lr1, residuals_per_step, rollout_bank_size in baseline_methods + methods:
                        one_true, one_born_metric, one_base_metric = one_step_quality(
                            rte,
                            precond,
                            residual,
                            base_kind,
                            tsa_harmonics=tsa_harmonics,
                        )
                        si, sf, sc, st, s_hist, s_times, s_calls = solve_stationary(
                            rte,
                            source,
                            precond,
                            args.maxiter,
                            args.tol,
                        )
                        fi, ff, fc, ft, f_hist, f_times, f_calls = solve_fgmres(
                            rte,
                            source,
                            precond,
                            args.fgmres_maxiter,
                            args.tol,
                            real_stacked=args.fgmres_mode == "real-stacked",
                        )

                        s_base_calls = [c * calls for c in s_calls]
                        f_base_calls = [c * calls for c in f_calls]

                        s_tol = tolerance_snapshot(s_hist, s_times, s_base_calls, args.tol)
                        f_tol = tolerance_snapshot(f_hist, f_times, f_base_calls, args.tol)

                        tol_metrics = {
                            "stationary_hit_tol": s_tol["hit_tol"],
                            "stationary_time_to_tol": s_tol["time_to_tol"],
                            "stationary_base_calls_to_tol": s_tol["base_calls_to_tol"],
                            "fgmres_hit_tol": f_tol["hit_tol"],
                            "fgmres_time_to_tol": f_tol["time_to_tol"],
                            "fgmres_base_calls_to_tol": f_tol["base_calls_to_tol"],
                        }
                        for tol_val in tol_grid:
                            key = tol_key(tol_val)
                            s_snap = tolerance_snapshot(s_hist, s_times, s_base_calls, tol_val)
                            f_snap = tolerance_snapshot(f_hist, f_times, f_base_calls, tol_val)
                            tol_metrics[f"stationary_hit_{key}"] = s_snap["hit_tol"]
                            tol_metrics[f"stationary_time_{key}"] = s_snap["time_to_tol"]
                            tol_metrics[f"stationary_base_calls_{key}"] = s_snap["base_calls_to_tol"]
                            tol_metrics[f"fgmres_hit_{key}"] = f_snap["hit_tol"]
                            tol_metrics[f"fgmres_time_{key}"] = f_snap["time_to_tol"]
                            tol_metrics[f"fgmres_base_calls_{key}"] = f_snap["base_calls_to_tol"]
                        row = {
                            "size": size,
                            "case": case,
                            "split": split,
                            "base_kind": base_kind,
                            "tsa_harmonics": tsa_harmonics,
                            "requested_tsa_harmonics": args.tsa_harmonics,
                            "method": label,
                            "base_calls_per_apply": calls,
                            "one_step_true": one_true,
                            "one_step_born_metric": one_born_metric,
                            "one_step_base_metric": one_base_metric,
                            "stationary_iters": si,
                            "stationary_final": sf,
                            "stationary_factor": sc,
                            "stationary_time": st,
                            "fgmres_iters": fi,
                            "fgmres_final": ff,
                            "fgmres_factor": fc,
                            "fgmres_time": ft,
                            "train_loss_initial": loss0,
                            "train_loss_final": loss1,
                            "train_lr_initial": lr0,
                            "train_lr_final": lr1,
                            "residual_mode": args.residual_mode,
                            "residuals_per_step": residuals_per_step,
                            "rollout_bank_size": rollout_bank_size,
                            "n_rollout": args.n_rollout,
                            "n_random": args.n_random,
                            "random_modes": args.random_modes,
                            "aux_sources": args.aux_sources,
                            "scheduler": args.scheduler,
                            "scheduler_step_size": args.scheduler_step_size,
                            "scheduler_gamma": args.scheduler_gamma,
                            "scheduler_eta_min": args.scheduler_eta_min,
                            "scheduler_min_lr_ratio": args.scheduler_min_lr_ratio,
                            "scheduler_pct_start": args.scheduler_pct_start,
                            "gate_scale": args.gate_scale if getattr(precond, "form", "") in ("residual", "defect2", "moment_defect2") else "",
                            "scale_aware_gates": args.scale_aware_gates if getattr(precond, "form", "") in ("residual", "defect2", "moment_defect2") else "",
                            "scale_gate_channel_mix": args.scale_gate_channel_mix if getattr(precond, "form", "") in ("residual", "defect2", "moment_defect2") else "",
                            "extended_coeff_features": (args.extended_coeff_features or args.scale_aware_gates) if getattr(precond, "form", "") in ("residual", "defect2", "moment_defect2") else "",
                            "metric_weight": args.metric_weight,
                            "physical_weight": args.physical_weight,
                            "correction_weight": args.correction_weight,
                            "rollout_weight": args.rollout_weight,
                            "rollout_steps": args.rollout_steps,
                            "rollout_decay": args.rollout_decay,
                            "rollout_space": args.rollout_space,
                            "stability_weight": args.stability_weight,
                            "contraction_target": args.contraction_target,
                            "stability_space": args.stability_space,
                            "teacher_depth": args.teacher_depth,
                            "teacher_kind": (args.teacher_kind if args.teacher_kind else base_kind) if args.teacher_depth > 0 else "",
                            "teacher_relaxation": args.teacher_relaxation if args.teacher_depth > 0 else "",
                            "teacher_weight": args.teacher_weight,
                            "moment_harmonics": args.moment_harmonics if getattr(precond, "form", "") in ("moment_defect2", "weighted_moment_tsa", "coeff_weighted_moment_tsa", "coeff_mixed_moment_tsa", "learned_rp_moment_tsa", "residual_subspace_moment_tsa") else "",
                            "moment_weight_scale": args.moment_weight_scale if getattr(precond, "form", "") in ("weighted_moment_tsa", "coeff_weighted_moment_tsa", "coeff_mixed_moment_tsa", "learned_rp_moment_tsa", "residual_subspace_moment_tsa") else "",
                            "moment_transport_relaxation": getattr(precond, "transport_relaxation", ""),
                            "moment_synthetic_relaxation": args.moment_synthetic_relaxation if getattr(precond, "form", "") in ("coeff_weighted_moment_tsa", "coeff_mixed_moment_tsa", "learned_rp_moment_tsa", "residual_subspace_moment_tsa") else "",
                            "learn_scalar_moment": args.learn_scalar_moment if getattr(precond, "form", "") in ("coeff_weighted_moment_tsa", "coeff_mixed_moment_tsa", "learned_rp_moment_tsa", "residual_subspace_moment_tsa") else "",
                            "learned_base_router": bool(getattr(precond, "base_module", None) is not None),
                            "learned_base_candidates": getattr(getattr(precond, "base_module", None), "router_candidates", ""),
                            "learned_base_use_residual": getattr(getattr(precond, "base_module", None), "use_residual_summary", ""),
                            "learned_base_temperature": getattr(getattr(precond, "base_module", None), "temperature", ""),
                            "learned_base_route_scale": getattr(getattr(precond, "base_module", None), "route_scale", ""),
                            "learned_base_relaxation_scale": getattr(getattr(precond, "base_module", None), "relaxation_scale", ""),
                            "fgmres_mode": args.fgmres_mode,
                        }
                        row.update(tol_metrics)
                        rows.append(row)

                        for idx, relres in enumerate(s_hist):
                            trace_rows.append(
                                {
                                    "size": size,
                                    "case": case,
                                    "split": split,
                                    "base_kind": base_kind,
                                    "tsa_harmonics": tsa_harmonics,
                                    "method": label,
                                    "solver": "stationary",
                                    "iter": idx,
                                    "relres": relres,
                                    "elapsed": s_times[idx],
                                    "preconditioner_calls": s_calls[idx],
                                    "base_calls": s_base_calls[idx],
                                    "tol": args.tol,
                                    "hit_tol": relres <= args.tol,
                                }
                            )
                        for idx, relres in enumerate(f_hist):
                            trace_rows.append(
                                {
                                    "size": size,
                                    "case": case,
                                    "split": split,
                                    "base_kind": base_kind,
                                    "tsa_harmonics": tsa_harmonics,
                                    "method": label,
                                    "solver": "fgmres",
                                    "iter": idx,
                                    "relres": relres,
                                    "elapsed": f_times[idx],
                                    "preconditioner_calls": f_calls[idx],
                                    "base_calls": f_base_calls[idx],
                                    "tol": args.tol,
                                    "hit_tol": relres <= args.tol,
                                }
                            )
                        print(
                            f"  {split:7s} {label:18s} one={one_true:.2e} "
                            f"stat={si:3d}/{sf:.2e} fgmres={fi:3d}/{ff:.2e}"
                        )

    csv_path = ROOT / args.output_csv
    md_path = ROOT / args.output_md
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.trace_csv:
        trace_path = ROOT / args.trace_csv
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trace_rows)
        print(f"wrote {trace_path}")

    write_report(rows, md_path, vars(args))
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
