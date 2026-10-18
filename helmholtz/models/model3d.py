from ._precision import preserve_default_dtype
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 3D MgNO preconditioned Born-Series solver.
#
# Direct 3D port of the MGNO chain in model2d.py
#   _MGNO_coeff_net -> _MgConv_DC_coeff -> MGNO_precond -> MGNO_BS
# with the CBS physics base (MPMG_BS_3D) mirroring ConvergentBornSeries3d
# (cbs.py): FFT over dim=[-3,-2,-1], p_dist2 = pz^2+py^2+px^2, 3D replicate
# padding and a 3D PBL absorbing layer.
#
# The preconditioner is nonlinear in the coefficient field but strictly linear
# in the residual rhs (coeff enters multiplicatively), so it can be used as an
# FGMRES preconditioner just like the 2D version.
# ---------------------------------------------------------------------------


class MPMG_BS_3D(nn.Module):
    """3D CBS physics base (isotropic cube), analogue of 2D MPMG_BS.

    Accepts scalar ``N`` and scalar ``boundary_width`` to match the train/
    inference calling convention (``N=data_cfg.size``); the padded domain is a
    cube of side ``N_pad = N + 2*boundary_width``.
    """

    @preserve_default_dtype
    def __init__(self,
                 freq=1,
                 sos_mean=3000.,
                 sos_max=3000.,
                 sos_min=3000.,
                 N=96,
                 dx=0.125,
                 boundary_width=8,
                 boundary_type='pbl',
                 u_channels_list=[5, 10, 15, 20, 25],
                 a_channels_list=[10, 15, 20, 25, 30],
                 mg_levels=[1, 2, 4, 8, 8],
                 fdtype=torch.float32,
                 device="cpu"):
        super().__init__()
        torch.set_default_dtype(fdtype)
        self.float_dtype = fdtype
        self.complex_dtype = torch.complex128 if fdtype == torch.float64 else torch.complex64
        self.device = device
        # PDE params
        self.freq = freq
        self.omega0 = 2 * torch.pi * freq
        self.dx = dx
        self.N = N
        self.sos_max = sos_max
        self.sos_min = sos_min
        self.sos_mean = sos_mean
        # grid
        self.boundary_width = boundary_width
        self.N_pad = self.N + self.boundary_width * 2
        # 3D spatial frequencies
        p = 2 * torch.pi * torch.fft.fftfreq(self.N_pad, d=self.dx)
        self.pz_range = p.view(-1, 1, 1)
        self.py_range = p.view(1, -1, 1)
        self.px_range = p.view(1, 1, -1)
        self.p_dist2 = (self.pz_range ** 2 + self.py_range ** 2 + self.px_range ** 2).to(self.device)
        # boundary setting
        self.boundary_type = boundary_type
        self.Bl = int((self.N_pad - self.N) / 2)
        self.Br = int((self.N_pad - self.N) / 2)
        # 3D boundary distance (0 inside the region of interest)
        c = torch.cat((torch.arange(self.Bl, 0, -1), torch.zeros(self.N), torch.arange(1, self.Br + 1)))
        z = c.view(-1, 1, 1)
        y = c.view(1, -1, 1)
        x = c.view(1, 1, -1)
        self.boundary_dist = torch.sqrt(z ** 2 + y ** 2 + x ** 2).to(self.device)
        # preconditioner is set by subclasses; keep attribute for parity
        self.precond = None

    def setup(self, sos, src=None):
        self.sos = sos
        self.setup_boundary()

        # CBS params
        self.k0 = ((1 / torch.mean(self.sos, dim=[-3, -2, -1])) * self.omega0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.epsilon = torch.amax(
            torch.abs(self.inv_sos2 * self.omega0 ** 2 - self.k0 ** 2), dim=[-3, -2, -1]
        ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # scatter potential
        self.V = self.inv_sos2 * self.omega0 ** 2 - self.k0 ** 2 - 1.0j * self.epsilon
        # g0_tilde
        self.g0_tilde = 1 / (self.p_dist2 - self.k0 ** 2 - 1.0j * self.epsilon)
        # rhs
        if src is not None:
            self.b = F.pad(src, (self.boundary_width,) * 6, mode='replicate')
        # preconditioner coefficients
        coeff1 = 2 * (1 / self.sos - 1 / self.sos_mean) / (1 / self.sos_min - 1 / self.sos_max)
        coeff2 = (self.V.imag / self.epsilon + 1) * 3
        coeff = torch.cat([coeff1, coeff2], dim=1).to(torch.float32)
        self.a_list, self.s_list = self.precond.setup(coeff)

    def G(self, u):
        return torch.fft.ifftn(self.g0_tilde * torch.fft.fftn(u, dim=[-3, -2, -1], norm='backward'),
                               dim=[-3, -2, -1], norm='backward')

    def G_inv(self, u):
        return torch.fft.ifftn(1 / self.g0_tilde * torch.fft.fftn(u, dim=[-3, -2, -1], norm='backward'),
                               dim=[-3, -2, -1], norm='backward')

    def A_op(self, u):
        return (self.G_inv(u) - self.V * u) * (self.dx ** 2)

    def cal_residual(self, u, rhs):
        return rhs - self.A_op(u)

    def setup_boundary(self):
        self.sos = F.pad(self.sos, (self.Bl, self.Br) * 3, mode='replicate')
        self.inv_sos2 = 1 / self.sos ** 2
        factor = torch.abs(torch.mean(self.inv_sos2, dim=[-3, -2, -1])).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        if self.boundary_type == "abc":
            absorb = self.set_abc_layer(self.inv_sos2.size(-1), self.Br, 0.5)
            self.inv_sos2 = self.inv_sos2 - factor * absorb * 1.j
        elif self.boundary_type == "pbl":
            k0 = (1 / torch.mean(self.sos, dim=[-3, -2, -1])) * self.omega0 * self.dx  # k0 in 1/pixels
            k0 = k0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            absorb = self.set_pbl_layer(r=self.boundary_dist, c=k0 / 2, k0=k0)
            self.inv_sos2 = self.inv_sos2 + factor * absorb.to(self.device)
        else:
            raise ValueError(f"Unknown boundary_type: {self.boundary_type}")

    def set_pbl_layer(self, r, c, k0):
        f_boundary_curve = 1 / k0 ** 2 * (c ** 5 * r ** 3 * (4.0 + (2.0j * k0 - c) * r)) / \
            (24 + 24 * c * r + 12 * c ** 2 * r ** 2 + 4 * c ** 3 * r ** 3 + c ** 4 * r ** 4)
        return f_boundary_curve

    def set_abc_layer(self, domain_size, width, value):
        abc_coord = torch.arange(width, device=self.device)
        layer_outer = value * (1 - abc_coord / width) ** 2
        layer_1d = torch.zeros((domain_size,), device=self.device)
        layer_1d[:width] = layer_outer
        layer_1d[-width:] = torch.flip(layer_outer, dims=[0])
        lz, ly, lx = torch.meshgrid(layer_1d, layer_1d, layer_1d, indexing='ij')
        return lz + ly + lx


"""
3D MGNO preconditioner and Born-Series solver

Design mirrors the 2D MGNO chain in model2d.py:
  - setup(coeff): coeff -> _MGNO_coeff_net_3d (nonlinear) -> a_list[l] per level
  - forward(rhs, a_list, _): rhs -> V-cycle block, each level gated by a_list[l]
No hardcoded spatial sizes.
"""


class _MGNO_coeff_net_3d(nn.Module):
    """Nonlinear coefficient encoder producing per-level 3D feature maps.

    coeff (B, 2, D, H, W) -> a_list: [(B, ch, D), (B, ch, D/2), ..., (B, ch, D/2^(L-1))]
    """
    def __init__(self, num_levels, num_channel):
        super().__init__()
        self.num_levels = num_levels
        self.lifting = nn.Conv3d(2, num_channel, kernel_size=3, padding=1, bias=True)
        self.nonlinear = nn.ModuleList([
            nn.Sequential(
                nn.GELU(),
                nn.Conv3d(num_channel, num_channel, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv3d(num_channel, num_channel, kernel_size=3, padding=1, bias=True),
            ) for _ in range(num_levels)
        ])
        self.restrict = nn.ModuleList([
            nn.Sequential(
                nn.GELU(),
                nn.Conv3d(num_channel, num_channel, kernel_size=3, stride=2, padding=1, bias=True),
            ) for _ in range(num_levels - 1)
        ])

    def forward(self, coeff):
        a = self.lifting(coeff)
        a_list = []
        for l in range(self.num_levels):
            a_list.append(self.nonlinear[l](a))
            if l < self.num_levels - 1:
                a = self.restrict[l](a)
        return a_list


class _MgConv_DC_coeff_3d(nn.Module):
    """3D V-cycle MgConv with per-level coefficient gating.

    At each level l the smoother correction is multiplied elementwise by a_list[l]:
        init step  : u  = a_list[l] * S(f)
        smooth step: u += a_list[l] * S(f - A(u))
    """
    def __init__(self, num_iteration, num_channel, padding_mode='zeros', bias=False):
        super().__init__()
        self.num_iteration = num_iteration
        L = len(num_iteration)
        self.RTlayers = nn.ModuleList([
            nn.ConvTranspose3d(num_channel, num_channel, kernel_size=4, stride=2, padding=1, bias=False)
            for _ in range(L - 1)
        ])
        for l, (n_pre, n_post) in enumerate(num_iteration):
            setattr(self, f'pre_S_{l}', nn.ModuleList([
                nn.Conv3d(num_channel, num_channel, 3, padding=1, bias=bias, padding_mode=padding_mode)
                for _ in range(n_pre)
            ]))
            setattr(self, f'pre_A_{l}', nn.ModuleList([
                nn.Identity() if (l == 0 and i == 0) else
                nn.Conv3d(num_channel, num_channel, 3, padding=1, bias=bias, padding_mode=padding_mode)
                for i in range(n_pre)
            ]))
            setattr(self, f'post_S_{l}', nn.ModuleList([
                nn.Conv3d(num_channel, num_channel, 3, padding=1, bias=bias, padding_mode=padding_mode)
                for _ in range(n_post)
            ]))
            setattr(self, f'post_A_{l}', nn.ModuleList([
                nn.Conv3d(num_channel, num_channel, 3, padding=1, bias=bias, padding_mode=padding_mode)
                for _ in range(n_post)
            ]))
            if l < L - 1:
                setattr(self, f'Pi_{l}', nn.Conv3d(num_channel, num_channel, 3, stride=2, padding=1,
                                                   bias=False, padding_mode=padding_mode))
                setattr(self, f'R_{l}', nn.Conv3d(num_channel, num_channel, 3, stride=2, padding=1,
                                                  bias=False, padding_mode=padding_mode))

    def forward(self, f, a_list):
        L = len(self.num_iteration)
        out_list = [None] * L
        u = None
        for l in range(L):
            pre_S = getattr(self, f'pre_S_{l}')
            pre_A = getattr(self, f'pre_A_{l}')
            for i, (S, A) in enumerate(zip(pre_S, pre_A)):
                if l == 0 and i == 0:
                    u = a_list[l] * S(f)
                else:
                    u = u + a_list[l] * S(f - A(u))
            out_list[l] = (u, f)
            if l < L - 1:
                f = getattr(self, f'R_{l}')(f)
                u = getattr(self, f'Pi_{l}')(u)
        for j in range(L - 2, -1, -1):
            u, f = out_list[j]
            correction = self.RTlayers[j](out_list[j + 1][0])
            # Match the fine grid when stride-two restriction rounded up.
            u = u + correction[..., :u.size(-3), :u.size(-2), :u.size(-1)]
            for S, A in zip(getattr(self, f'post_S_{j}'), getattr(self, f'post_A_{j}')):
                u = u + a_list[j] * S(f - A(u))
            out_list[j] = (u, f)
        return out_list[0][0]


class MGNO_precond_3d(nn.Module):
    """3D MgNO-based preconditioner with nonlinear coefficient conditioning.

    Mirrors the 2D MGNO_precond:
      setup(coeff) runs coeff through _MGNO_coeff_net_3d -> a_list per level.
      forward(rhs, a_list, _) lifts rhs, runs the V-cycle gated by a_list,
      then projects to 2 output channels.
    """
    @preserve_default_dtype
    def __init__(self, num_channel_u=32, num_iteration=None, activation='gelu'):
        super().__init__()
        torch.set_default_dtype(torch.float32)
        if num_iteration is None:
            num_iteration = [[2, 2], [2, 2], [2, 2], [2, 2]]
        num_levels = len(num_iteration)
        ch = num_channel_u

        self.coeff_net = _MGNO_coeff_net_3d(num_levels, ch)
        self.lifting_rhs = nn.Conv3d(2, ch, kernel_size=1, bias=False)
        # Single V-cycle: stacking linear V-cycles adds no expressiveness.
        self.vcycle = _MgConv_DC_coeff_3d(num_iteration, ch)
        self.proj_u = nn.Conv3d(ch, 2, kernel_size=1, bias=False)

    def setup(self, coeff):
        """coeff: (B, 2, N_pad, N_pad, N_pad) float32 -> (a_list, None)."""
        a_list = self.coeff_net(coeff)
        return a_list, None

    def forward(self, rhs, a_list, _s_list):
        x = self.lifting_rhs(rhs)
        x = self.vcycle(x, a_list)
        return self.proj_u(x)


class MGNO_BS_3D(MPMG_BS_3D):
    """3D Born-Series solver preconditioned by MgNO (MgConv_DC-based)."""

    def __init__(self, loss_type, num_channel_u=32,
                 num_iteration=None, activation='gelu', **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.precond = MGNO_precond_3d(
            num_channel_u=num_channel_u,
            num_iteration=num_iteration,
            activation=activation,
        ).to(self.device)

    def forward(self, u):
        if self.loss_type == 1:
            residual = self.b - self.A_op(u)
        else:
            residual = self.G(self.V * u + self.b / self.dx ** 2) - u
        residual = torch.cat([residual.real, residual.imag], dim=1).to(torch.float32)
        update = (self.precond(residual, self.a_list, self.s_list)
                  .permute(0, 2, 3, 4, 1).unsqueeze(1).contiguous())
        u = u + torch.view_as_complex(update).to(self.complex_dtype)
        return u


def trainable_params_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)
