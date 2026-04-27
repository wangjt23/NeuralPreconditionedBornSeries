"""
MGCFNN network building blocks, adapted from
  https://gitee.com/xiehuohuo77/mgcfnn.git/model.py

Changes from the original:
  - Removed the `from configclass import global_is_solve` import.
  - Set global_is_solve = False so that Fourier weight matrices are
    always recomputed from the learnable parameters on every forward
    pass (correct behaviour during training).
  - Removed visualization helpers and __main__ block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Controls whether Fourier2d_cmplx caches its weight matrix after
# the first setup call.  False = always recompute (correct for training).
global_is_solve = False


###############################################################################
# Level-0 components
###############################################################################

class ResizeChannel2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.conv(x)


class DownSample2d(nn.Module):
    """Halve spatial resolution (stride-2 conv)."""
    def __init__(self, channels, out_channels=None, kernel_size=3, bias=False):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size, 2, bias=bias)

    def forward(self, x):
        return self.conv(x)


class UpSample2d(nn.Module):
    """Double spatial resolution (transposed conv)."""
    def __init__(self, channels, out_channels=None, kernel_size=3, bias=False):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self.conv = nn.ConvTranspose2d(out_channels, channels, kernel_size, 2, bias=bias)

    def forward(self, x, size=None):
        x = self.conv(x)
        if size is not None:
            x = F.interpolate(x, size=size)
        return x


class SameSizeConv2d(nn.Module):
    """Spatial-size-preserving 2-D conv with optional nonlinearity."""
    def __init__(self, channels, kernel_size, padding_mode, active='none'):
        super().__init__()
        stride = 1
        padding = kernel_size // 2
        bias = (active != 'none')
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding,
                              padding_mode=padding_mode, bias=bias)
        self.active = active

    def forward(self, x):
        x = self.conv(x)
        if self.active == 'relu':
            x = torch.relu(x)
        elif self.active == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.active == 'tanh':
            x = torch.tanh(x)
        elif self.active == 'gelu':
            x = F.gelu(x)
        return x


class Fourier2d_cmplx(nn.Module):
    """
    Full-resolution complex-valued Fourier layer.
    Learnable weights are stored at (2*wave_number)^2 resolution and
    bilinearly interpolated to the actual input size at forward time.
    """
    def __init__(self, channels, wave_number):
        super().__init__()
        self.channels = channels
        self.wave_number = wave_number
        scale = 1.0 / (channels * channels)
        self.weights = nn.Parameter(
            scale * torch.randn(channels, channels, 2 * wave_number, 2 * wave_number, 2)
        )
        self.setup_w = 0
        # Stored as a parameter so it is part of the module state but not trained.
        self.x_size = nn.Parameter(torch.zeros(1))

    def complex_mul2d(self, input, weights):
        return torch.einsum("bixy,xyio->boxy", input, weights)

    def setup(self, x_size, xx_size):
        weights = (self.weights
                   .permute(0, 1, 4, 2, 3)
                   .reshape(self.channels, self.channels * 2,
                            2 * self.wave_number, 2 * self.wave_number))
        weights = F.interpolate(weights, size=(x_size, x_size), mode='bilinear',
                                align_corners=False)
        weights = (weights
                   .reshape(self.channels, self.channels, 2, x_size, x_size)
                   .permute(0, 1, 3, 4, 2))
        weights = torch.view_as_complex(weights.contiguous())

        if xx_size > x_size:
            w_space = torch.fft.ifft2(weights)
            w_pad = torch.zeros(self.channels, self.channels, xx_size, xx_size, 2,
                                device=self.weights.device,
                                dtype=self.weights.dtype)
            w_pad = torch.view_as_complex(w_pad)
            h = x_size // 2
            r = x_size - h
            w_pad[..., :h, :h]  = w_space[..., :h, :h]
            w_pad[..., -r:, :h] = w_space[..., -r:, :h]
            w_pad[..., :h, -r:] = w_space[..., :h, -r:]
            w_pad[..., -r:, -r:] = w_space[..., -r:, -r:]
            weights = torch.fft.fft2(w_pad)

        self.w = weights.permute(2, 3, 0, 1).contiguous()
        if global_is_solve:
            self.setup_w = 1

    def forward(self, x):
        if self.x_size.item() == 0:
            self.x_size.data.fill_(x.shape[-1])
        x_size = int(self.x_size.item())
        if self.setup_w == 0:
            self.setup(x_size, x.shape[-1])

        x_ft = torch.fft.fft2(x)
        out_ft = self.complex_mul2d(x_ft, self.w)
        return torch.fft.ifft2(out_ft)


class Fourier2d_cmplx_padded(Fourier2d_cmplx):
    """
    Fourier layer that operates on real-valued tensors.
    Splits real/imag channels, zero-pads in space for stability,
    applies the complex Fourier layer, then concatenates real/imag back.
    """
    def __init__(self, channels, wave_number, padding_factor=2):
        super().__init__(channels // 2, int(wave_number * padding_factor))
        self.padding_factor = padding_factor

    def forward(self, x):
        x_size = x.shape[-1]
        c = x.shape[1]
        # Convert real tensor (c channels) to complex (c/2 channels)
        x = x[:, :c // 2] + 1j * x[:, c // 2:]
        # Pad
        xx_size = int(self.padding_factor * x_size)
        x = F.pad(x, (0, xx_size - x_size, 0, xx_size - x_size))
        x = super().forward(x)
        # Crop and convert back to real
        x = x[:, :, :x_size, :x_size]
        return torch.cat([x.real, x.imag], dim=1)


def net_type_module(net_type):
    if net_type in ('conv2d', 'nonconv2d'):
        return SameSizeConv2d
    elif net_type == 'fourier2d_cmplx':
        return Fourier2d_cmplx_padded
    raise ValueError(f'Unknown net_type: {net_type}')


###############################################################################
# Level-1 architecture blocks
###############################################################################

class ResidualBlock(nn.Module):
    """
    Stacked same-size-conv residual block.
    Optionally multiplies input by a coefficient map before each sweep.
    """
    def __init__(self, net_type, net_param_dict, sweeps,
                 apply_coeff='none', short_cut_net=True, share_type='none'):
        super().__init__()
        net_module = net_type_module(net_type)
        if share_type == 'none':
            self.nets = nn.ModuleList([net_module(**net_param_dict) for _ in range(sweeps)])
        elif share_type == 'n':
            self.net_n = net_module(**net_param_dict)
        elif share_type == 'n1':
            self.net_n = net_module(**net_param_dict)
            self.net_1 = net_module(**net_param_dict)
        self.apply_coeff = apply_coeff
        channels = net_param_dict['channels']
        if short_cut_net:
            self.short_cuts = nn.ModuleList(
                [nn.Conv2d(channels, channels, 1, bias=False) for _ in range(sweeps)]
            )
        self.sweeps = sweeps
        self.short_cut_net = short_cut_net
        self.share_type = share_type

    def forward(self, x, coeff=None):
        share_type = self.share_type
        if share_type == 'none':
            nets = self.nets
        elif share_type == 'n':
            nets = [self.net_n] * self.sweeps
        elif share_type == 'n1':
            nets = [self.net_n] * (self.sweeps - 1) + [self.net_1]

        for i in range(self.sweeps):
            if self.apply_coeff == 'pre-dot':
                out = nets[i](torch.mul(x, coeff))
            else:
                out = nets[i](x)
            if self.short_cut_net:
                x = self.short_cuts[i](x) + out
            else:
                x = x + out
        return x


class DownMLCycle(nn.Module):
    def __init__(self, resblk_param, restrict_param, levels):
        super().__init__()
        self.levels = levels
        self.lvl_nets  = nn.ModuleList([ResidualBlock(**resblk_param) for _ in range(levels)])
        self.restricts = nn.ModuleList([DownSample2d(**restrict_param) for _ in range(levels - 1)])

    def forward(self, x, coeffs=None):
        if coeffs is None:
            coeffs = [None] * self.levels
        xs = []
        for lvl in range(self.levels):
            x = self.lvl_nets[lvl](x, coeffs[lvl])
            xs.append(x)
            if lvl < self.levels - 1:
                x = self.restricts[lvl](x)
        return xs


class UpMLCycle(nn.Module):
    def __init__(self, resblk_param, interp_param, levels):
        super().__init__()
        self.levels = levels
        self.lvl_nets = nn.ModuleList([ResidualBlock(**resblk_param) for _ in range(levels)])
        self.interps  = nn.ModuleList([UpSample2d(**interp_param) for _ in range(levels - 1)])

    def forward(self, down_xs, coeffs=None):
        if coeffs is None:
            coeffs = [None] * self.levels
        x = down_xs[self.levels - 1]
        for lvl in range(self.levels - 1, -1, -1):
            x = self.lvl_nets[lvl](x, coeffs[lvl])
            if lvl > 0:
                x = self.interps[lvl - 1](x, size=down_xs[lvl - 1].shape[-1])
                x = x + down_xs[lvl - 1]
        return x


class DiffDownCycle(DownMLCycle):
    """Like DownMLCycle but each level can have its own block parameters."""
    def __init__(self, resblk_params, restrict_params, levels):
        nn.Module.__init__(self)
        self.levels = levels
        self.lvl_nets  = nn.ModuleList([ResidualBlock(**resblk_params[i]) for i in range(levels)])
        self.restricts = nn.ModuleList([DownSample2d(**restrict_params[i]) for i in range(levels - 1)])


class DiffUpCycle(UpMLCycle):
    """Like UpMLCycle but each level can have its own block parameters."""
    def __init__(self, resblk_params, interp_params, levels):
        nn.Module.__init__(self)
        self.levels = levels
        self.lvl_nets = nn.ModuleList([ResidualBlock(**resblk_params[i]) for i in range(levels)])
        self.interps  = nn.ModuleList([UpSample2d(**interp_params[i]) for i in range(levels - 1)])


class MLCycle(nn.Module):
    def __init__(self, resblk_param, transfer_param, levels, ml_type='same'):
        super().__init__()
        if ml_type == 'same':
            self.down_net = DownMLCycle(resblk_param, transfer_param, levels)
            self.up_net   = UpMLCycle(resblk_param, transfer_param, levels)
        elif ml_type == 'diff':
            self.down_net = DiffDownCycle(resblk_param, transfer_param, levels)
            self.up_net   = DiffUpCycle(resblk_param, transfer_param, levels)

    def forward(self, x, coeffs=None):
        down_xs = self.down_net(x, coeffs)
        return self.up_net(down_xs, coeffs)


###############################################################################
# Factory helpers
###############################################################################

def arch_name_module(arch_name):
    mapping = {
        'coeff_conv_net': ResidualBlock,
        'coeff_down_net': DownMLCycle,
        'coeff_ml_net':   MLCycle,
        'coeff_diff_down_net': DiffDownCycle,
        'fnn':   ResidualBlock,
        'mgcnn': MLCycle,
    }
    if arch_name in mapping:
        return mapping[arch_name]
    raise ValueError(f'Unknown arch_name: {arch_name}')


###############################################################################
# Level-2 network flows
###############################################################################

class MyFlow(nn.Module):
    def __init__(self, coeff_net_param, solve_net_param):
        super().__init__()
        self.coeff_net = arch_name_module(coeff_net_param['arch_name'])(**coeff_net_param['param'])
        self.solve_net = arch_name_module(solve_net_param['arch_name'])(**solve_net_param['param'])
        self.coeff_info = None

    def setup(self, coeff):
        self.coeff_info = self.coeff_net(coeff)

    def forward(self, x, coeff=None):
        if coeff is not None:
            self.coeff_info = self.coeff_net(coeff)
        return self.solve_net(x, self.coeff_info)


class MyFlow2(nn.Module):
    def __init__(self, coeff_net1_param, coeff_net2_param, solve_net_param):
        super().__init__()
        self.coeff_net1 = arch_name_module(coeff_net1_param['arch_name'])(**coeff_net1_param['param'])
        self.coeff_net2 = arch_name_module(coeff_net2_param['arch_name'])(**coeff_net2_param['param'])
        self.solve_net  = arch_name_module(solve_net_param['arch_name'])(**solve_net_param['param'])
        self.coeff_info = None

    def setup(self, coeff):
        self.coeff_info = self.coeff_net2(self.coeff_net1(coeff))

    def forward(self, x, coeff=None):
        if coeff is not None:
            self.setup(coeff)
        return self.solve_net(x, self.coeff_info)


def flow_name_module(flow_name):
    if flow_name == 'myflow':
        return MyFlow
    elif flow_name == 'myflow2':
        return MyFlow2
    raise ValueError(f'Unknown flow_name: {flow_name}')


class MyFlows(nn.Module):
    def __init__(self, flow_params):
        super().__init__()
        self.flows = nn.ModuleList(
            [flow_name_module(fp['flow_name'])(**fp['param']) for fp in flow_params]
        )

    def setup(self, coeff):
        for flow in self.flows:
            flow.setup(coeff)

    def forward(self, x, coeff=None):
        for flow in self.flows:
            x = flow(x, coeff)
        return x


###############################################################################
# Level-3 top-level model
###############################################################################

class MyModel(nn.Module):
    """
    Full MGCFNN model.

    Parameters
    ----------
    in_channels : int
        Channels of the input tensor (e.g. 2 for real+imag).
    out_channels : int
        Channels of the output tensor.
    coeff_in_channels : int
        Channels of the coefficient tensor (e.g. 2).
    flow_param : dict
        Architecture configuration dict with keys
        ``'channels'`` (int) and ``'param'`` (list of flow dicts).
    """
    def __init__(self, in_channels, out_channels, coeff_in_channels, flow_param):
        super().__init__()
        channels = flow_param['channels']
        self.innet       = ResizeChannel2d(in_channels, channels)
        self.outnet      = ResizeChannel2d(channels, out_channels)
        self.coeff_innet = ResizeChannel2d(coeff_in_channels, channels, bias=True)
        self.flow_net    = MyFlows(flow_param['param'])

    def setup(self, coeff):
        coeff = self.coeff_innet(coeff)
        self.flow_net.setup(coeff)

    def forward(self, x, coeff=None):
        if coeff is not None:
            self.setup(coeff)
        x = self.innet(x)
        x = self.flow_net(x)
        return self.outnet(x)
