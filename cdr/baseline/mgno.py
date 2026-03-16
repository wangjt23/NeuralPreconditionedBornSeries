import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Basic MG iteration modules ---------------- #

class MgIte(nn.Module):
    def __init__(self, A, S):
        super().__init__()
        self.A = A  # operator (u -> f)
        self.S = S  # smoother (f -> u)
    def forward(self, out):
        if isinstance(out, tuple):
            u, f = out
            # one smoothing iteration: u_{k+1} = u_k + S(f - A u_k)
            u = u + self.S(f - self.A(u))
        else:
            # initialization (no current u)
            f = out
            u = self.S(f)
        return (u, f)

class MgIte_init(nn.Module):
    def __init__(self, S):
        super().__init__()
        self.S = S
    def forward(self, f):
        u = self.S(f)
        return (u, f)

class Restrict(nn.Module):
    """
    Restricts (u,f) to coarser grid. Optionally recomputes residual f := R(f - A u).
    Pi: restrict u
    R : restrict residual / f
    A : fine-grid operator (optional for residual computation)
    """
    def __init__(self, Pi=None, R=None, A=None):
        super().__init__()
        self.Pi = Pi
        self.R = R
        self.A = A
    def forward(self, out):
        u, f = out
        if self.A is not None:
            f = self.R(f - self.A(u))
        else:
            f = self.R(f)
        u = self.Pi(u)
        return (u, f)

# ---------------- Helper: size match (center crop/pad) ---------------- #

def match_size(src, ref):
    """
    Center-crop or pad src to match ref spatial size.
    src, ref: (N,C,H,W)
    """
    sh, sw = src.shape[-2:]
    rh, rw = ref.shape[-2:]
    dh, dw = rh - sh, rw - sw
    if dh == 0 and dw == 0:
        return src
    # pad if target bigger
    pad_left = dw // 2 if dw > 0 else 0
    pad_right = dw - pad_left if dw > 0 else 0
    pad_top = dh // 2 if dh > 0 else 0
    pad_bottom = dh - pad_top if dh > 0 else 0
    if dw > 0 or dh > 0:
        src = F.pad(src, (pad_left, pad_right, pad_top, pad_bottom))
    # crop if target smaller
    sh2, sw2 = src.shape[-2:]
    extra_h = sh2 - rh
    extra_w = sw2 - rw
    if extra_h > 0:
        top = extra_h // 2
        src = src[:, :, top: top + rh, :]
    if extra_w > 0:
        left = extra_w // 2
        src = src[:, :, :, left: left + rw]
    return src

# ---------------- Original MgConv (generic) with GroupNorm fix ---------------- #

class MgConv(nn.Module):
    """
    A multigrid block defined by num_iteration:
    num_iteration: list of [num_pre_smooth, num_post_smooth] for each level (fine -> coarse).
    """
    def __init__(self, num_iteration, num_channel_u, num_channel_f,
                 padding_mode='zeros', bias=False, use_res=False, elementwise_affine=True):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        # Replace shape-dependent LayerNorm with GroupNorm(1,C), which is shape-agnostic
        self.norm_layer_list = nn.ModuleList([
            nn.GroupNorm(1, num_channel_u, affine=elementwise_affine)
            for _ in range(len(num_iteration)-1)
        ])

        # Upsampling operators (transpose conv)
        self.RTlayers = nn.ModuleList([
            nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=4, stride=2, padding=1, bias=False)
            for _ in range(len(num_iteration)-1)
        ])

        # Build downward (cumulative) sequences
        layers_cumulative = []
        for lvl, (n_pre, n_post) in enumerate(num_iteration):
            post_smooth_layers = []
            for i in range(n_pre):
                S = nn.Conv2d(num_channel_f if (lvl == 0 and i == 0) else num_channel_f,
                              num_channel_u, 3, 1, 1, bias=bias, padding_mode=padding_mode)
                if lvl == 0 and i == 0:
                    layers_cumulative.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(num_channel_u, num_channel_f, 3, 1, 1, bias=bias, padding_mode=padding_mode)
                    layers_cumulative.append(MgIte(A, S))
            # post-smoothing
            if n_post == 0:
                post_smooth_layers.append(nn.Identity())
            else:
                for _ in range(n_post):
                    S = nn.Conv2d(num_channel_f, num_channel_u, 3, 1, 1, bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d(num_channel_u, num_channel_f, 3, 1, 1, bias=bias, padding_mode=padding_mode)
                    post_smooth_layers.append(MgIte(A, S))

            setattr(self, f'layer{lvl}', nn.Sequential(*layers_cumulative))
            setattr(self, f'post_smooth_layer{lvl}', nn.Sequential(*post_smooth_layers))

            # Restrict (prepare for next level)
            if lvl < len(num_iteration) - 1:
                A = nn.Conv2d(num_channel_u, num_channel_f, 3, 1, 1, bias=bias, padding_mode=padding_mode)
                Pi = nn.Conv2d(num_channel_u, num_channel_u, 3, 2, 1, bias=False, padding_mode=padding_mode)
                R = nn.Conv2d(num_channel_f, num_channel_f, 3, 2, 1, bias=False, padding_mode=padding_mode)
                if use_res:
                    layers_cumulative = [Restrict(Pi, R, A)]
                else:
                    layers_cumulative = [Restrict(Pi=Pi, R=R)]

    def forward(self, f):
        out_per_level = [None] * len(self.num_iteration)
        out = f
        # Down / cumulative
        for lvl in range(len(self.num_iteration)):
            out = getattr(self, f'layer{lvl}')(out)
            out_per_level[lvl] = out

        # Up (coarse -> fine)
        for j in range(len(self.num_iteration)-2, -1, -1):
            u, f = out_per_level[j]
            up_u = self.RTlayers[j](out_per_level[j+1][0])
            up_u = match_size(up_u, u)
            u_post = self.norm_layer_list[j](u + up_u)
            out = (u_post, f)
            out_per_level[j] = getattr(self, f'post_smooth_layer{j}')(out)
        return out_per_level[0][0]

# ---------------- Modified MgConv with same fixes ---------------- #

class MgConv(nn.Module):
    """
    Specialized multigrid block (original had hard-coded LayerNorm shapes).
    We replace them with shape-agnostic GroupNorm(1,C) + generic upsample alignment.
    """
    def __init__(self, num_iteration, num_channel_u, num_channel_f,
                 padding_mode='reflect', bias=False, elementwise_affine=True, init=False):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        self.norm_layer_list = nn.ModuleList([
            nn.GroupNorm(1, num_channel_u, affine=True if j == 0 else elementwise_affine)
            for j in range(len(num_iteration)-1)
        ])

        # Use a simple consistent transposed conv (double size ideally)
        self.RTlayers = nn.ModuleList([
            nn.ConvTranspose2d(num_channel_u, num_channel_u, 4, 2, 1, bias=False)
            for _ in range(len(num_iteration)-1)
        ])

        layers_cumulative = []
        for lvl, (n_pre, n_post) in enumerate(num_iteration):
            post_smooth_layers = []
            for i in range(n_pre):
                S = nn.Conv2d(num_channel_f if (lvl == 0 and i == 0) else num_channel_f,
                              num_channel_u, 3, 1, 1, bias=bias, padding_mode='reflect')
                if lvl == 0 and i == 0:
                    layers_cumulative.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(num_channel_u, num_channel_f, 3, 1, 1, bias=bias, padding_mode='reflect')
                    layers_cumulative.append(MgIte(A, S))
            if n_post == 0:
                post_smooth_layers.append(nn.Identity())
            else:
                for _ in range(n_post):
                    S = nn.Conv2d(num_channel_f, num_channel_u, 3, 1, 1, bias=bias, padding_mode='reflect')
                    A = nn.Conv2d(num_channel_u, num_channel_f, 3, 1, 1, bias=bias, padding_mode='reflect')
                    post_smooth_layers.append(MgIte(A, S))

            setattr(self, f'layer{lvl}', nn.Sequential(*layers_cumulative))
            setattr(self, f'post_smooth_layer{lvl}', nn.Sequential(*post_smooth_layers))

            if lvl < len(num_iteration) - 1:
                # stride=2 with padding=1 保持尺度近似对半（ceil）
                Pi = nn.Conv2d(num_channel_u, num_channel_u, 3, 2, 1, bias=False, padding_mode='zeros')
                R  = nn.Conv2d(num_channel_f, num_channel_f, 3, 2, 1, bias=False, padding_mode='zeros')
                layers_cumulative = [Restrict(Pi=Pi, R=R)]

    def forward(self, f):
        out_per_level = [None] * len(self.num_iteration)
        out = f
        for lvl in range(len(self.num_iteration)):
            out = getattr(self, f'layer{lvl}')(out)
            out_per_level[lvl] = out
        for j in range(len(self.num_iteration)-2, -1, -1):
            u, ff = out_per_level[j]
            up_u = self.RTlayers[j](out_per_level[j+1][0])
            up_u = match_size(up_u, u)
            u_post = self.norm_layer_list[j](u + up_u)
            out = (u_post, ff)
            out_per_level[j] = getattr(self, f'post_smooth_layer{j}')(out)
        return out_per_level[0][0]

# ---------------- High level MgNO model ---------------- #

class MgNO(nn.Module):
    """
    num_iteration: e.g. [[1,0],[1,0],[1,0],[1,0],[2,0]]
    num_layer: how many stacked MgConv blocks
    """
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes,
                 num_iteration, normalizer=None, output_dim=1, activation='gelu', init=False, if_mlp=False):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.conv_list = nn.ModuleList()
        # 第一层：输入通道 = num_channel_f
        self.conv_list.append(MgConv(num_iteration, num_channel_u, num_channel_f, init=init))
        # 后续层：输入通道 = num_channel_u
        for _ in range(num_layer - 1):
            self.conv_list.append(MgConv(num_iteration, num_channel_u, num_channel_u, init=init))

        if if_mlp:
            # 用另一个多网格块当“MLP”
            self.mlp = MgConv(num_iteration, 1, num_channel_u, init=init)
        else:
            self.mlp = nn.Conv2d(num_channel_u, output_dim, kernel_size=1)

        self.normalizer = normalizer
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            raise NameError('invalid activation')

    def forward(self, u):
        # u: (B, num_channel_f, H, W)
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u))
        out = self.mlp(u)
        if self.normalizer:
            out = torch.squeeze(out, 1)
            out = self.normalizer.decode(out)
        return out

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MgNO(
        num_layer=4,
        num_channel_u=20,
        num_channel_f=1,
        num_classes=1,
        num_iteration=[[1,0], [1,0], [1,0], [1,0], [2,0]],
        activation='gelu',
        output_dim=1,
        if_mlp=False
    ).to(device)

    for size in [70, 101, 129]:
        x = torch.randn(2, 1, size, size, device=device)
        y = model(x)
        print(f"Input {x.shape} -> Output shape: {y.shape}")
        (y.mean()).backward()
    print("Success!")