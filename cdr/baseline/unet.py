import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def norm_act(in_ch, groups=8):
    return nn.Sequential(
        nn.GroupNorm(num_groups=min(groups, in_ch), num_channels=in_ch),
        nn.SiLU(inplace=True)
    )

class ResidualDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False)
        self.na1 = norm_act(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False)
        self.na2 = norm_act(out_ch)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.na1(x)
        x = self.conv2(x)
        x = self.na2(x)
        if self.proj is not None:
            identity = self.proj(identity)
        return x + identity


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, bias=False),
            ResidualDoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ResidualDoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base*2, base*4, base*8, base*16

        self.enc1 = ResidualDoubleConv(in_ch, c1)
        self.enc2 = Down(c1, c2)
        self.enc3 = Down(c2, c3)
        self.enc4 = Down(c3, c4)
        self.enc5 = Down(c4, c5)

        self.up4 = Up(c5 + c4, c4)
        self.up3 = Up(c4 + c3, c3)
        self.up2 = Up(c3 + c2, c2)
        self.up1 = Up(c2 + c1, c1)

        self.out_conv = nn.Conv2d(c1, out_ch, 1)

    @staticmethod
    def _pad_to_32(x):
        _, _, h, w = x.shape
        nh = math.ceil(h / 32) * 32
        nw = math.ceil(w / 32) * 32
        ph, pw = nh - h, nw - w
        top, bottom = ph // 2, ph - ph // 2
        left, right = pw // 2, pw - pw // 2
        return F.pad(x, (left, right, top, bottom)), (left, right, top, bottom)

    @staticmethod
    def _unpad(x, pad):
        left, right, top, bottom = pad
        if (left + right + top + bottom) == 0:
            return x
        return x[:, :, top: x.size(2)-bottom, left: x.size(3)-right]
    def get_grid(self, shape, device):
        batchsize, _, size_x, size_y = shape
        
        # Create a 1D tensor for x coordinates
        gridx_1d = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        # Reshape to (1, size_x, 1) and expand to (batchsize, size_x, size_y)
        # This is a memory-efficient operation that reuses the data from the original tensor.
        gridx = gridx_1d.view(1, size_x, 1).expand(batchsize, size_x, size_y)
        
        # Create a 1D tensor for y coordinates
        gridy_1d = torch.linspace(0, 1, size_y, dtype=torch.float, device=device)
        # Reshape to (1, 1, size_y) and expand to (batchsize, size_x, size_y)
        gridy = gridy_1d.view(1, 1, size_y).expand(batchsize, size_x, size_y)
        return torch.stack([gridx, gridy], dim=1)
    
    def forward(self, x):

        x_pad, pad = self._pad_to_32(x)
        
        grid = self.get_grid(x_pad.shape, x.device)
        x_pad = torch.cat((x_pad, grid), dim=1)

        e1 = self.enc1(x_pad)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d4 = self.up4(e5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.out_conv(d1)
        out = self._unpad(out, pad)
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_ch=3, out_ch=1).to(device)
    for size in [70, 71, 96, 123]:
        x = torch.randn(2, 3, size, size, device=device)
        y = model(x)
        print(f"{tuple(x.shape)} -> {tuple(y.shape)}")
        y.mean().backward()


'''
Version 2
'''
# """ Parts of the U-Net model """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             # nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             # nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2, bias=False),
#             # nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)


# class UNet(nn.Module):
#     def __init__(self, in_ch, out_ch, bilinear=False):
#         super(UNet, self).__init__()
#         self.in_channels = in_ch
#         self.out_channels = out_ch
#         self.bilinear = bilinear

#         self.inc = (DoubleConv(in_ch, 32))
#         self.down1 = (Down(32, 64))
#         self.down2 = (Down(64, 128))
#         self.down3 = (Down(128, 256))
#         factor = 2 if bilinear else 1
#         self.down4 = (Down(256, 512 // factor))
#         self.up1 = (Up(512, 256 // factor, bilinear))
#         self.up2 = (Up(256, 128 // factor, bilinear))
#         self.up3 = (Up(128, 64 // factor, bilinear))
#         self.up4 = (Up(64, 32, bilinear))
#         self.outc = (OutConv(32, out_ch))

#     def forward(self, x):
#         # grid = self.get_grid(x.shape, x.device)
#         # x = torch.cat((x, grid), dim=1)
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits
#     def get_grid(self, shape, device):
#         batchsize, _, size_x, size_y = shape
        
#         # Create a 1D tensor for x coordinates
#         gridx_1d = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
#         # Reshape to (1, size_x, 1) and expand to (batchsize, size_x, size_y)
#         # This is a memory-efficient operation that reuses the data from the original tensor.
#         gridx = gridx_1d.view(1, size_x, 1).expand(batchsize, size_x, size_y)
        
#         # Create a 1D tensor for y coordinates
#         gridy_1d = torch.linspace(0, 1, size_y, dtype=torch.float, device=device)
#         # Reshape to (1, 1, size_y) and expand to (batchsize, size_x, size_y)
#         gridy = gridy_1d.view(1, 1, size_y).expand(batchsize, size_x, size_y)
#         return torch.stack([gridx, gridy], dim=1)

# if __name__ == "__main__":
#     x = torch.rand(10,3,70,70)
#     model = UNet(3,1)

#     y = model(x)
#     print(y.shape)