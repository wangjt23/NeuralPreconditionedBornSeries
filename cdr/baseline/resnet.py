import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

# --- 辅助模块 (与您之前的AttU-Net一致) ---
def norm_act(in_ch, groups=8):
    if in_ch < groups:
        groups = 1
    return nn.Sequential(
        nn.GroupNorm(num_groups=groups, num_channels=in_ch),
        nn.SiLU(inplace=True)
    )

class ResidualDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False)
        self.na1 = norm_act(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False)
        self.na2 = norm_act(out_ch)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        x = self.na2(self.conv2(self.na1(self.conv1(x))))
        return x + identity

# --- 解码器上采样模块 ---
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # in_ch 已经包含了跳跃连接的通道数
        self.conv = ResidualDoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# --- 核心模型：ResNet-UNet ---
class ResNetUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_model='resnet34'):
        """
        初始化 ResNet-UNet 模型.
        :param in_ch: 输入通道数.
        :param out_ch: 输出通道数 (回归任务通常为1).
        :param base_model: 使用的ResNet骨干, 'resnet34' 或 'resnet50' 是常用选择.
        """
        super().__init__()
        
        # 1. 加载ResNet骨干网络 (不使用预训练权重，因为您的任务是科学计算)
        if base_model == 'resnet34':
            backbone = models.resnet34(weights=None)
            enc_channels = [64, 64, 128, 256, 512]
        elif base_model == 'resnet50':
            backbone = models.resnet50(weights=None)
            enc_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported base model: {base_model}")

        # 2. 修改ResNet的第一个卷积层以适应不同的输入通道数
        self.encoder_init = nn.Conv2d(in_ch, enc_channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        # 复制ResNet的其余部分作为编码器
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1 # -> enc_channels[1]
        self.layer2 = backbone.layer2 # -> enc_channels[2]
        self.layer3 = backbone.layer3 # -> enc_channels[3]
        self.layer4 = backbone.layer4 # -> enc_channels[4]

        # 3. 构建解码器
        # 解码器通道数，通常与编码器对称
        dec_channels = [enc_channels[3], enc_channels[2], enc_channels[1], enc_channels[0]]
        
        self.up4 = UpBlock(enc_channels[4] + enc_channels[3], dec_channels[0])
        self.up3 = UpBlock(dec_channels[0] + enc_channels[2], dec_channels[1])
        self.up2 = UpBlock(dec_channels[1] + enc_channels[1], dec_channels[2])
        self.up1 = UpBlock(dec_channels[2] + enc_channels[0], dec_channels[3])
        
        # 最后的上采样层，处理stride=2的初始卷积和maxpool
        self.final_up = UpBlock(dec_channels[3], dec_channels[3])
        self.out_conv = nn.Conv2d(dec_channels[3], out_ch, 1)

    @staticmethod
    def _pad(x, divisible_by=32):
        _, _, h, w = x.shape
        nh = math.ceil(h / divisible_by) * divisible_by
        nw = math.ceil(w / divisible_by) * divisible_by
        if nh == h and nw == w: return x, (0,0,0,0)
        ph, pw = nh - h, nw - w
        top, bottom = ph // 2, ph - ph // 2
        left, right = pw // 2, pw - pw // 2
        return F.pad(x, (left, right, top, bottom)), (left, right, top, bottom)

    @staticmethod
    def _unpad(x, pad):
        left, right, top, bottom = pad
        if sum(pad) == 0: return x
        return x[:, :, top: x.size(2) - bottom, left: x.size(3) - right]

    def forward(self, x):
        x_pad, pad = self._pad(x)

        # --- 编码器路径 ---
        e0 = self.encoder_init(x_pad)
        e1 = self.relu(self.bn1(e0))
        e1_pool = self.maxpool(e1)
        
        e2 = self.layer1(e1_pool)
        e3 = self.layer2(e2)
        e4 = self.layer3(e3)
        e5 = self.layer4(e4) # 这是瓶颈层的特征

        # --- 解码器路径 ---
        d4 = self.up4(e5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1) # 注意：这里的跳跃连接用的是maxpool之前的e1
        
        # 最后的上采样，回到原始尺寸的一半
        d0 = self.final_up(d1, None) # 没有跳跃连接
        
        out = self.out_conv(d0)
        out = self._unpad(out, pad)
        return out

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 您可以轻松切换 'resnet34' 或 'resnet50'
    model = ResNetUNet(in_ch=3, out_ch=1, base_model='resnet34').to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model. __class__.__name__} with resnet34 backbone")
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    for size in [70, 71, 96, 123]:
        x = torch.randn(2, 3, size, size, device=device)
        y = model(x)
        print(f"Input: {tuple(x.shape)} -> Output: {tuple(y.shape)}")
        assert x.shape[2:] == y.shape[2:], "Output shape must match input shape"
        # 测试反向传播
        y.mean().backward()
    
    print("\nModel test passed!")