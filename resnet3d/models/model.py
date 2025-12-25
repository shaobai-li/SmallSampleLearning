import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(c, affine=True)
        self.conv2 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(c, affine=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class Down3D(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv = nn.Conv3d(
            c_in, c_out, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.norm = nn.InstanceNorm3d(c_out, affine=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Up3D(nn.Module):
    def __init__(self, c_in, c_out, scale):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv3d(c_in, c_out, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(c_out, affine=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=self.scale,
            mode="trilinear", align_corners=False
        )
        return self.act(self.norm(self.conv(x)))


# -------------------------
# AE_3D 主体
# -------------------------
class AE_3D(nn.Module):
    def __init__(self):
        super().__init__()

        # -------- Encoder --------
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.InstanceNorm3d(32, affine=True),
            nn.SiLU(inplace=True),
            ResBlock3D(32),
        )

        # 下采样策略：
        # H/W 更积极，D 相对保守（医学影像友好）
        self.down1 = nn.Sequential(
            Down3D(32, 64, stride=(1,2,2)),
            ResBlock3D(64)
        )
        self.down2 = nn.Sequential(
            Down3D(64, 128, stride=(2,2,2)),
            ResBlock3D(128)
        )
        self.down3 = nn.Sequential(
            Down3D(128, 256, stride=(2,2,2)),
            ResBlock3D(256)
        )
        self.down4 = nn.Sequential(
            Down3D(256, 512, stride=(2,2,2)),
            ResBlock3D(512)
        )

        # -------- Decoder --------
        self.up4 = nn.Sequential(
            Up3D(512, 256, scale=(2,2,2)),
            ResBlock3D(256)
        )
        self.up3 = nn.Sequential(
            Up3D(256, 128, scale=(2,2,2)),
            ResBlock3D(128)
        )
        self.up2 = nn.Sequential(
            Up3D(128, 64, scale=(2,2,2)),
            ResBlock3D(64)
        )
        self.up1 = nn.Sequential(
            Up3D(64, 32, scale=(1,2,2)),
            ResBlock3D(32)
        )

        self.out_conv = nn.Conv3d(32, 1, kernel_size=1)
        self.out_act = nn.Identity()

        # -------- Bottleneck (for latent representation) --------
        self.bottleneck = nn.Linear(512, 256)

    # -------- 编码 --------
    def encode(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x  # [B, 8C, D', H', W']

    # -------- 获取潜在表示（关键） --------
    def get_latent(self, x):
        z = self.encode(x)                 # [B,512,8,8,8]
        z = z.mean(dim=(2,3,4))             # GAP → [B,512]
        z = self.bottleneck(z)              # → [B,256]
        return z

    # -------- 前向（重建） --------
    def forward(self, x):
        z = self.encode(x)
        x = self.up4(z)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.out_act(self.out_conv(x))

        return x

def main():

    input_tensor = torch.randn(1, 1, 64, 128, 128)
    model = AE_3D()
    print("input shape:", input_tensor.shape)
    latent = model.get_latent(input_tensor)
    print("latent shape:", latent.shape)
    output = model(input_tensor)
    print("output shape:", output.shape)

if __name__ == "__main__":
    main()