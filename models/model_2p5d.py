import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    """2D Residual Block with InstanceNorm"""
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(c, affine=True)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(c, affine=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class Down2D(nn.Module):
    """2D Downsampling Block"""
    def __init__(self, c_in, c_out, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.norm = nn.InstanceNorm2d(c_out, affine=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Up2D(nn.Module):
    """2D Upsampling Block"""
    def __init__(self, c_in, c_out, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(c_out, affine=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=self.scale,
            mode="bilinear", align_corners=False
        )
        return self.act(self.norm(self.conv(x)))


# -------------------------
# AE_2p5D 主体
# -------------------------
class AE_2p5D(nn.Module):
    """
    2.5D Autoencoder for Slice Interpolation
    Input: [B, 2, H, W] (adjacent slices)
    Output: [B, 1, H, W] (interpolated middle slice)
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # -------- Encoder --------
        # 输入: 2通道 (z-1, z+1 两个切片)
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.SiLU(inplace=True),
            ResBlock2D(32),
        )

        self.down1 = nn.Sequential(
            Down2D(32, 64, stride=2),
            ResBlock2D(64)
        )
        self.down2 = nn.Sequential(
            Down2D(64, 128, stride=2),
            ResBlock2D(128)
        )
        self.down3 = nn.Sequential(
            Down2D(128, 256, stride=2),
            ResBlock2D(256)
        )
        self.down4 = nn.Sequential(
            Down2D(256, 512, stride=2),
            ResBlock2D(512)
        )

        # -------- Decoder --------
        self.up4 = nn.Sequential(
            Up2D(512, 256, scale=2),
            ResBlock2D(256)
        )
        self.up3 = nn.Sequential(
            Up2D(256, 128, scale=2),
            ResBlock2D(128)
        )
        self.up2 = nn.Sequential(
            Up2D(128, 64, scale=2),
            ResBlock2D(64)
        )
        self.up1 = nn.Sequential(
            Up2D(64, 32, scale=2),
            ResBlock2D(32)
        )

        # 输出: 1通道 (预测的中间切片)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.out_act = nn.Identity()

        # -------- Bottleneck (for latent representation) --------
        self.bottleneck = nn.Linear(512, latent_dim)

    # -------- 编码 --------
    def encode(self, x):
        """
        Args:
            x: [B, 2, H, W] input slices
        Returns:
            [B, 512, H', W'] encoded features
        """
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x

    # -------- 获取潜在表示（关键） --------
    def get_latent(self, x):
        """
        Args:
            x: [B, 2, H, W] input slices
        Returns:
            [B, latent_dim] latent vector
        """
        z = self.encode(x)           # [B, 512, H', W']
        z = z.mean(dim=(2, 3))       # GAP → [B, 512]
        z = self.bottleneck(z)       # → [B, latent_dim]
        return z

    # -------- 解码 --------
    def decode(self, z):
        """
        Args:
            z: [B, 512, H', W'] encoded features
        Returns:
            [B, 1, H, W] reconstructed slice
        """
        x = self.up4(z)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.out_act(self.out_conv(x))
        return x

    # -------- 前向（切片插值重建） --------
    def forward(self, x):
        """
        Args:
            x: [B, 2, H, W] input (adjacent slices z-1 and z+1)
        Returns:
            [B, 1, H, W] output (predicted middle slice z)
        """
        z = self.encode(x)
        out = self.decode(z)
        return out


# -------------------------
# AE_2p5D with Skip Connections (U-Net style)
# -------------------------
class UNet_2p5D(nn.Module):
    """
    2.5D U-Net for Slice Interpolation with Skip Connections
    Better reconstruction quality through skip connections
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # -------- Encoder --------
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.SiLU(inplace=True),
            ResBlock2D(32),
        )

        self.down1 = Down2D(32, 64, stride=2)
        self.res1 = ResBlock2D(64)

        self.down2 = Down2D(64, 128, stride=2)
        self.res2 = ResBlock2D(128)

        self.down3 = Down2D(128, 256, stride=2)
        self.res3 = ResBlock2D(256)

        self.down4 = Down2D(256, 512, stride=2)
        self.res4 = ResBlock2D(512)

        # -------- Decoder (with skip connections) --------
        self.up4 = Up2D(512, 256, scale=2)
        self.res_up4 = ResBlock2D(512)  # 256 + 256 skip
        self.conv_up4 = nn.Conv2d(512, 256, 1)

        self.up3 = Up2D(256, 128, scale=2)
        self.res_up3 = ResBlock2D(256)  # 128 + 128 skip
        self.conv_up3 = nn.Conv2d(256, 128, 1)

        self.up2 = Up2D(128, 64, scale=2)
        self.res_up2 = ResBlock2D(128)  # 64 + 64 skip
        self.conv_up2 = nn.Conv2d(128, 64, 1)

        self.up1 = Up2D(64, 32, scale=2)
        self.res_up1 = ResBlock2D(64)   # 32 + 32 skip
        self.conv_up1 = nn.Conv2d(64, 32, 1)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.out_act = nn.Identity()

        # -------- Bottleneck --------
        self.bottleneck = nn.Linear(512, latent_dim)

    def encode(self, x):
        """Returns all encoder features for skip connections"""
        e0 = self.stem(x)                # [B, 32, H, W]
        e1 = self.res1(self.down1(e0))   # [B, 64, H/2, W/2]
        e2 = self.res2(self.down2(e1))   # [B, 128, H/4, W/4]
        e3 = self.res3(self.down3(e2))   # [B, 256, H/8, W/8]
        e4 = self.res4(self.down4(e3))   # [B, 512, H/16, W/16]
        return e0, e1, e2, e3, e4

    def get_latent(self, x):
        """Get latent representation for feature extraction"""
        _, _, _, _, e4 = self.encode(x)
        z = e4.mean(dim=(2, 3))          # GAP → [B, 512]
        z = self.bottleneck(z)           # → [B, latent_dim]
        return z

    def forward(self, x):
        # Encode with skip connections
        e0, e1, e2, e3, e4 = self.encode(x)

        # Decode with skip connections
        d4 = self.up4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.conv_up4(self.res_up4(d4))

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.conv_up3(self.res_up3(d3))

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.conv_up2(self.res_up2(d2))

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.conv_up1(self.res_up1(d1))

        out = self.out_act(self.out_conv(d1))
        return out


def main():
    # Test AE_2p5D
    print("=" * 50)
    print("Testing AE_2p5D")
    print("=" * 50)

    input_tensor = torch.randn(2, 2, 128, 128)  # [B, 2, H, W]
    model = AE_2p5D(latent_dim=256)

    print(f"Input shape: {input_tensor.shape}")
    latent = model.get_latent(input_tensor)
    print(f"Latent shape: {latent.shape}")
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    # Test UNet_2p5D
    print("\n" + "=" * 50)
    print("Testing UNet_2p5D")
    print("=" * 50)

    model_unet = UNet_2p5D(latent_dim=256)

    print(f"Input shape: {input_tensor.shape}")
    latent_unet = model_unet.get_latent(input_tensor)
    print(f"Latent shape: {latent_unet.shape}")
    output_unet = model_unet(input_tensor)
    print(f"Output shape: {output_unet.shape}")

    # Count parameters
    num_params_unet = sum(p.numel() for p in model_unet.parameters())
    print(f"Number of parameters: {num_params_unet:,}")


if __name__ == "__main__":
    main()

