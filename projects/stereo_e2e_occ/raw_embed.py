"""
RAW Bayer 图像 Patch 嵌入

将 12-bit 单通道 Bayer RAW (1, H, W) 转换为特征图 (C, H/16, W/16)
- 可学习 2x2 卷积解码 RGGB 四通道
- 三级 stride-2 卷积 stem 逐步降采样
"""
import torch
import torch.nn as nn


class RAWPatchEmbed(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        # 可学习 RGGB 解包: (1, H, W) → (4, H/2, W/2)
        self.rggb_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

        # 三级 stem: 4→32→64→128→embed_dim, 总 stride=16
        self.stem = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x):
        """
        x: [B, N, 1, H, W]  N=2 (左右眼)
        returns: [B, N, C, Hf, Wf]  C=embed_dim, Hf=H/16, Wf=W/16
        """
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.rggb_conv(x)
        x = self.stem(x)
        _, D, Hf, Wf = x.shape
        return x.view(B, N, D, Hf, Wf)


class StereoPatchEmbed(nn.Module):
    """双目 RAW Patch 嵌入 + LayerNorm"""
    def __init__(self, config):
        super().__init__()
        self.patch_embed = RAWPatchEmbed(embed_dim=config.embed_dim)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, images):
        """
        images: [B, 2, 1, H, W]
        returns: [B, 2, C, Hf, Wf]
        """
        feats = self.patch_embed(images)  # [B, 2, C, Hf, Wf]
        # LayerNorm on channel dim
        feats = feats.permute(0, 1, 3, 4, 2)  # [B, 2, Hf, Wf, C]
        feats = self.norm(feats)
        feats = feats.permute(0, 1, 4, 2, 3)  # [B, 2, C, Hf, Wf]
        return feats
