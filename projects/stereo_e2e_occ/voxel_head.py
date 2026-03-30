"""
体素输出头

三级层次上采样: fine(36x30x16) → mid(72x60x32) → output(72x60x32)
与参考版区别:
  - fine: 80x80x16 → 36x30x16
  - 只需一次上采样到 72x60x32 (参考版需 80→200→400 两次)
  - 通道数减半以匹配 embed_dim=128
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        C = config.embed_dim  # 128
        self.voxel_size = config.voxel_size  # (72, 60, 32)

        # 降通道: C → C//2 → C//4
        self.reduce = nn.Sequential(
            nn.Conv3d(C, C // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(C // 2),
            nn.GELU(),
            nn.Conv3d(C // 2, C // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(C // 4),
            nn.GELU(),
        )

        # 上采样到目标尺寸后精炼
        mid_ch = C // 4  # 32
        self.refine = nn.Sequential(
            nn.Conv3d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_ch),
            nn.GELU(),
        )
        self.skip = nn.Conv3d(mid_ch, mid_ch, kernel_size=1)

        # 分类头
        self.cls_head = nn.Conv3d(mid_ch, config.num_classes, kernel_size=3, padding=1)
        self.cls_skip = nn.Conv3d(mid_ch, config.num_classes, kernel_size=1)

    def forward(self, x):
        """
        x: [B, fx, fy, fz, C]  (36, 30, 16, 128)
        returns: [B, num_classes, 72, 60, 32]  raw logits
        """
        x = x.permute(0, 4, 1, 2, 3)  # [B, C, 36, 30, 16]

        # 降通道
        x = self.reduce(x)  # [B, 32, 36, 30, 16]

        # 上采样到目标尺寸
        vx, vy, vz = self.voxel_size
        x = F.interpolate(x, size=(vx, vy, vz), mode='trilinear', align_corners=False)  # [B, 32, 72, 60, 32]

        # 精炼 + skip
        x = self.refine(x) + self.skip(x)  # [B, 32, 72, 60, 32]

        # 分类
        logits = self.cls_head(x) + self.cls_skip(x)  # [B, 18, 72, 60, 32]
        return logits
