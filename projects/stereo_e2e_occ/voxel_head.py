"""
体素输出头

三级层次上采样: fine(36x30x16) → mid(72x60x32) → output(72x60x32)
多任务输出:
  - semantic:    [B, 18, 72, 60, 32]  语义分类 logits
  - flow:        [B,  2, 72, 60, 32]  速度 (vx, vy) m/s
  - orientation: [B,  1, 72, 60, 32]  航向角 (弧度, [-π,π])
  - angular_vel: [B,  1, 72, 60, 32]  角速度 ωz (rad/s)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _reg_head(in_ch, out_ch):
    """轻量回归头：Conv3d + skip。"""
    return nn.ModuleDict({
        'conv': nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        'skip': nn.Conv3d(in_ch, out_ch, kernel_size=1),
    })


class VoxelHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        C = config.embed_dim  # 128
        self.voxel_size = config.voxel_size  # (72, 60, 32)
        self.config = config

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

        # 语义分类头
        self.cls_head = nn.Conv3d(mid_ch, config.num_classes, kernel_size=3, padding=1)
        self.cls_skip = nn.Conv3d(mid_ch, config.num_classes, kernel_size=1)

        # 回归头（多任务）
        if config.predict_flow:
            self.flow_head = _reg_head(mid_ch, 2)       # (vx, vy)
        if config.predict_orientation:
            self.orient_head = _reg_head(mid_ch, 1)     # 航向角
        if config.predict_angular_vel:
            self.angvel_head = _reg_head(mid_ch, 1)     # ωz

    def forward(self, x):
        """
        x: [B, fx, fy, fz, C]  (36, 30, 16, 128)
        returns dict:
          'semantic':    [B, num_classes, 72, 60, 32]
          'flow':        [B, 2, 72, 60, 32]  (若 predict_flow)
          'orientation': [B, 1, 72, 60, 32]  (若 predict_orientation)
          'angular_vel': [B, 1, 72, 60, 32]  (若 predict_angular_vel)
        """
        x = x.permute(0, 4, 1, 2, 3)  # [B, C, 36, 30, 16]
        x = self.reduce(x)             # [B, 32, 36, 30, 16]

        vx, vy, vz = self.voxel_size
        x = F.interpolate(x, size=(vx, vy, vz), mode='trilinear', align_corners=False)

        x = self.refine(x) + self.skip(x)  # [B, 32, 72, 60, 32]

        out = {}
        out['semantic'] = self.cls_head(x) + self.cls_skip(x)

        if self.config.predict_flow:
            out['flow'] = self.flow_head['conv'](x) + self.flow_head['skip'](x)
        if self.config.predict_orientation:
            out['orientation'] = self.orient_head['conv'](x) + self.orient_head['skip'](x)
        if self.config.predict_angular_vel:
            out['angular_vel'] = self.angvel_head['conv'](x) + self.angvel_head['skip'](x)

        return out
