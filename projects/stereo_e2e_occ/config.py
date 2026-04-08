"""
双目 RAW12 端到端占用网络配置

与 8 目参考版本的核心差异:
  - 相机数: 8 → 2 (左右眼立体对)
  - 图像: 960x1280 → 1080x1280 (f-theta 鱼眼 157.2deg)
  - 输入: RAW 12-bit Bayer DNG (1 通道 uint16)
  - 体素: 400x400x32 (0.2m) → 72x60x32 (0.1m)
  - 感知范围: [-40,40]^2 → [-3.6,3.6]x[-3.0,3.0]x[-0.7,2.5]
  - 基线: 环视 → 80mm Y 轴立体对
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class StereoOccConfig:
    # --- 相机 ---
    num_cameras: int = 2
    image_size: Tuple[int, int] = (1080, 1280)  # (H, W)
    raw_channels: int = 1  # Bayer RAW 单通道

    # 鱼眼内参 (f-theta 等距投影, 来自 stereo_voxel_capture_dng.py)
    focal_length_px: float = 648.148  # fx = fy
    cx: float = 640.0
    cy: float = 540.0
    max_fov_deg: float = 157.2  # 对角 FOV
    baseline_m: float = 0.08  # 双目基线 80mm, Y 轴方向

    # --- 模型容量 (针对小体素网格缩减) ---
    embed_dim: int = 128  # 8 目用 256, 双目体素小可减半
    num_heads: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 2

    # --- 分辨率梯级 ---
    # 粗查询: 体素 72x60x32 的 ~1/6 下采样
    coarse_size: Tuple[int, int, int] = (12, 10, 8)  # 960 queries
    # 精细查询: 体素 72x60x32 的 ~1/2 下采样
    fine_size: Tuple[int, int, int] = (36, 30, 16)  # 17280 queries
    # 最终输出
    voxel_size: Tuple[int, int, int] = (72, 60, 32)

    # --- 类别 (与 stereo_voxel 语义类一致) ---
    num_classes: int = 18
    ignore_index: int = 255  # UNOBSERVED

    # --- 感知范围 (米, 与 voxel_grid.py 一致) ---
    voxel_range: Tuple[float, ...] = (-3.6, -3.0, -0.7, 3.6, 3.0, 2.5)
    voxel_resolution: float = 0.1  # 每体素 0.1m

    # 地面层索引 (z_ground_index=7, 物理高度 ~0.05m)
    z_ground_index: int = 7

    # --- 注意力 ---
    num_sample_points: int = 4  # 可变形注意力采样点
    dropout: float = 0.1

    # --- 功能开关 ---
    use_ray_encoding: bool = True
    use_self_attention: bool = True
    use_fine_self_attention: bool = False  # 17K queries 可以开, 但默认关
    use_stereo_fusion: bool = True  # 双目特征融合 (新增)

    # --- 时序融合 ---
    use_temporal: bool = False  # 默认关闭, 双目场景先做单帧
    use_ego_motion: bool = True
    temporal_frames: int = 2
    memory_dim: int = 128  # 与 embed_dim 一致

    # --- 多任务输出头 ---
    predict_flow: bool = True         # 输出体素速度 (vx, vy)
    predict_orientation: bool = True  # 输出 NPC 航向角
    predict_angular_vel: bool = True  # 输出 NPC 角速度 ωz

    @property
    def feat_size(self) -> Tuple[int, int]:
        """RAWPatchEmbed 16x 下采样后的特征图尺寸
        注意: Conv stride 向上取整, 1080/16=67.5 → 实际输出 68
        公式: ceil(ceil(ceil(ceil(H/2)/2)/2)/2) 对应 rggb(s2)+stem(s2,s2,s2)
        """
        import math
        h, w = self.image_size
        for _ in range(4):  # 4 次 stride=2
            h = math.ceil(h / 2)
            w = math.ceil(w / 2)
        return (h, w)  # (68, 80)

    @property
    def num_coarse_queries(self) -> int:
        return self.coarse_size[0] * self.coarse_size[1] * self.coarse_size[2]

    @property
    def num_fine_queries(self) -> int:
        return self.fine_size[0] * self.fine_size[1] * self.fine_size[2]

    @property
    def x_range(self) -> Tuple[float, float]:
        return (self.voxel_range[0], self.voxel_range[3])

    @property
    def y_range(self) -> Tuple[float, float]:
        return (self.voxel_range[1], self.voxel_range[4])

    @property
    def z_range(self) -> Tuple[float, float]:
        return (self.voxel_range[2], self.voxel_range[5])
