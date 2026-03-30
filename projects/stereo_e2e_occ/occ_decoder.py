"""
占用解码器

粗 → 精两阶段解码:
  粗: 12x10x8 (960 queries) + 自注意力 + 可变形交叉注意力
  精: 36x30x16 (17280 queries) + 可变形交叉注意力 (无自注意力)
  + 空间一致性 3D 卷积

与参考版区别:
  - 体素尺寸: 25x25x8 / 80x80x16 → 12x10x8 / 36x30x16
  - config 传入可变形注意力以正确计算投影
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from .position_encoding import SineCosinePositionEncoding3D
    from .deformable_attention import DeformableDecoderLayer
except (ImportError, ValueError):
    from position_encoding import SineCosinePositionEncoding3D
    from deformable_attention import DeformableDecoderLayer


class OccupancyDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 粗查询: 可学习初始化
        self.coarse_query = nn.Parameter(
            torch.randn(1, config.num_coarse_queries, config.embed_dim) * 0.02
        )
        pos_3d_gen = SineCosinePositionEncoding3D(config.embed_dim)
        cx, cy, cz = config.coarse_size
        fx, fy, fz = config.fine_size
        self.register_buffer('coarse_pos', pos_3d_gen(cx, cy, cz, 'cpu'))
        self.register_buffer('fine_pos', pos_3d_gen(fx, fy, fz, 'cpu'))

        # 粗解码层
        self.coarse_layers = nn.ModuleList([
            DeformableDecoderLayer(
                config.embed_dim, config.num_heads, config.num_cameras,
                config.num_sample_points, dropout=config.dropout,
                use_self_attn=config.use_self_attention, config=config,
            )
            for _ in range(config.decoder_layers)
        ])

        # 粗→精投影
        self.coarse_to_fine = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
        )

        # 精解码层
        self.fine_layers = nn.ModuleList([
            DeformableDecoderLayer(
                config.embed_dim, config.num_heads, config.num_cameras,
                config.num_sample_points, dropout=config.dropout,
                use_self_attn=config.use_fine_self_attention, config=config,
            )
            for _ in range(config.decoder_layers)
        ])

        # 参考点 buffer
        self.register_buffer('coarse_ref', self._create_reference_points(config.coarse_size))
        self.register_buffer('fine_ref', self._create_reference_points(config.fine_size))

        # 时序融合 (可选)
        if config.use_temporal:
            try:
                from .temporal_fusion import TemporalFusionModule
            except (ImportError, ValueError):
                from temporal_fusion import TemporalFusionModule
            self.temporal_fusion = TemporalFusionModule(
                dim=config.embed_dim, num_heads=config.num_heads,
                dropout=config.dropout, use_checkpoint=True, config=config,
            )

        # 精阶段空间一致性��积
        self.fine_spatial_conv = nn.Sequential(
            nn.Conv3d(config.embed_dim, config.embed_dim, kernel_size=3, padding=1, groups=config.embed_dim),
            nn.BatchNorm3d(config.embed_dim),
            nn.GELU(),
        )

        self.checkpoint_coarse = False
        self.checkpoint_fine = True

    def _create_reference_points(self, size):
        x = torch.linspace(0, 1, size[0])
        y = torch.linspace(0, 1, size[1])
        z = torch.linspace(0, 1, size[2])
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        ref = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return ref.view(-1, 3)

    def forward(self, image_feats, intrinsics=None, extrinsics=None, memory=None, ego_motion=None):
        B = image_feats.shape[0]
        device = image_feats.device
        cx, cy, cz = self.config.coarse_size

        # 预计算外参逆矩阵 (所有 decoder 层共享, 避免重复求逆)
        # 外参是刚体变换, 用解析公式: R_inv = R^T, t_inv = -R^T @ t
        # 比 torch.inverse() 更快且 FP16 下数值更稳定
        if extrinsics is not None:
            R = extrinsics[..., :3, :3]            # [B, N, 3, 3]
            t = extrinsics[..., :3, 3:]            # [B, N, 3, 1]
            R_inv = R.transpose(-1, -2)            # [B, N, 3, 3]
            t_inv = -torch.matmul(R_inv, t)        # [B, N, 3, 1]
            inv_extrinsics = torch.zeros_like(extrinsics)
            inv_extrinsics[..., :3, :3] = R_inv
            inv_extrinsics[..., :3, 3:] = t_inv
            inv_extrinsics[..., 3, 3] = 1.0
        else:
            inv_extrinsics = None

        # --- 粗阶段 ---
        query = self.coarse_query.expand(B, -1, -1) + self.coarse_pos.unsqueeze(0)
        ref = self.coarse_ref.unsqueeze(0).expand(B, -1, -1)

        for layer in self.coarse_layers:
            if self.checkpoint_coarse and self.training:
                query = checkpoint(layer, query, ref, image_feats, intrinsics, inv_extrinsics, use_reentrant=False)
            else:
                query = layer(query, ref, image_feats, intrinsics, inv_extrinsics)

        coarse_feats = query.view(B, cx, cy, cz, -1).permute(0, 4, 1, 2, 3)  # [B, C, cx, cy, cz]

        # --- 时序融合 (可选) ---
        new_memory = None
        if self.config.use_temporal and hasattr(self, 'temporal_fusion'):
            coarse_flat = coarse_feats.permute(0, 2, 3, 4, 1).flatten(1, 3)
            fused_flat, new_memory = self.temporal_fusion(
                coarse_flat, memory, ego_motion=ego_motion, spatial_shape=(cx, cy, cz),
            )
            coarse_feats = fused_flat.view(B, cx, cy, cz, self.config.embed_dim).permute(0, 4, 1, 2, 3)

        # --- 精阶段 ---
        fx, fy, fz = self.config.fine_size
        fine_feats = F.interpolate(coarse_feats, size=(fx, fy, fz), mode='trilinear', align_corners=False)
        fine_feats = fine_feats.permute(0, 2, 3, 4, 1).reshape(B, -1, self.config.embed_dim)
        fine_feats = self.coarse_to_fine(fine_feats)

        query = fine_feats + self.fine_pos.unsqueeze(0)
        ref = self.fine_ref.unsqueeze(0).expand(B, -1, -1)

        for layer in self.fine_layers:
            if self.checkpoint_fine and self.training:
                query = checkpoint(layer, query, ref, image_feats, intrinsics, inv_extrinsics, use_reentrant=False)
            else:
                query = layer(query, ref, image_feats, intrinsics, inv_extrinsics)

        # 空间一致性卷积
        query_vol = query.view(B, fx, fy, fz, -1).permute(0, 4, 1, 2, 3).contiguous()
        query_vol = query_vol + self.fine_spatial_conv(query_vol)
        output = query_vol.permute(0, 2, 3, 4, 1)  # [B, fx, fy, fz, C]

        return output, new_memory
