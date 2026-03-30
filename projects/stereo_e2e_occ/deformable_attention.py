"""
可变形交叉注意力

将 3D 体素查询投影到 2D 鱼眼图像特征上进行采样
核心修改:
  - get_reference_points 使用 f-theta 等距投影 (而非 pinhole)
  - 感知范围从 [-40,40] 改为 config.voxel_range
  - num_cameras=2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, num_cameras, num_points=4, dropout=0.1, config=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_cameras = num_cameras
        self.num_points = num_points
        self.head_dim = dim // num_heads
        self.config = config

        self.sampling_offsets = nn.Linear(dim, num_cameras * num_heads * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_cameras * num_heads * num_points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.constant_(self.attention_weights.bias, 0.0)

    def get_reference_points(self, query_coords, intrinsics, inv_extrinsics, H, W):
        """
        将 [0,1] 归一化体素坐标 → 世界坐标 → 相机坐标 → f-theta 鱼眼图像坐标

        query_coords: [B, Q, 3] in [0, 1]
        intrinsics: [B, N, 3, 3]
        inv_extrinsics: [B, N, 4, 4]  预计算的 extrinsics 逆矩阵
        returns: [B, N, Q, 2] 归一化图像坐标 [-1, 1]
        """
        B, Q, _ = query_coords.shape
        N = self.num_cameras

        # [0,1] → 世界坐标 (米)
        if self.config is not None:
            xmin, ymin, zmin, xmax, ymax, zmax = self.config.voxel_range
        else:
            xmin, ymin, zmin, xmax, ymax, zmax = -3.6, -3.0, -0.7, 3.6, 3.0, 2.5

        real_x = query_coords[..., 0] * (xmax - xmin) + xmin
        real_y = query_coords[..., 1] * (ymax - ymin) + ymin
        real_z = query_coords[..., 2] * (zmax - zmin) + zmin
        world_points = torch.stack([real_x, real_y, real_z, torch.ones_like(real_x)], dim=-1)  # [B, Q, 4]

        # 扩展到 N 个相机
        world_points = world_points.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, Q, 4]

        # 世界坐标 → 相机坐标 (使用预计算的逆矩阵)
        cam_points = torch.matmul(
            inv_extrinsics.unsqueeze(2), world_points.unsqueeze(-1)
        ).squeeze(-1)  # [B, N, Q, 4]
        cam_xyz = cam_points[..., :3]  # [B, N, Q, 3]

        # f-theta 等距鱼眼投影: r = f * theta, theta = arctan2(sqrt(x^2+y^2), z)
        cam_x = cam_xyz[..., 0]
        cam_y = cam_xyz[..., 1]
        cam_z = cam_xyz[..., 2]

        r_3d = torch.sqrt(cam_x ** 2 + cam_y ** 2 + 1e-8)
        theta = torch.atan2(r_3d, cam_z)  # 入射角
        phi = torch.atan2(cam_y, cam_x)   # 方位角

        # 内参
        fx = intrinsics[:, :, 0, 0].unsqueeze(-1)  # [B, N, 1]
        cx = intrinsics[:, :, 0, 2].unsqueeze(-1)
        cy = intrinsics[:, :, 1, 2].unsqueeze(-1)

        # r_img = f * theta (等距模型)
        r_img = fx * theta
        u = r_img * torch.cos(phi) + cx
        v = r_img * torch.sin(phi) + cy

        # 归一化到 [-1, 1]
        # u, v 是原图像素坐标, grid_sample 采样特征图时 [-1,1] 覆盖整张特征图
        # 由于特征图是原图的 stride-16 下采样, 空间对应关系线性, 用原图尺寸归一化即可
        orig_H, orig_W = self.config.image_size if self.config is not None else (1080, 1280)
        u_norm = 2.0 * u / (orig_W - 1) - 1.0
        v_norm = 2.0 * v / (orig_H - 1) - 1.0

        ref_points = torch.stack([u_norm, v_norm], dim=-1)  # [B, N, Q, 2]

        # FOV 有效性掩码: 相机后方 (cam_z<0) 或超出 FOV 的点标记为无效
        max_fov_rad = self.config.max_fov_deg / 2.0 * (3.14159265 / 180.0) if self.config is not None else 1.372
        fov_mask = (cam_z > 0) & (theta < max_fov_rad)  # [B, N, Q]

        return ref_points, fov_mask

    def forward(self, query, query_coords, image_feats, intrinsics=None, inv_extrinsics=None):
        """
        query: [B, Q, C]
        query_coords: [B, Q, 3] in [0,1]
        image_feats: [B, N, C, H, W]
        inv_extrinsics: [B, N, 4, 4] 预计算的外参逆矩阵
        returns: [B, Q, C]
        """
        B, Q, C = query.shape
        _, N, _, H, W = image_feats.shape

        # 1. 参考点 + FOV 掩码
        if intrinsics is not None and inv_extrinsics is not None:
            reference_points, fov_mask = self.get_reference_points(query_coords, intrinsics, inv_extrinsics, H, W)
        else:
            reference_points = torch.zeros(B, N, Q, 2, device=query.device)
            fov_mask = torch.ones(B, N, Q, dtype=torch.bool, device=query.device)

        # 2. 偏移与权重
        offsets = self.sampling_offsets(query)
        offsets = offsets.view(B, Q, N, self.num_heads, self.num_points, 2)
        offsets = offsets.tanh() * 0.5

        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(B, Q, N, self.num_heads, self.num_points)
        # 将超出 FOV 的参考点权重置零 (减轻网络学习无效投影的负担)
        # fov_mask: [B, N, Q] → [B, Q, N, 1, 1]
        fov_weight_mask = fov_mask.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        attn_weights = attn_weights * fov_weight_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 3. Value 投影
        value_proj = self.value_proj(image_feats.permute(0, 1, 3, 4, 2))
        value_proj = value_proj.view(B, N, H, W, self.num_heads, self.head_dim)

        # 4. 全相机+全头批量采样 (N 和 num_heads 合并到 batch, 1 次 grid_sample)
        # reference_points: [B, N, Q, 2]
        # offsets: [B, Q, N, num_heads, num_points, 2]
        ref_exp = reference_points.permute(0, 1, 2, 3)  # [B, N, Q, 2]
        ref_exp = ref_exp.unsqueeze(3).unsqueeze(4)       # [B, N, Q, 1, 1, 2]
        locs = ref_exp + offsets.permute(0, 2, 1, 3, 4, 5)  # [B, N, Q, num_heads, num_points, 2]

        # value: [B, N, H, W, num_heads, head_dim] → [B*N*num_heads, head_dim, H, W]
        v_all = value_proj.permute(0, 1, 4, 5, 2, 3).reshape(
            B * N * self.num_heads, self.head_dim, H, W)
        # locs: [B, N, Q, num_heads, num_points, 2] → [B*N*num_heads, Q*num_points, 1, 2]
        locs_all = locs.permute(0, 1, 3, 2, 4, 5).reshape(
            B * N * self.num_heads, Q * self.num_points, 1, 2)

        # 单次 grid_sample (替代原来 N×num_heads 次循环)
        sampled = F.grid_sample(
            v_all, locs_all,
            mode='bilinear', align_corners=False, padding_mode='zeros',
        )  # [B*N*num_heads, head_dim, Q*num_points, 1]

        # 还原维度: [B, N, num_heads, head_dim, Q, num_points]
        sampled = sampled.view(B, N, self.num_heads, self.head_dim, Q, self.num_points)
        # → [B, Q, N, num_heads, num_points, head_dim]
        sampled = sampled.permute(0, 4, 1, 2, 5, 3)
        # attn_weights: [B, Q, N, num_heads, num_points] → 加权求和
        weighted = (sampled * attn_weights.unsqueeze(-1)).sum(dim=4)  # [B, Q, N, num_heads, head_dim]
        output = weighted.sum(dim=2)  # [B, Q, num_heads, head_dim] (对 N 个相机求和)

        output = output.view(B, Q, C)
        output = self.output_proj(output)
        return self.dropout(output)


class DeformableDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, num_cameras, num_points=4, mlp_ratio=4.0, dropout=0.1,
                 use_self_attn=True, config=None):
        super().__init__()
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            self.norm_self = nn.LayerNorm(dim)

        self.cross_attn = DeformableCrossAttention(dim, num_heads, num_cameras, num_points, dropout, config=config)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, query_coords, image_feats, intrinsics=None, inv_extrinsics=None):
        if self.use_self_attn:
            q = self.norm_self(query)
            query = query + self.self_attn(q, q, q)[0]
        query = query + self.cross_attn(self.norm_cross(query), query_coords, image_feats, intrinsics, inv_extrinsics)
        query = query + self.mlp(self.norm_mlp(query))
        return query
