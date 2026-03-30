"""
位置编码模块

1. SineCosinePositionEncoding3D: 3D 体素空间的正弦余弦位置编码
2. FisheyeRayEncoding: f-theta 鱼眼射线方向编码 (替代参考版的 RayDirectionEncoding)
   - 使用等距投影模型 theta = r / f
   - 将像素坐标转换为世界空间射线方向
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SineCosinePositionEncoding3D(nn.Module):
    """3D 体素网格的正弦-余弦位置编码"""
    def __init__(self, dim, temperature=10000):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, x, y, z, device):
        """
        生成 (x, y, z) 网格的位置编码
        returns: [x*y*z, dim]
        """
        d = self.dim // 6
        dim_t = torch.arange(d, device=device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / d)

        xs = torch.arange(x, device=device).float()
        ys = torch.arange(y, device=device).float()
        zs = torch.arange(z, device=device).float()

        pos_x = xs.view(-1, 1, 1, 1) / dim_t.view(1, 1, 1, -1)
        pos_y = ys.view(1, -1, 1, 1) / dim_t.view(1, 1, 1, -1)
        pos_z = zs.view(1, 1, -1, 1) / dim_t.view(1, 1, 1, -1)

        px = torch.stack([pos_x.sin(), pos_x.cos()], -1).flatten(-2)
        py = torch.stack([pos_y.sin(), pos_y.cos()], -1).flatten(-2)
        pz = torch.stack([pos_z.sin(), pos_z.cos()], -1).flatten(-2)

        px = px.expand(x, y, z, -1)
        py = py.expand(x, y, z, -1)
        pz = pz.expand(x, y, z, -1)

        pe = torch.cat([px, py, pz], dim=-1)
        if pe.shape[-1] < self.dim:
            pe = F.pad(pe, (0, self.dim - pe.shape[-1]))
        return pe.view(-1, self.dim)


class FisheyeRayEncoding(nn.Module):
    """
    f-theta 鱼眼射线方向编码

    与参考版 RayDirectionEncoding 的区别:
    - 使用 f-theta 等距投影模型: theta = r / f (而非通用 pinhole 反投影)
    - 从 calibration.json 的 fx/fy/cx/cy 直接计算
    - 只有 2 个相机 (左右眼), 不需要环视处理
    """
    def __init__(self, dim: int, image_size: Tuple[int, int], num_cameras: int = 2,
                 num_freqs: int = 10, feat_size: Tuple[int, int] = None):
        super().__init__()
        self.dim = dim
        self.image_size = image_size  # (H, W) = (1080, 1280) 原始图像分辨率
        self.num_freqs = num_freqs
        self.num_cameras = num_cameras

        # 预计算像素网格 (在原图分辨率上均匀采样 feat_size 个点)
        if feat_size is not None:
            fH, fW = feat_size
            orig_H, orig_W = image_size
            v_coords, u_coords = torch.meshgrid(
                torch.linspace(0, orig_H - 1, fH),
                torch.linspace(0, orig_W - 1, fW),
                indexing='ij',
            )
            self.register_buffer('u_coords', u_coords)
            self.register_buffer('v_coords', v_coords)

        # 输入: xyz (3) + sinusoidal (3 * 2 * num_freqs)
        self.input_dim = 3 + 3 * 2 * num_freqs
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def _sinusoidal_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """正弦位置编码: x [*, 3] → [*, 3 + 3*2*num_freqs]"""
        freq_bands = 2.0 ** torch.linspace(0, self.num_freqs - 1, self.num_freqs, device=x.device)
        x_expanded = x.unsqueeze(-1) * freq_bands  # [*, 3, F]
        sin_x = torch.sin(x_expanded * torch.pi)
        cos_x = torch.cos(x_expanded * torch.pi)
        embeddings = torch.stack([sin_x, cos_x], dim=-1).flatten(start_dim=-3)
        return torch.cat([x, embeddings], dim=-1)

    def get_rays_from_params(self, intrinsics, extrinsics, H, W):
        """
        f-theta 等距投影: theta = r / f
        像素 → 相机系射线 → 世界系射线

        intrinsics: [B, N, 3, 3]  (fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2])
        extrinsics: [B, N, 4, 4]
        H, W: 特征图尺寸 (68, 80)
        returns: [B, N, H, W, 3]  世界空间射线方向
        """
        B, N, _, _ = intrinsics.shape
        device = intrinsics.device

        # 使用预计算的像素网格 (若可用), 否则动态生成
        if hasattr(self, 'u_coords') and self.u_coords.shape == (H, W):
            u_coords = self.u_coords
            v_coords = self.v_coords
        else:
            orig_H, orig_W = self.image_size
            v_coords, u_coords = torch.meshgrid(
                torch.linspace(0, orig_H - 1, H, device=device),
                torch.linspace(0, orig_W - 1, W, device=device),
                indexing='ij',
            )

        # 从内参读光心和焦距
        cx = intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
        cy = intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1)
        f = intrinsics[..., 0, 0].unsqueeze(-1).unsqueeze(-1)   # fx ≈ fy

        dx = u_coords.unsqueeze(0).unsqueeze(0) - cx  # [B, N, H, W]
        dy = v_coords.unsqueeze(0).unsqueeze(0) - cy

        # f-theta 等距投影反投影
        r = torch.sqrt(dx ** 2 + dy ** 2)
        phi = torch.atan2(dy, dx)
        theta = r / f  # 等距模型: r = f * theta → theta = r / f

        # 球坐标 → 笛卡尔 (相机坐标系: Z 前, X 右, Y 下)
        sin_theta = torch.sin(theta)
        cam_x = sin_theta * torch.cos(phi)
        cam_y = sin_theta * torch.sin(phi)
        cam_z = torch.cos(theta)
        cam_dirs = torch.stack([cam_x, cam_y, cam_z], dim=-1)  # [B, N, H, W, 3]

        # 相机 → 世界
        R = extrinsics[..., :3, :3]  # [B, N, 3, 3]
        world_dirs = torch.einsum('bnij,bnhwj->bnhwi', R, cam_dirs)

        return world_dirs

    def forward(self, x, intrinsics=None, extrinsics=None):
        """
        x: [B, N, C, H, W]  图像特征 (仅用于获取维度)
        returns: [B, N, dim, H, W]
        """
        B, N, C, H, W = x.shape
        if intrinsics is None:
            return torch.zeros(B, N, self.dim, H, W, device=x.device)

        rays = self.get_rays_from_params(intrinsics, extrinsics, H, W)  # [B, N, H, W, 3]
        encoded_rays = self._sinusoidal_encoding(rays)
        encoded = self.proj(encoded_rays)  # [B, N, H, W, dim]
        return encoded.permute(0, 1, 4, 2, 3)  # [B, N, dim, H, W]
