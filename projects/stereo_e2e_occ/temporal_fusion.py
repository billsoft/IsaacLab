"""
时序融合模块

与参考版完全相同的结构, 仅调整 voxel_range 默认值
- Ego-Motion 对齐: 3D grid_sample warp
- GRU 门控更新
- FlashAttention (PyTorch SDPA)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class EfficientTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, query, key, value):
        B, Q, C = query.shape
        q = self.q_proj(query).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous().view(B, Q, C)
        return self.o_proj(output)


class GRUGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.update_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.reset_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.candidate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Tanh())

    def forward(self, current, memory):
        concat = torch.cat([current, memory], dim=-1)
        z = self.update_gate(concat)
        r = self.reset_gate(concat)
        concat_reset = torch.cat([current, r * memory], dim=-1)
        h_candidate = self.candidate(concat_reset)
        return (1 - z) * memory + z * h_candidate


class TemporalFusionModule(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, use_checkpoint=True, config=None):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.config = config

        self.temporal_attn = EfficientTemporalAttention(dim, num_heads, dropout)
        self.gate = GRUGate(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def _init_memory(self, B, Q, C, device):
        return torch.zeros(B, Q, C, device=device)

    def align_memory(self, memory, ego_motion, spatial_shape):
        """将上一帧 memory warp 到当前帧坐标系"""
        if ego_motion is None:
            return memory

        B, Q, C = memory.shape
        H, W, D = spatial_shape
        device = memory.device

        if self.config is not None:
            xmin, ymin, zmin, xmax, ymax, zmax = self.config.voxel_range
        else:
            xmin, ymin, zmin, xmax, ymax, zmax = -3.6, -3.0, -0.7, 3.6, 3.0, 2.5

        scale = torch.tensor([
            (xmax - xmin) / 2, (ymax - ymin) / 2, (zmax - zmin) / 2, 1.0,
        ], device=device)
        offset = torch.tensor([
            (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, 0.0,
        ], device=device)

        mem_vol = memory.view(B, H, W, D, C).permute(0, 4, 3, 1, 2)  # [B, C, D, H, W]

        zs = torch.linspace(-1, 1, D, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_d, grid_h, grid_w = torch.meshgrid(zs, ys, xs, indexing='ij')
        ones = torch.ones_like(grid_d)
        grid_homo = torch.stack([grid_w, grid_h, grid_d, ones], dim=-1)
        grid_homo = grid_homo.unsqueeze(0).expand(B, -1, -1, -1, -1)
        grid_flat = grid_homo.view(B, -1, 4)

        grid_flat_world = grid_flat * scale + offset
        T_inv = torch.linalg.inv(ego_motion)
        grid_warped_world = torch.bmm(grid_flat_world, T_inv.transpose(1, 2))
        grid_warped_norm = (grid_warped_world - offset) / scale
        grid_warped = grid_warped_norm[..., :3].view(B, D, H, W, 3)

        aligned_vol = F.grid_sample(
            mem_vol, grid_warped, mode='bilinear', padding_mode='zeros', align_corners=False,
        )
        aligned = aligned_vol.permute(0, 3, 4, 2, 1).reshape(B, Q, C)
        return aligned

    def _forward_impl(self, current, memory, ego_motion=None, spatial_shape=None):
        if memory is not None and ego_motion is not None and spatial_shape is not None:
            memory = self.align_memory(memory, ego_motion, spatial_shape)
        if memory is None:
            B, Q, C = current.shape
            memory = self._init_memory(B, Q, C, current.device)

        q = self.norm1(current)
        k = self.norm1(memory)
        attn_out = self.temporal_attn(q, k, k)
        fused = current + attn_out
        fused = fused + self.ffn(self.norm2(fused))
        new_memory = self.gate(fused, memory)
        return fused, new_memory

    def forward(self, current, memory=None, ego_motion=None, spatial_shape=None):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, current, memory, ego_motion, spatial_shape, use_reentrant=False)
        return self._forward_impl(current, memory, ego_motion, spatial_shape)
