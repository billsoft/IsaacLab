"""VoxelGridV2 —— 继承 VoxelGrid，新增 flow 和 uint16 实例支持
===============================================================
不修改原 voxel_grid.py，通过继承扩展。
"""

import os
import sys

import numpy as np

# 添加 scripts 目录到 path，复用现有 VoxelGrid
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from voxel_grid import VoxelGrid


class VoxelGridV2(VoxelGrid):
    """扩展体素网格：instance→uint16，新增 flow/flow_mask/orientation/angular_vel。"""

    INSTANCE_UNOBSERVED: int = 65535

    def __init__(self):
        super().__init__()
        # 覆盖父类的 int32 instance 为 uint16
        self.instance = np.zeros((self.NX, self.NY, self.NZ), dtype=np.uint16)
        # flow 字段：速度 (vx, vy) + 掩码
        self.flow = np.zeros((self.NX, self.NY, self.NZ, 2), dtype=np.float16)
        self.flow_mask = np.zeros((self.NX, self.NY, self.NZ), dtype=np.uint8)
        # 方向字段：航向角（体素局部坐标系，弧度，[-π,π]）+ 角速度 ωz（rad/s）
        self.orientation = np.zeros((self.NX, self.NY, self.NZ), dtype=np.float16)
        self.angular_vel = np.zeros((self.NX, self.NY, self.NZ), dtype=np.float16)

    def reset(self):
        """重置所有字段。"""
        self.semantic[:] = self.UNOBSERVED
        self.instance[:] = 0
        self.flow[:] = 0
        self.flow_mask[:] = 0
        self.orientation[:] = 0
        self.angular_vel[:] = 0

    def save(self, path_prefix: str):
        """保存体素数据（semantic + instance + flow）。"""
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        np.savez_compressed(f"{path_prefix}_semantic.npz", data=self.semantic)
        np.savez_compressed(f"{path_prefix}_instance.npz", data=self.instance)
        np.savez_compressed(f"{path_prefix}_flow.npz",
                            flow=self.flow, flow_mask=self.flow_mask,
                            orientation=self.orientation, angular_vel=self.angular_vel)

    @classmethod
    def load(cls, path_prefix: str) -> "VoxelGridV2":
        """从文件加载体素数据（兼容旧版无 orientation/angular_vel 的 npz）。"""
        vg = cls()
        vg.semantic = np.load(f"{path_prefix}_semantic.npz")["data"]

        inst = np.load(f"{path_prefix}_instance.npz")["data"]
        vg.instance = inst.astype(np.uint16)

        flow_path = f"{path_prefix}_flow.npz"
        if os.path.isfile(flow_path):
            data = np.load(flow_path)
            vg.flow = data["flow"]
            vg.flow_mask = data["flow_mask"]
            if "orientation" in data:
                vg.orientation = data["orientation"]
            if "angular_vel" in data:
                vg.angular_vel = data["angular_vel"]

        return vg

    def flow_stats(self) -> dict:
        """flow 统计信息。"""
        dynamic_count = int(self.flow_mask.sum())
        if dynamic_count == 0:
            return {"dynamic_voxels": 0}

        speeds = np.sqrt(
            self.flow[..., 0].astype(np.float32) ** 2
            + self.flow[..., 1].astype(np.float32) ** 2
        )
        masked_speeds = speeds[self.flow_mask == 1]
        return {
            "dynamic_voxels": dynamic_count,
            "mean_speed_ms": float(masked_speeds.mean()),
            "max_speed_ms": float(masked_speeds.max()),
            "stationary_ratio": float((masked_speeds < 0.01).mean()),
        }
