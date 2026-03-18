"""
体素网格核心模块
================
纯 Python/NumPy 实现，不依赖 Isaac Sim，可单独测试。

坐标系：以相机地面投影点为原点
  - X: 宽边方向 (1280px), 72 格, ±3.6m
  - Y: 窄边方向 (1080px), 60 格, ±3.0m
  - Z: 高度方向, 32 格, -0.7m ~ +2.5m

使用：
    from voxel_grid import VoxelGrid
    vg = VoxelGrid()
    world_pts = vg.get_world_centers(cam_pos=[3,1,3], cam_yaw=0.0)
    vg.semantic[10, 20, 7] = 6  # 标记为 PERSON
    vg.save("output/voxel/frame_000042")
"""

import json
import os

import numpy as np


class VoxelGrid:
    """语义体素网格。"""

    # 网格尺寸
    NX: int = 72   # 宽边 (X)
    NY: int = 60   # 窄边 (Y)
    NZ: int = 32   # 高度 (Z)

    # 物理参数
    VOXEL_SIZE: float = 0.1   # 米/格
    Z_MIN: float = -0.7       # 最低高度
    Z_MAX: float = 2.5        # 最高高度
    Z_GROUND_INDEX: int = 7   # 地面分界索引

    # 中心索引（体素网格原点对应的索引）
    CENTER_X: int = 36  # NX // 2
    CENTER_Y: int = 30  # NY // 2

    # 特殊值
    UNOBSERVED: int = 255

    def __init__(self):
        self.semantic = np.full((self.NX, self.NY, self.NZ), self.UNOBSERVED, dtype=np.uint8)
        self.instance = np.zeros((self.NX, self.NY, self.NZ), dtype=np.int32)
        self._local_centers = self._precompute_centers()

    def _precompute_centers(self) -> np.ndarray:
        """预计算所有体素中心的局部坐标 (NX, NY, NZ, 3)。"""
        ix = np.arange(self.NX)
        iy = np.arange(self.NY)
        iz = np.arange(self.NZ)
        gi, gj, gk = np.meshgrid(ix, iy, iz, indexing="ij")

        centers = np.stack([
            (gi - self.CENTER_X + 0.5) * self.VOXEL_SIZE,
            (gj - self.CENTER_Y + 0.5) * self.VOXEL_SIZE,
            (gk - self.Z_GROUND_INDEX + 0.5) * self.VOXEL_SIZE,
        ], axis=-1)  # (72, 60, 32, 3)
        return centers

    @property
    def local_centers(self) -> np.ndarray:
        """所有体素中心的局部坐标 (NX, NY, NZ, 3)。"""
        return self._local_centers

    @property
    def local_centers_flat(self) -> np.ndarray:
        """展平的局部坐标 (NX*NY*NZ, 3)。"""
        return self._local_centers.reshape(-1, 3)

    def voxel_to_local(self, i: int, j: int, k: int) -> tuple:
        """单个体素索引 → 局部坐标（米）。"""
        x = (i - self.CENTER_X + 0.5) * self.VOXEL_SIZE
        y = (j - self.CENTER_Y + 0.5) * self.VOXEL_SIZE
        z = (k - self.Z_GROUND_INDEX + 0.5) * self.VOXEL_SIZE
        return (x, y, z)

    def local_to_voxel(self, x: float, y: float, z: float) -> tuple:
        """局部坐标 → 最近的体素索引（可能越界，调用者需检查）。"""
        i = int(np.floor(x / self.VOXEL_SIZE + self.CENTER_X))
        j = int(np.floor(y / self.VOXEL_SIZE + self.CENTER_Y))
        k = int(np.floor(z / self.VOXEL_SIZE + self.Z_GROUND_INDEX))
        return (i, j, k)

    def in_bounds(self, i: int, j: int, k: int) -> bool:
        return 0 <= i < self.NX and 0 <= j < self.NY and 0 <= k < self.NZ

    def get_world_centers(self, cam_pos, cam_yaw: float = 0.0) -> np.ndarray:
        """局部坐标 → 世界坐标。

        Args:
            cam_pos: (3,) 相机世界位置 [x, y, z]
            cam_yaw: 相机绕世界 Z 轴的偏航角（弧度）

        Returns:
            (NX, NY, NZ, 3) 世界坐标
        """
        cam_pos = np.asarray(cam_pos, dtype=np.float64)
        ground_proj = np.array([cam_pos[0], cam_pos[1], 0.0])

        local = self._local_centers.reshape(-1, 3).copy()

        # 绕 Z 轴旋转 XY
        if abs(cam_yaw) > 1e-6:
            cos_y, sin_y = np.cos(cam_yaw), np.sin(cam_yaw)
            x_rot = local[:, 0] * cos_y - local[:, 1] * sin_y
            y_rot = local[:, 0] * sin_y + local[:, 1] * cos_y
            local[:, 0] = x_rot
            local[:, 1] = y_rot

        local += ground_proj
        return local.reshape(self.NX, self.NY, self.NZ, 3)

    def get_world_centers_flat(self, cam_pos, cam_yaw: float = 0.0) -> np.ndarray:
        """返回展平的世界坐标 (NX*NY*NZ, 3)。"""
        return self.get_world_centers(cam_pos, cam_yaw).reshape(-1, 3)

    def reset(self):
        """重置为 unobserved。"""
        self.semantic[:] = self.UNOBSERVED
        self.instance[:] = 0

    def save(self, path_prefix: str):
        """保存体素数据。

        生成文件：
          {path_prefix}_semantic.npz
          {path_prefix}_instance.npz
        """
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        np.savez_compressed(f"{path_prefix}_semantic.npz", data=self.semantic)
        np.savez_compressed(f"{path_prefix}_instance.npz", data=self.instance)

    @classmethod
    def load(cls, path_prefix: str) -> "VoxelGrid":
        """从文件加载体素数据。"""
        vg = cls()
        vg.semantic = np.load(f"{path_prefix}_semantic.npz")["data"]
        vg.instance = np.load(f"{path_prefix}_instance.npz")["data"]
        return vg

    def stats(self) -> dict:
        """统计体素占用情况。"""
        from semantic_classes import CLASS_NAMES, FREE
        total = self.NX * self.NY * self.NZ
        observed = np.sum(self.semantic != self.UNOBSERVED)
        free = np.sum(self.semantic == FREE)
        occupied = observed - free
        class_counts = {}
        for cid in range(18):
            cnt = int(np.sum(self.semantic == cid))
            if cnt > 0:
                class_counts[CLASS_NAMES.get(cid, str(cid))] = cnt
        return {
            "total": total,
            "observed": int(observed),
            "free": int(free),
            "occupied": int(occupied),
            "unobserved": int(total - observed),
            "class_counts": class_counts,
        }

    @classmethod
    def get_config(cls) -> dict:
        """返回体素配置字典（用于保存 voxel_config.json）。"""
        return {
            "voxel_size": cls.VOXEL_SIZE,
            "grid_shape": [cls.NX, cls.NY, cls.NZ],
            "x_range": [-(cls.CENTER_X) * cls.VOXEL_SIZE, (cls.NX - cls.CENTER_X) * cls.VOXEL_SIZE],
            "y_range": [-(cls.CENTER_Y) * cls.VOXEL_SIZE, (cls.NY - cls.CENTER_Y) * cls.VOXEL_SIZE],
            "z_range": [cls.Z_MIN, cls.Z_MAX],
            "z_ground_index": cls.Z_GROUND_INDEX,
            "num_classes": 18,
            "coordinate_system": "camera_nadir_centered",
        }


# ============================================================================
# 自测
# ============================================================================
if __name__ == "__main__":
    vg = VoxelGrid()
    print(f"VoxelGrid: {vg.NX}x{vg.NY}x{vg.NZ} = {vg.NX*vg.NY*vg.NZ:,} voxels")
    print(f"Voxel size: {vg.VOXEL_SIZE}m")
    print(f"X range: [{vg.voxel_to_local(0,0,0)[0]:.2f}, {vg.voxel_to_local(71,0,0)[0]:.2f}] m")
    print(f"Y range: [{vg.voxel_to_local(0,0,0)[1]:.2f}, {vg.voxel_to_local(0,59,0)[1]:.2f}] m")
    print(f"Z range: [{vg.voxel_to_local(0,0,0)[2]:.2f}, {vg.voxel_to_local(0,0,31)[2]:.2f}] m")
    print(f"Ground (0,0,0) → local: {vg.voxel_to_local(36, 30, 7)}")

    # 测试世界坐标变换
    cam = [5.0, 3.0, 3.0]
    world = vg.get_world_centers(cam, cam_yaw=0.0)
    print(f"\nCamera at {cam}, yaw=0:")
    print(f"  Voxel(36,30,7) world = {world[36,30,7]}")  # 应该是 (5.05, 3.05, 0.05)
    print(f"  Voxel(0,0,0) world = {world[0,0,0]}")

    # 测试旋转
    import math
    world_rot = vg.get_world_centers(cam, cam_yaw=math.pi/2)
    print(f"  Voxel(36,30,7) world (yaw=90°) = {world_rot[36,30,7]}")

    print(f"\nConfig: {json.dumps(VoxelGrid.get_config(), indent=2)}")
