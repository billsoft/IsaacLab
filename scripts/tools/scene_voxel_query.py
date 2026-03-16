# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
场景体素查询工具
================
给定世界坐标，返回该体素格子内的所有 Actor（NPC、机器人、静态障碍物）。

两种查询后端：
  1. PhysX overlap_box  —— 动态 actor（有碰撞体）实时查询，精确
  2. USD BBoxCache      —— 静态场景（货架、墙壁等）预建索引，快速

典型用途：
  • 遍历三维空间生成带标注体素图（用于 ML 数据集）
  • 检查某格是否被占用
  • 获取格内所有 actor 的 prim path 和语义类型

可单独运行（demo 模式）：
    isaaclab.bat -p scripts/tools/scene_voxel_query.py --headless
    isaaclab.bat -p scripts/tools/scene_voxel_query.py  # 带 GUI

也可作为模块被其他脚本导入：
    from scripts.tools.scene_voxel_query import SceneVoxelQuery, VoxelGrid
"""

from __future__ import annotations

import dataclasses
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ActorInfo:
    """单个 Actor 的信息"""
    prim_path: str          # USD prim 路径，全局唯一 ID
    actor_type: str         # 语义类型：human / robot / shelf / wall / object
    world_pos: tuple        # 世界坐标 (x, y, z)，浮点元组


@dataclasses.dataclass
class VoxelCell:
    """单个体素格子的查询结果"""
    voxel_idx: tuple        # 格子三维索引 (ix, iy, iz)
    world_center: tuple     # 格子中心世界坐标 (x, y, z)
    actors: list            # List[ActorInfo]

    @property
    def is_occupied(self) -> bool:
        return len(self.actors) > 0

    @property
    def actor_types(self) -> set:
        return {a.actor_type for a in self.actors}


# ─────────────────────────────────────────────────────────────────────────────
# 类型分类器（可由使用者替换）
# ─────────────────────────────────────────────────────────────────────────────

def default_classifier(prim_path: str) -> str:
    """
    根据 prim 路径推断语义类型。
    使用者可传入自定义函数替换此逻辑。
    """
    p = prim_path.lower()
    if "/characters/" in p:                       return "human"
    if "worker" in p or "people" in p:            return "human"
    if "/robot" in p or "/h1" in p:               return "robot"
    if "/shelf" in p or "/rack" in p:             return "shelf"
    if "/wall" in p or "/floor" in p:             return "static"
    if "/forklift" in p or "/vehicle" in p:       return "vehicle"
    return "object"


# ─────────────────────────────────────────────────────────────────────────────
# 核心查询类
# ─────────────────────────────────────────────────────────────────────────────

class SceneVoxelQuery:
    """
    体素空间查询器。

    用法：
        vq = SceneVoxelQuery(voxel_size=0.5)
        vq.build_static_index(stage)          # 场景加载后调一次

        # 仿真循环里随时查询：
        cell = vq.query_world_pos((3.0, 0.5, 1.0))
        print(cell.actors)                    # List[ActorInfo]

        # 按体素索引查询（配合 VoxelGrid 使用）：
        cell = vq.query_voxel_idx((10, 1, 4), grid)
    """

    def __init__(
        self,
        voxel_size: float = 0.5,
        dynamic_roots: tuple[str, ...] = ("/World/Characters", "/World/Robots"),
        classifier: Callable[[str], str] | None = None,
    ):
        """
        Args:
            voxel_size:    体素边长（米）。
            dynamic_roots: 动态 actor 所在的 prim 根路径，这些路径下的 prim
                           不进静态索引，通过 PhysX 实时查询。
            classifier:    自定义类型分类函数 (prim_path: str) -> str。
                           传 None 使用内置 default_classifier。
        """
        self.voxel_size    = voxel_size
        self.dynamic_roots = dynamic_roots
        self._classifier   = classifier or default_classifier
        self._static_index: list[dict] = []
        self._physx_ok     = False
        self._physx_iface  = None

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def build_static_index(self, stage) -> int:
        """
        遍历场景所有静态 prim，计算包围盒，建立空间索引。
        场景加载完成、仿真开始前调用一次。

        Returns:
            静态 prim 数量。
        """
        import numpy as np
        from pxr import Usd, UsdGeom

        self._static_index = []
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_],
        )

        for prim in stage.Traverse():
            if not prim.IsActive():
                continue
            path_str = str(prim.GetPath())
            # 动态 actor 不进静态索引
            if any(path_str.startswith(r) for r in self.dynamic_roots):
                continue
            try:
                bbox = bbox_cache.ComputeWorldBound(prim)
                rng  = bbox.ComputeAlignedRange()
                if rng.IsEmpty():
                    continue
                self._static_index.append({
                    "path":  path_str,
                    "type":  self._classifier(path_str),
                    "min":   np.array(rng.GetMin()),
                    "max":   np.array(rng.GetMax()),
                    "center": np.array(rng.GetMidpoint()),
                })
            except Exception:
                continue

        # 初始化 PhysX 接口
        try:
            from omni.physx import get_physx_scene_query_interface
            self._physx_iface = get_physx_scene_query_interface()
            self._physx_ok    = True
        except Exception:
            self._physx_ok = False

        return len(self._static_index)

    # ── 查询接口 ──────────────────────────────────────────────────────────────

    def query_world_pos(self, world_pos: tuple, voxel_size: float | None = None) -> VoxelCell:
        """
        给世界坐标，返回该位置所在体素的内容。

        Args:
            world_pos:  (x, y, z) 世界坐标。
            voxel_size: 覆盖默认体素尺寸（可选）。

        Returns:
            VoxelCell，包含该格内所有 ActorInfo。
        """
        size = voxel_size or self.voxel_size
        dynamic = self._query_dynamic(world_pos, size)
        static  = self._query_static(world_pos, size)
        return VoxelCell(
            voxel_idx   = (0, 0, 0),   # 无网格定义时为占位符
            world_center = world_pos,
            actors      = dynamic + static,
        )

    def query_voxel_idx(self, voxel_idx: tuple, grid: "VoxelGrid") -> VoxelCell:
        """
        给体素索引 (ix, iy, iz)，通过 VoxelGrid 换算坐标后查询。

        Args:
            voxel_idx: (ix, iy, iz) 体素格子索引。
            grid:      VoxelGrid 实例（提供索引→坐标换算）。

        Returns:
            VoxelCell。
        """
        center = grid.idx_to_world(voxel_idx)
        cell   = self.query_world_pos(center, grid.voxel_size)
        cell.voxel_idx = voxel_idx
        return cell

    # ── 批量扫描 ──────────────────────────────────────────────────────────────

    def scan_grid(
        self,
        grid: "VoxelGrid",
        skip_empty: bool = True,
    ) -> list[VoxelCell]:
        """
        扫描 VoxelGrid 的每个格子，返回所有（非空）格子的 VoxelCell 列表。

        Args:
            grid:       要扫描的 VoxelGrid。
            skip_empty: True 时跳过空格（无 actor 的格子），节省内存。

        Returns:
            List[VoxelCell]，按 (ix, iy, iz) 顺序。
        """
        results = []
        nx, ny, nz = grid.dims
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    cell = self.query_voxel_idx((ix, iy, iz), grid)
                    if skip_empty and not cell.is_occupied:
                        continue
                    results.append(cell)
        return results

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    def _query_dynamic(self, pos: tuple, size: float) -> list[ActorInfo]:
        """PhysX overlap_box 查询动态 actor（有碰撞体）"""
        if not self._physx_ok:
            return []

        import carb

        hits    = []
        half    = size / 2.0
        cx, cy, cz = pos

        def on_hit(hit):
            try:
                path = str(hit.rigid_body)
                # 获取 prim 世界坐标
                import omni.usd
                from pxr import UsdGeom, Usd
                stage = omni.usd.get_context().get_stage()
                prim  = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                    t = mat.ExtractTranslation()
                    wpos = (float(t[0]), float(t[1]), float(t[2]))
                else:
                    wpos = pos
                hits.append(ActorInfo(
                    prim_path  = path,
                    actor_type = self._classifier(path),
                    world_pos  = wpos,
                ))
            except Exception:
                pass
            return True   # True = 继续报告所有命中

        try:
            self._physx_iface.overlap_box(
                halfExtent = carb.Float3(half, half, half),
                pos        = carb.Float3(cx, cy, cz),
                rot        = carb.Float4(0.0, 0.0, 0.0, 1.0),
                reportFn   = on_hit,
                anyHit     = False,
            )
        except Exception:
            pass
        return hits

    def _query_static(self, pos: tuple, size: float) -> list[ActorInfo]:
        """BBoxCache 索引查询静态 prim"""
        import numpy as np

        c    = np.array(pos, dtype=float)
        half = size / 2.0
        vmin = c - half
        vmax = c + half

        result = []
        for entry in self._static_index:
            if np.all(entry["max"] >= vmin) and np.all(entry["min"] <= vmax):
                result.append(ActorInfo(
                    prim_path  = entry["path"],
                    actor_type = entry["type"],
                    world_pos  = tuple(entry["center"].tolist()),
                ))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 体素网格定义（坐标系和索引换算）
# ─────────────────────────────────────────────────────────────────────────────

class VoxelGrid:
    """
    定义体素网格的空间范围和分辨率，提供坐标↔索引换算。

    不存储体素数据本身（数据由 SceneVoxelQuery.scan_grid 返回）。

    示例：
        # 仓库场景：X[-10,10], Y[0,3], Z[-5,5]，0.5m 分辨率
        grid = VoxelGrid(origin=(-10, 0, -5), size=(20, 3, 10), voxel_size=0.5)
        print(grid.dims)    # (40, 6, 20)

        idx = grid.world_to_idx((3.0, 0.5, 1.0))   # → (26, 1, 12)
        ctr = grid.idx_to_world((26, 1, 12))        # → (3.25, 0.25, 1.25)
    """

    def __init__(self, origin: tuple, size: tuple, voxel_size: float = 0.5):
        """
        Args:
            origin:     网格起点（最小角）世界坐标 (x, y, z)。
            size:       网格尺寸 (dx, dy, dz)，单位米。
            voxel_size: 每个体素的边长，单位米。
        """
        import numpy as np

        self.origin     = np.array(origin, dtype=float)
        self.size       = np.array(size,   dtype=float)
        self.voxel_size = float(voxel_size)
        self.dims       = tuple((self.size / self.voxel_size).astype(int).tolist())
        # dims = (nx, ny, nz)

    @property
    def total_voxels(self) -> int:
        nx, ny, nz = self.dims
        return nx * ny * nz

    def world_to_idx(self, world_pos: tuple) -> tuple:
        """世界坐标 → 体素索引（clamp 到合法范围内）"""
        import numpy as np

        idx = ((np.array(world_pos) - self.origin) / self.voxel_size).astype(int)
        idx = np.clip(idx, 0, np.array(self.dims) - 1)
        return tuple(idx.tolist())

    def idx_to_world(self, voxel_idx: tuple) -> tuple:
        """体素索引 → 格子中心世界坐标"""
        import numpy as np

        center = self.origin + (np.array(voxel_idx) + 0.5) * self.voxel_size
        return tuple(center.tolist())

    def is_valid_idx(self, voxel_idx: tuple) -> bool:
        nx, ny, nz = self.dims
        ix, iy, iz = voxel_idx
        return 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz

    def __repr__(self) -> str:
        return (
            f"VoxelGrid(origin={tuple(self.origin.tolist())}, "
            f"size={tuple(self.size.tolist())}, "
            f"voxel_size={self.voxel_size}, "
            f"dims={self.dims}, "
            f"total={self.total_voxels})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 独立运行 Demo（isaaclab.bat -p scripts/tools/scene_voxel_query.py）
# ─────────────────────────────────────────────────────────────────────────────

def _run_demo():
    """加载仓库场景，构建体素网格，打印占用情况。"""
    import argparse
    import time
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Scene voxel query demo")
    parser.add_argument("--voxel_size", type=float, default=0.5)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import omni.usd
    import omni.timeline
    import carb
    from pxr import UsdLux, UsdGeom, Gf

    ASSET_ROOT = "D:/code/IsaacLab/Assets/Isaac/5.1"
    SCENE_USD  = f"{ASSET_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

    # 加载场景
    ctx = omni.usd.get_context()
    ctx.open_stage(SCENE_USD)
    for _ in range(300):
        simulation_app.update()
        if ctx.get_stage().GetPrimAtPath("/World").IsValid():
            break
        time.sleep(0.05)

    stage = ctx.get_stage()
    print(f"\n[Demo] 场景已加载：{SCENE_USD}")

    # 构建查询器
    vq   = SceneVoxelQuery(voxel_size=args.voxel_size)
    grid = VoxelGrid(origin=(-10, 0, -5), size=(20, 3, 10), voxel_size=args.voxel_size)
    n    = vq.build_static_index(stage)
    print(f"[Demo] 静态索引：{n} 个 prim")
    print(f"[Demo] {grid}")

    # 启动时间轴（激活 PhysX）
    tl = omni.timeline.get_timeline_interface()
    tl.set_start_time(0)
    tl.set_end_time(1e9)
    tl.play()
    for _ in range(30):
        simulation_app.update()

    # 单点查询示例
    test_points = [(0.0, 0.5, 0.0), (3.0, 0.5, 2.0), (-5.0, 0.5, -3.0)]
    print("\n[Demo] 单点查询测试：")
    for pt in test_points:
        cell = vq.query_world_pos(pt)
        tag  = "占用" if cell.is_occupied else "空"
        print(f"  {pt} → [{tag}]  actors={[a.actor_type for a in cell.actors]}")

    # 全网格扫描
    print("\n[Demo] 全网格扫描（仅打印占用格子）...")
    occupied = vq.scan_grid(grid, skip_empty=True)
    print(f"[Demo] 占用格子数：{len(occupied)} / {grid.total_voxels}")
    for cell in occupied[:10]:   # 只打印前 10 个
        print(f"  idx={cell.voxel_idx}  center={cell.world_center}"
              f"  types={cell.actor_types}")

    tl.stop()
    simulation_app.close()


if __name__ == "__main__":
    _run_demo()
