"""
多目相机体素地图生成工具 —— 自动驾驶 / 机器人训练数据
===========================================================
功能：
  1. 以 ego（多目摄像机组）为中心定义体素网格（如 400×400×32）
  2. 获取 ego 在 Isaac Sim 世界坐标系中的位姿
  3. 将每个体素中心从 ego 局部坐标系转换到世界坐标系
  4. 查询世界坐标处的 actor（静态物体 + 动态 NPC）
  5. 填充体素：free=0, 静态障碍物=语义ID, 动态NPC=actor_id

输出：
  • voxel_grid.npy  — (X, Y, Z) uint16 数组，值为语义标签
  • voxel_meta.json — 元数据（ego pose、体素参数、语义映射表）

体素坐标约定（ego-centric, Z-up）：
  • X: ego 前方（+X=前, -X=后）
  • Y: ego 左方（+Y=左, -Y=右）
  • Z: 竖直方向（+Z=上, -Z=下）
  • 原点: ego 摄像机位置

运行方式：
    isaaclab.bat -p scripts/npc/generate_voxel_grid.py
    isaaclab.bat -p scripts/npc/generate_voxel_grid.py --headless --voxel_size 0.5 --grid_x 200 --grid_y 200 --grid_z 16
"""

import argparse
import json
import math
import os
import sys
import time

# ---------------------------------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Voxel Grid Generator for Autonomous Driving")
parser.add_argument("--ego_prim", type=str, default="/World/Ego",
                    help="Ego camera prim path (will create if not exists)")
parser.add_argument("--ego_pos", type=float, nargs=3, default=[0.0, 0.0, 1.5],
                    help="Ego position in world (x y z)")
parser.add_argument("--ego_yaw", type=float, default=0.0,
                    help="Ego heading in degrees (0=+X)")
parser.add_argument("--voxel_size", type=float, default=0.5,
                    help="Voxel size in meters (default 0.5m)")
parser.add_argument("--grid_x", type=int, default=400,
                    help="Grid size along X (front/back)")
parser.add_argument("--grid_y", type=int, default=400,
                    help="Grid size along Y (left/right)")
parser.add_argument("--grid_z", type=int, default=32,
                    help="Grid size along Z (up/down)")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory")
parser.add_argument("--scene_usd", type=str, default=None,
                    help="Scene USD to load (optional, uses current stage if not set)")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--use_physx", action="store_true",
                    help="Also query dynamic actors via PhysX overlap (slower but more accurate)")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# 2. SimulationApp
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

# ---------------------------------------------------------------------------
# 3. Imports after SimulationApp
# ---------------------------------------------------------------------------
import numpy as np
import carb
import omni.usd
from pxr import Gf, Usd, UsdGeom, Sdf


# ===========================================================================
# 核心模块
# ===========================================================================

# ---------------------------------------------------------------------------
# A. 语义分类器：将 USD prim 路径映射到语义标签
# ---------------------------------------------------------------------------
class SemanticClassifier:
    """
    根据 prim 路径和类型名推断语义类别。
    可扩展：添加自定义规则或使用 USD semantic tag。
    """

    # 语义类别定义
    CLASSES = {
        0:  "free",
        1:  "ground",
        2:  "wall",
        3:  "ceiling",
        4:  "shelf",
        5:  "rack",
        6:  "box",
        7:  "pallet",
        8:  "forklift",
        9:  "vehicle",
        10: "npc_pedestrian",
        11: "npc_worker",
        12: "robot",
        13: "static_obstacle",
        14: "unknown_occupied",
    }

    # 路径关键词 → 语义ID
    PATH_RULES = [
        ("ground",    1), ("floor",    1), ("Ground",    1), ("Floor",    1),
        ("wall",      2), ("Wall",      2),
        ("ceiling",   3), ("Ceiling",   3), ("Roof",      3),
        ("shelf",     4), ("Shelf",     4), ("Shelving",  4),
        ("rack",      5), ("Rack",      5),
        ("box",       6), ("Box",       6), ("Carton",    6), ("crate",    6),
        ("pallet",    7), ("Pallet",    7),
        ("forklift",  8), ("Forklift",  8),
        ("vehicle",   9), ("Vehicle",   9), ("Car",       9), ("Truck",    9),
        ("Character", 10), ("character", 10), ("people",   10), ("People",  10),
        ("Worker",    11), ("worker",    11), ("construction", 11),
        ("robot",     12), ("Robot",     12), ("Carter",   12),
    ]

    @classmethod
    def classify(cls, prim_path: str, type_name: str = "") -> int:
        """返回语义标签ID"""
        path_lower = prim_path.lower()
        for keyword, label in cls.PATH_RULES:
            if keyword.lower() in path_lower:
                return label
        # 有几何体但未识别 → unknown_occupied
        if type_name in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule"):
            return 14  # unknown_occupied
        return 13  # static_obstacle

    @classmethod
    def get_class_name(cls, label: int) -> str:
        return cls.CLASSES.get(label, f"class_{label}")


# ---------------------------------------------------------------------------
# B. 场景 BBox 索引器
# ---------------------------------------------------------------------------
class SceneBBoxIndex:
    """
    遍历 USD Stage 构建所有可见 Mesh prim 的世界空间 AABB 索引。
    用于快速判断某个体素是否与某个 prim 重叠。
    """

    def __init__(self):
        self.entries = []  # [{"path", "type", "label", "min", "max"}, ...]

    def build(self, stage, skip_paths=None):
        """遍历 stage 构建索引"""
        skip_paths = skip_paths or []
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        count = 0
        for prim in stage.Traverse():
            if not prim.IsActive():
                continue

            prim_path = str(prim.GetPath())

            # 跳过指定路径前缀
            if any(prim_path.startswith(sp) for sp in skip_paths):
                continue

            type_name = prim.GetTypeName()

            # 只索引有几何形状的 prim
            if type_name not in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule",
                                 "Xform", "SkelRoot"):
                continue

            try:
                bbox = bbox_cache.ComputeWorldBound(prim)
                rng = bbox.ComputeAlignedRange()
                if rng.IsEmpty():
                    continue

                bmin = np.array(rng.GetMin())
                bmax = np.array(rng.GetMax())

                # 过滤极小的 prim（噪声）
                size = bmax - bmin
                if np.any(size < 0.01):
                    continue

                label = SemanticClassifier.classify(prim_path, type_name)

                self.entries.append({
                    "path": prim_path,
                    "type": type_name,
                    "label": label,
                    "min": bmin,
                    "max": bmax,
                })
                count += 1
            except Exception:
                continue

        print(f"[Voxel] BBox index built: {count} entries")
        return count

    def query_point(self, point: np.ndarray) -> list:
        """查询包含该点的所有 prim"""
        results = []
        for entry in self.entries:
            if np.all(point >= entry["min"]) and np.all(point <= entry["max"]):
                results.append(entry)
        return results


# ---------------------------------------------------------------------------
# C. 体素网格生成器
# ---------------------------------------------------------------------------
class VoxelGridGenerator:
    """
    以 ego 为中心生成体素网格，遍历每个体素查询场景内容。

    坐标系约定：
      ego 局部坐标系（Z-up）：
        X = 前方, Y = 左方, Z = 上方
        原点 = ego 摄像机位置

      体素网格索引 (i, j, k)：
        i ∈ [0, grid_x): 前后方向，i=grid_x/2 为 ego 位置
        j ∈ [0, grid_y): 左右方向，j=grid_y/2 为 ego 位置
        k ∈ [0, grid_z): 上下方向，k=0 为最低层
    """

    def __init__(self, grid_x: int, grid_y: int, grid_z: int,
                 voxel_size: float, ego_pos: np.ndarray, ego_yaw_deg: float):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.voxel_size = voxel_size
        self.ego_pos = ego_pos.copy()
        self.ego_yaw_deg = ego_yaw_deg

        # ego → world 旋转矩阵（绕 Z 轴旋转 yaw）
        yaw_rad = math.radians(ego_yaw_deg)
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        self.R_ego2world = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ])

        # 体素网格（uint16，0=free）
        self.grid = np.zeros((grid_x, grid_y, grid_z), dtype=np.uint16)

        # 体素中心在 ego 局部坐标系中的偏移
        # ego 在网格中心：i=grid_x/2, j=grid_y/2, k 从地面开始
        self.origin_offset = np.array([
            -grid_x / 2.0 * voxel_size,
            -grid_y / 2.0 * voxel_size,
            -ego_pos[2],  # k=0 对应世界 z=0（地面）
        ])

    def voxel_to_ego(self, i, j, k):
        """体素索引 → ego 局部坐标"""
        x = (i + 0.5) * self.voxel_size + self.origin_offset[0]
        y = (j + 0.5) * self.voxel_size + self.origin_offset[1]
        z = (k + 0.5) * self.voxel_size + self.origin_offset[2]
        return np.array([x, y, z])

    def ego_to_world(self, ego_point):
        """ego 局部坐标 → 世界坐标"""
        return self.R_ego2world @ ego_point + self.ego_pos

    def voxel_to_world(self, i, j, k):
        """体素索引 → 世界坐标"""
        return self.ego_to_world(self.voxel_to_ego(i, j, k))

    def generate_all_world_centers(self):
        """
        批量生成所有体素中心的世界坐标 (高效向量化版本)。
        返回 (grid_x * grid_y * grid_z, 3) 的数组。
        """
        # 生成 ego 局部坐标网格
        ix = np.arange(self.grid_x)
        iy = np.arange(self.grid_y)
        iz = np.arange(self.grid_z)

        # 体素中心在 ego 局部坐标
        cx = (ix + 0.5) * self.voxel_size + self.origin_offset[0]
        cy = (iy + 0.5) * self.voxel_size + self.origin_offset[1]
        cz = (iz + 0.5) * self.voxel_size + self.origin_offset[2]

        # 构建 3D meshgrid 并展平
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
        ego_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)  # (N, 3)

        # 批量旋转 + 平移：world = R @ ego + t
        world_points = (ego_points @ self.R_ego2world.T) + self.ego_pos  # (N, 3)

        return world_points

    def fill_from_bbox_index(self, bbox_index: SceneBBoxIndex):
        """
        高效填充：对每个 actor 的 BBox，找出与之重叠的体素并填充。
        比逐体素查询快几个数量级。
        """
        vs = self.voxel_size
        total_filled = 0

        for entry in bbox_index.entries:
            actor_min = entry["min"]
            actor_max = entry["max"]
            label = entry["label"]

            # 将 actor BBox 从世界坐标转换到体素索引范围
            # 先转到 ego 坐标，再转到体素索引
            # 简化：直接在世界坐标中计算体素索引范围

            # 世界坐标 → ego 坐标（逆变换）
            R_inv = self.R_ego2world.T
            t_inv = -R_inv @ self.ego_pos

            # BBox 8 个角点在世界坐标中
            corners_world = np.array([
                [actor_min[0], actor_min[1], actor_min[2]],
                [actor_max[0], actor_min[1], actor_min[2]],
                [actor_min[0], actor_max[1], actor_min[2]],
                [actor_max[0], actor_max[1], actor_min[2]],
                [actor_min[0], actor_min[1], actor_max[2]],
                [actor_max[0], actor_min[1], actor_max[2]],
                [actor_min[0], actor_max[1], actor_max[2]],
                [actor_max[0], actor_max[1], actor_max[2]],
            ])

            # 转到 ego 坐标
            corners_ego = (corners_world @ R_inv.T) + t_inv

            # 在 ego 坐标中的 AABB
            ego_min = corners_ego.min(axis=0)
            ego_max = corners_ego.max(axis=0)

            # 转换到体素索引
            i_min = int(np.floor((ego_min[0] - self.origin_offset[0]) / vs))
            i_max = int(np.ceil((ego_max[0] - self.origin_offset[0]) / vs))
            j_min = int(np.floor((ego_min[1] - self.origin_offset[1]) / vs))
            j_max = int(np.ceil((ego_max[1] - self.origin_offset[1]) / vs))
            k_min = int(np.floor((ego_min[2] - self.origin_offset[2]) / vs))
            k_max = int(np.ceil((ego_max[2] - self.origin_offset[2]) / vs))

            # 裁剪到网格范围
            i_min = max(0, i_min)
            i_max = min(self.grid_x, i_max)
            j_min = max(0, j_min)
            j_max = min(self.grid_y, j_max)
            k_min = max(0, k_min)
            k_max = min(self.grid_z, k_max)

            if i_min >= i_max or j_min >= j_max or k_min >= k_max:
                continue

            # 填充体素（高优先级标签覆盖低优先级，但不覆盖已有的动态标签）
            # 动态 NPC (10, 11) 优先级高于静态障碍物
            sub = self.grid[i_min:i_max, j_min:j_max, k_min:k_max]
            mask = (sub == 0) | (label >= 10 and sub < 10)
            if isinstance(mask, np.ndarray):
                sub[mask] = label
            else:
                if mask:
                    self.grid[i_min:i_max, j_min:j_max, k_min:k_max] = label

            filled = (i_max - i_min) * (j_max - j_min) * (k_max - k_min)
            total_filled += filled

        return total_filled

    def fill_with_physx(self, stage):
        """
        使用 PhysX overlap_box 精确查询动态物体。
        仅对已标记为 free 的体素进行查询（优化性能）。
        注意：需要 physics simulation 处于运行状态。
        """
        try:
            from omni.physx import get_physx_scene_query_interface
        except ImportError:
            print("[Voxel] PhysX not available, skipping dynamic query.")
            return 0

        physx_iface = get_physx_scene_query_interface()
        half = self.voxel_size / 2.0
        filled = 0

        # 只查询 free 且在合理高度范围内的体素
        free_indices = np.argwhere(self.grid == 0)
        total = len(free_indices)
        if total == 0:
            return 0

        print(f"[Voxel] PhysX querying {total} free voxels...")

        for idx, (i, j, k) in enumerate(free_indices):
            if idx % 50000 == 0 and idx > 0:
                print(f"  PhysX progress: {idx}/{total} ({idx*100//total}%)")

            world_pos = self.voxel_to_world(i, j, k)

            hits = []

            def on_hit(hit):
                hits.append(str(hit.rigid_body))
                return False  # 只要第一个

            try:
                physx_iface.overlap_box(
                    halfExtent=carb.Float3(half, half, half),
                    pos=carb.Float3(float(world_pos[0]), float(world_pos[1]), float(world_pos[2])),
                    rot=carb.Float4(0, 0, 0, 1),
                    reportFn=on_hit,
                    anyHit=True,
                )
            except Exception:
                continue

            if hits:
                label = SemanticClassifier.classify(hits[0])
                self.grid[i, j, k] = label
                filled += 1

        return filled

    def get_statistics(self) -> dict:
        """统计体素分布"""
        unique, counts = np.unique(self.grid, return_counts=True)
        total = self.grid.size
        stats = {}
        for label, count in zip(unique, counts):
            name = SemanticClassifier.get_class_name(int(label))
            stats[name] = {
                "label": int(label),
                "count": int(count),
                "ratio": round(count / total * 100, 2),
            }
        return stats


# ===========================================================================
# 主函数
# ===========================================================================
def main():
    print("\n" + "=" * 60)
    print("[Voxel] Multi-camera Voxel Grid Generator")
    print("=" * 60)

    ego_pos = np.array(args.ego_pos)
    total_voxels = args.grid_x * args.grid_y * args.grid_z
    range_x = args.grid_x * args.voxel_size
    range_y = args.grid_y * args.voxel_size
    range_z = args.grid_z * args.voxel_size

    print(f"\n[Config]")
    print(f"  Ego position:  ({ego_pos[0]:.1f}, {ego_pos[1]:.1f}, {ego_pos[2]:.1f})")
    print(f"  Ego heading:   {args.ego_yaw:.1f} deg")
    print(f"  Voxel size:    {args.voxel_size} m")
    print(f"  Grid:          {args.grid_x} x {args.grid_y} x {args.grid_z}")
    print(f"  Total voxels:  {total_voxels:,}")
    print(f"  Range:         {range_x:.0f}m x {range_y:.0f}m x {range_z:.0f}m")

    # 加载场景
    if args.scene_usd:
        print(f"\n[Step 1] Loading scene: {args.scene_usd}")
        ctx = omni.usd.get_context()
        ctx.open_stage(args.scene_usd)
        for _ in range(200):
            simulation_app.update()
            if ctx.get_stage().GetPrimAtPath("/World").IsValid():
                break
            import time as _t
            _t.sleep(0.05)
    else:
        print("\n[Step 1] Using current stage (no --scene_usd specified)")
        # 加载默认仓库场景
        assets_root = "D:/code/IsaacLab/Assets/Isaac/5.1"
        scene_usd = f"{assets_root}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
        if os.path.isfile(scene_usd):
            print(f"  Loading default: {scene_usd}")
            ctx = omni.usd.get_context()
            ctx.open_stage(scene_usd)
            for _ in range(300):
                simulation_app.update()
                if ctx.get_stage().GetPrimAtPath("/World").IsValid():
                    break
                import time as _t
                _t.sleep(0.05)

    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("[Voxel] ERROR: No stage loaded.", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)

    # 等待渲染稳定
    for _ in range(30):
        simulation_app.update()

    # 构建场景 BBox 索引
    print("\n[Step 2] Building scene BBox index...")
    bbox_index = SceneBBoxIndex()
    num_entries = bbox_index.build(stage, skip_paths=["/World/defaultGroundPlane"])
    print(f"  Indexed {num_entries} scene objects")

    # 打印各语义类别的 actor 数量
    label_counts = {}
    for entry in bbox_index.entries:
        name = SemanticClassifier.get_class_name(entry["label"])
        label_counts[name] = label_counts.get(name, 0) + 1
    for name, count in sorted(label_counts.items()):
        print(f"    {name}: {count}")

    # 生成体素网格
    print("\n[Step 3] Generating voxel grid...")
    t0 = time.time()
    generator = VoxelGridGenerator(
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        grid_z=args.grid_z,
        voxel_size=args.voxel_size,
        ego_pos=ego_pos,
        ego_yaw_deg=args.ego_yaw,
    )

    # BBox 填充（快速）
    filled = generator.fill_from_bbox_index(bbox_index)
    t1 = time.time()
    print(f"  BBox fill: {filled:,} voxel assignments in {t1-t0:.2f}s")

    # PhysX 填充（可选，较慢）
    if args.use_physx:
        print("\n[Step 3b] PhysX dynamic actor query...")
        physx_filled = generator.fill_with_physx(stage)
        print(f"  PhysX fill: {physx_filled:,} additional voxels")

    # 统计
    print("\n[Step 4] Voxel statistics:")
    stats = generator.get_statistics()
    for name, info in sorted(stats.items(), key=lambda x: -x[1]["count"]):
        print(f"  {name:20s}: {info['count']:>10,} ({info['ratio']:5.1f}%)")

    # 保存
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "voxel_data")
    os.makedirs(output_dir, exist_ok=True)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    grid_file = os.path.join(output_dir, f"voxel_grid_{timestamp_str}.npy")
    meta_file = os.path.join(output_dir, f"voxel_meta_{timestamp_str}.json")

    # 保存体素网格
    np.save(grid_file, generator.grid)

    # 保存元数据
    meta = {
        "timestamp": timestamp_str,
        "ego_position": ego_pos.tolist(),
        "ego_yaw_deg": args.ego_yaw,
        "voxel_size_m": args.voxel_size,
        "grid_shape": [args.grid_x, args.grid_y, args.grid_z],
        "range_m": [range_x, range_y, range_z],
        "origin_offset": generator.origin_offset.tolist(),
        "rotation_matrix_ego2world": generator.R_ego2world.tolist(),
        "semantic_classes": SemanticClassifier.CLASSES,
        "statistics": stats,
        "coordinate_convention": {
            "ego_frame": "X=forward, Y=left, Z=up (Z-up, right-hand)",
            "voxel_index": "i=X(forward), j=Y(left), k=Z(up)",
            "ego_at_grid_center": f"i={args.grid_x//2}, j={args.grid_y//2}",
        },
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"[Voxel] Generation complete!")
    print(f"  Grid file:  {grid_file} ({os.path.getsize(grid_file)/1024/1024:.1f} MB)")
    print(f"  Meta file:  {meta_file}")
    print(f"  Grid shape: {generator.grid.shape}")
    print(f"  dtype:      {generator.grid.dtype}")
    print(f"{'='*60}")

    simulation_app.close()


if __name__ == "__main__":
    main()
