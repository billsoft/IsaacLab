"""诊断仓库场景：查找所有碰撞体的位置范围。

运行：isaaclab.bat -p projects/stereo_voxel/scripts/diagnose_scene.py
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 640, "height": 480})

import os
import carb
import numpy as np
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

# 加载场景
ASSETS_ROOT = "D:/code/IsaacLab/Assets/Isaac/5.1"
SCENE_USD = f"{ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.append(SCENE_USD)
for _ in range(30):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()
mpu = UsdGeom.GetStageMetersPerUnit(stage)
print(f"metersPerUnit = {mpu}")
print(f"Stage unit: {'centimeters' if abs(mpu - 0.01) < 0.001 else 'meters' if abs(mpu - 1.0) < 0.01 else f'{mpu}'}")

# 1. 遍历所有 prim，找有碰撞属性的
print("\n=== Collision Prims ===")
collision_prims = []
all_prims = []
for prim in stage.Traverse():
    all_prims.append(prim.GetPath().pathString)
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        path = prim.GetPath().pathString
        # 获取世界变换
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            xf = xformable.ComputeLocalToWorldTransform(0)
            pos = xf.ExtractTranslation()
            collision_prims.append((path, pos))

print(f"Total prims: {len(all_prims)}")
print(f"Collision prims: {len(collision_prims)}")

# 统计碰撞体位置范围
if collision_prims:
    positions = np.array([[p[1][0], p[1][1], p[1][2]] for p in collision_prims])
    print(f"\nCollision prim position ranges (stage units):")
    print(f"  X: [{positions[:,0].min():.1f}, {positions[:,0].max():.1f}]")
    print(f"  Y: [{positions[:,1].min():.1f}, {positions[:,1].max():.1f}]")
    print(f"  Z: [{positions[:,2].min():.1f}, {positions[:,2].max():.1f}]")
    print(f"\nIn meters:")
    print(f"  X: [{positions[:,0].min()*mpu:.2f}, {positions[:,0].max()*mpu:.2f}]")
    print(f"  Y: [{positions[:,1].min()*mpu:.2f}, {positions[:,1].max()*mpu:.2f}]")
    print(f"  Z: [{positions[:,2].min()*mpu:.2f}, {positions[:,2].max()*mpu:.2f}]")

    # 打印前20个碰撞体
    print(f"\nFirst 30 collision prims:")
    for path, pos in collision_prims[:30]:
        short = path.split("/")[-1] if "/" in path else path
        parent = "/".join(path.split("/")[:-1])
        print(f"  {pos[0]:8.1f} {pos[1]:8.1f} {pos[2]:8.1f}  {short:<30s}  {parent}")

# 2. 用 PhysX 做一个大范围 overlap 查询
print("\n=== PhysX Large Overlap Query ===")
from omni.physx import get_physx_scene_query_interface
physx_sqi = get_physx_scene_query_interface()

# 用 step 初始化 PhysX
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(10):
    simulation_app.update()
timeline.pause()
simulation_app.update()

# 查询整个仓库范围 (100m x 100m x 10m in meters -> stage units)
half_x = 50.0 / mpu  # 50m in stage units
half_y = 50.0 / mpu
half_z = 5.0 / mpu
center_x = 0.0
center_y = 0.0
center_z = 1.5 / mpu  # 1.5m height

big_hits = []
def on_big_hit(hit):
    big_hits.append(hit.rigid_body)
    return True

physx_sqi.overlap_box(
    carb.Float3(half_x, half_y, half_z),
    carb.Float3(center_x, center_y, center_z),
    carb.Float4(0, 0, 0, 1),
    on_big_hit,
    False,
)
print(f"Large overlap (100m x 100m x 10m centered at origin): {len(big_hits)} hits")
unique_hits = sorted(set(big_hits))
for h in unique_hits[:50]:
    print(f"  {h}")

# 3. 测试几个特定点
print("\n=== Point Overlap Tests ===")
test_points_m = [
    (0, 0, 0.05),     # 地面
    (0, 0, 1.0),      # 1m height
    (5, 0, 0.05),     # 5m away
    (10, 0, 0.05),    # 10m away
    (-10, 0, 0.05),
    (0, 10, 0.05),
    (0, -10, 0.05),
]
for mx, my, mz in test_points_m:
    sx, sy, sz = mx/mpu, my/mpu, mz/mpu
    hits = []
    def on_pt_hit(hit, _hits=hits):
        _hits.append(hit.rigid_body)
        return True
    physx_sqi.overlap_box(
        carb.Float3(25.0, 25.0, 25.0),  # 0.25m cube in stage units if cm, or 25cm
        carb.Float3(sx, sy, sz),
        carb.Float4(0, 0, 0, 1),
        on_pt_hit,
        False,
    )
    hit_names = [h.split("/")[-1] for h in set(hits)]
    print(f"  ({mx:6.1f}, {my:6.1f}, {mz:4.2f})m → stage({sx:8.1f}, {sy:8.1f}, {sz:6.1f}) → {len(hits)} hits: {hit_names[:5]}")

simulation_app.close()
