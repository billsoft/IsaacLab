"""体素地面层诊断脚本
================================
分析体素数据中地面层(z=7)、地上层、地下层的占用情况，
采样关键点帮助定位地面缺失问题。

用法：python diag_voxel.py [frame_path]
  默认：D:/code/IsaacLab/projects/stereo_voxel/output/voxel/frame_000000
"""
import sys
import os
import json
import numpy as np

# 默认路径
DEFAULT_PREFIX = "D:/code/IsaacLab/projects/stereo_voxel/output/voxel/frame_000000"
prefix = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PREFIX

# 加载数据
sem_path = f"{prefix}_semantic.npz"
meta_path = prefix.replace("/voxel/", "/meta/") + ".json"

print(f"=== 体素地面诊断 ===")
print(f"数据文件: {sem_path}")

d = np.load(sem_path)
g = d['data']  # shape: (NX, NY, NZ) = (72, 60, 32)
NX, NY, NZ = g.shape
print(f"网格形状: {NX}x{NY}x{NZ} = {NX*NY*NZ:,} 体素")

# 读取 meta
meta = None
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"相机位置: {meta.get('camera_pos')}")
    print(f"体素原点(世界): {meta.get('voxel_origin_world')}")

CLASS_NAMES = {
    0: "free", 1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck",
    5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist",
    9: "road", 10: "parking", 11: "sidewalk", 12: "other-ground",
    13: "building", 14: "fence", 15: "vegetation", 16: "trunk",
    17: "general-object", 255: "unobserved",
}

# 体素参数
VOXEL_SIZE = 0.1
Z_GROUND_INDEX = 7
CENTER_X, CENTER_Y = NX // 2, NY // 2

def voxel_to_local(i, j, k):
    x = (i - CENTER_X + 0.5) * VOXEL_SIZE
    y = (j - CENTER_Y + 0.5) * VOXEL_SIZE
    z = (k - Z_GROUND_INDEX + 0.5) * VOXEL_SIZE
    return x, y, z

# ============================================================
# 1. 总体类别分布
# ============================================================
print("\n" + "=" * 60)
print("1. 总体类别分布")
print("=" * 60)
unique, counts = np.unique(g, return_counts=True)
for u, c in zip(unique, counts):
    name = CLASS_NAMES.get(int(u), f"unknown-{u}")
    print(f"  [{int(u):3d}] {name:<20s}: {int(c):7d}")
total_occ = int(np.sum((g > 0) & (g != 255)))
print(f"\n  占用: {total_occ} / {g.size} ({total_occ/g.size*100:.1f}%)")

# ============================================================
# 2. 逐层分析 (Z 轴)
# ============================================================
print("\n" + "=" * 60)
print("2. 逐层 Z 分析 (Z_GROUND_INDEX=7, z=0.05m)")
print("=" * 60)
print(f"  {'层':>3s}  {'z(m)':>7s}  {'free':>6s}  {'occup':>6s}  {'unobs':>6s}  {'occ%':>6s}  {'主类别'}")
print(f"  {'---':>3s}  {'-----':>7s}  {'----':>6s}  {'-----':>6s}  {'-----':>6s}  {'----':>6s}  {'------'}")

layer_size = NX * NY
for k in range(NZ):
    layer = g[:, :, k]
    z_m = (k - Z_GROUND_INDEX + 0.5) * VOXEL_SIZE
    n_free = int(np.sum(layer == 0))
    n_unobs = int(np.sum(layer == 255))
    n_occ = layer_size - n_free - n_unobs
    occ_pct = n_occ / layer_size * 100 if layer_size > 0 else 0

    # 找该层主要占用类别
    occ_mask = (layer > 0) & (layer != 255)
    if np.any(occ_mask):
        vals, cnts = np.unique(layer[occ_mask], return_counts=True)
        top_idx = np.argmax(cnts)
        top_name = CLASS_NAMES.get(int(vals[top_idx]), f"?{vals[top_idx]}")
        top_str = f"{top_name}({int(cnts[top_idx])})"
    else:
        top_str = "-"

    marker = " <<<< GROUND" if k == Z_GROUND_INDEX else ""
    print(f"  {k:3d}  {z_m:+7.2f}  {n_free:6d}  {n_occ:6d}  {n_unobs:6d}  {occ_pct:5.1f}%  {top_str}{marker}")

# ============================================================
# 3. 地面层 (z=7) 详细空间分析
# ============================================================
print("\n" + "=" * 60)
print("3. 地面层 (z=7) XY 占用热力图")
print("=" * 60)

ground = g[:, :, Z_GROUND_INDEX]
ground_occ = (ground > 0) & (ground != 255)
ground_free = ground == 0

print(f"  地面层总体: 占用={int(ground_occ.sum())}, free={int(ground_free.sum())}, "
      f"unobs={int((ground==255).sum())}")
print(f"  占用率: {ground_occ.sum()/layer_size*100:.1f}%")

# 按象限分析
q_names = ["X+Y+", "X-Y+", "X-Y-", "X+Y-"]
q_slices = [
    (slice(CENTER_X, NX), slice(CENTER_Y, NY)),  # X+ Y+
    (slice(0, CENTER_X), slice(CENTER_Y, NY)),    # X- Y+
    (slice(0, CENTER_X), slice(0, CENTER_Y)),     # X- Y-
    (slice(CENTER_X, NX), slice(0, CENTER_Y)),    # X+ Y-
]
print(f"\n  象限占用分析 (中心 = voxel[{CENTER_X},{CENTER_Y}] = 世界原点投影):")
for name, (sx, sy) in zip(q_names, q_slices):
    q = ground[sx, sy]
    q_occ = int(np.sum((q > 0) & (q != 255)))
    q_free = int(np.sum(q == 0))
    q_total = q.size
    print(f"    {name}: 占用={q_occ:5d}, free={q_free:5d}, total={q_total}, occ%={q_occ/q_total*100:.1f}%")

# X 轴切片（固定 Y=CENTER_Y，遍历 X）
print(f"\n  地面层 X 切片 (Y={CENTER_Y}, 经过中心的一行):")
row = ground[:, CENTER_Y]
for i in range(0, NX, 4):
    val = int(row[i])
    name = CLASS_NAMES.get(val, f"?{val}")
    x_m = (i - CENTER_X + 0.5) * VOXEL_SIZE
    print(f"    x[{i:2d}] x={x_m:+5.2f}m: {name}")

# Y 轴切片
print(f"\n  地面层 Y 切片 (X={CENTER_X}, 经过中心的一列):")
col = ground[CENTER_X, :]
for j in range(0, NY, 4):
    val = int(col[j])
    name = CLASS_NAMES.get(val, f"?{val}")
    y_m = (j - CENTER_Y + 0.5) * VOXEL_SIZE
    print(f"    y[{j:2d}] y={y_m:+5.2f}m: {name}")

# ============================================================
# 4. 地面边界分析 — 找 free/occupied 分界线
# ============================================================
print("\n" + "=" * 60)
print("4. 地面层 free/occupied 边界分析")
print("=" * 60)

# 找每行(固定X)中，第一个 occupied 和最后一个 occupied 的 Y
print(f"  每行(X方向) occupied Y 范围:")
has_boundary = False
for i in range(0, NX, 6):
    row_occ = np.where(ground_occ[i, :])[0]
    x_m = (i - CENTER_X + 0.5) * VOXEL_SIZE
    if len(row_occ) > 0:
        y_min, y_max = row_occ[0], row_occ[-1]
        y_min_m = (y_min - CENTER_Y + 0.5) * VOXEL_SIZE
        y_max_m = (y_max - CENTER_Y + 0.5) * VOXEL_SIZE
        coverage = len(row_occ) / NY * 100
        print(f"    X[{i:2d}] x={x_m:+5.2f}m: Y=[{y_min}..{y_max}] "
              f"y=[{y_min_m:+.2f}..{y_max_m:+.2f}]m, 覆盖={coverage:.0f}%")
        if coverage < 90:
            has_boundary = True
    else:
        print(f"    X[{i:2d}] x={x_m:+5.2f}m: 全 free")

# 同理，每列(固定Y)中 occupied X 范围
print(f"\n  每列(Y方向) occupied X 范围:")
for j in range(0, NY, 6):
    col_occ = np.where(ground_occ[:, j])[0]
    y_m = (j - CENTER_Y + 0.5) * VOXEL_SIZE
    if len(col_occ) > 0:
        x_min, x_max = col_occ[0], col_occ[-1]
        x_min_m = (x_min - CENTER_X + 0.5) * VOXEL_SIZE
        x_max_m = (x_max - CENTER_X + 0.5) * VOXEL_SIZE
        coverage = len(col_occ) / NX * 100
        print(f"    Y[{j:2d}] y={y_m:+5.2f}m: X=[{x_min}..{x_max}] "
              f"x=[{x_min_m:+.2f}..{x_max_m:+.2f}]m, 覆盖={coverage:.0f}%")
    else:
        print(f"    Y[{j:2d}] y={y_m:+5.2f}m: 全 free")

# ============================================================
# 5. Z=6 vs Z=7 vs Z=8 对比 (地面上下层)
# ============================================================
print("\n" + "=" * 60)
print("5. 地面 ±1 层对比 (z=6, z=7, z=8)")
print("=" * 60)

for k in [6, 7, 8]:
    layer = g[:, :, k]
    z_m = (k - Z_GROUND_INDEX + 0.5) * VOXEL_SIZE
    occ = int(np.sum((layer > 0) & (layer != 255)))
    free = int(np.sum(layer == 0))
    unobs = int(np.sum(layer == 255))
    label = "地面下" if k < Z_GROUND_INDEX else ("地面" if k == Z_GROUND_INDEX else "地面上")
    print(f"  z={k} ({z_m:+.2f}m) [{label}]: occ={occ}, free={free}, unobs={unobs}, "
          f"occ%={occ/layer_size*100:.1f}%")

    # XY 统计
    occ_mask = (layer > 0) & (layer != 255)
    if np.any(occ_mask):
        occ_coords = np.argwhere(occ_mask)
        x_range = (occ_coords[:, 0].min(), occ_coords[:, 0].max())
        y_range = (occ_coords[:, 1].min(), occ_coords[:, 1].max())
        print(f"    occupied X范围: [{x_range[0]}..{x_range[1]}], Y范围: [{y_range[0]}..{y_range[1]}]")

# ============================================================
# 6. PhysX 边界效应分析
# ============================================================
print("\n" + "=" * 60)
print("6. PhysX 边界效应分析")
print("=" * 60)

# z=7 体素中心在 z=+0.05m，fine_half=0.05m → PhysX box 下边界恰好在 z=0
# 如果地面平面在 z=0，PhysX overlap_box 的边界恰好接触，可能不稳定
print(f"  Z_GROUND_INDEX = {Z_GROUND_INDEX}")
print(f"  z=7 体素中心 = {(Z_GROUND_INDEX - Z_GROUND_INDEX + 0.5) * VOXEL_SIZE:+.3f}m")
print(f"  fine_half = {VOXEL_SIZE/2:.3f}m")
print(f"  z=7 PhysX box 下边界 = {0.05 - 0.05:+.3f}m (恰好在地面 z=0)")
print(f"  z=6 体素中心 = {(6 - Z_GROUND_INDEX + 0.5) * VOXEL_SIZE:+.3f}m")
print(f"  z=6 PhysX box 上边界 = {-0.05 + 0.05:+.3f}m (恰好在地面 z=0)")
print()

# 比较 z=7 的 occupied 区域是否是 z=6 的子集（即 z=7 边界不稳定导致部分丢失）
if NZ > 8:
    z6_occ = (g[:, :, 6] > 0) & (g[:, :, 6] != 255)
    z7_occ = (g[:, :, 7] > 0) & (g[:, :, 7] != 255)
    z8_occ = (g[:, :, 8] > 0) & (g[:, :, 8] != 255)

    z7_only = z7_occ & ~z6_occ  # z=7 有但 z=6 没有
    z6_only = z6_occ & ~z7_occ  # z=6 有但 z=7 没有
    both = z6_occ & z7_occ

    print(f"  z=6 occupied: {int(z6_occ.sum())}")
    print(f"  z=7 occupied: {int(z7_occ.sum())}")
    print(f"  z=8 occupied: {int(z8_occ.sum())}")
    print(f"  z=6∩z=7 (两层都有): {int(both.sum())}")
    print(f"  z=6 only (z=7缺失): {int(z6_only.sum())}")
    print(f"  z=7 only (z=6缺失): {int(z7_only.sum())}")

    if z6_only.sum() > 0:
        missing_coords = np.argwhere(z6_only)
        print(f"\n  z=7 缺失但 z=6 有的位置采样 (前10):")
        for idx in range(min(10, len(missing_coords))):
            i, j = missing_coords[idx]
            x_m, y_m, _ = voxel_to_local(i, j, 7)
            cls_z6 = CLASS_NAMES.get(int(g[i, j, 6]), "?")
            cls_z7 = CLASS_NAMES.get(int(g[i, j, 7]), "?")
            print(f"    [{i:2d},{j:2d}] world=({x_m:+.2f},{y_m:+.2f})m: "
                  f"z=6→{cls_z6}, z=7→{cls_z7}")

# ============================================================
# 7. 建议修复方案
# ============================================================
print("\n" + "=" * 60)
print("7. 诊断结论与修复建议")
print("=" * 60)

z7_total_occ = int(((g[:, :, 7] > 0) & (g[:, :, 7] != 255)).sum())
z7_expected = NX * NY  # 理想情况下地面层应该全部是 occupied（地板）
coverage_pct = z7_total_occ / z7_expected * 100

print(f"  地面层(z=7) 覆盖率: {coverage_pct:.1f}% ({z7_total_occ}/{z7_expected})")

if coverage_pct < 90:
    print(f"\n  ⚠ 地面层覆盖不足!")
    print(f"  根因: PhysX overlap_box 的 z=7 体素下边界恰好在 z=0.0m (地面平面)")
    print(f"  PhysX 对恰好接触边界的碰撞检测不稳定 → 部分体素 miss")
    print()
    print(f"  修复方案 A: 微调地面层体素中心下移 epsilon")
    print(f"    在 fill_voxel_grid() 中，对 z=Z_GROUND_INDEX 层的")
    print(f"    world_centers_stage z坐标减少 0.001m (1mm)，确保 box 与地面重叠")
    print()
    print(f"  修复方案 B: 增大地面层 fine_half")
    print(f"    对 z=Z_GROUND_INDEX 层使用 fine_half * 1.1，增加 5mm 余量")
    print()
    print(f"  修复方案 C: 后处理填充")
    print(f"    如果 z=6 (地下层) 有 occupied，且 z=7 (地面层) 对应位置 free，")
    print(f"    则将 z=7 填充为 other-ground(12)")
else:
    print(f"\n  ✓ 地面层覆盖正常")

print("\n=== 诊断完成 ===")
