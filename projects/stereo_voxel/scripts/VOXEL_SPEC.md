# 体素空间规格定义

## 1. 相机参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 分辨率 | 1280 x 1080 | 宽 x 高 |
| 像素尺寸 | 2.7 um | 正方形像素 |
| 焦距 | 1.75 mm | |
| 基线 | 80 mm | 双目瞳距 |
| 对角 FOV | 157.20° | |
| 水平 FOV (1280 方向) | 115.63° (标称) / **113.1°** (等距模型计算) | |
| 垂直 FOV (1080 方向) | 96.8° (标称) / **95.5°** (等距模型计算) | |
| 吊装高度 | **3.0 m** | 垂直向下俯视 |

> 注：水平/垂直 FOV 在原始规格中可能标注互换（宽边 > 窄边但水平 FOV < 垂直 FOV），
> 以等距模型 `theta = r / f` 重新计算为准。

## 2. 地面覆盖范围计算

相机垂直向下俯视，高度 h = 3.0m。

鱼眼等距投影模型中，入射角 theta 与像面距离 r 的关系：`r = f * theta`，
而地面覆盖半径与入射角的关系：`ground_half = h * tan(theta)`。

### 窄边（1080 方向）—— 取有效 FOV = 90°

```
半角 theta_narrow = 45°
ground_half_narrow = 3.0 * tan(45°) = 3.0 m
ground_narrow = 2 * 3.0 = 6.0 m
```

取有效 FOV 90° 而非满幅 95.5°，原因：
- 鱼眼边缘畸变严重，体素映射精度下降
- 留约 3° 边缘余量，保证中心区域标注质量

### 宽边（1280 方向）—— 取有效 FOV = 100°

```
半角 theta_wide = 50°
ground_half_wide = 3.0 * tan(50°) = 3.0 * 1.1918 = 3.575 m
ground_wide = 2 * 3.575 = 7.15 m ≈ 7.2 m
```

取有效 FOV 100° 而非满幅 113.1°，同样留边缘余量。

### 修正后的覆盖矩形

| 方向 | 满幅 FOV | 有效 FOV | 覆盖距离 | 精确值 |
|------|---------|---------|---------|--------|
| 窄边 (1080) | 95.5° | 90° | **6.0 m** | 6.000 m |
| 宽边 (1280) | 113.1° | 100° | **7.2 m** | 7.150 m |

## 3. 体素网格定义

### 3.1 网格尺寸

| 维度 | 物理范围 | 格子数 | 分辨率 |
|------|---------|--------|--------|
| X (宽边方向) | 7.2 m | **72** | 0.1 m |
| Y (窄边方向) | 6.0 m | **60** | 0.1 m |
| Z (高度方向) | 3.2 m | **32** | 0.1 m |

**总体素数**：72 x 60 x 32 = **138,240 个**

### 3.2 高度分层

相机吊装高度 3.0m，体素 Z 范围以**地面为零点**：

```
Z 范围：-0.7m ~ +2.5m（共 3.2m = 32 格）

  +2.5m ┬─ 格子 31（最高层，接近相机）
        │  ... 25 格地上空间
  0.0m  ┼─ 格子 7 ← 地面分界（地面 = 格子 6 和 7 之间）
        │  ... 7 格地下空间
  -0.7m ┴─ 格子 0（最低层，地下斜坡/凹槽）
```

- **地上**：25 格（0.0m ~ +2.5m），覆盖行人全身高度（~1.8m）+ 头顶余量
- **地下**：7 格（-0.7m ~ 0.0m），覆盖斜坡、凹槽、地面厚度
- **地面分界索引**：`Z_GROUND_INDEX = 7`（即 voxel_z = 7 对应世界 z = 0）

### 3.3 体素局部坐标系

原点定义：**相机正下方地面投影点** = 体素网格中心。

```
体素局部坐标系（右手系）：

        +Z (向上)
         |
         |    +Y (窄边方向)
         |   /
         |  /
         | /
         +--------→ +X (宽边方向)

原点 = 地面中心点 = (X=36, Y=30, Z=7) 对应的格子
```

体素索引 (i, j, k) 到局部米制坐标的转换：

```python
VOXEL_SIZE = 0.1  # 米

# 网格中心索引
CENTER_X = 36  # 72 / 2
CENTER_Y = 30  # 60 / 2
CENTER_Z = 7   # 地面分界

def voxel_to_local(i, j, k):
    """体素索引 → 局部坐标（米），原点在地面中心。"""
    x = (i - CENTER_X + 0.5) * VOXEL_SIZE
    y = (j - CENTER_Y + 0.5) * VOXEL_SIZE
    z = (k - CENTER_Z + 0.5) * VOXEL_SIZE
    return (x, y, z)

# 示例：
# voxel (0, 0, 0)   → (-3.55, -2.95, -0.65) m  左下后地下角
# voxel (36, 30, 7)  → (0.05, 0.05, 0.05) m     地面中心偏一点
# voxel (71, 59, 31) → (3.55, 2.95, 2.45) m      右上前最高角
```

### 3.4 局部坐标 → 世界坐标转换

相机在世界坐标系中的位置 `cam_pos = (cx, cy, cz)`，姿态四元数 `cam_quat`。
相机垂直向下俯视时，地面投影点 = `(cx, cy, 0)`（假设地面 z=0）。

```python
import numpy as np
from scipy.spatial.transform import Rotation

def local_to_world(local_xyz, cam_pos, cam_quat):
    """体素局部坐标 → 世界坐标。

    Args:
        local_xyz: (N, 3) 局部坐标数组
        cam_pos: (3,) 相机世界位置 [x, y, z]
        cam_quat: (4,) 相机四元数 [w, x, y, z]

    Returns:
        (N, 3) 世界坐标
    """
    # 相机地面投影点 = 体素网格原点的世界坐标
    ground_proj = np.array([cam_pos[0], cam_pos[1], 0.0])

    # 相机绕 Z 轴的偏航角（yaw）决定体素 XY 平面的朝向
    # 纯俯视相机只有 yaw 自由度影响体素方向
    r = Rotation.from_quat([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]])  # scipy 用 xyzw
    euler = r.as_euler('ZYX', degrees=False)
    yaw = euler[0]  # 绕世界 Z 轴的旋转

    # 构建 2D 旋转矩阵（只旋转 XY，Z 不变）
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot_2d = np.array([[cos_y, -sin_y],
                        [sin_y,  cos_y]])

    world = np.zeros_like(local_xyz)
    world[:, :2] = local_xyz[:, :2] @ rot_2d.T  # 旋转 XY
    world[:, 2] = local_xyz[:, 2]                # Z 保持不变
    world += ground_proj                          # 平移到世界位置

    return world
```

### 3.5 遍历体素获取世界坐标

```python
# 预计算所有体素中心的局部坐标
NX, NY, NZ = 72, 60, 32

# 生成所有体素中心的局部坐标 (72*60*32 = 138240 个点)
ix = np.arange(NX)
iy = np.arange(NY)
iz = np.arange(NZ)
gi, gj, gk = np.meshgrid(ix, iy, iz, indexing='ij')  # (72, 60, 32)

local_centers = np.stack([
    (gi - CENTER_X + 0.5) * VOXEL_SIZE,
    (gj - CENTER_Y + 0.5) * VOXEL_SIZE,
    (gk - CENTER_Z + 0.5) * VOXEL_SIZE,
], axis=-1)  # shape (72, 60, 32, 3)

# 拍照时将局部坐标转为世界坐标
flat_local = local_centers.reshape(-1, 3)          # (138240, 3)
flat_world = local_to_world(flat_local, cam_pos, cam_quat)  # (138240, 3)
world_centers = flat_world.reshape(NX, NY, NZ, 3)  # (72, 60, 32, 3)
```

## 4. 体素数据结构

每帧采集生成一个体素张量，包含两个通道：

```python
# 每帧输出
voxel_semantic = np.zeros((NX, NY, NZ), dtype=np.uint8)   # 语义类别 (0-17, 255=unobserved)
voxel_instance = np.zeros((NX, NY, NZ), dtype=np.int32)   # 物体实例 ID (0=无)

# 填充逻辑（伪代码）：
for i, j, k in all_voxels:
    world_pos = world_centers[i, j, k]
    hit_prim = raycast_or_overlap_query(world_pos, voxel_size=0.1)
    if hit_prim is None:
        voxel_semantic[i, j, k] = FREE  # 0 = 空气
        voxel_instance[i, j, k] = 0
    else:
        obj_type = get_object_type(hit_prim)
        voxel_semantic[i, j, k] = lookup_class_id(obj_type)
        voxel_instance[i, j, k] = get_instance_id(hit_prim)
```

## 5. 与双目图像的同步

每个数据帧包含：

| 文件 | 内容 | 格式 |
|------|------|------|
| `left/frame_NNNNNN.png` | 左眼 RGB | 1280x1080 PNG |
| `right/frame_NNNNNN.png` | 右眼 RGB | 1280x1080 PNG |
| `voxel/frame_NNNNNN_semantic.npz` | 语义体素 | (72,60,32) uint8 |
| `voxel/frame_NNNNNN_instance.npz` | 实例体素 | (72,60,32) int32 |
| `meta/frame_NNNNNN.json` | 帧元数据 | JSON |

帧元数据示例：
```json
{
  "frame_id": 42,
  "timestamp_sec": 4.2,
  "camera_pos": [3.0, 1.5, 3.0],
  "camera_quat": [0.707, 0.0, 0.0, 0.707],
  "voxel_origin_world": [3.0, 1.5, 0.0],
  "voxel_size": 0.1,
  "voxel_shape": [72, 60, 32],
  "z_ground_index": 7
}
```
