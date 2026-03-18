# 双目鱼眼 + 体素标注 数据采集系统 v2 设计

## 1. 设计目标

在 Isaac Sim 仓库场景中，同步采集：
- 双目鱼眼图像对（左/右 RGB）
- 对应的 3D 语义体素 ground truth（72x60x32, 0.1m 分辨率）

用于训练语义占用网格 (Semantic Occupancy) 预测网络。

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     stereo_voxel_capture.py                     │
│                      （新主脚本，统一入口）                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 1: 初始化                                                 │
│  ├─ SimulationApp 创建                                           │
│  ├─ NPC 扩展加载                                                 │
│  ├─ 场景预加载 (open_stage)                                      │
│  └─ IRA NPC 设置 (SimulationManager)                            │
│                                                                 │
│  PHASE 2: 相机创建                                               │
│  ├─ /World/StereoRig Xform                                      │
│  ├─ UsdGeom.Camera x2 + ApplyAPI(FthetaAPI)                    │
│  └─ Replicator render_product + annotator                       │
│                                                                 │
│  PHASE 3: 体素系统初始化                                          │
│  ├─ 预计算 138,240 个体素中心局部坐标                              │
│  └─ PhysX overlap query 初始化                                   │
│                                                                 │
│  PHASE 4: 主循环                                                 │
│  │                                                               │
│  │  每帧 (timeline.play 状态下):                                  │
│  │  ├─ simulation_app.update()                                   │
│  │  ├─ 每 N 步 → 冻结 + 采集:                                    │
│  │  │   ├─ timeline.pause()          ← 冻结世界                  │
│  │  │   ├─ render 2 帧等管线刷新                                  │
│  │  │   ├─ annotator.get_data()      ← 双目图像                  │
│  │  │   ├─ 读取相机世界坐标/姿态                                   │
│  │  │   ├─ 体素遍历 + 世界坐标转换                                 │
│  │  │   ├─ PhysX overlap 查询填充体素                              │
│  │  │   ├─ 异步保存 (图像 + 体素 + 元数据)                         │
│  │  │   └─ timeline.play()           ← 恢复世界                  │
│  │  └─ 非采集帧 → 正常 update                                    │
│  │                                                               │
│  └─ GUI: Play/Stop 控制录制                                      │
│     Headless: 自动采集 N 帧后退出                                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  依赖模块（import）                                              │
│  ├─ semantic_classes.py    ← 语义映射 lookup_class_id()          │
│  ├─ voxel_grid.py          ← 体素坐标/遍历/填充（新建）           │
│  └─ VOXEL_SPEC.md          ← 参数定义文档                        │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 文件结构

```
projects/stereo_voxel/
├── scripts/
│   ├── test_single_camera.py      # 保留，单目测试参考
│   ├── test_stereo_pair.py        # 保留，双目+NPC 参考（不改动）
│   ├── inspect_scene_objects.py   # 保留，场景物体扫描
│   ├── semantic_classes.py        # 保留，语义分类 + 映射表
│   ├── voxel_grid.py              # 【新建】体素网格核心模块
│   ├── stereo_voxel_capture.py    # 【新建】v2 主脚本
│   ├── VOXEL_SPEC.md              # 体素规格文档
│   └── DESIGN_v2.md               # 本文件
├── output/
│   ├── left/                      # 左眼图像
│   ├── right/                     # 右眼图像
│   ├── voxel/                     # 体素数据（新增）
│   ├── meta/                      # 帧元数据（新增）
│   ├── calibration.json           # 相机标定
│   └── scene_object_types.txt     # 场景物体清单
└── PLAN.md                        # 项目总计划
```

## 4. 新建模块设计

### 4.1 voxel_grid.py — 体素网格核心

纯 Python/NumPy 模块，不依赖 Isaac Sim，可单独测试。

```python
class VoxelGrid:
    """语义体素网格。

    坐标系: 以相机地面投影点为原点
    - X: 宽边方向 (1280px)，72 格，±3.6m
    - Y: 窄边方向 (1080px)，60 格，±3.0m
    - Z: 高度方向，32 格，-0.7m ~ +2.5m
    """

    NX, NY, NZ = 72, 60, 32
    VOXEL_SIZE = 0.1  # 米
    Z_MIN = -0.7      # 地下最低点
    Z_MAX = 2.5       # 地上最高点
    Z_GROUND_INDEX = 7  # 地面分界索引

    def __init__(self):
        self.semantic = np.full((self.NX, self.NY, self.NZ), 255, dtype=np.uint8)  # 255=unobserved
        self.instance = np.zeros((self.NX, self.NY, self.NZ), dtype=np.int32)
        self._local_centers = self._precompute_centers()

    def _precompute_centers(self) -> np.ndarray:
        """预计算所有体素中心的局部坐标 (NX, NY, NZ, 3)。"""
        ...

    def get_world_centers(self, cam_pos, cam_yaw) -> np.ndarray:
        """局部坐标 → 世界坐标，返回 (NX, NY, NZ, 3)。"""
        ...

    def fill_from_physx(self, world_centers, query_fn, classify_fn):
        """用 PhysX overlap 查询填充体素。

        Args:
            world_centers: (NX*NY*NZ, 3) 世界坐标
            query_fn: overlap_box(center, half_extent) → list[prim]
            classify_fn: prim → (semantic_id, instance_id)
        """
        ...

    def save(self, path_prefix):
        """保存为 .npz 文件。"""
        np.savez_compressed(f"{path_prefix}_semantic.npz", data=self.semantic)
        np.savez_compressed(f"{path_prefix}_instance.npz", data=self.instance)

    def reset(self):
        """重置为 unobserved。"""
        self.semantic[:] = 255
        self.instance[:] = 0
```

### 4.2 stereo_voxel_capture.py — 主采集脚本

#### 关键设计决策

**Q: 为什么不改 test_stereo_pair.py？**
- test_stereo_pair.py 已验证 NPC + 双目共存，是宝贵的参考
- 体素生成涉及 PhysX 查询、坐标变换、同步冻结等大量新逻辑
- 在已验证脚本上大改容易引入回归 bug
- 新脚本可以从 test_stereo_pair.py 复制验证过的模式

**Q: 为什么合并到一个脚本而不是拆成拍照+体素两个？**
- **同步是核心需求**：体素 GT 必须精确对应拍照瞬间的世界状态
- 两个脚本无法共享同一个 SimulationApp 实例
- 冻结/恢复逻辑必须在同一进程中控制
- IRA NPC 管道只能初始化一次

**Q: 采集频率？**
- 目标：每 3 秒采集 1 帧（~0.33 FPS）
- Isaac Sim 默认 30 FPS → 每 ~90 个 update() 采集一次
- 命令行可调：`--capture_interval 90`（单位：simulation steps）
- 每帧采集耗时估算：
  - 冻结 + 渲染刷新：~100ms
  - 双目图像获取：~30ms
  - 体素遍历 + PhysX 查询：~200-500ms（138K 查询，可批量化）
  - 异步保存：不阻塞
  - 恢复：~10ms
  - **总计：~0.5s 冻结时间/帧**

## 5. 同步机制 — 冻结采集

这是整个系统最关键的设计点。

### 问题

NPC 在持续行走，如果拍照和体素生成不同步：
- 拍照时 NPC 在位置 A
- 体素查询时 NPC 已移动到位置 B
- → 训练数据 GT 与输入图像不对应

### 方案：timeline.pause() 冻结

```python
def capture_frame():
    timeline = omni.timeline.get_timeline_interface()

    # 1. 冻结世界
    timeline.pause()

    # 2. 刷新渲染管线（pause 后仍需渲染当前帧）
    for _ in range(3):
        simulation_app.update()

    # 3. 采集双目图像
    rgb_l = annot_left.get_data()
    rgb_r = annot_right.get_data()

    # 4. 读取相机世界坐标
    cam_xform = UsdGeom.Xformable(stage.GetPrimAtPath(rig_path))
    cam_pos = get_world_position(cam_xform)
    cam_quat = get_world_orientation(cam_xform)

    # 5. 体素填充（世界已冻结，所有物体静止）
    voxel = VoxelGrid()
    world_centers = voxel.get_world_centers(cam_pos, cam_yaw)
    voxel.fill_from_physx(world_centers, physx_query, classify_fn)

    # 6. 异步保存
    async_save_images(rgb_l, rgb_r, frame_id)
    voxel.save(f"output/voxel/frame_{frame_id:06d}")
    save_meta(frame_id, cam_pos, cam_quat)

    # 7. 恢复世界
    timeline.play()
```

### 为什么用 timeline.pause() 而不是 physics.pause()？

- `timeline.pause()` 冻结一切：物理、动画图、NPC 行为脚本
- `physics.pause()` 只冻结 PhysX，动画图和 NPC 导航仍在更新
- NPC 走路由 `omni.anim.people` 行为脚本驱动，必须用 timeline 级别冻结

## 6. 体素填充策略

### 方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **PhysX overlap box** | 准确，支持碰撞体 | 需要物体有碰撞体 |
| USD BBox 查询 | 不需要碰撞体 | 只有 AABB，不精确 |
| Raycast 从体素中心向6方向 | 能检测表面 | 很慢，138K x 6 |
| Mesh voxelization | 最精确 | 复杂，需要遍历三角面 |

**选择：PhysX overlap box**

Isaac Sim 场景中大多数物体都有碰撞体。
对于 0.1m 分辨率，overlap box 精度足够。

```python
from omni.physx import get_physx_scene_query_interface

physx_sqi = get_physx_scene_query_interface()

def query_voxel(center, half_extent=0.05):
    """在世界坐标 center 处放一个 0.1m 立方体，查询重叠物体。"""
    hits = []
    def on_hit(hit):
        hits.append(hit)
        return True  # 继续查询

    physx_sqi.overlap_box(
        half_extent=carb.Float3(half_extent, half_extent, half_extent),
        pos=carb.Float3(*center),
        rot=carb.Float4(0, 0, 0, 1),
        report_fn=on_hit,
        any_hit=False,  # 获取所有重叠体
    )
    return hits
```

### 批量优化

138,240 个体素逐一查询太慢。优化策略：

1. **分层跳过**：先用大 box（如 1m³）粗查，无 hit 的区域整块标 FREE
2. **空间局部性**：按 Z 层遍历，每层 72x60=4320 查询
3. **早停**：地面以下大部分是 FREE（除非有凹槽），可以用地面检测快速填充

```python
# 粗查 → 细查 两阶段
COARSE_SIZE = 5  # 5x5x5 体素为一个粗查块
for cx in range(0, NX, COARSE_SIZE):
    for cy in range(0, NY, COARSE_SIZE):
        for cz in range(0, NZ, COARSE_SIZE):
            # 粗查：0.5m 立方体
            coarse_center = get_block_center(cx, cy, cz)
            if not has_any_overlap(coarse_center, half=0.25):
                # 整块标 FREE
                voxel.semantic[cx:cx+5, cy:cy+5, cz:cz+5] = FREE
                continue
            # 细查：逐个 0.1m 体素
            for i, j, k in block_voxels(cx, cy, cz):
                ...
```

## 7. 物体分类流程

```
PhysX overlap hit
    → hit.rigid_body (USD prim path)
    → 向上遍历找最近的有意义 Xform（去掉 Mesh/Collider 层级）
    → get_object_type(prim)  # 提取物体类型名
    → lookup_class_id(type_name)  # semantic_classes.py
    → (semantic_id, instance_id)
```

instance_id 生成：对每个唯一的 USD prim path 分配递增 ID。

```python
_instance_map = {}
_next_id = 1

def get_instance_id(prim_path: str) -> int:
    global _next_id
    if prim_path not in _instance_map:
        _instance_map[prim_path] = _next_id
        _next_id += 1
    return _instance_map[prim_path]
```

## 8. 输出格式

### 目录结构

```
output/
├── left/
│   ├── frame_000000.png     # 1280x1080 RGB
│   ├── frame_000001.png
│   └── ...
├── right/
│   ├── frame_000000.png
│   └── ...
├── voxel/
│   ├── frame_000000_semantic.npz   # uint8 (72, 60, 32)
│   ├── frame_000000_instance.npz   # int32 (72, 60, 32)
│   └── ...
├── meta/
│   ├── frame_000000.json
│   └── ...
├── calibration.json          # 相机内参 + 基线
├── voxel_config.json         # 体素参数（网格尺寸、分辨率等）
└── instance_map.json         # instance_id → prim_path 映射
```

### voxel_config.json

```json
{
  "voxel_size": 0.1,
  "grid_shape": [72, 60, 32],
  "x_range": [-3.6, 3.6],
  "y_range": [-3.0, 3.0],
  "z_range": [-0.7, 2.5],
  "z_ground_index": 7,
  "num_classes": 18,
  "coordinate_system": "camera_nadir_centered"
}
```

## 9. 命令行接口

```bash
# GUI 模式：手动 Play/Stop，可拖拽相机
isaaclab.bat -p projects/stereo_voxel/scripts/stereo_voxel_capture.py

# 无头批量采集
isaaclab.bat -p projects/stereo_voxel/scripts/stereo_voxel_capture.py \
    --headless \
    --num_frames 200 \
    --camera_height 3.0 \
    --camera_x 0.0 --camera_y 0.0 \
    --num_characters 3 \
    --capture_interval 90

# 不加 NPC（纯静态场景标注）
isaaclab.bat -p projects/stereo_voxel/scripts/stereo_voxel_capture.py \
    --headless --no_npc --num_frames 50
```

## 10. 实现计划

### Step 1: voxel_grid.py

纯 NumPy 模块，可单独 `python voxel_grid.py` 测试：
- VoxelGrid 类
- 坐标变换（voxel ↔ local ↔ world）
- save/load .npz
- 可视化辅助（matplotlib 3D 散点图）

### Step 2: stereo_voxel_capture.py 基础框架

从 test_stereo_pair.py 复制验证过的模式：
- IRA NPC 初始化
- 相机创建 + ftheta
- Replicator annotator
- 主循环 + Play/Stop

### Step 3: 集成 PhysX overlap 查询

- 冻结/恢复逻辑
- overlap_box 批量查询
- 物体分类 + instance ID

### Step 4: 完整同步采集

- 冻结 → 拍照 → 体素 → 保存 → 恢复
- 帧元数据 JSON
- calibration + voxel_config

### Step 5: 性能优化

- 粗查/细查两阶段
- 异步保存 npz
- 采集间隔自适应

## 11. 风险与注意事项

| 风险 | 影响 | 缓解 |
|------|------|------|
| NPC 无碰撞体 | 体素中检测不到行人 | 检查角色 USD 是否有 PhysicsCollider，必要时手动添加 |
| timeline.pause 后 IRA 状态异常 | NPC 恢复后行为异常 | 测试 pause/play 循环，如异常则改用减速方案 |
| 138K overlap 查询太慢 | 每帧冻结时间过长 | 粗查跳过 + 降低采集频率 |
| 相机 ftheta 被 IRA 覆盖 | 回退到针孔 | 已验证：ApplyAPI 方式不受 open_stage 影响 |
| 场景物体无 prim 类型信息 | 全部归 GENERAL_OBJECT | 可接受，后续补充映射 |
