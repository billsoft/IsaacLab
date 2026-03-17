# 双目鱼眼相机体素生成器 —— 详细技术计划

## 1. 项目目标

在 Isaac Sim 仓库场景中，以俯视视角创建一对双目鱼眼摄像头，
离地 2.5m 垂直向下拍摄 NPC 行人。同时以 5 FPS 保存左右眼图像，
为后续体素 occupancy grid 生成提供原始数据。

## 2. 硬件规格（SC132GS × 2）

| 参数 | 值 | 说明 |
|------|-----|------|
| 传感器 | 1/4" CMOS, 全局快门 | SC132GS |
| 分辨率 | 1280 × 1080 | 每只眼 |
| 像素尺寸 | 2.7 μm × 2.7 μm | |
| 基线（瞳距） | 80 mm | 左右眼间距 |
| 焦距 | 1.75 mm | |
| 光圈 | F2.0 | |
| 对角 FOV | 157.20° | |
| 水平 FOV | 96.8° (存疑，见下) | |
| 垂直 FOV | 115.63° (存疑，见下) | |

### FOV 分析与投影模型选择

**为什么不用小孔模型**：对角 FOV=157° 远超 180° 的理论上限（小孔模型 FOV < 180°），
即使 ~157° 在小孔模型下畸变极严重，必须用鱼眼投影。

**等距投影模型（Equidistant / Kannala-Brandt）**：
- 公式：`r = f * θ`（r=像面距离, f=焦距, θ=入射角）
- 适合 FOV > 120° 的广角/鱼眼镜头
- Isaac Sim 支持：`fisheyeKannalaBrandtK3` 或 OpenCV fisheye (`opencv.fisheye`)

**FOV 验证**：
```
对角 FOV = 157.20° → 半角 θ_max = 78.6°
f = 1.75mm, pixel_size = 2.7μm

等距投影：r = f * θ
  r_max = 1.75mm * (78.6° * π/180) = 1.75 * 1.372 = 2.401mm

传感器对角线 = √(1280² + 1080²) * 2.7μm = 1675.8 * 0.0027mm = 4.525mm
传感器对角半长 = 2.262mm

r_max(2.401mm) vs sensor_half_diag(2.262mm) → 基本匹配
（差异 ~6% 来自实际镜头与理想等距模型的偏差，可通过畸变系数修正）
```

**水平/垂直 FOV 重新计算**（等距模型）：
```
sensor_half_w = 1280/2 * 2.7μm = 1.728mm
sensor_half_h = 1080/2 * 2.7μm = 1.458mm

θ_h = r_h / f = 1.728 / 1.75 = 0.9874 rad = 56.6° → 水平 FOV ≈ 113.1°
θ_v = r_v / f = 1.458 / 1.75 = 0.8331 rad = 47.7° → 垂直 FOV ≈ 95.5°

原始规格水平 96.8° / 垂直 115.63° 可能是标注反了（宽 > 高 但 水平FOV < 垂直FOV）
或者使用了不同的投影模型。以实际传感器计算为准。
```

## 3. Isaac Sim 中的实现方案

### 3.1 API 选择

有两套 API 可用：

| API | 来源 | 鱼眼支持 | 适用场景 |
|-----|------|---------|---------|
| **IsaacSim Camera 类** | `isaacsim.sensors.camera.Camera` | `set_opencv_fisheye_properties()` | 独立脚本，更底层 |
| **IsaacLab FisheyeCameraCfg** | `isaaclab.sim.spawners.sensors` | `FisheyeCameraCfg` dataclass | IsaacLab 环境集成 |

**选择**：使用 **IsaacSim Camera 类** + `set_opencv_fisheye_properties()`。
理由：
- 我们的 NPC 脚本已使用 `SimulationApp`（非 AppLauncher）
- IsaacSim Camera 有完整的 `get_rgb()`, `get_depth()` 接口
- OpenCV fisheye 模型参数直观（fx, fy, cx, cy, k1-k4）
- 官方示例 `camera_opencv_fisheye.py` 完整可参考

### 3.2 相机参数计算

```python
# SC132GS 物理参数
pixel_size_um = 2.7        # μm
focal_length_mm = 1.75     # mm
f_stop = 2.0
width, height = 1280, 1080
baseline_mm = 80           # 双目基线

# 转换为 Isaac Sim 需要的单位
pixel_size_m = pixel_size_um * 1e-6                    # 2.7e-6 m
focal_length_m = focal_length_mm * 1e-3                # 1.75e-3 m
horizontal_aperture_m = pixel_size_m * width           # 3.456e-3 m
vertical_aperture_m = pixel_size_m * height            # 2.916e-3 m

# OpenCV fisheye 内参（像素单位）
fx = focal_length_mm / pixel_size_um * 1000            # 1.75/2.7*1000 = 648.15 pixels
fy = fx                                                # 正方形像素
cx = width / 2.0                                       # 640.0
cy = height / 2.0                                      # 540.0

# Kannala-Brandt 畸变系数（等距模型 k1≈0, k2≈0, k3≈0, k4≈0 为理想等距）
# 实际镜头的 k1-k4 需要标定，先用零值（纯等距投影）
fisheye_k = [0.0, 0.0, 0.0, 0.0]
```

### 3.3 相机布局

```
俯视图（Z-up 场景，相机在 z=2.5m 向下看）：

世界坐标系：
  Z ↑ (up)
  │
  │   相机在 z=2.5m
  │   ┌────┬────┐
  │   │ L  │  R │  ← 80mm 基线（沿 Y 轴）
  │   └────┴────┘
  │       │
  │       ↓ 向下看 (-Z)
  │
  ├──── Y (left)
  │
  X (forward)

左眼位置: (ego_x, ego_y + 0.04, 2.5)   # +40mm
右眼位置: (ego_x, ego_y - 0.04, 2.5)   # -40mm

朝向: 绕 X 轴旋转 +90°（从 -Z 看向地面）
  原始相机 -Z = 光轴方向
  旋转后 -Z → -Y... 不对，需要仔细算：

  Camera 默认光轴：-Z（Isaac Sim 约定）
  需要光轴指向世界 -Z（向下看地面）

  但相机默认已经是 -Z 方向... 如果不旋转，相机就是看向 -Z。
  在 Z-up 场景中，-Z = 向下 ✓

  所以：orientation = 单位四元数（不旋转）即可！

  等等——Isaac Sim 相机约定是：
  - 默认光轴方向取决于 USD 相机约定
  - USD 相机：看向 -Z，up = +Y
  - 但 Isaac Sim 用的是 OpenGL 约定还是 USD 约定？

  查看官方示例 camera_opencv_fisheye.py：
    camera_rotation_as_euler = [0, 90, 0]  # 绕 Y 轴 90°

  **已确认**：Isaac Sim Camera 默认光轴 = +X（不是 USD 的 -Z）
  惯例：+X = 前(光轴), +Y = 左, +Z = 上
  euler [0, 90, 0] 绕 Y 轴 90° 使 +X 旋转到 -Z（向下看）✓
```

### 3.4 关键 Isaac Sim Camera API

```python
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

# 创建相机
camera = Camera(
    prim_path="/World/Stereo/Left",
    position=np.array([x, y, z]),
    frequency=5,                        # 5 FPS 采集
    resolution=(1280, 1080),
    orientation=rot_utils.euler_angles_to_quats(euler, degrees=True),
)

# 初始化（需要在 world.reset() 之后）
camera.initialize()

# 设置物理参数
camera.set_focal_length(focal_length_m)
camera.set_horizontal_aperture(horizontal_aperture_m)
camera.set_vertical_aperture(vertical_aperture_m)
camera.set_lens_aperture(f_stop)
camera.set_clipping_range(0.1, 100.0)

# 设置 OpenCV 鱼眼畸变
camera.set_opencv_fisheye_properties(
    cx=cx, cy=cy, fx=fx, fy=fy,
    fisheye=[k1, k2, k3, k4]
)

# 获取图像
rgb = camera.get_rgb()           # (H, W, 3) uint8
depth = camera.get_depth()       # (H, W) float32, meters
# 也可获取：get_pointcloud(), get_semantic_segmentation() 等
```

### 3.5 图像保存策略

```
projects/stereo_voxel/output/
├── left/
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
├── right/
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
├── depth_left/          # 可选：深度图
│   └── ...
├── depth_right/         # 可选：深度图
│   └── ...
├── timestamps.csv       # 时间戳 + ego pose
└── calibration.json     # 相机内外参
```

**timestamps.csv 格式**：
```csv
frame_id,timestamp_sec,ego_x,ego_y,ego_z,ego_yaw
0,0.000,0.0,0.0,2.5,0.0
1,0.200,0.0,0.0,2.5,0.0
...
```

**calibration.json**：
```json
{
  "left": {
    "fx": 648.15, "fy": 648.15, "cx": 640.0, "cy": 540.0,
    "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
    "baseline_to_right_m": 0.08,
    "position_in_ego": [0.0, 0.04, 0.0],
    "model": "opencv_fisheye_equidistant"
  },
  "right": { ... }
}
```

## 4. 与 NPC 系统集成

基于 `npc_people_demo.py` 的 IRA SimulationManager 方案：

```
启动流程：
  1. SimulationApp (非 AppLauncher)
  2. enable extensions (动画 + 导航)
  3. IRA SimulationManager 加载场景 + NPC
  4. 等待 setup 完成
  5. 创建双目相机 (isaacsim.sensors.camera.Camera)
  6. 设置鱼眼参数
  7. 启动仿真 + 数据采集循环
  8. 每帧: world.step() → camera.get_rgb() → 保存图像
```

**注意**：IRA `SimulationManager` 使用异步 `run_data_generation_async()`，
需要确认相机采集是否能在同一主循环中进行。
可能需要改为手动 `simulation_app.update()` 循环（不使用 IRA 的异步跑法）。

## 5. 实现步骤

### Step 1: 创建最小可行相机测试 ✅ 已实现
- `scripts/test_single_camera.py`
- 在仓库场景中创建单个下视相机（不加 NPC）
- 验证鱼眼参数是否正确
- 保存图像 + 深度图确认 FOV 和畸变

### Step 2: 添加双目（左右眼） ✅ 已实现
- `scripts/test_stereo_pair.py`
- 80mm 基线（沿 Y 轴）
- 验证两个相机同时工作
- 保存左右眼图像对 + calibration.json

### Step 3+4: 集成 NPC + 完善输出 ✅ 已实现
- `scripts/stereo_capture.py`
- IRA SimulationManager 管理 NPC 行走
- 在 NPC 行走过程中 5 FPS 定时采集
- 输出 timestamps.csv + calibration.json + 可选深度图

### Step 5: 体素生成（后续）
- 读取双目图像 → 深度估计 → 3D 点云 → 体素填充
- 或直接用 Isaac Sim depth buffer → 体素

## 6. 需要验证的关键问题

| # | 问题 | 验证方法 |
|---|------|---------|
| 1 | Camera 默认光轴方向在 Z-up 场景中是什么 | 创建不旋转相机，渲染一帧看画面内容 |
| 2 | `set_opencv_fisheye_properties` 对 157° FOV 是否有效 | 渲染已知几何体，检查边缘畸变 |
| 3 | 两个 Camera 能否同时渲染 | 创建两个 Camera 实例，各自 get_rgb() |
| 4 | IRA 仿真循环中能否插入 Camera 采集 | 在 simulation_app.update() 后尝试 get_rgb() |
| 5 | 图像保存性能（5FPS × 2 × 1280×1080）| 测量每帧保存耗时 |
| 6 | `isaacsim.sensors.camera.Camera` 在 `SimulationApp` 下是否需要 `World` 对象 | 参考官方示例 |

## 7. 文件结构

```
projects/stereo_voxel/
├── PLAN.md                          # 本文件
├── scripts/
│   ├── stereo_capture.py            # 主脚本：NPC + 双目采集
│   ├── test_single_camera.py        # Step1: 单相机测试
│   └── test_stereo_pair.py          # Step2: 双目测试
└── output/
    ├── left/                        # 左眼图像
    ├── right/                       # 右眼图像
    ├── timestamps.csv
    └── calibration.json
```

## 8. 依赖

- `isaacsim.sensors.camera.Camera` — 相机创建和渲染
- `isaacsim.core.api.World` — 仿真世界管理
- `isaacsim.core.utils.numpy.rotations` — 四元数工具
- `isaacsim.replicator.agent.core.SimulationManager` — NPC 管理
- `cv2` (OpenCV) — 图像保存
- `numpy` — 数值计算

## 9. 参考源码

| 文件 | 用途 |
|------|------|
| `D:\code\IsaacSim\source\standalone_examples\api\isaacsim.sensors.camera\camera_opencv_fisheye.py` | OpenCV 鱼眼相机创建的完整官方示例 |
| `D:\code\IsaacSim\source\standalone_examples\api\isaacsim.anim.people\npc_walk_back_and_forth.py` | NPC IRA 方案的官方示例（我们的参考脚本） |
| `D:\code\IsaacLab\source\isaaclab\isaaclab\sim\spawners\sensors\sensors_cfg.py` | FisheyeCameraCfg 定义 |
| `D:\code\IsaacLab\source\isaaclab\isaaclab\sensors\camera\camera.py` | IsaacLab Camera 类 |
| `D:\code\IsaacSim\source\extensions\isaacsim.sensors.camera\isaacsim\sensors\camera\camera.py` | IsaacSim Camera 完整 API |
| `D:\code\IsaacLab\scripts\npc\npc_people_demo.py` | 我们已验证可用的 NPC 脚本 |
