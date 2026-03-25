# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ 最高优先级规则：Python 环境限制

**🔴 严禁使用本地系统 Python！必须使用 Isaac Sim 内置 Python**

本项目运行在 **Windows Isaac Lab** 环境，唯一可用的 Python 解释器路径：
```
D:\code\IsaacLab\_isaac_sim\python.bat
```

### 🚫 绝对禁止的操作
- ❌ `python script.py` - 使用系统 Python
- ❌ `pip install package` - 使用系统 pip  
- ❌ `conda activate` - 使用 conda 环境
- ❌ 在 CMD/PowerShell 中直接运行 Python 脚本

### ✅ 唯一正确的操作方式
```bat
# 运行脚本
isaaclab.bat -p <脚本路径>

# 安装包
isaaclab.bat -p -m pip install <包名>

# 直接使用 Isaac Sim Python
D:\code\IsaacLab\_isaac_sim\python.bat <脚本路径>
D:\code\IsaacLab\_isaac_sim\python.bat -m pip install <包名>

# 检查包
isaaclab.bat -p -m pip list
```

### 📋 原因说明
Isaac Sim 的 C++ 扩展模块与内置 Python 3.11.13 紧密绑定，使用其他 Python 环境会导致 DLL 加载失败和运行时错误。

## 沟通语言

**始终用中文（简体中文）与用户沟通。** 所有回复、解释、问题均使用中文。
**严禁使用朝鲜语（한국어）和日语（日本語/ひらがな/カタカナ）文字。** 若需引用代码注释中的外文，保留原文但解释用中文。

## Overview

Isaac Lab is a GPU-accelerated robotics simulation framework built on NVIDIA Isaac Sim. It supports reinforcement learning, imitation learning, and motion planning with sim-to-real transfer. Requires Isaac Sim 4.5/5.0/5.1 installed separately.

## Python 环境说明（重复强调）

**再次强调：本项目只能使用 Isaac Sim 内置 Python**
```
D:\code\IsaacLab\_isaac_sim\python.bat
```

### 安装包
```bat
:: 推荐方式
isaaclab.bat -p -m pip install <包名>

:: 直接方式
D:\code\IsaacLab\_isaac_sim\python.bat -m pip install <包名>
```

### 运行脚本
```bat
isaaclab.bat -p <脚本路径>
```

### 检查包
```bat
isaaclab.bat -p -m pip list
isaaclab.bat -p -c "import <包名>; print('OK')"
```

## Common Commands

All commands use `isaaclab.bat` (Windows) or `isaaclab.sh` (Linux/macOS):

```bash
# Install all extensions in development mode
isaaclab.bat -i

# Run linting (pre-commit hooks: ruff + codespell)
isaaclab.bat -l

# Run tests
isaaclab.bat -t

# Build documentation
isaaclab.bat -d

# Run a Python script with Isaac Sim environment
isaaclab.bat -p <script.py>

# Launch Isaac Sim
isaaclab.bat -s
```

Run a single test directly:
```bash
python -m pytest source/isaaclab/test/path/to/test_file.py -v
```

## Code Style

- **Line length**: 120 characters
- **Formatter/linter**: Ruff (`ruff check --fix`, `ruff format`)
- **Docstrings**: Google style
- **Import order**: future → stdlib → third-party → omniverse-extensions (isaacsim, omni, pxr, carb) → isaaclab → isaaclab_assets/contrib/rl/mimic/tasks → first-party → local
- `__init__.py` files allow unused imports (F401 ignored)

## Architecture

### Extension Structure

The codebase is organized as independent installable extensions under `source/`:

| Extension | Package | Purpose |
|-----------|---------|---------|
| `source/isaaclab/` | `isaaclab` | Core framework |
| `source/isaaclab_assets/` | `isaaclab_assets` | Robot/environment USD assets |
| `source/isaaclab_tasks/` | `isaaclab_tasks` | Pre-built RL training environments (30+) |
| `source/isaaclab_rl/` | `isaaclab_rl` | RL framework integrations (RSL-RL, SKRL, RL Games, SB3) |
| `source/isaaclab_mimic/` | `isaaclab_mimic` | Imitation learning (Apache 2.0) |
| `source/isaaclab_contrib/` | `isaaclab_contrib` | Community contributions |

Each extension contains: `setup.py`, `config/extension.toml`, the package code, `test/`, and `docs/`.

### Core `isaaclab` Package Modules

- **`app/`** — `AppLauncher`: entry point that initializes Isaac Sim before any other imports
- **`envs/`** — Environment base classes:
  - `ManagerBasedEnv` / `ManagerBasedRLEnv`: modular environments controlled by managers
  - `DirectRLEnv` / `DirectMARLEnv`: direct-control environments without manager overhead
- **`managers/`** — Modular MDP components: observation, action, reward, termination, randomization, curriculum, recorder managers
- **`scene/`** — `InteractiveScene`: composes assets, sensors, and terrain
- **`sim/`** — Simulation context, spawners (rigid body, articulation, light, deformable), URDF/MJCF→USD converters
- **`assets/`** — `RigidObject`, `Articulation`, `DeformableObject` wrappers
- **`sensors/`** — Camera (RTX/pinhole), RayCaster (LIDAR), IMU, ContactSensor, FrameTransformer
- **`actuators/`** — Motor model abstractions (ideal, DCMotor, impedance control)
- **`controllers/`** — IK solvers (differential IK, pink-ik), operational space control
- **`terrains/`** — Procedural terrain generation
- **`devices/`** — Gamepad, keyboard, SpaceMouse, OpenXR input devices

### Two Environment Patterns

**Manager-based** (recommended for new tasks): Declare environment logic as config dataclasses; swap components without subclassing.
```python
class MyEnvCfg(ManagerBasedRLEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
```

**Direct** (lower overhead, full control): Subclass `DirectRLEnv` and override `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`.

### Task Registration

Tasks in `isaaclab_tasks` are registered as Gymnasium environments via `gym.register()` in each task's `__init__.py`. Task configs follow the naming convention `Isaac-<TaskName>-<Robot>-v<N>`.

### AppLauncher Pattern

Every script must launch Isaac Sim first via `AppLauncher` before importing any `isaaclab` or `isaacsim` modules:
```python
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
# All other imports come AFTER this
import isaaclab.sim as sim_utils
```

## NPC 人物动画（重要经验）

### 骨骼系统不兼容问题

Isaac Sim 官方角色有两套完全不兼容的骨骼系统：
- **NVIDIA Biped**（81 关节）：`Root/Pelvis/L_UpLeg/...` — 仅 `biped_demo_meters.usd` 使用
- **RL Bones / Reallusion**（101 关节）：`RL_BoneRoot/Hip/Pelvis/L_Thigh/...` — 所有纹理角色使用

两套骨骼关节名 **0% 匹配**。直接用 `UsdSkel.BindingAPI` 绑定 `.skelanim.usd` 到 RL Bones 角色会静默失败 → T-pose。
**永远不要**手动绑定 skelanim 到纹理角色。

### 正确方案：IRA SimulationManager

使用 `isaacsim.replicator.agent.core.simulation.SimulationManager`，它自动处理：
- 场景加载、角色生成
- 骨骼重定向（`omni.anim.retarget.core`）
- 动画图驱动（`omni.anim.graph.core`）
- NavMesh 烘焙和寻路
- Behavior Script 挂载

**必须启用的扩展**（顺序重要）：
```python
from isaacsim.core.utils.extensions import enable_extension
for ext in [
    "omni.anim.timeline", "omni.anim.graph.bundle", "omni.anim.graph.core",
    "omni.anim.retarget.core", "omni.anim.navigation.core",
    "omni.anim.navigation.bundle", "omni.anim.people", "omni.kit.scripting",
]:
    enable_extension(ext)
```

**关键流程**：
```python
enable_extension("isaacsim.replicator.agent.core")
from isaacsim.replicator.agent.core.simulation import SimulationManager

sim_manager = SimulationManager()
sim_manager.load_config_file("config.yaml")  # YAML 配置
sim_manager.set_up_simulation_from_config_file()  # 异步 setup
await sim_manager.run_data_generation_async(will_wait_until_complete=True)  # 异步运行
```

**GoTo 命令格式**（角色行走控制）：
```
Character GoTo 4.0 0.0 0.0 0       # 走到 (4,0,0)，面朝 rotation=0°
Character GoTo -4.0 0.0 0.0 180    # 走回 (-4,0,0)，面朝 180°
Character Idle 5                     # 原地站立 5 秒
```

IRA 角色命名规则：第 0 个→`Character`，第 1 个→`Character_01`，第 10 个→`Character_10`

**参考脚本**：
- Isaac Sim 官方：`D:\code\IsaacSim\source\standalone_examples\api\isaacsim.anim.people\npc_walk_back_and_forth.py`
- IsaacLab 移植：`scripts/npc/npc_people_demo.py`

### 角色朝向约定（Z-up 场景直接操作 XformOp 时）

`_new` / `F_*` / `M_*` 角色本地前向是 **+Y**（Z-up Reallusion 约定）：
```python
# 要让角色面朝运动方向 (dx, dy)：yaw = atan2(-dx, dy)
yaw = np.degrees(np.arctan2(-d[0], d[1]))
# Gf.Rotation(Gf.Vec3d(0,0,1), yaw)
```
`xformOp:orient` 精度为 **Quatd**（double），不是 Quatf。

## 鱼眼相机（重要经验）

### Isaac Sim 5.x 相机投影 API

**`cameraProjectionType` 已废弃！** Isaac Sim 5.x 不识别 `fisheyeEquidistant` / `fisheyePolynomial` 等旧值，
渲染器会报 `Unknown projection type, defaulting to pinhole` 并回退到小孔模型。

**正确方式：`OmniLensDistortion` schema**：
```python
from isaacsim.sensors.camera import Camera

camera = Camera(prim_path="/World/Camera", ...)
camera.initialize()

# f-theta 鱼眼投影（原生渲染，支持 >180° FOV）
camera.set_ftheta_properties(
    nominal_width=1280.0,
    nominal_height=1080.0,
    optical_center=(640.0, 540.0),
    max_fov=157.2,                                    # 对角 FOV（度）
    distortion_coefficients=[0, 0.001543, 0, 0, 0],   # 等距: k1 = 1/fx
)

# 其他可用模型（都是 OmniLensDistortion 下的）：
# camera.set_opencv_fisheye_properties(...)    # OpenCV 鱼眼（后处理畸变，不改变 FOV！）
# camera.set_kannala_brandt_properties(...)    # Kannala-Brandt K3
```

**关键区别**：
- `set_ftheta_properties()` → 设置 `omni:lensdistortion:model = "ftheta"`，RTX 原生鱼眼投影，`max_fov` 直接控制渲染 FOV
- `set_opencv_fisheye_properties()` → 只是后处理畸变，不改变底层渲染 FOV，不适合超广角
- Isaac Lab 的 `FisheyeCameraCfg` 仍使用旧 `cameraProjectionType`，在 5.x 可能不工作

**直接操作 USD prim 设置 ftheta（IRA 场景下必须用此方式）**：
```python
# 关键：必须先 ApplyAPI 注册 schema，渲染器才会识别 ftheta 属性！
# 不 ApplyAPI → 属性存在但渲染器忽略 → 退化为普通广角相机
prim.ApplyAPI("OmniLensDistortionFthetaAPI")
prim.GetAttribute("omni:lensdistortion:model").Set("ftheta")

# 属性名必须用 k0-k4（不是 p0-p4！早期文档和某些示例用 p0-p4 是错的）
prim.GetAttribute("omni:lensdistortion:ftheta:k0").Set(0.0)
prim.GetAttribute("omni:lensdistortion:ftheta:k1").Set(float(K1_EQUIDISTANT))
prim.GetAttribute("omni:lensdistortion:ftheta:k2").Set(0.0)
prim.GetAttribute("omni:lensdistortion:ftheta:k3").Set(0.0)
prim.GetAttribute("omni:lensdistortion:ftheta:k4").Set(0.0)

# opticalCenter 是元组（不是分开的 opticalCentreX/Y！）
prim.GetAttribute("omni:lensdistortion:ftheta:opticalCenter").Set((cx, cy))
prim.GetAttribute("omni:lensdistortion:ftheta:maxFov").Set(157.2)

# fStop=0 禁用景深模糊（重要：不设置会导致远处模糊）
prim.GetAttribute("fStop").Set(0.0)
```

### f-theta 多项式公式

```
θ = k0 + k1·r + k2·r² + k3·r³ + k4·r⁴
```
- θ = 入射角（弧度），r = 像素距离（从光心到像素点的距离）
- 等距投影（r = f·θ）→ k0=0, k1=1/fx, 其余=0
- fx = focal_length_mm / pixel_size_um × 1000（像素焦距）

### 相机渲染管线延迟

Isaac Sim 的 `camera.get_rgb()` 有 **1-2 帧管线延迟**：
- `world.step(render=True)` 后立即读取可能得到上一帧或全黑帧
- **解决方案**：预热 ≥30 帧，每次采集前 step 3 次再读取，检查亮度跳过黑帧

### Isaac Sim Camera 坐标约定

- 默认光轴 = **+X**（不是 USD 的 -Z）
- 坐标系：+X = 前(光轴), +Y = 左, +Z = 上
- 向下俯视：`euler = [0, 0, 90]`（绕 Z 轴 90°，使 +X 指向世界 -Z）

### set_focal_length / set_horizontal_aperture 单位

API 接受 **stage units**（通常是米），内部乘以 `USD_CAMERA_TENTHS_TO_STAGE_UNIT=10` 存入 USD。
注意 `UsdGeom.GetStageMetersPerUnit(stage)` 检测场景单位是否为米。

## 双目相机+IRA NPC数据采集（完整经验）

### 问题背景

在 Isaac Sim 中同时运行 IRA（Isaac Replicator Agent）管理 NPC 行走和双目鱼眼相机采集时遇到两个关键问题：
1. **图像全黑**：`Camera.get_rgb()` 返回 shape 正确但 mean=0.00
2. **严重卡顿**：NPC 走路卡顿，每帧采集导致性能下降

### 根因分析

#### 1. 全黑图像的根本原因

`isaacsim.sensors.camera.Camera` 类在 `initialize()` 时注册了 `_acquisition_callback` 监听渲染事件。但 IRA 的 `open_stage()` 会触发 `StageEventType.OPENED` 事件，导致 `Camera._stage_open_callback_fn` 被调用：

```python
def _stage_open_callback_fn(self, event):
    self._acquisition_callback = None  # ← 数据采集回调被清除！
    self._stage_open_callback = None
    self._timer_reset_callback = None
```

即使我们在 IRA `open_stage` 之后才创建 Camera，`OPENED` 事件的异步延迟可能在 `initialize()` **之后**才到达，仍然清掉回调，导致 `get_rgb()` 返回全黑。

#### 2. 卡顿的根本原因

每帧同步执行：
```python
rgb = camera.get_rgb()          # GPU→CPU 同步拷贝，阻塞渲染管线
cv2.imwrite(path, rgb)         # 磁盘 IO，阻塞主线程
```

- 1280×1080×3 = 4MB/帧，双目 = 8MB/帧
- 每帧都做同步 IO → 严重卡顿

### 正确解决方案

#### 方案对比

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| 修补 Camera.get_rgb() | 无需重写 | 不稳定，治标不治本 | ❌ 不推荐 |
| IRA 完整数据管线 | 官方支持，高性能 | 仅支持针孔相机，不支持鱼眼 | ❌ 不满足需求 |
| **USD prim + Replicator annotator** | **绕过 Camera 类，支持鱼眼，异步高性能** | 需要手动设置 USD 属性 | ✅ **最终选择** |

#### 实现细节

**1. 直接创建 USD Camera prim 并设置鱼眼属性**
```python
# 创建相机 prim（绕过 Camera 类）
cam_prim = UsdGeom.Camera.Define(stage, cam_path)

# 设置 xform
xf = UsdGeom.Xformable(cam_prim.GetPrim())
xf.AddTranslateOp().Set(Gf.Vec3d(0.0, float(y_offset), 0.0))
xf.AddOrientOp().Set(Gf.Quatf(float(cam_quat[0]), 
                            Gf.Vec3f(float(cam_quat[1]), float(cam_quat[2]), float(cam_quat[3]))))

# 直接在 USD prim 上设置 ftheta 鱼眼属性
def set_fisheye_on_prim(stage, cam_prim_path):
    prim = stage.GetPrimAtPath(cam_prim_path)
    # 必须 ApplyAPI，否则渲染器忽略 ftheta 属性
    prim.ApplyAPI("OmniLensDistortionFthetaAPI")
    prim.GetAttribute("omni:lensdistortion:model").Set("ftheta")
    prim.GetAttribute("omni:lensdistortion:ftheta:nominalWidth").Set(float(CAM_W))
    prim.GetAttribute("omni:lensdistortion:ftheta:nominalHeight").Set(float(CAM_H))
    # opticalCenter 是元组！不是 opticalCentreX/Y
    prim.GetAttribute("omni:lensdistortion:ftheta:opticalCenter").Set((float(cx), float(cy)))
    prim.GetAttribute("omni:lensdistortion:ftheta:maxFov").Set(float(DIAG_FOV_DEG))
    # k0-k4（不是 p0-p4！）
    prim.GetAttribute("omni:lensdistortion:ftheta:k0").Set(0.0)
    prim.GetAttribute("omni:lensdistortion:ftheta:k1").Set(float(K1_EQUIDISTANT))
    prim.GetAttribute("omni:lensdistortion:ftheta:k2").Set(0.0)
    prim.GetAttribute("omni:lensdistortion:ftheta:k3").Set(0.0)
    prim.GetAttribute("omni:lensdistortion:ftheta:k4").Set(0.0)
    prim.GetAttribute("fStop").Set(0.0)  # 禁用景深模糊
```

**2. 使用 omni.replicator annotator 异步采集**
```python
# 创建 render product 和 annotator（绕过 Camera 类）
rp_left = rep.create.render_product(left_cam_path, resolution=(CAM_W, CAM_H))
rp_right = rep.create.render_product(right_cam_path, resolution=(CAM_W, CAM_H))

annot_left = rep.AnnotatorRegistry.get_annotator("rgb")
annot_left.attach([rp_left.path])

annot_right = rep.AnnotatorRegistry.get_annotator("rgb")
annot_right.attach([rp_right.path])

# 在主循环中异步获取数据
def try_capture():
    data_l = annot_left.get_data()
    data_r = annot_right.get_data()
    
    if data_l is not None and data_r is not None:
        # annotator 返回 RGBA，取 RGB 通道
        rgb_l = data_l[:, :, :3] if data_l.ndim == 3 else data_l
        rgb_r = data_r[:, :, :3] if data_r.ndim == 3 else data_r
        
        # 跳过全黑帧
        if rgb_l.mean() < 1.0 or rgb_r.mean() < 1.0:
            return False
            
        # 异步保存（不阻塞渲染线程）
        async_save_image(left_path, rgb_l)
        async_save_image(right_path, rgb_r)
        return True
    return False
```

**3. 异步图像存储**
```python
from concurrent.futures import ThreadPoolExecutor

_save_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="img_saver")
_pending_saves = []

def async_save_image(path, rgb_array):
    """将 RGB numpy array 异步保存为 PNG（不阻塞渲染线程）。"""
    def _save(p, arr):
        bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(p, bgr)
    future = _save_pool.submit(_save, path, rgb_array.copy())
    _pending_saves.append(future)

def wait_pending_saves():
    """等待所有异步保存完成。"""
    for f in _pending_saves:
        f.result()
    _pending_saves.clear()
```

### 性能优化要点

1. **异步存储**：`ThreadPoolExecutor` 将磁盘 IO 移到后台线程
2. **采集间隔**：`--capture_interval 3` 每 3 步采集一帧，降低负载
3. **跳过黑帧**：`rgb.mean() < 1.0` 过滤无效帧
4. **预热阶段**：30 帧让渲染管线稳定

### 验证结果

- **1607 对左右图像成功保存**
- **图像亮度正常**：mean=117~121（不再是 0.00）
- **NPC 走动流畅**：无卡顿
- **鱼眼投影正确**：ftheta 模型，157.2° 对角 FOV

### 关键经验总结

1. **永远不要在 IRA 场景中使用 `Camera.get_rgb()`**：`_stage_open_callback_fn` 会清掉回调
2. **直接使用 omni.replicator annotator**：绕过 Camera 类，天然同步渲染管线
3. **异步存储是必须的**：磁盘 IO 必须移到后台线程
4. **USD prim 属性设置要精确**：`Gf.Quatf` vs `Gf.Quatd`，`Sdf.ValueTypeNames.Float`
5. **IRA + 自定义相机可以共存**：先手动 open_stage，再让 IRA 跳过 stage 加载

### 参考脚本

完整实现见：`projects/stereo_voxel/scripts/test_stereo_pair.py`

## PhysX 体素查询地面边界效应（重要经验）

### 问题现象

使用 PhysX `overlap_box` 填充语义体素网格时，地面层（z=Z_GROUND_INDEX=7）只有约 **51%** 的体素被正确检测为 occupied，另一半被误标为 free。缺失区域呈干净的直线分界（X=CENTER_X），而非随机分布。

### 根因分析

体素网格参数：
- `VOXEL_SIZE = 0.1m`，`Z_GROUND_INDEX = 7`
- z=7 体素中心 = `(7 - 7 + 0.5) * 0.1 = +0.05m`
- `fine_half = VOXEL_SIZE / 2 = 0.05m`
- PhysX box z 范围 = `[0.05 - 0.05, 0.05 + 0.05] = [0.00, 0.10]m`

地面平面（GroundPlane）碰撞体在 z=0.0m。PhysX `overlap_box` 的下边界 **恰好** 在 z=0.0m，属于精确边界接触。PhysX 对这种零重叠的边界碰撞检测 **不稳定** —— 部分体素检测到碰撞，部分漏检，取决于浮点精度和内部空间划分。

### 修复方案

**双重保障**（已在 `stereo_voxel_capture.py` 的 `fill_voxel_grid()` 中实现）：

**方案 A（主修复）**：将地面层查询中心 z 坐标下移 2mm
```python
Z_GI = voxel_grid.Z_GROUND_INDEX
GROUND_Z_NUDGE = 0.002 * meters_to_stage  # 2mm
world_centers_stage[:, :, Z_GI, 2] -= GROUND_Z_NUDGE
```
效果：PhysX box 下边界从 z=0.0 变为 z=-0.002m，确保与地面明确重叠。

**方案 C（后处理兜底）**：如果 z=Z_GI 仍为 free 但 z=Z_GI-1（地下层）有 occupied，用 `OTHER_GROUND(12)` 填充
```python
miss_mask = (ground == FREE) & (underground > 0) & (underground != UNOBSERVED)
voxel_grid.semantic[:, :, Z_GI][miss_mask] = OTHER_GROUND
```

### 修复效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| z=7 占用率 | 51.4% (2220/4320) | 100.0% (4320/4320) |
| 总占用体素 | 32838 | 34938 |
| 四象限覆盖 | X-=100%, X+=2.8% | 全部 100% |

### 关键教训

1. **PhysX overlap 在精确边界上不可靠**：box 边界恰好接触碰撞面时，结果不确定
2. **始终给边界查询留 epsilon 余量**：1-2mm 的 nudge 足够消除不稳定性
3. **用 `diag_voxel.py` 诊断**：逐层分析 + 象限分析 + 边界检测可快速定位问题
4. **后处理兜底是必要的**：即使主修复有效，利用地下层数据兜底更安全

### 诊断工具

`projects/stereo_voxel/scripts/diag_voxel.py` —— 体素地面诊断脚本，分析：
- 总体类别分布
- 逐层 Z 占用率
- 地面层 XY 象限占用
- free/occupied 边界检测
- z±1 层对比
- PhysX 边界效应量化

## Isaac Sim Python 环境限制（重要经验）

### GUI 不可用

Isaac Sim 内置 Python 的 `cv2` 是 **headless 编译**（无 `highgui` 模块），且 **没有 `_tkinter`**。
因此以下 GUI 操作全部不可用：
- ❌ `cv2.imshow()` / `cv2.namedWindow()` → 报 `The function is not implemented`
- ❌ `matplotlib` TkAgg 后端 → 报 `No module named '_tkinter'`
- ❌ 任何依赖 tkinter 的 GUI（filedialog 等）

**替代方案**：
- 可视化工具生成 **HTML 文件**，用 `webbrowser.open()` 在浏览器中查看
- 图像处理用 `cv2.imread/imwrite`（文件读写正常，只是 GUI 不行）
- `PIL/Pillow` 可用于图像加载和 base64 编码
- `tifffile` 可用于 DNG/TIFF 读写（Isaac Sim Python 自带）

### stdout 缓冲

通过 `isaaclab.bat` 运行脚本时，stdout 被 `cmd //c` 管道**全缓冲**，`print()` 输出可能延迟到脚本结束才显示。

**解决方案**：在脚本顶部强制刷新：
```python
import builtins
_original_print = builtins.print
def _flush_print(*a, **kw):
    kw.setdefault("flush", True)
    _original_print(*a, **kw)
builtins.print = _flush_print
```

或用文件日志记录关键进度。

## Camera 类 vs USD prim 朝向差异（重要经验）

### 问题

`isaacsim.sensors.camera.Camera` 类和直接操作 USD prim 使用**相同的 euler 角**会产生**不同的朝向**。

### 根因

Camera 类假设光轴为 **+X**，但 USD 相机标准光轴为 **-Z**。Camera 类在内部自动叠加一个 **R_y(-90°)** 旋转来做 +X → -Z 的转换：

```
Q_usd_final = Q_user × Q_y(-90°)
```

直接操作 USD prim 的 `xformOp:orient` 时，**没有**这个内部转换。

### 实际影响

| 设置方式 | euler = [0, 90, 0] 的效果 |
|----------|--------------------------|
| Camera 类 | 向下俯视（正确） |
| USD prim 直接 | 侧面朝向（错误） |

### 正确做法

- 用 **Camera 类**：直接传入想要的 euler，类内部处理转换
- 用 **USD prim 直接操作**：要么手动补偿 `Q_usd = Q_isaac × Q_y(-90°)`，要么使用 `euler = [0, 0, 90]`（绕 Z 轴 90° 使 -Z 指向地面，这是 `stereo_voxel_capture.py` 的做法）

**经验**：两种方式不要混用。选定一种后保持一致。`stereo_voxel_capture.py` 系列脚本统一使用 USD prim 直接操作 + `euler = [0, 0, 90]`。

## opencv_fisheye 模型 ≠ 针孔（重要经验）

### 常见误解

`set_opencv_fisheye_properties(k=[0,0,0,0])` **不是**针孔投影。

### 实际行为

即使所有畸变系数 k 都为 0，`opencv_fisheye` 模型仍然激活**等距鱼眼投影**（r = f·θ），产生约 **148°** 对角 FOV。针孔模型相同焦距下仅 **89°**。

这是因为 `opencv_fisheye` 的基础模型就是等距投影，k 参数只是在此基础上叠加高阶畸变。

### 与 ftheta 的关系

`ftheta` 模型 `k1=1/fx, 其余=0` 在数学上等价于 `opencv_fisheye k=[0,0,0,0]`——都是 r = f·θ 等距投影。区别在于：
- `ftheta` 是 **RTX 原生渲染**，`maxFov` 直接控制渲染 FOV
- `opencv_fisheye` 是**后处理畸变**，不改变底层渲染 FOV

**结论**：需要超广角鱼眼时，始终用 `ftheta`。

## 伪 RAW/DNG 管线（开发经验）

### 概述

从 Isaac Sim RGB 渲染图生成逼真的 12-bit Bayer RAW DNG 文件，模拟真实传感器输出。

### 管线流程

```
RGB(uint8) → sRGB 解码 → linear(float32) → RGGB Bayer CFA 马赛克 → 传感器噪声 → 12-bit 量化(uint16) → DNG
```

### 关键实现

**DNG 写入：纯 tifffile，不需要 exiftool**
```python
import tifffile
import numpy as np

# bayer_uint16: (H, W) uint16 Bayer RAW 数据
tifffile.imwrite(
    path, bayer_uint16,
    photometric='cfa',            # CFA (Color Filter Array)
    compression='none',
    extratags=[
        (50706, 1, 4, (1, 4, 0, 0)),    # DNGVersion 1.4
        (50710, 1, 1, (2,)),              # CFAPlaneColor
        (50711, 3, 1, (1,)),              # CFALayout
        (33421, 3, 2, (2, 2)),            # CFARepeatPatternDim
        (33422, 1, 4, (0, 1, 1, 2)),     # CFAPattern (RGGB)
        (50714, 3, 1, (black_level,)),    # BlackLevel
        (50717, 3, 1, (white_level,)),    # WhiteLevel
        (50721, 12, 9, color_matrix),     # ColorMatrix1
        (50778, 3, 1, (21,)),             # CalibrationIlluminant1 (D65)
    ],
)
```

**传感器噪声模型（5 种来源）**：
- PRNU（像素响应非均匀性）— 乘性，每个传感器固定
- 散粒噪声（Shot noise）— 泊松分布，与信号强度成正比
- 暗电流（Dark current）— 高斯，与曝光时间正比
- 读出噪声（Read noise）— 高斯，固定标准差
- 行噪声（Row noise）— 每行共模偏移

### 组件包

`projects/stereo_voxel/scripts/rawcam/` — 可复用的 RAW 相机仿真组件：
- `core/`：纯 Python 层（RawConverter, DngWriter, NoiseModel），不依赖 Isaac Sim
- `sim/`：Isaac Sim 封装层（StereoRawRig, RawCamera），处理 USD prim 创建和 annotator
- `configs/`：数据类（SensorConfig, NoiseConfig, OutputConfig）+ 传感器预设（SC132GS 等）

### 端到端验证方法

验证 DNG 变体脚本与原版一致性的标准流程：
```bash
# 1. 分别运行两个脚本（无 NPC，消除随机性）
isaaclab.bat -p .../stereo_voxel_capture.py --headless --num_frames 5 --no_npc --capture_interval 30
isaaclab.bat -p .../stereo_voxel_capture_dng.py --headless --num_frames 5 --no_npc --capture_interval 30

# 2. 自动化对比
python .../compare_capture_outputs.py
```

**验证指标**：
- RGB NCC（归一化互相关）> 0.99 = 像素级一致（渲染非确定性导致 < 1.0）
- 体素 match_rate = 1.0（无 NPC 时静态场景完全确定）
- DNG 值域在 12-bit 范围内 [0, 4095]
- calibration.json 全部参数匹配

### 变体脚本开发原则

**始终复制原版最小修改**，不要从零重写。确保相机设置、场景加载、体素查询代码完全一致，仅替换图像保存管线。

## Licenses

- Core code: BSD-3-Clause
- `isaaclab_mimic`: Apache 2.0 (separate `LICENSE-mimic`)
