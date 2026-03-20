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

# 属性名必须用 k0-k4（不是 p0-p4！）
prim.GetAttribute("omni:lensdistortion:ftheta:k0").Set(0.0)
prim.GetAttribute("omni:lensdistortion:ftheta:k1").Set(float(K1_EQUIDISTANT))
# ...

# opticalCenter 是元组（不是分开的 opticalCentreX/Y！）
prim.GetAttribute("omni:lensdistortion:ftheta:opticalCenter").Set((cx, cy))
prim.GetAttribute("omni:lensdistortion:ftheta:maxFov").Set(157.2)
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
    attrs = {
        "omni:lensdistortion:model": "ftheta",
        "omni:lensdistortion:ftheta:nominalWidth": float(CAM_W),
        "omni:lensdistortion:ftheta:nominalHeight": float(CAM_H),
        "omni:lensdistortion:ftheta:opticalCentreX": float(cx),
        "omni:lensdistortion:ftheta:opticalCentreY": float(cy),
        "omni:lensdistortion:ftheta:maxFov": float(DIAG_FOV_DEG),
        "omni:lensdistortion:ftheta:p0": 0.0,
        "omni:lensdistortion:ftheta:p1": float(K1_EQUIDISTANT),
        "omni:lensdistortion:ftheta:p2": 0.0,
        "omni:lensdistortion:ftheta:p3": 0.0,
        "omni:lensdistortion:ftheta:p4": 0.0,
        "fStop": 0.0,
    }
    for attr_name, val in attrs.items():
        attr = prim.GetAttribute(attr_name)
        if not attr:
            if isinstance(val, float):
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
            elif isinstance(val, str):
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.String)
        if attr:
            attr.Set(val)
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

## Licenses

- Core code: BSD-3-Clause
- `isaaclab_mimic`: Apache 2.0 (separate `LICENSE-mimic`)
