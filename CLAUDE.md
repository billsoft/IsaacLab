# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 沟通语言

**始终用中文（简体中文）与用户沟通。** 所有回复、解释、问题均使用中文。
**严禁使用朝鲜语（한국어）和日语（日本語/ひらがな/カタカナ）文字。** 若需引用代码注释中的外文，保留原文但解释用中文。

## Overview

Isaac Lab is a GPU-accelerated robotics simulation framework built on NVIDIA Isaac Sim. It supports reinforcement learning, imitation learning, and motion planning with sim-to-real transfer. Requires Isaac Sim 4.5/5.0/5.1 installed separately.

## Python 环境（重要）

本项目运行在 **Windows Isaac Lab 路线 A** 环境，Python 解释器为 Isaac Sim 内置版本：
```
D:\code\IsaacLab\_isaac_sim\python.bat
```

**绝对不要**使用系统 Python、conda 或裸 `pip` 命令。

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

## Licenses

- Core code: BSD-3-Clause
- `isaaclab_mimic`: Apache 2.0 (separate `LICENSE-mimic`)
