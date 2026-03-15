# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 沟通语言

**始终用中文与用户沟通。** 所有回复、解释、问题均使用中文。

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

## Licenses

- Core code: BSD-3-Clause
- `isaaclab_mimic`: Apache 2.0 (separate `LICENSE-mimic`)
