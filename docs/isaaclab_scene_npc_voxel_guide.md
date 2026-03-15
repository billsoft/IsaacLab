# Isaac Lab 场景 / NPC / 体素数据集完整使用指南

> 适用版本：Isaac Sim 5.1 + Isaac Lab (main)
> 运行环境：`isaaclab.bat -p <脚本>`（Windows Isaac Sim 内置 Python）

---

## 目录

1. [加载不同场景](#1-加载不同场景)
2. [加载 NPC 并控制行走](#2-加载-npc-并控制行走)
3. [获取场景与 Actor 信息](#3-获取场景与-actor-信息)
4. [体素化空间并获取每格 Actor](#4-体素化空间并获取每格-actor)
5. [多视角相机 + 体素运动数据集准备](#5-多视角相机--体素运动数据集准备)
6. [完整脚本模板](#6-完整脚本模板)

---

## 1. 加载不同场景

### 1.1 本地资产路径

```
D:/code/IsaacLab/Assets/Isaac/5.1/Isaac/Environments/
├── Digital_Twin_Warehouse/
│   └── small_warehouse_digital_twin.usd   ← 数字孪生仓库（推荐）
├── Simple_Warehouse/
│   └── full_warehouse.usd
├── Hospital/
│   └── hospital.usd
├── Office/
│   └── office.usd
└── Grid/
    └── default_environment.usd            ← 空网格地面（调试用）
```

### 1.2 代码加载方式

```python
import omni.usd
import time

SCENE_USD = "D:/code/IsaacLab/Assets/Isaac/5.1/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd"

def load_scene(simulation_app):
    context = omni.usd.get_context()
    stage = context.get_stage()

    # 若已有场景则跳过
    if stage.GetPrimAtPath("/World").IsValid():
        return stage

    context.open_stage(SCENE_USD)
    for _ in range(300):          # 最多等 30 秒
        simulation_app.update()
        stage = context.get_stage()
        if stage.GetPrimAtPath("/World").IsValid():
            break
        time.sleep(0.1)
    return context.get_stage()
```

### 1.3 在代码中切换场景

```python
# 关闭当前场景，打开新场景
context = omni.usd.get_context()
context.new_stage()                              # 清空
context.open_stage("另一个场景.usd")
```

### 1.4 空场景快速创建（无 USD 文件）

```python
from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils

# 地面平面
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/ground", cfg)

# 灯光
cfg_light = sim_utils.DomeLightCfg(intensity=3000.0)
cfg_light.func("/World/light", cfg_light)
```

---

## 2. 加载 NPC 并控制行走

### 2.1 可用人物模型路径

```
D:/code/IsaacLab/Assets/Isaac/5.1/Isaac/People/Characters/
├── male_adult_construction_01_new/    ← 建筑工人（男）
├── male_adult_police_01/              ← 警察（男）
├── F_Business_02/                     ← 商务人士（女）
├── F_Medical_01/                      ← 医护人员（女）
├── F_Nurse_01/                        ← 护士
└── ...（共 24 个角色）
```

### 2.2 命令文件格式（people_commands.txt）

```text
# 格式：AgentName CommandName [params...]
# AgentName = /World/Characters/ 下的 prim 名称

# GoTo：走向目标点，最后一个参数是朝向角（度）
Worker_01 GoTo 8.0 0.0 2.0 0
Worker_01 Idle 2.0              # 等待 2 秒
Worker_01 GoTo -4.0 0.0 2.0 180

# LookAround：原地环顾
Worker_02 LookAround 3.0

# 所有命令执行完后自动循环（需设置 number_of_loop=inf）
```

**坐标系说明：** X 正方向向右、Z 正方向向前（Isaac Sim 默认 Y-up 但 NPC 命令用 XZ 平面移动）

### 2.3 完整 NPC 加载代码

```python
import carb
import carb.settings
import omni.kit.app
import omni.kit.commands
import omni.usd
from pxr import Sdf, UsdGeom, Gf

ASSET_ROOT    = "D:/code/IsaacLab/Assets/Isaac/5.1"
PEOPLE_ROOT   = f"{ASSET_ROOT}/Isaac/People/Characters"
CHARACTER_ROOT = "/World/Characters"

# ── 步骤 1：启用扩展 ──────────────────────────────────────────────
def enable_npc_extensions():
    mgr = omni.kit.app.get_app().get_extension_manager()
    for ext in [
        "omni.anim.graph.core",
        "omni.anim.retarget.core",
        "omni.anim.navigation.core",
        "omni.kit.scripting",
        "omni.anim.people",
    ]:
        if not mgr.is_extension_enabled(ext):
            mgr.set_extension_enabled_immediate(ext, True)

# ── 步骤 2：配置 NPC 系统 ─────────────────────────────────────────
def configure_npc(cmd_file_path: str):
    s = carb.settings.get_settings()
    s.set("/exts/omni.anim.people/command_settings/command_file_path",
          cmd_file_path.replace("\\", "/"))
    s.set("/exts/omni.anim.people/command_settings/number_of_loop", "inf")
    s.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", False)
    s.set("/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", False)
    s.set("/persistent/exts/omni.anim.people/character_prim_path", CHARACTER_ROOT)

# ── 步骤 3：添加角色到场景 ────────────────────────────────────────
def add_npc(stage, name: str, usd_path: str, position: tuple):
    """
    Args:
        name:     角色名，同时作为命令文件中的 AgentName
        usd_path: 角色 USD 文件路径
        position: (x, y, z) 世界坐标初始位置
    """
    prim_path = f"{CHARACTER_ROOT}/{name}"
    if stage.GetPrimAtPath(prim_path).IsValid():
        return  # 已存在则跳过

    # 获取 BehaviorScript 路径
    ext_path = omni.kit.app.get_app().get_extension_manager() \
                   .get_extension_path_by_module("omni.anim.people")
    behavior_script = (ext_path + "/omni/anim/people/scripts/character_behavior.py").replace("\\", "/")

    # 确保根节点存在
    if not stage.GetPrimAtPath(CHARACTER_ROOT).IsValid():
        stage.DefinePrim(CHARACTER_ROOT, "Xform")

    # 创建 Xform + 引用 USD
    xform_prim = stage.DefinePrim(prim_path, "Xform")
    xform_prim.GetReferences().AddReference(usd_path)

    # 设置初始位置
    xform = UsdGeom.Xformable(xform_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    # 挂载 BehaviorScript
    omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(prim_path)])
    attr = xform_prim.GetAttribute("omni:scripting:scripts")
    scripts = list(attr.Get()) if attr.Get() else []
    if behavior_script not in scripts:
        scripts.insert(0, behavior_script)
        attr.Set(scripts)
```

### 2.4 动态修改 NPC 命令（运行时）

```python
# 方法一：修改命令文件（NPC 循环完当前周期后生效）
with open("commands/people_commands.txt", "w") as f:
    f.write("Worker_01 GoTo 0.0 0.0 0.0 0\n")
    f.write("Worker_01 Idle 1.0\n")

# 方法二：直接调用 NPC 系统 API（立即生效）
from omni.anim.people.scripts.commands.goto_command import GotoCommand
# 暂无公开 API 支持直接注入命令，推荐方式是修改命令文件
```

### 2.5 停止 / 重置 NPC

```python
import carb.settings
s = carb.settings.get_settings()
s.set("/exts/omni.anim.people/command_settings/number_of_loop", "0")  # 停止循环
```

---

## 3. 获取场景与 Actor 信息

### 3.1 获取所有 NPC 当前位置

```python
# 方法一：GlobalCharacterPositionManager（推荐）
from omni.anim.people.scripts.global_character_position_manager import GlobalCharacterPositionManager

def get_all_npc_positions() -> dict:
    """返回 {prim_path: (x, y, z)} 字典"""
    mgr = GlobalCharacterPositionManager.get_instance()
    return dict(mgr._character_positions)

# 使用示例
positions = get_all_npc_positions()
for path, pos in positions.items():
    print(f"{path}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
```

```python
# 方法二：直接读 USD XformOp（备用）
from pxr import UsdGeom
import omni.usd

def get_npc_position_usd(prim_path: str) -> tuple:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(0)
    t = mat.ExtractTranslation()
    return (t[0], t[1], t[2])
```

### 3.2 获取 NPC 运动速度和朝向

```python
from pxr import UsdGeom, Gf
import omni.usd

def get_npc_velocity_and_facing(prim_path: str, dt: float = 1/60):
    """
    通过两帧位置差估算速度向量和朝向。
    dt: 帧间隔秒数（默认 60 FPS）
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)

    # 当前帧和上一帧的 world transform
    mat_now  = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t_now    = mat_now.ExtractTranslation()

    # 速度需要两帧数据，建议在主循环中缓存上一帧位置
    # velocity = (pos_now - pos_prev) / dt

    # 朝向：从旋转矩阵提取 forward 向量（-Z 轴在 Isaac Sim Y-up 坐标系）
    rot = mat_now.ExtractRotation()
    forward = rot.TransformDir(Gf.Vec3d(0, 0, -1))
    return t_now, forward
```

### 3.3 获取场景所有物理 Actor

```python
# 使用 PhysX 接口枚举所有 RigidBody
import omni.physx
from pxr import UsdPhysics, Usd
import omni.usd

def get_all_rigid_bodies() -> list:
    """返回场景中所有 RigidBody prim 路径列表"""
    stage = omni.usd.get_context().get_stage()
    rigid_bodies = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(str(prim.GetPath()))
    return rigid_bodies

# 获取刚体的物理状态（位置/速度）
from isaacsim.core.utils.stage import get_stage_units
import numpy as np

def get_rigid_body_state(prim_path: str) -> dict:
    from isaacsim.core.prims import RigidPrim
    obj = RigidPrim(prim_path=prim_path)
    pos, orient = obj.get_world_pose()       # numpy arrays
    vel_linear, vel_angular = obj.get_velocities()
    return {
        "position":        pos,              # [x, y, z]
        "orientation":     orient,           # quaternion [w, x, y, z]
        "linear_velocity": vel_linear,       # [vx, vy, vz] m/s
        "angular_velocity": vel_angular,     # [wx, wy, wz] rad/s
    }
```

### 3.4 枚举场景中所有 NPC 角色类型

```python
def get_npc_actor_info() -> list:
    """
    返回每个 NPC 的基本信息字典列表
    """
    stage = omni.usd.get_context().get_stage()
    result = []
    chars_prim = stage.GetPrimAtPath("/World/Characters")
    if not chars_prim.IsValid():
        return result
    for child in chars_prim.GetChildren():
        path = str(child.GetPath())
        name = child.GetName()
        # 判断是否有 BehaviorScript（即是否为受控 NPC）
        has_behavior = child.HasAttribute("omni:scripting:scripts")
        # 获取位置
        xform = UsdGeom.Xformable(child)
        mat = xform.ComputeLocalToWorldTransform(0)
        pos = mat.ExtractTranslation()
        result.append({
            "prim_path":    path,
            "name":         name,
            "actor_type":   "NPC_Character",
            "has_behavior": has_behavior,
            "position":     (pos[0], pos[1], pos[2]),
        })
    return result
```

---

## 4. 体素化空间并获取每格 Actor

### 4.1 体素网格设计

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

@dataclass
class VoxelGrid:
    """
    三维均匀体素网格

    Attributes:
        origin:      世界坐标系下网格原点 (x, y, z)
        voxel_size:  每个体素的边长（米）
        grid_size:   网格维度 (nx, ny, nz)
    """
    origin:     np.ndarray         # shape (3,)
    voxel_size: float              # 例如 0.5 米
    grid_size:  Tuple[int,int,int] # (nx, ny, nz)

    # 每个体素存储的信息
    # actor_ids[i,j,k]   = 该格中所有 actor 的 prim_path 列表
    # actor_types[i,j,k] = 该格中所有 actor 的类型字符串列表
    actor_ids:   Dict = field(default_factory=dict)
    actor_types: Dict = field(default_factory=dict)

    def world_to_voxel(self, world_pos: np.ndarray) -> Optional[Tuple[int,int,int]]:
        """世界坐标 → 体素索引，越界返回 None"""
        idx = ((world_pos - self.origin) / self.voxel_size).astype(int)
        nx, ny, nz = self.grid_size
        if (0 <= idx[0] < nx) and (0 <= idx[1] < ny) and (0 <= idx[2] < nz):
            return tuple(idx)
        return None

    def voxel_to_world_center(self, voxel_idx: Tuple[int,int,int]) -> np.ndarray:
        """体素索引 → 该体素中心的世界坐标"""
        return self.origin + (np.array(voxel_idx) + 0.5) * self.voxel_size
```

### 4.2 将 NPC 映射到体素

```python
def update_voxel_grid_with_npcs(grid: VoxelGrid) -> VoxelGrid:
    """
    遍历所有 NPC，将其当前位置写入体素网格

    体素内容格式：
        actor_ids[(i,j,k)]   = ["/World/Characters/Worker_01", ...]
        actor_types[(i,j,k)] = ["NPC_walking", ...]
    """
    # 清空上一帧数据
    grid.actor_ids.clear()
    grid.actor_types.clear()

    try:
        from omni.anim.people.scripts.global_character_position_manager import \
            GlobalCharacterPositionManager
        mgr = GlobalCharacterPositionManager.get_instance()
        char_positions = mgr._character_positions
    except Exception:
        char_positions = {}

    for prim_path, pos in char_positions.items():
        if pos is None:
            continue
        if hasattr(pos, 'x'):
            world_pos = np.array([pos.x, pos.y, pos.z])
        else:
            world_pos = np.array([pos[0], pos[1], pos[2]])

        voxel_idx = grid.world_to_voxel(world_pos)
        if voxel_idx is None:
            continue

        if voxel_idx not in grid.actor_ids:
            grid.actor_ids[voxel_idx]   = []
            grid.actor_types[voxel_idx] = []

        grid.actor_ids[voxel_idx].append(prim_path)
        grid.actor_types[voxel_idx].append("NPC_Character")

    return grid
```

### 4.3 将静态场景物体也映射到体素

```python
from pxr import UsdGeom, Gf, UsdPhysics
import omni.usd

STATIC_ACTOR_TYPES = {
    "UsdGeom.Mesh":          "StaticMesh",
    "UsdPhysics.RigidBody":  "RigidBody",
    "UsdGeom.Xform":         "Xform",
}

def populate_voxel_static_scene(grid: VoxelGrid):
    """将场景中的静态/物理物体投影到体素网格（仅初始化时调用一次）"""
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        # 跳过 NPC（已在动态更新中处理）
        if str(prim.GetPath()).startswith("/World/Characters"):
            continue
        # 只处理有几何体的 prim
        if not prim.IsA(UsdGeom.Gprim) and not prim.IsA(UsdGeom.Xform):
            continue

        xformable = UsdGeom.Xformable(prim)
        mat = xformable.ComputeLocalToWorldTransform(0)
        pos = mat.ExtractTranslation()
        world_pos = np.array([pos[0], pos[1], pos[2]])

        voxel_idx = grid.world_to_voxel(world_pos)
        if voxel_idx is None:
            continue

        if voxel_idx not in grid.actor_ids:
            grid.actor_ids[voxel_idx]   = []
            grid.actor_types[voxel_idx] = []

        prim_path = str(prim.GetPath())
        actor_type = "RigidBody" if prim.HasAPI(UsdPhysics.RigidBodyAPI) else "StaticObject"
        grid.actor_ids[voxel_idx].append(prim_path)
        grid.actor_types[voxel_idx].append(actor_type)
```

### 4.4 查询体素内容

```python
def query_voxel(grid: VoxelGrid, world_pos: np.ndarray) -> dict:
    """查询世界坐标点所在体素中的所有 actor"""
    voxel_idx = grid.world_to_voxel(world_pos)
    if voxel_idx is None:
        return {"voxel_idx": None, "actors": []}

    actors = []
    ids    = grid.actor_ids.get(voxel_idx, [])
    types  = grid.actor_types.get(voxel_idx, [])
    for aid, atype in zip(ids, types):
        actors.append({"prim_path": aid, "actor_type": atype})

    return {
        "voxel_idx":   voxel_idx,
        "world_center": grid.voxel_to_world_center(voxel_idx).tolist(),
        "actors":       actors,
    }

def get_occupied_voxels(grid: VoxelGrid) -> List[dict]:
    """返回所有被占据的体素（含 NPC 或物体）的信息列表"""
    result = []
    for voxel_idx, ids in grid.actor_ids.items():
        types = grid.actor_types.get(voxel_idx, [])
        result.append({
            "voxel_idx":    voxel_idx,
            "world_center": grid.voxel_to_world_center(voxel_idx).tolist(),
            "actor_ids":    ids,
            "actor_types":  types,
        })
    return result
```

---

## 5. 多视角相机 + 体素运动数据集准备

### 5.1 在场景中创建多个固定相机

```python
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from pxr import Gf
import numpy as np

def create_overhead_cameras(stage, num_cameras: int = 4, height: float = 8.0):
    """
    在场景中创建俯视相机阵列

    Args:
        num_cameras: 相机数量，沿圆形均匀分布
        height:      相机高度（米）
    Returns:
        相机 prim 路径列表
    """
    from pxr import UsdGeom
    camera_paths = []
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        x = 6.0 * np.cos(angle)   # 圆半径 6 米
        z = 6.0 * np.sin(angle)
        cam_path = f"/World/Cameras/cam_{i:02d}"

        cam_prim = stage.DefinePrim(cam_path, "Camera")
        xform = UsdGeom.Xformable(cam_prim)
        xform.ClearXformOpOrder()
        # 位置
        xform.AddTranslateOp().Set(Gf.Vec3d(x, height, z))
        # 朝向：俯视（绕 X 轴旋转 -90 度指向下方）
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-90, np.degrees(angle), 0))

        camera_paths.append(cam_path)
    return camera_paths
```

### 5.2 使用 Isaac Lab TiledCamera 采集图像

```python
import torch
from isaaclab.sensors import TiledCameraCfg, TiledCamera
import isaaclab.sim as sim_utils

CAMERA_CFG = TiledCameraCfg(
    prim_path="/World/Cameras/cam_.*",   # 正则匹配所有相机
    update_period=0,                      # 每帧更新
    height=480,
    width=640,
    data_types=["rgb", "depth"],          # 采集 RGB + 深度
    spawn=None,                           # 使用场景中已有相机
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="world",
    ),
)

# 初始化相机传感器
camera = TiledCamera(CAMERA_CFG)

def capture_all_cameras(camera: TiledCamera) -> dict:
    """采集所有相机当帧图像"""
    camera.update(dt=0.0)
    data = camera.data
    return {
        "rgb":   data.output["rgb"],    # shape: (N, H, W, 4) uint8
        "depth": data.output["depth"],  # shape: (N, H, W, 1) float32
    }
```

### 5.3 每帧数据集记录：体素 + NPC 状态 + 图像

```python
import json, os, time
import numpy as np

class DatasetRecorder:
    """
    逐帧记录体素占用状态、NPC 位置/速度、多视角图像

    数据集结构：
        dataset/
        ├── frame_000000/
        │   ├── meta.json          ← 体素 + NPC 状态
        │   ├── cam_00_rgb.png
        │   ├── cam_00_depth.npy
        │   ├── cam_01_rgb.png
        │   └── ...
        └── ...
    """

    def __init__(self, output_dir: str, grid: VoxelGrid):
        self.output_dir = output_dir
        self.grid = grid
        self.frame_idx = 0
        self._prev_positions = {}   # 用于计算速度
        os.makedirs(output_dir, exist_ok=True)

    def record_frame(self, images: dict):
        """
        Args:
            images: {"rgb": tensor(N,H,W,4), "depth": tensor(N,H,W,1)}
        """
        frame_dir = os.path.join(self.output_dir, f"frame_{self.frame_idx:06d}")
        os.makedirs(frame_dir, exist_ok=True)

        # ── 1. 更新体素 ──────────────────────────────────────────
        update_voxel_grid_with_npcs(self.grid)

        # ── 2. 获取 NPC 当前状态 ──────────────────────────────────
        try:
            from omni.anim.people.scripts.global_character_position_manager import \
                GlobalCharacterPositionManager
            char_positions = dict(GlobalCharacterPositionManager.get_instance()._character_positions)
        except Exception:
            char_positions = {}

        npc_states = []
        for path, pos in char_positions.items():
            if pos is None:
                continue
            curr = np.array([pos.x if hasattr(pos, 'x') else pos[0],
                             pos.y if hasattr(pos, 'y') else pos[1],
                             pos.z if hasattr(pos, 'z') else pos[2]])
            prev = self._prev_positions.get(path, curr)
            vel  = curr - prev   # 帧间位移（未除以 dt，可按 FPS 换算）
            speed = float(np.linalg.norm(vel))
            voxel_idx = self.grid.world_to_voxel(curr)
            npc_states.append({
                "prim_path":    path,
                "position":     curr.tolist(),
                "velocity":     vel.tolist(),
                "speed":        speed,
                "voxel_idx":    list(voxel_idx) if voxel_idx else None,
            })
            self._prev_positions[path] = curr

        # ── 3. 序列化体素占用 ──────────────────────────────────────
        occupied = [
            {
                "voxel_idx":    list(k),
                "world_center": self.grid.voxel_to_world_center(k).tolist(),
                "actor_ids":    v,
                "actor_types":  self.grid.actor_types.get(k, []),
            }
            for k, v in self.grid.actor_ids.items()
        ]

        # ── 4. 写 meta.json ───────────────────────────────────────
        meta = {
            "frame":         self.frame_idx,
            "timestamp":     time.time(),
            "grid_origin":   self.grid.origin.tolist(),
            "voxel_size":    self.grid.voxel_size,
            "grid_size":     list(self.grid.grid_size),
            "npcs":          npc_states,
            "occupied_voxels": occupied,
        }
        with open(os.path.join(frame_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # ── 5. 保存图像 ───────────────────────────────────────────
        import cv2
        rgb_np = images["rgb"].cpu().numpy() if hasattr(images["rgb"], "cpu") else images["rgb"]
        dep_np = images["depth"].cpu().numpy() if hasattr(images["depth"], "cpu") else images["depth"]
        for cam_i in range(rgb_np.shape[0]):
            rgb_bgr = cv2.cvtColor(rgb_np[cam_i, :, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(frame_dir, f"cam_{cam_i:02d}_rgb.png"), rgb_bgr)
            np.save(os.path.join(frame_dir, f"cam_{cam_i:02d}_depth.npy"), dep_np[cam_i])

        self.frame_idx += 1
```

### 5.4 数据集后处理：加载并使用

```python
import json, glob, numpy as np

def load_dataset_frame(frame_dir: str) -> dict:
    with open(os.path.join(frame_dir, "meta.json")) as f:
        meta = json.load(f)
    return meta

def get_npc_trajectory(dataset_dir: str, npc_name: str) -> np.ndarray:
    """
    从数据集中提取某个 NPC 的轨迹
    Returns: shape (T, 3) 的位置序列
    """
    frames = sorted(glob.glob(os.path.join(dataset_dir, "frame_*/meta.json")))
    traj = []
    for fp in frames:
        with open(fp) as f:
            meta = json.load(f)
        for npc in meta["npcs"]:
            if npc_name in npc["prim_path"]:
                traj.append(npc["position"])
    return np.array(traj)

def build_voxel_occupancy_sequence(dataset_dir: str,
                                   grid_size: tuple) -> np.ndarray:
    """
    返回体素占用序列，shape (T, nx, ny, nz)
    0 = 空，1 = NPC，2 = 静态物体
    """
    frames = sorted(glob.glob(os.path.join(dataset_dir, "frame_*/meta.json")))
    T = len(frames)
    nx, ny, nz = grid_size
    occupancy = np.zeros((T, nx, ny, nz), dtype=np.uint8)

    for t, fp in enumerate(frames):
        with open(fp) as f:
            meta = json.load(f)
        for vox in meta["occupied_voxels"]:
            i, j, k = vox["voxel_idx"]
            types = vox["actor_types"]
            if "NPC_Character" in types:
                occupancy[t, i, j, k] = 1
            else:
                occupancy[t, i, j, k] = 2
    return occupancy
```

---

## 6. 完整脚本模板

```python
"""
Isaac Lab 多 NPC 体素数据集采集脚本
运行：isaaclab.bat -p scripts/dataset/collect_voxel_dataset.py
"""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Voxel NPC Dataset Collector")
parser.add_argument("--num_frames", type=int, default=1000)
parser.add_argument("--voxel_size", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── 以下 import 必须在 AppLauncher 之后 ────────────────────────────
import numpy as np
import omni.usd

from scripts.npc.npc_people_demo import (   # 复用已有 NPC 工具函数
    enable_extensions, configure_people_settings,
    load_scene, add_characters,
)

SCENE_USD  = "D:/code/IsaacLab/Assets/Isaac/5.1/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd"
CMD_FILE   = "D:/code/IsaacLab/scripts/npc/commands/people_commands.txt"
OUTPUT_DIR = "D:/code/IsaacLab/dataset/warehouse_npc"

def main():
    # 1. 启用扩展
    enable_extensions()
    simulation_app.update()

    # 2. 配置 NPC
    configure_people_settings()

    # 3. 加载场景
    stage = load_scene()

    # 4. 添加 NPC
    add_characters(stage)
    for _ in range(10):
        simulation_app.update()

    # 5. 初始化体素网格（以场景为中心，15m×4m×15m，0.5m 分辨率）
    grid = VoxelGrid(
        origin     = np.array([-7.5, 0.0, -7.5]),
        voxel_size = 0.5,
        grid_size  = (30, 8, 30),
    )
    populate_voxel_static_scene(grid)

    # 6. 创建相机（可选）
    camera_paths = create_overhead_cameras(stage, num_cameras=4, height=8.0)
    # camera = TiledCamera(CAMERA_CFG)  # 取消注释启用图像采集

    # 7. 初始化记录器
    recorder = DatasetRecorder(OUTPUT_DIR, grid)

    # 8. 主循环
    frame = 0
    while simulation_app.is_running() and frame < args.num_frames:
        simulation_app.update()
        frame += 1

        if frame % 5 == 0:   # 每 5 帧记录一次（约 12 FPS）
            # images = capture_all_cameras(camera)
            images = {"rgb": np.zeros((4, 480, 640, 4), dtype=np.uint8),
                      "depth": np.zeros((4, 480, 640, 1), dtype=np.float32)}
            recorder.record_frame(images)

    simulation_app.close()

if __name__ == "__main__":
    main()
```

---

## 附录：常用命令速查

| 操作 | 代码片段 |
|------|---------|
| 获取 stage | `omni.usd.get_context().get_stage()` |
| 获取 prim | `stage.GetPrimAtPath("/World/xxx")` |
| 遍历所有 prim | `for p in stage.Traverse(): ...` |
| NPC 当前位置 | `GlobalCharacterPositionManager.get_instance()._character_positions` |
| 体素查询 | `grid.world_to_voxel(np.array([x,y,z]))` |
| 获取 USD 世界变换 | `UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)` |
| 获取刚体速度 | `RigidPrim(prim_path).get_velocities()` |
| 运行 N 帧 | `for _ in range(N): simulation_app.update()` |

---

*生成时间：2026-03-15 | Isaac Sim 5.1 + Isaac Lab main*
