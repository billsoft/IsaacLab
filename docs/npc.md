# NPC 角色动画集成指南

## 核心结论

| 方案 | 动画 | Standalone 兼容 | 复杂度 | 推荐 |
|------|------|----------------|--------|------|
| Isaac People + omni.anim.people | ✅ | ❌ GUI 限定（Python OGN 注册失败） | 高 | 仅 GUI |
| **Mixamo + UsdSkel 直接播放** | ✅ | ✅ | 中 | **推荐** |
| 胶囊体占位符 | N/A | ✅ | 低 | 纯数据收集 |

**为什么 Mixamo + UsdSkel 在 Standalone 下可行：**

Isaac People 角色的动画依赖 `omni.anim.graph.core` 的 locomotion 状态机选择动画片段——
这个系统在 Standalone 模式下 Python OGN 注册失败，动画无法播放，角色保持 T-pose。

Mixamo 导出的 USD 包含直接的 UsdSkel 时间采样数据（SkelAnimation prim），
不需要任何状态机——`timeline.play()` 时间轴推进，骨骼关键帧自动插值，角色自然行走。
位置控制仍用我们的 XformOp Wrapper 方案。

---

## 工作流程

### 第一步：从 Mixamo 下载资产

访问 **mixamo.com**（Adobe 账号免费注册）：

1. 搜索选择一个人物角色（推荐 "Y Bot" 或任意人形角色）
2. 先下载**带皮肤的角色 + Walk 动画**（Format: FBX, Skin: With Skin, Frames: 30fps）
3. 再下载其他动画（如 Idle）时选 **Without Skin**，绑定到同一个角色

推荐动画：
- `Walking`（标准行走循环，约 1 秒/循环）
- `Idle`（站立等待）
- `Female Walking`（女性角色）

### 第二步：Blender 处理（必须步骤——添加 Root Bone）

Mixamo 默认没有根骨骼，Hips 直接是骨架根。这会导致行走时角色漂移。
我们需要加一个固定在原点的 Root 骨骼作为 Hips 的父级。

**安装 Blender**（免费，官网下载 4.x 版本）

**操作步骤：**

```
1. File → Import → FBX → 选择下载的 FBX 文件
   （Import 设置：Armature → Automatic Bone Orientation ✓）

2. 在 Outliner 里选中角色的 Armature

3. 进入 Edit Mode（Tab 键）

4. 添加根骨骼：
   Shift+A → Bone
   将新骨骼命名为 "Root"
   位置设为原点 (0, 0, 0)，方向朝上

5. 将 "Hips" 骨骼的 Parent 设为 "Root"
   选中 Hips → Bone Properties → Parent → 选 Root

6. 回到 Object Mode

7. 导出 FBX：
   File → Export → FBX
   设置：
     - Apply Scalings: FBX Units Scale
     - Forward: -Z Forward
     - Up: Y Up
     - Armature → Add Leaf Bones: ✗ 不勾选
     - Baked Animation ✓
```

### 第三步：FBX 转 USD（使用 Isaac Sim 内置转换器）

Isaac Sim 内置了 `omni.kit.asset_converter`，可以直接将 FBX 转为 USD：

**方式 A：命令行转换脚本**（推荐，一次性执行）

```python
# scripts/tools/convert_fbx_to_usd.py
# 运行：isaaclab.bat -p scripts/tools/convert_fbx_to_usd.py

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input",  required=True, help="输入 FBX 路径")
parser.add_argument("--output", required=True, help="输出 USD 路径")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import asyncio
import omni.kit.asset_converter as converter

async def convert(input_path: str, output_path: str):
    ctx = converter.AssetConverterContext()
    ctx.ignore_animation     = False   # 保留动画数据
    ctx.export_animations    = True
    ctx.bake_all_blend_shapes = False
    ctx.merge_all_meshes     = False

    instance = converter.get_instance()
    task = instance.create_converter_task(input_path, output_path, None, ctx)
    success = await task.wait_until_finished()
    if success:
        print(f"[Convert] 成功：{output_path}")
    else:
        print(f"[Convert] 失败：{task.get_status_string()}")

asyncio.ensure_future(convert(args.input, args.output))
for _ in range(200):
    simulation_app.update()

simulation_app.close()
```

**运行示例：**
```bat
isaaclab.bat -p scripts/tools/convert_fbx_to_usd.py ^
  --input  D:/Downloads/mixamo_walk.fbx ^
  --output D:/code/IsaacLab/Assets/Custom/Characters/mixamo_worker/mixamo_worker.usd ^
  --headless
```

**方式 B：Isaac Sim GUI 手动转换**
```
File → Import → FBX
保存为 USD 即可
```

### 第四步：验证 USD 结构

转换成功的 USD 应包含：

```
/Root                     ← 场景根
  /Armature               ← SkelRoot prim（UsdSkelBindingAPI）
    /Armature/Skeleton    ← Skeleton prim（关节层级）
    /mixamorig:Hips/...   ← 网格 prim（蒙皮绑定）
    /mixamorig:Hips_anim  ← SkelAnimation prim（行走关键帧数据）
```

验证脚本（可选）：
```python
# 在 Python 中快速检查 USD 结构
from pxr import Usd, UsdSkel
stage = Usd.Stage.Open("D:/code/IsaacLab/Assets/Custom/Characters/mixamo_worker/mixamo_worker.usd")
for prim in stage.Traverse():
    if prim.IsA(UsdSkel.Root) or prim.IsA(UsdSkel.Animation):
        print(prim.GetPath(), prim.GetTypeName())
# 应看到 SkelRoot 和 SkelAnimation 各至少一个
```

---

## 与巡逻脚本的集成

完整集成方案：`scripts/npc/npc_mixamo_patrol.py`

### 关键设计

```
/World/Characters/
  Worker_01/               ← Wrapper Xform（PatrolAgent 控制位置和朝向）
    Char/                  ← Mixamo USD 引用（timeline 推进时 UsdSkel 自动播放行走动画）
```

`PatrolAgent` 每帧：
1. 更新 Wrapper 的 `translate` op → 角色移动到新位置
2. 更新 Wrapper 的 `rotateY` op → 角色面朝行进方向
3. `timeline.play()` 让时间轴自动推进 → SkelAnimation 关键帧自动插值 → 腿/手臂动作

动画循环：设置 `timeline.set_looping(True)` 加上行走循环的时长（通常 30-60 帧）。

---

## 文件存放建议

```
D:/code/IsaacLab/Assets/Custom/Characters/
  mixamo_worker_male/
    mixamo_worker_male.usd       ← 主 USD（含骨架 + 行走动画）
    mixamo_worker_male_idle.usd  ← 可选：独立 idle 动画
  mixamo_worker_female/
    mixamo_worker_female.usd
```

---

## 已知问题和解决方案

| 问题 | 原因 | 解决 |
|------|------|------|
| 转换后角色漂移/滑步 | Mixamo 无 Root bone，Hips 直接平移 | Blender 加 Root bone（见第二步） |
| 角色侧身倒地 | FBX 导出轴设置错误 | Blender 导出时 Forward=-Z, Up=Y |
| 动画不循环 | 时间轴到末尾停止 | `timeline.set_looping(True)` |
| 转换后无动画数据 | 转换时 ignore_animation=True | 确认 `ctx.ignore_animation = False` |
| T-pose（无关键帧） | SkelAnimation prim 路径错误 | 验证 USD 结构（见第四步） |
| 规模比例错误（角色很小或很大） | FBX 单位 cm vs m | Blender 导出 Apply Scalings: FBX Units Scale |

---

## 与 Isaac People 的对比

| 特性 | Isaac People | Mixamo + UsdSkel |
|------|-------------|-----------------|
| 动画驱动方式 | omni.anim.graph.core 状态机 | UsdSkel 时间采样（直接播放） |
| Standalone 模式 | ❌ Python OGN 注册失败 | ✅ timeline.play() 直接工作 |
| 自定义动画种类 | 少（官方提供） | 多（几千个 Mocap） |
| NavMesh 路径跟随 | ✅ 内置 | ❌ 需要自己用 XformOp 驱动 |
| 资产来源 | 随 Isaac Sim 分发 | 需要下载 + 转换 |
| 准备工作量 | 零 | 中等（Blender + 转换） |
