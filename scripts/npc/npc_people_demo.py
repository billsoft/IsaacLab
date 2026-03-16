"""
NPC 行人演示脚本 —— omni.anim.people
=========================================
在 Digital_Twin_Warehouse 场景中添加 3 个行走的 NPC，
按指定路径点循环移动，并实时打印位置。

运行方式：
    isaaclab.bat -p scripts/npc/npc_people_demo.py
    isaaclab.bat -p scripts/npc/npc_people_demo.py --headless   (无头模式)
"""

import argparse
import os

# ── 1. AppLauncher 必须最先初始化 ──────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="NPC people demo with omni.anim.people")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── 2. 其余 import 必须在 AppLauncher 之后 ────────────────────────────────────
import asyncio
import time

import carb
import carb.settings
import omni.client
import omni.kit.app
import omni.kit.commands
import omni.usd
from pxr import Sdf, UsdGeom, Gf

# ── 路径常量 ──────────────────────────────────────────────────────────────────
ASSET_ROOT = "D:/code/IsaacLab/Assets/Isaac/5.1"
SCENE_USD   = f"{ASSET_ROOT}/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd"
PEOPLE_ROOT = f"{ASSET_ROOT}/Isaac/People/Characters"
CMD_FILE    = os.path.abspath(os.path.join(os.path.dirname(__file__), "commands/people_commands.txt"))

# 3 个 NPC：名字、USD 文件、初始位置 (x, y, z)
NPC_CONFIG = [
    {
        "name":     "Worker_01",
        "usd":      f"{PEOPLE_ROOT}/male_adult_construction_01_new/male_adult_construction_01_new.usd",
        "position": (0.0, 0.0, 2.0),
    },
    {
        "name":     "Worker_02",
        "usd":      f"{PEOPLE_ROOT}/F_Business_02/F_Business_02.usd",
        "position": (0.0, 0.0, 0.0),
    },
    {
        "name":     "Worker_03",
        "usd":      f"{PEOPLE_ROOT}/F_Medical_01/F_Medical_01.usd",
        "position": (2.0, 0.0, 0.0),
    },
]

CHARACTER_ROOT_PRIM = "/World/Characters"


def enable_extensions():
    """启用 omni.anim.people 及其依赖扩展"""
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
            carb.log_info(f"[NPC] Enabled extension: {ext}")


def configure_people_settings():
    """配置 omni.anim.people 的 carb 设置"""
    s = carb.settings.get_settings()

    # 命令文件路径（forward slash，omni.client 要求）
    cmd_path = CMD_FILE.replace("\\", "/")
    s.set("/exts/omni.anim.people/command_settings/command_file_path", cmd_path)

    # 无限循环
    s.set("/exts/omni.anim.people/command_settings/number_of_loop", "inf")

    # 启用 NavMesh（GoTo 命令需要 NavMesh 驱动移动）
    s.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
    s.set("/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", False)

    # 角色 prim 根路径
    s.set("/persistent/exts/omni.anim.people/character_prim_path", CHARACTER_ROOT_PRIM)

    carb.log_info(f"[NPC] Command file: {cmd_path}")


def load_scene():
    """加载仓库场景"""
    context = omni.usd.get_context()
    stage = context.get_stage()

    # 如果 Stage 已有内容就直接用，否则打开 USD 场景
    if stage.GetPrimAtPath("/World").IsValid():
        carb.log_info("[NPC] Existing stage detected, skipping scene load.")
        return stage

    carb.log_info(f"[NPC] Opening scene: {SCENE_USD}")
    context.open_stage(SCENE_USD)

    # 等待场景加载完成（最多 30 秒）
    for _ in range(300):
        simulation_app.update()
        stage = context.get_stage()
        if stage.GetPrimAtPath("/World").IsValid():
            break
        time.sleep(0.1)

    carb.log_info("[NPC] Scene loaded.")
    return context.get_stage()


def add_navmesh_volume(stage):
    """
    在场景中添加 NavMesh Volume prim，覆盖整个仓库地面区域。
    omni.anim.navigation.core 需要至少一个 NavMeshVolume 才能烘焙导航网格。
    """
    vol_path = "/World/NavMesh/Volume"
    if stage.GetPrimAtPath(vol_path).IsValid():
        carb.log_info("[NPC] NavMesh volume already exists, skipping.")
        return

    # 确保父节点存在
    if not stage.GetPrimAtPath("/World/NavMesh").IsValid():
        stage.DefinePrim("/World/NavMesh", "Xform")

    # 创建 NavMeshVolume（需要 omni.anim.navigation.schema）
    try:
        from omni.anim.navigation.core import NavmeshVolumeScheme
        NavmeshVolumeScheme.Apply(stage.DefinePrim(vol_path, "Cube"))
    except Exception:
        # 若 schema API 不可用，手动用 Cube + 语义标签代替
        vol_prim = stage.DefinePrim(vol_path, "Cube")
        vol_prim.CreateAttribute("omni:navmesh:isVolume", Sdf.ValueTypeNames.Bool).Set(True)

    # 设置 Volume 覆盖范围：仓库地面约 20m×1m×20m（XYZ），居中于原点
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(vol_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.5, 0.0))   # 中心稍高于地面
    xform.AddScaleOp().Set(Gf.Vec3f(20.0, 1.0, 20.0))      # 半边长，实际范围 ±20m

    carb.log_info(f"[NPC] NavMesh volume added at {vol_path}")


def bake_navmesh():
    """触发 NavMesh 烘焙并等待完成"""
    try:
        import omni.anim.navigation.core as nav_core
        nav = nav_core.acquire_interface()
        nav.start_navmesh_baking()
        carb.log_info("[NPC] NavMesh baking started.")
    except Exception as e:
        carb.log_warn(f"[NPC] NavMesh baking skipped: {e}")


def add_characters(stage):
    """在 /World/Characters/ 下添加 NPC 并挂载 CharacterBehavior 脚本"""
    # 获取 CharacterBehavior 脚本的绝对路径
    ext_path = omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module("omni.anim.people")
    behavior_script = (ext_path + "/omni/anim/people/scripts/character_behavior.py").replace("\\", "/")
    carb.log_info(f"[NPC] Behavior script: {behavior_script}")

    # 确保 /World/Characters 存在
    if not stage.GetPrimAtPath(CHARACTER_ROOT_PRIM).IsValid():
        stage.DefinePrim(CHARACTER_ROOT_PRIM, "Xform")

    for cfg in NPC_CONFIG:
        prim_path = f"{CHARACTER_ROOT_PRIM}/{cfg['name']}"

        # 跳过已存在的 prim
        if stage.GetPrimAtPath(prim_path).IsValid():
            carb.log_info(f"[NPC] {prim_path} already exists, skipping.")
            continue

        carb.log_info(f"[NPC] Adding {cfg['name']} from {cfg['usd']}")

        # 定义 Xform prim 并添加 USD 引用
        xform_prim = stage.DefinePrim(prim_path, "Xform")
        xform_prim.GetReferences().AddReference(cfg["usd"])

        # 设置初始位置
        x, y, z = cfg["position"]
        xform = UsdGeom.Xformable(xform_prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(x, y, z))

        # 挂载 CharacterBehavior BehaviorScript
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(prim_path)])
        attr = xform_prim.GetAttribute("omni:scripting:scripts")
        current = list(attr.Get()) if attr.Get() else []
        if behavior_script not in current:
            current.insert(0, behavior_script)
            attr.Set(current)

        carb.log_info(f"[NPC] {cfg['name']} added at ({x}, {y}, {z})")


def get_npc_positions():
    """读取所有 NPC 的当前世界坐标"""
    # ── 方法1：GlobalCharacterPositionManager（尝试多种键格式）──────────────
    mgr_positions = {}
    try:
        from omni.anim.people.scripts.global_character_position_manager import GlobalCharacterPositionManager
        mgr = GlobalCharacterPositionManager.get_instance()
        all_keys = set(mgr._character_positions.keys())

        for name in [c["name"] for c in NPC_CONFIG]:
            prim_path = f"{CHARACTER_ROOT_PRIM}/{name}"
            pos = (
                mgr._character_positions.get(prim_path)
                or mgr._character_positions.get(f"{prim_path}/SkelRoot")
                or mgr._character_positions.get(name)   # 只用名字作为键
            )
            # 若还是 None，尝试遍历所有键找包含该名字的条目
            if pos is None:
                for key, val in mgr._character_positions.items():
                    if name in key and val is not None:
                        pos = val
                        break
            mgr_positions[name] = pos
    except Exception:
        pass

    # ── 方法2：直接读 USD XformOp（始终作为兜底，且作为运动真值）──────────
    stage = omni.usd.get_context().get_stage()
    positions = {}
    for cfg in NPC_CONFIG:
        name = cfg["name"]
        prim = stage.GetPrimAtPath(f"{CHARACTER_ROOT_PRIM}/{name}")
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            mat = xform.ComputeLocalToWorldTransform(0)
            t = mat.ExtractTranslation()
            usd_pos = (t[0], t[1], t[2])
        else:
            usd_pos = None

        # 优先用 manager 结果（包含动画驱动的位置），否则用 USD XformOp
        positions[name] = mgr_positions.get(name) or usd_pos
    return positions


def main():
    # ── Step 1: 启用扩展 ──────────────────────────────────────────────────────
    print("[NPC] Step 1: Enabling extensions...")
    enable_extensions()
    simulation_app.update()

    # ── Step 2: 配置 omni.anim.people 设置 ───────────────────────────────────
    print("[NPC] Step 2: Configuring people settings...")
    configure_people_settings()

    # ── Step 3: 加载场景 ──────────────────────────────────────────────────────
    print("[NPC] Step 3: Loading scene...")
    stage = load_scene()

    # ── Step 4: 添加 NavMesh Volume 并烘焙导航网格 ───────────────────────────
    print("[NPC] Step 4: Setting up NavMesh...")
    add_navmesh_volume(stage)
    for _ in range(5):
        simulation_app.update()
    bake_navmesh()
    # 等待烘焙完成（通常需要几帧）
    for _ in range(30):
        simulation_app.update()

    # ── Step 5: 添加 NPC 角色 + 挂载行为脚本 ─────────────────────────────────
    print("[NPC] Step 5: Adding NPC characters...")
    add_characters(stage)

    # 让 Kit 处理一帧，使 Scripting API 生效
    for _ in range(10):
        simulation_app.update()

    # ── Step 6: 开始仿真，读取位置 ────────────────────────────────────────────
    print("[NPC] Step 6: Starting simulation... Press Ctrl+C to stop.")
    print("[NPC] NPCs will walk along paths defined in commands/people_commands.txt")
    print("-" * 60)

    frame = 0
    report_interval = 120  # 每 120 帧打印一次位置（约 2 秒）

    try:
        while simulation_app.is_running():
            simulation_app.update()
            frame += 1

            if frame % report_interval == 0:
                positions = get_npc_positions()
                print(f"\n[NPC] Frame {frame} — NPC Positions:")
                for name, pos in positions.items():
                    if pos is not None:
                        if hasattr(pos, 'x'):
                            print(f"  {name}: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
                        else:
                            print(f"  {name}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                    else:
                        print(f"  {name}: position not yet available")

    except KeyboardInterrupt:
        print("\n[NPC] Stopped by user.")

    simulation_app.close()


if __name__ == "__main__":
    main()
