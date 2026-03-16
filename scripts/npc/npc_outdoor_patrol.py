"""
NPC 室外巡逻演示脚本
====================
场景：Simple_Warehouse（大型开放仓库，加阳光照明）
NPC：3 人各自沿一条直线来回巡逻，实时打印位置

运行方式：
    isaaclab.bat -p scripts/npc/npc_outdoor_patrol.py
    isaaclab.bat -p scripts/npc/npc_outdoor_patrol.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="NPC outdoor patrol demo")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── 以下 import 必须在 AppLauncher 之后 ────────────────────────────────────
import time
import numpy as np

import carb
import carb.settings
import omni.kit.app
import omni.kit.commands
import omni.usd
from pxr import Sdf, Gf, UsdGeom, UsdLux

# ── 路径常量 ──────────────────────────────────────────────────────────────
ASSET_ROOT   = "D:/code/IsaacLab/Assets/Isaac/5.1"
# 大型室内仓库，地面开阔，适合巡逻（约 40m×20m 可用区域）
SCENE_USD    = f"{ASSET_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
PEOPLE_ROOT  = f"{ASSET_ROOT}/Isaac/People/Characters"
CMD_FILE     = "D:/code/IsaacLab/scripts/npc/commands/outdoor_commands.txt"

CHARACTER_ROOT = "/World/Characters"

# ── NPC 配置：名字 / 模型 / 起点 / 终点（直线巡逻）────────────────────────
NPC_PATROL = [
    {
        "name":  "Worker_01",
        "usd":   f"{PEOPLE_ROOT}/male_adult_construction_01_new/male_adult_construction_01_new.usd",
        "start": ( 8.0, 0.0,  2.0),   # 沿 X 轴巡逻（Z=2 排）
        "end":   (-8.0, 0.0,  2.0),
    },
    {
        "name":  "Worker_02",
        "usd":   f"{PEOPLE_ROOT}/F_Business_02/F_Business_02.usd",
        "start": ( 8.0, 0.0,  0.0),   # 中间排
        "end":   (-8.0, 0.0,  0.0),
    },
    {
        "name":  "Worker_03",
        "usd":   f"{PEOPLE_ROOT}/F_Medical_01/F_Medical_01.usd",
        "start": ( 8.0, 0.0, -2.0),   # 另一排
        "end":   (-8.0, 0.0, -2.0),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 1：启用扩展
# ─────────────────────────────────────────────────────────────────────────────
def enable_extensions():
    mgr = omni.kit.app.get_app().get_extension_manager()
    for ext in [
        "omni.anim.graph.core",
        "omni.anim.retarget.core",
        "omni.anim.navigation.schema",   # NavMesh schema（OmniNavVolume 类型）
        "omni.anim.navigation.core",
        "omni.kit.scripting",
        "omni.anim.people",
    ]:
        if not mgr.is_extension_enabled(ext):
            mgr.set_extension_enabled_immediate(ext, True)
            carb.log_info(f"[NPC] Enabled: {ext}")


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 2：生成命令文件
# ─────────────────────────────────────────────────────────────────────────────
def write_command_file():
    """根据 NPC_PATROL 配置动态生成命令文件"""
    import os
    os.makedirs(os.path.dirname(CMD_FILE), exist_ok=True)
    with open(CMD_FILE, "w", encoding="utf-8") as f:
        f.write("# 自动生成的 NPC 巡逻命令\n")
        for npc in NPC_PATROL:
            name = npc["name"]
            sx, sy, sz = npc["start"]
            ex, ey, ez = npc["end"]
            # 计算朝向角（绕 Y 轴，0=+X，90=+Z，180=-X，270=-Z）
            dx = ex - sx
            dz = ez - sz
            angle_to_end = float(np.degrees(np.arctan2(dx, dz)))
            angle_to_start = float(np.degrees(np.arctan2(-dx, -dz)))
            f.write(f"{name} GoTo {ex} {ey} {ez} {angle_to_end:.0f}\n")
            f.write(f"{name} Idle 0.5\n")
            f.write(f"{name} GoTo {sx} {sy} {sz} {angle_to_start:.0f}\n")
            f.write(f"{name} Idle 0.5\n")
    carb.log_info(f"[NPC] Command file written: {CMD_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 3：配置 omni.anim.people
# ─────────────────────────────────────────────────────────────────────────────
def configure_people():
    s = carb.settings.get_settings()
    cmd_path = CMD_FILE.replace("\\", "/")
    s.set("/exts/omni.anim.people/command_settings/command_file_path", cmd_path)
    s.set("/exts/omni.anim.people/command_settings/number_of_loop", "inf")
    s.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
    s.set("/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", False)
    s.set("/persistent/exts/omni.anim.people/character_prim_path", CHARACTER_ROOT)
    carb.log_info(f"[NPC] Config done, cmd: {cmd_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 4：加载场景 + 添加阳光
# ─────────────────────────────────────────────────────────────────────────────
def load_scene_with_sun():
    context = omni.usd.get_context()
    carb.log_info(f"[NPC] Opening: {SCENE_USD}")
    context.open_stage(SCENE_USD)

    for _ in range(300):
        simulation_app.update()
        stage = context.get_stage()
        if stage.GetPrimAtPath("/World").IsValid():
            break
        time.sleep(0.1)

    stage = context.get_stage()
    carb.log_info("[NPC] Scene loaded.")

    # ── 添加定向阳光（如果场景没有足够光照）────────────────────────────────
    sun_path = "/World/Sun"
    if not stage.GetPrimAtPath(sun_path).IsValid():
        sun = UsdLux.DistantLight.Define(stage, sun_path)
        sun.CreateIntensityAttr(5000.0)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
        sun.CreateAngleAttr(0.53)        # 太阳角直径（度）
        # 斜 45° 从右上方照射
        xform = UsdGeom.Xformable(sun.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 45.0, 0.0))
        carb.log_info("[NPC] Sun light added.")

    # ── 天空穹顶光（环境光）────────────────────────────────────────────────
    sky_path = "/World/SkyDome"
    if not stage.GetPrimAtPath(sky_path).IsValid():
        sky = UsdLux.DomeLight.Define(stage, sky_path)
        sky.CreateIntensityAttr(800.0)
        sky.CreateColorAttr(Gf.Vec3f(0.5, 0.7, 1.0))
        carb.log_info("[NPC] Sky dome added.")

    return stage


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 5：添加 NavMesh Volume（关键：使用 OmniNavVolume prim 类型）
# ─────────────────────────────────────────────────────────────────────────────
def add_navmesh_and_bake(stage):
    vol_path = "/World/NavMesh/NavVolume"

    if not stage.GetPrimAtPath("/World/NavMesh").IsValid():
        stage.DefinePrim("/World/NavMesh", "Xform")

    if not stage.GetPrimAtPath(vol_path).IsValid():
        # OmniNavVolume 是 omni.anim.navigation.schema 注册的 prim 类型
        # 必须使用这个类型，普通 Cube+属性 不会被识别
        nav_prim = stage.DefinePrim(vol_path, "OmniNavVolume")

        xform = UsdGeom.Xformable(nav_prim)
        xform.ClearXformOpOrder()
        # 覆盖仓库地面区域：X[-15,15] × Y[0,4] × Z[-10,10]
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 2.0, 0.0))  # 中心高 2m
        xform.AddScaleOp().Set(Gf.Vec3f(15.0, 2.0, 10.0))    # 半边长

        carb.log_info(f"[NPC] OmniNavVolume created at {vol_path}")
    else:
        carb.log_info("[NPC] NavMesh volume exists.")

    # 等一帧让 prim 生效
    for _ in range(5):
        simulation_app.update()

    # 触发 NavMesh 烘焙
    try:
        import omni.anim.navigation.core as nav_core
        nav_iface = nav_core.acquire_interface()
        nav_iface.start_navmesh_baking()
        carb.log_info("[NPC] NavMesh baking triggered.")
    except Exception as e:
        carb.log_warn(f"[NPC] NavMesh bake failed: {e}")

    # 等待烘焙（最多 5 秒）
    for _ in range(60):
        simulation_app.update()
    carb.log_info("[NPC] NavMesh ready.")


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 6：添加 NPC 角色
# ─────────────────────────────────────────────────────────────────────────────
def add_characters(stage):
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_path = ext_mgr.get_extension_path_by_module("omni.anim.people")
    behavior_script = (ext_path + "/omni/anim/people/scripts/character_behavior.py").replace("\\", "/")

    if not stage.GetPrimAtPath(CHARACTER_ROOT).IsValid():
        stage.DefinePrim(CHARACTER_ROOT, "Xform")

    for npc in NPC_PATROL:
        prim_path = f"{CHARACTER_ROOT}/{npc['name']}"
        if stage.GetPrimAtPath(prim_path).IsValid():
            carb.log_info(f"[NPC] {npc['name']} already exists.")
            continue

        xform_prim = stage.DefinePrim(prim_path, "Xform")
        xform_prim.GetReferences().AddReference(npc["usd"])

        x, y, z = npc["start"]
        xform = UsdGeom.Xformable(xform_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))

        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(prim_path)])
        attr = xform_prim.GetAttribute("omni:scripting:scripts")
        scripts = list(attr.Get()) if attr.Get() else []
        if behavior_script not in scripts:
            scripts.insert(0, behavior_script)
            attr.Set(scripts)

        carb.log_info(f"[NPC] Added {npc['name']} at ({x}, {y}, {z})")


# ─────────────────────────────────────────────────────────────────────────────
# 位置读取：GlobalCharacterPositionManager + USD XformOp 双保险
# ─────────────────────────────────────────────────────────────────────────────
def get_positions() -> dict:
    # 方法1：尝试 GlobalCharacterPositionManager
    mgr_pos = {}
    try:
        from omni.anim.people.scripts.global_character_position_manager import \
            GlobalCharacterPositionManager
        mgr = GlobalCharacterPositionManager.get_instance()
        for npc in NPC_PATROL:
            name = npc["name"]
            path = f"{CHARACTER_ROOT}/{name}"
            pos = (mgr._character_positions.get(path)
                   or mgr._character_positions.get(f"{path}/SkelRoot")
                   or mgr._character_positions.get(name))
            if pos is None:
                for k, v in mgr._character_positions.items():
                    if name in k and v is not None:
                        pos = v
                        break
            mgr_pos[name] = pos
    except Exception:
        pass

    # 方法2：USD XformOp（始终可读，是角色世界坐标的真值）
    stage = omni.usd.get_context().get_stage()
    result = {}
    for npc in NPC_PATROL:
        name = npc["name"]
        prim = stage.GetPrimAtPath(f"{CHARACTER_ROOT}/{name}")
        if prim.IsValid():
            mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            t = mat.ExtractTranslation()
            usd_p = (t[0], t[1], t[2])
        else:
            usd_p = None
        result[name] = mgr_pos.get(name) or usd_p
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("[NPC] 室外 NPC 巡逻演示 — Simple_Warehouse + 阳光")
    print("="*60)

    print("\n[NPC] Step 1: 启用扩展...")
    enable_extensions()
    simulation_app.update()

    print("[NPC] Step 2: 生成命令文件...")
    write_command_file()

    print("[NPC] Step 3: 配置 omni.anim.people...")
    configure_people()

    print("[NPC] Step 4: 加载场景 + 添加阳光...")
    stage = load_scene_with_sun()

    print("[NPC] Step 5: 创建 NavMesh Volume 并烘焙...")
    add_navmesh_and_bake(stage)

    print("[NPC] Step 6: 添加 NPC 角色...")
    add_characters(stage)

    # 给 CharacterBehavior 脚本几帧初始化时间
    print("[NPC] Step 7: 等待初始化...")
    for _ in range(60):
        simulation_app.update()

    print("\n[NPC] Step 8: 开始仿真（Ctrl+C 停止）")
    print("  Worker_01: X轴 +8 ↔ -8  (Z= 2.0)")
    print("  Worker_02: X轴 +8 ↔ -8  (Z= 0.0)")
    print("  Worker_03: X轴 +8 ↔ -8  (Z=-2.0)")
    print("-"*60)

    frame = 0
    report_every = 120   # 每 120 帧打印一次（约 2 秒@60FPS）
    prev_pos = {npc["name"]: np.array(npc["start"]) for npc in NPC_PATROL}

    try:
        while simulation_app.is_running():
            simulation_app.update()
            frame += 1

            if frame % report_every == 0:
                positions = get_positions()
                print(f"\n[NPC] Frame {frame}")
                for name, pos in positions.items():
                    if pos is not None:
                        x = pos.x if hasattr(pos, 'x') else pos[0]
                        y = pos.y if hasattr(pos, 'y') else pos[1]
                        z = pos.z if hasattr(pos, 'z') else pos[2]
                        curr = np.array([x, y, z])
                        vel  = curr - prev_pos[name]
                        speed = np.linalg.norm(vel) * 60  # 帧速度 → m/s（60FPS）
                        print(f"  {name}: pos=({x:6.2f}, {y:5.2f}, {z:5.2f})  "
                              f"speed={speed:.2f} m/s")
                        prev_pos[name] = curr
                    else:
                        print(f"  {name}: 位置未就绪")

    except KeyboardInterrupt:
        print("\n[NPC] 用户停止.")

    simulation_app.close()


if __name__ == "__main__":
    main()
