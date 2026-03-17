"""诊断脚本：IRA 加载后 Stage 状态检查
用于确认 IRA 完成后：
1. Stage 的 prim 树结构
2. 能否在根节点或 /World 下添加新 prim
3. 新 prim 是否能持续存在
"""

import argparse
import os
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--num_characters", type=int, default=2)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 1080})

import carb
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, UsdGeom, Usd

# ---------------------------------------------------------------------------
LOCAL_ASSETS_PATH = "D:/code/IsaacLab/Assets/Isaac/5.1"
ASSETS_ROOT = LOCAL_ASSETS_PATH.replace("\\", "/")
SCENE_USD = f"{ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"


def print_stage_tree(stage, label, max_depth=3):
    """打印 Stage 的 prim 树。"""
    print(f"\n{'='*60}")
    print(f"  STAGE TREE: {label}")
    print(f"{'='*60}")
    root = stage.GetPseudoRoot()
    _print_tree(root, 0, max_depth)
    print(f"{'='*60}\n")


def _print_tree(prim, depth, max_depth):
    if depth > max_depth:
        children = list(prim.GetChildren())
        if children:
            print(f"{'  ' * depth}... ({len(children)} children)")
        return
    for child in prim.GetChildren():
        type_name = child.GetTypeName()
        type_str = f" [{type_name}]" if type_name else ""
        print(f"{'  ' * depth}{child.GetName()}{type_str}")
        _print_tree(child, depth + 1, max_depth)


# ---------------------------------------------------------------------------
# IRA NPC config (修复：去掉 robot section 避免空 command_file 错误)
# ---------------------------------------------------------------------------
def make_config(num_chars):
    char_folder = f"{ASSETS_ROOT}/Isaac/People/Characters/"
    # 生成命令文件
    lines = []
    for i in range(num_chars):
        name = "Character" if i == 0 else (f"Character_0{i}" if i < 10 else f"Character_{i}")
        for _ in range(50):
            lines.append(f"{name} GoTo 4.0 {i*3.0:.1f} 0.0 0")
            lines.append(f"{name} GoTo -4.0 {i*3.0:.1f} 0.0 180")

    temp_dir = tempfile.gettempdir()
    cmd_path = os.path.join(temp_dir, "debug_npc_cmds.txt")
    with open(cmd_path, "w") as f:
        f.write("\n".join(lines))

    config = (
        "isaacsim.replicator.agent:\n"
        "  version: 0.7.0\n"
        "  global:\n"
        "    seed: 42\n"
        "    simulation_length: 90000\n"
        "  scene:\n"
        f"    asset_path: {SCENE_USD}\n"
        "  character:\n"
        f"    asset_path: {char_folder}\n"
        f"    command_file: debug_npc_cmds.txt\n"
        f"    num: {num_chars}\n"
        "  replicator:\n"
        "    writer: IRABasicWriter\n"
        "    parameters:\n"
        "      rgb: false\n"
    )
    cfg_path = os.path.join(temp_dir, "debug_npc_config.yaml")
    with open(cfg_path, "w") as f:
        f.write(config)
    print(f"[Debug] Config: {cfg_path}")
    print(f"[Debug] Commands: {cmd_path}")
    return cfg_path


# ===========================================================================
# STEP 1: IRA 加载
# ===========================================================================
NPC_EXTENSIONS = [
    "omni.anim.timeline",
    "omni.anim.graph.bundle",
    "omni.anim.graph.core",
    "omni.anim.retarget.core",
    "omni.anim.navigation.core",
    "omni.anim.navigation.bundle",
    "omni.anim.people",
    "omni.kit.scripting",
]
print("[Debug] Enabling NPC extensions...")
for ext in NPC_EXTENSIONS:
    enable_extension(ext)
    simulation_app.update()

enable_extension("isaacsim.replicator.agent.core")
simulation_app.update()
simulation_app.update()

from isaacsim.replicator.agent.core.simulation import SimulationManager

sim_manager = SimulationManager()
settings = carb.settings.get_settings()
settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
settings.set("/app/omni.graph.scriptnode/enable_opt_in", False)

config_file = make_config(args.num_characters)
can_load = sim_manager.load_config_file(config_file)
print(f"[Debug] load_config_file returned: {can_load}")

if can_load:
    setup_done = [False]

    def on_setup_done(e):
        setup_done[0] = True

    sim_manager.register_set_up_simulation_done_callback(on_setup_done)
    sim_manager.set_up_simulation_from_config_file()

    tick = 0
    max_ticks = 3000  # ~100s 超时保护
    while not setup_done[0] and not simulation_app.is_exiting() and tick < max_ticks:
        simulation_app.update()
        tick += 1
        if tick % 300 == 0:
            print(f"[Debug] Waiting for IRA setup... tick={tick}")

    if setup_done[0]:
        print(f"[Debug] IRA setup done after {tick} ticks")
    else:
        print(f"[Debug] IRA setup TIMED OUT after {tick} ticks (continuing anyway)")

    # 额外等待确保所有异步操作完成
    for i in range(60):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    print_stage_tree(stage, "AFTER IRA SETUP + 60 ticks")

    # ===========================================================================
    # STEP 2: 尝试在不同位置创建 prim
    # ===========================================================================
    print("[Debug] Creating test prims...")

    # 方案 A: /World/StereoRig (在 /World 下)
    xf_a = UsdGeom.Xform.Define(stage, "/World/StereoRig")
    xf_a.AddTranslateOp().Set(Gf.Vec3d(0, 0, 2.5))
    print(f"  /World/StereoRig created: valid={stage.GetPrimAtPath('/World/StereoRig').IsValid()}")

    # 方案 B: /StereoRig (根节点)
    xf_b = UsdGeom.Xform.Define(stage, "/StereoRig")
    xf_b.AddTranslateOp().Set(Gf.Vec3d(1, 0, 2.5))
    print(f"  /StereoRig created: valid={stage.GetPrimAtPath('/StereoRig').IsValid()}")

    for _ in range(10):
        simulation_app.update()

    print_stage_tree(stage, "AFTER creating test prims + 10 ticks")

    # 再等一会看是否消失
    for _ in range(60):
        simulation_app.update()

    print(f"  After 60 more ticks:")
    print(f"    /World/StereoRig valid: {stage.GetPrimAtPath('/World/StereoRig').IsValid()}")
    print(f"    /StereoRig valid: {stage.GetPrimAtPath('/StereoRig').IsValid()}")

    print_stage_tree(stage, "AFTER 60 more ticks")

else:
    print("[Debug] ERROR: load_config_file failed!")
    # 即使 IRA 失败，也手动加载场景并测试 prim 创建
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().subLayerPaths.append(SCENE_USD)
    for _ in range(30):
        simulation_app.update()
    print_stage_tree(stage, "AFTER manual scene load")

    xf_a = UsdGeom.Xform.Define(stage, "/World/StereoRig")
    xf_a.AddTranslateOp().Set(Gf.Vec3d(0, 0, 2.5))
    xf_b = UsdGeom.Xform.Define(stage, "/StereoRig")
    xf_b.AddTranslateOp().Set(Gf.Vec3d(1, 0, 2.5))
    for _ in range(10):
        simulation_app.update()
    print_stage_tree(stage, "AFTER creating test prims")

# ===========================================================================
# STEP 3: 保持运行等待用户查看
# ===========================================================================
print("\n[Debug] GUI running. Check Stage panel. Close window to exit.")
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
