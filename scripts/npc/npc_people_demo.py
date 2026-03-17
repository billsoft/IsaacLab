"""
NPC 行人演示脚本 —— IRA SimulationManager 方案
=================================================
基于官方 IRA（Isaac Replicator Agent）系统，通过 SimulationManager
自动处理角色加载、骨骼重定向、动画图和 NavMesh。

重要：使用 SimulationApp 直接初始化（不用 AppLauncher），因为 IsaacLab 的
.kit 文件不包含 NavMesh schema 等动画系统必需的 C++ 原生插件。
SimulationApp 使用 Isaac Sim 完整默认配置，一切动画/导航功能开箱即用。

运行方式：
    isaaclab.bat -p scripts/npc/npc_people_demo.py
    isaaclab.bat -p scripts/npc/npc_people_demo.py --headless
    isaaclab.bat -p scripts/npc/npc_people_demo.py --num_characters 5 --walk_distance 10
"""

import argparse
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# 1. Parse arguments BEFORE creating SimulationApp
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="NPC Walk Back and Forth Demo")
parser.add_argument("--num_characters", type=int, default=3, help="Number of NPC characters")
parser.add_argument("--walk_distance", type=float, default=8.0, help="One-way walk distance in meters")
parser.add_argument("--duration", type=float, default=120.0, help="Simulation duration in seconds")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# 2. Create SimulationApp (NOT AppLauncher — need full Isaac Sim config for
#    NavMesh schema, animation graph, retarget plugins)
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

APP_CONFIG = {
    "headless": args.headless,
    "width": 1920,
    "height": 1080,
}
simulation_app = SimulationApp(APP_CONFIG)

# ---------------------------------------------------------------------------
# 3. Enable required extensions (after SimulationApp is created)
# ---------------------------------------------------------------------------
import carb
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.utils.extensions import enable_extension

REQUIRED_EXTENSIONS = [
    "omni.anim.timeline",
    "omni.anim.graph.bundle",
    "omni.anim.graph.core",
    "omni.anim.retarget.core",
    "omni.anim.navigation.core",
    "omni.anim.navigation.bundle",
    "omni.anim.people",
    "omni.kit.scripting",
]

print("[NPC Demo] Enabling required extensions...")
for ext in REQUIRED_EXTENSIONS:
    enable_extension(ext)
    simulation_app.update()

print("[NPC Demo] All extensions enabled.")
simulation_app.update()
simulation_app.update()

# ---------------------------------------------------------------------------
# 4. Resolve asset paths
# ---------------------------------------------------------------------------
LOCAL_ASSETS_PATH = "D:/code/IsaacLab/Assets/Isaac/5.1"

if not os.path.isdir(LOCAL_ASSETS_PATH):
    # Try Nucleus fallback
    try:
        from isaacsim.storage.native import get_assets_root_path
        LOCAL_ASSETS_PATH = get_assets_root_path()
    except Exception:
        pass

if not LOCAL_ASSETS_PATH or not os.path.isdir(LOCAL_ASSETS_PATH):
    print("[NPC Demo] ERROR: Cannot find Isaac assets. Check LOCAL_ASSETS_PATH.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)

ASSETS_ROOT = LOCAL_ASSETS_PATH.replace("\\", "/")
print(f"[NPC Demo] Using assets from: {ASSETS_ROOT}")

SCENE_USD = f"{ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

# Available character models
CHARACTER_MODELS = [
    "Isaac/People/Characters/F_Business_02/F_Business_02.usd",
    "Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd",
    "Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd",
    "Isaac/People/Characters/M_Medical_01/M_Medical_01.usd",
    "Isaac/People/Characters/male_adult_construction_03/male_adult_construction_03.usd",
    "Isaac/People/Characters/female_adult_police_03_new/female_adult_police_03_new.usd",
]

# Filter to models that actually exist locally
AVAILABLE_MODELS = [m for m in CHARACTER_MODELS if os.path.isfile(f"{ASSETS_ROOT}/{m}")]
if not AVAILABLE_MODELS:
    print("[NPC Demo] ERROR: No character models found.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)

print(f"[NPC Demo] Found {len(AVAILABLE_MODELS)} character model(s).")


# ---------------------------------------------------------------------------
# 5. Generate IRA command file for back-and-forth walking
# ---------------------------------------------------------------------------
def generate_command_file(num_characters: int, walk_distance: float, num_loops: int = 100) -> str:
    """
    Generate a command .txt file that makes each character walk back and forth.
    Characters are spaced along the Y-axis so they don't collide.
    IRA naming: 0->"Character", 1->"Character_01", 10->"Character_10"
    """
    lines = []
    spacing = 3.0
    start_x = -walk_distance / 2.0

    for i in range(num_characters):
        if i == 0:
            name = "Character"
        elif i < 10:
            name = f"Character_0{i}"
        else:
            name = f"Character_{i}"
        y = i * spacing
        x_a = start_x
        x_b = start_x + walk_distance

        for _ in range(num_loops):
            lines.append(f"{name} GoTo {x_b:.1f} {y:.1f} 0.0 0")
            lines.append(f"{name} GoTo {x_a:.1f} {y:.1f} 0.0 180")

    temp_dir = tempfile.gettempdir()
    cmd_file = os.path.join(temp_dir, "npc_walk_commands.txt")
    with open(cmd_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[NPC Demo] Command file written to: {cmd_file}")
    # Return relative path — IRA resolves relative to config file location
    return "npc_walk_commands.txt"


# ---------------------------------------------------------------------------
# 6. Generate IRA YAML config
# ---------------------------------------------------------------------------
def generate_config_file(num_characters: int, command_file: str) -> str:
    """Generate a minimal IRA YAML config file."""
    char_folder = f"{ASSETS_ROOT}/Isaac/People/Characters/"
    scene_fwd = SCENE_USD.replace("\\", "/")
    char_fwd = char_folder.replace("\\", "/")

    config = (
        "isaacsim.replicator.agent:\n"
        "  version: 0.7.0\n"
        "  global:\n"
        "    seed: 42\n"
        f"    simulation_length: {int(args.duration * 30)}\n"
        "  scene:\n"
        f"    asset_path: {scene_fwd}\n"
        "  character:\n"
        f"    asset_path: {char_fwd}\n"
        f"    command_file: {command_file}\n"
        f"    num: {num_characters}\n"
        "  robot:\n"
        "    nova_carter_num: 0\n"
        "    iw_hub_num: 0\n"
        '    command_file: ""\n'
        "  sensor:\n"
        "    camera_num: 0\n"
        "  replicator:\n"
        "    writer: IRABasicWriter\n"
        "    parameters:\n"
        "      rgb: false\n"
    )

    temp_dir = tempfile.gettempdir()
    config_file = os.path.join(temp_dir, "npc_walk_config.yaml")
    with open(config_file, "w") as f:
        f.write(config)

    print(f"[NPC Demo] Config file written to: {config_file}")
    print("--- Generated YAML ---")
    print(config)
    print("--- End YAML ---")
    return config_file


# ---------------------------------------------------------------------------
# 7. Main: Load scene and run simulation using IRA
# ---------------------------------------------------------------------------
def main():
    num_chars = min(args.num_characters, len(AVAILABLE_MODELS))
    if num_chars < args.num_characters:
        print(f"[NPC Demo] Only {num_chars} character models available, using {num_chars}.")

    # Generate command and config files
    cmd_file = generate_command_file(num_chars, args.walk_distance)
    config_file = generate_config_file(num_chars, cmd_file)

    # Enable IRA extension
    enable_extension("isaacsim.replicator.agent.core")
    simulation_app.update()
    simulation_app.update()

    # Import IRA SimulationManager
    from isaacsim.replicator.agent.core.simulation import SimulationManager

    sim_manager = SimulationManager()

    # Configure IRA settings
    settings = carb.settings.get_settings()
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
    settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
    settings.set("/app/omni.graph.scriptnode/enable_opt_in", False)

    # Load config
    can_load = sim_manager.load_config_file(config_file)
    if not can_load:
        print("[NPC Demo] ERROR: Failed to load IRA config file.", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)

    print("[NPC Demo] Setting up simulation (loading scene, spawning characters)...")
    print("[NPC Demo] This may take a while on first run...")

    # Setup simulation async
    setup_done = [False]

    def on_setup_done(e):
        setup_done[0] = True

    sim_manager.register_set_up_simulation_done_callback(on_setup_done)
    sim_manager.set_up_simulation_from_config_file()

    # Wait for setup to complete
    while not setup_done[0] and not simulation_app.is_exiting():
        simulation_app.update()

    if simulation_app.is_exiting():
        return

    print("[NPC Demo] Simulation setup complete!")
    print(f"[NPC Demo] {num_chars} NPC(s) will walk back and forth for {args.duration}s")
    print("[NPC Demo] Press Ctrl+C or close the window to stop.")

    # Start data generation (starts timeline and character behaviors)
    async def run_sim():
        await sim_manager.run_data_generation_async(will_wait_until_complete=True)

    from omni.kit.async_engine import run_coroutine

    task = run_coroutine(run_sim())

    try:
        frame = 0
        while not task.done() and not simulation_app.is_exiting():
            simulation_app.update()
            frame += 1
            if frame % 300 == 0:
                print(f"[NPC Demo] Running... frame {frame}")
    except KeyboardInterrupt:
        print("\n[NPC Demo] Interrupted by user.")
    finally:
        print("[NPC Demo] Shutting down...")
        simulation_app.close()


if __name__ == "__main__":
    main()
