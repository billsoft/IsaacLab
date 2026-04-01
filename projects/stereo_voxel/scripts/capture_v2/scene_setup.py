"""场景加载 + NPC 初始化
========================
封装 Isaac Sim 场景加载、IRA SimulationManager NPC 设置。
所有 omni.* 导入在函数内部延迟执行（Isaac Sim 必须先 SimulationApp）。
"""

import os
import tempfile

import numpy as np

from .constants import get_available_models, get_scene_usd


# ===========================================================================
# 场景探测
# ===========================================================================
def probe_scene_objects(stage, stage_mpu: float,
                        keywords=("RackFrame", "RackShelf", "PillarA")) -> dict:
    """扫描场景中名称包含关键词的顶层 Xform prim，返回世界坐标范围（米）。"""
    from pxr import UsdGeom

    results = {k: [] for k in keywords}
    root = stage.GetPrimAtPath("/Root")
    if not root.IsValid():
        return results

    for child in root.GetChildren():
        name = child.GetName()
        for kw in keywords:
            if kw in name:
                xformable = UsdGeom.Xformable(child)
                try:
                    xf = xformable.ComputeLocalToWorldTransform(0)
                    pos = xf.ExtractTranslation()
                    results[kw].append([
                        pos[0] * stage_mpu,
                        pos[1] * stage_mpu,
                        pos[2] * stage_mpu,
                    ])
                except Exception:
                    pass
                break

    out = {}
    for kw, positions in results.items():
        if positions:
            arr = np.array(positions)
            out[kw] = {
                "positions": arr,
                "bbox_min": arr.min(axis=0),
                "bbox_max": arr.max(axis=0),
                "count": len(arr),
            }
    return out


def suggest_camera_position(scene_info: dict) -> tuple[float, float]:
    """根据场景物体分布，推荐相机 XY 位置（米）。"""
    all_positions = []
    for info in scene_info.values():
        all_positions.append(info["positions"][:, :2])
    if not all_positions:
        return 0.0, 0.0
    all_xy = np.vstack(all_positions)
    return float(np.median(all_xy[:, 0])), float(np.median(all_xy[:, 1]))


# ===========================================================================
# NPC 配置文件生成
# ===========================================================================
def generate_npc_command_file(num_characters: int, walk_distance: float,
                              num_loops: int = 100) -> str:
    """生成 NPC 行走命令文件，返回文件名（在 tempdir 中）。"""
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
        x_a, x_b = start_x, start_x + walk_distance
        for _ in range(num_loops):
            lines.append(f"{name} GoTo {x_b:.1f} {y:.1f} 0.0 0")
            lines.append(f"{name} GoTo {x_a:.1f} {y:.1f} 0.0 180")

    cmd_file = os.path.join(tempfile.gettempdir(), "npc_walk_commands_v2.txt")
    with open(cmd_file, "w") as f:
        f.write("\n".join(lines))
    return "npc_walk_commands_v2.txt"


def generate_npc_config_file(num_characters: int, command_file: str,
                              assets_root: str, scene_usd: str) -> str:
    """生成 IRA YAML 配置文件，返回完整路径。"""
    char_folder = f"{assets_root}/Isaac/People/Characters/"
    config = (
        "isaacsim.replicator.agent:\n"
        "  version: 0.7.0\n"
        "  global:\n"
        "    seed: 42\n"
        "    simulation_length: 90000\n"
        "  scene:\n"
        f"    asset_path: {scene_usd}\n"
        "  character:\n"
        f"    asset_path: {char_folder}\n"
        f"    command_file: {command_file}\n"
        f"    num: {num_characters}\n"
        "  replicator:\n"
        "    writer: IRABasicWriter\n"
        "    parameters:\n"
        "      rgb: false\n"
    )
    config_file = os.path.join(tempfile.gettempdir(), "npc_walk_config_v2.yaml")
    with open(config_file, "w") as f:
        f.write(config)
    return config_file


# ===========================================================================
# 场景 + NPC 初始化（Phase 1 完整封装）
# ===========================================================================
def setup_scene(simulation_app, args, assets_root: str) -> tuple[bool, int]:
    """加载场景并初始化 NPC。

    Returns:
        (npc_ready, num_chars)
    """
    import carb
    import omni.kit.app
    import omni.usd

    from isaacsim.core.utils.extensions import enable_extension

    scene_usd = get_scene_usd(assets_root)
    available_models = get_available_models(assets_root)
    use_npc = not args.no_npc and len(available_models) > 0

    if not use_npc:
        if args.no_npc:
            print("[capture_v2] NPC disabled (--no_npc)")
        else:
            print("[capture_v2] No NPC models found")
        print(f"[capture_v2] Loading scene: {scene_usd}")
        stage = omni.usd.get_context().get_stage()
        stage.GetRootLayer().subLayerPaths.append(scene_usd)
        for _ in range(30):
            simulation_app.update()
        return False, 0

    num_chars = min(args.num_characters, len(available_models))
    print(f"[capture_v2] Found {len(available_models)} NPC model(s), using {num_chars}")

    # 启用 NPC 扩展
    NPC_EXTENSIONS = [
        "omni.anim.timeline", "omni.anim.graph.bundle", "omni.anim.graph.core",
        "omni.anim.retarget.core", "omni.anim.navigation.core",
        "omni.anim.navigation.bundle", "omni.anim.people", "omni.kit.scripting",
    ]
    print("[capture_v2] Enabling NPC extensions...")
    for ext in NPC_EXTENSIONS:
        enable_extension(ext)
        simulation_app.update()

    # 手动打开场景，等待 ASSETS_LOADED
    print(f"[capture_v2] Pre-loading scene: {scene_usd}")
    scene_loaded = [False]

    def on_scene_loaded(e):
        scene_loaded[0] = True

    _handle = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.usd.get_context().stage_event_name(
            omni.usd.StageEventType.ASSETS_LOADED),
        on_event=on_scene_loaded,
        observer_name="capture_v2/on_scene_preload",
    )

    import omni.kit.window.file
    old_ignore = carb.settings.get_settings().get("/app/file/ignoreUnsavedStage")
    carb.settings.get_settings().set("/app/file/ignoreUnsavedStage", True)
    omni.kit.window.file.open_stage(scene_usd, omni.usd.UsdContextInitialLoadSet.LOAD_ALL)
    carb.settings.get_settings().set(
        "/app/file/ignoreUnsavedStage",
        old_ignore if old_ignore is not None else False,
    )

    tick = 0
    while not scene_loaded[0] and not simulation_app.is_exiting() and tick < 1500:
        simulation_app.update()
        tick += 1
        if tick % 300 == 0:
            print(f"[capture_v2] Waiting for scene... tick={tick}")
    _handle = None
    print(f"[capture_v2] Scene loaded after {tick} ticks")

    for _ in range(30):
        simulation_app.update()

    # IRA setup
    print(f"[capture_v2] Setting up {num_chars} NPC(s) via IRA...")
    enable_extension("isaacsim.replicator.agent.core")
    simulation_app.update()
    simulation_app.update()

    from isaacsim.replicator.agent.core.simulation import SimulationManager

    sim_manager = SimulationManager()
    settings = carb.settings.get_settings()
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
    settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
    settings.set("/app/omni.graph.scriptnode/enable_opt_in", False)

    cmd_file = generate_npc_command_file(num_chars, args.walk_distance)
    config_file = generate_npc_config_file(num_chars, cmd_file, assets_root, scene_usd)

    can_load = sim_manager.load_config_file(config_file)
    npc_ready = False
    if can_load:
        setup_done = [False]

        def on_setup_done(e):
            setup_done[0] = True

        sim_manager.register_set_up_simulation_done_callback(on_setup_done)
        sim_manager.set_up_simulation_from_config_file()

        tick = 0
        while not setup_done[0] and not simulation_app.is_exiting() and tick < 3000:
            simulation_app.update()
            tick += 1
            if tick % 300 == 0:
                print(f"[capture_v2] Waiting for IRA setup... tick={tick}")

        if setup_done[0]:
            npc_ready = True
            print(f"[capture_v2] {num_chars} NPC(s) loaded after {tick} ticks!")
        else:
            print(f"[capture_v2] WARNING: IRA timeout at {tick} ticks")
    else:
        print("[capture_v2] WARNING: Failed to load NPC config")

    for _ in range(30):
        simulation_app.update()

    return npc_ready, num_chars
