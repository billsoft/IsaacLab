"""双目鱼眼 + 语义体素 GT 同步采集
=====================================
在 Isaac Sim 仓库场景中同步采集：
  1. 双目鱼眼图像对（左/右 RGB, 1280x1080）
  2. 对应的 3D 语义体素 ground truth（72x60x32, 0.1m 分辨率）

同步策略：timeline.pause() 冻结 → 拍照 + 体素查询 → timeline.play() 恢复

运行：
    isaaclab.bat -p projects/stereo_voxel/scripts/stereo_voxel_capture.py
    isaaclab.bat -p projects/stereo_voxel/scripts/stereo_voxel_capture.py --headless --num_frames 200
    isaaclab.bat -p projects/stereo_voxel/scripts/stereo_voxel_capture.py --no_npc --headless --num_frames 50
"""

import argparse
import glob
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Stereo fisheye + voxel GT capture")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--num_frames", type=int, default=200, help="Number of frames to capture (headless)")
parser.add_argument("--camera_height", type=float, default=3.0, help="Camera height (m)")
parser.add_argument("--camera_x", type=float, default=0.0, help="Camera X position (m)")
parser.add_argument("--camera_y", type=float, default=0.0, help="Camera Y position (m)")
parser.add_argument("--num_characters", type=int, default=3, help="Number of NPC characters")
parser.add_argument("--walk_distance", type=float, default=8.0, help="NPC walk distance (m)")
parser.add_argument("--capture_interval", type=int, default=90, help="Capture every N sim steps (~3s at 30FPS)")
parser.add_argument("--no_npc", action="store_true", help="Skip NPC loading (static scene only)")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 1080})

import carb
import cv2
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, UsdGeom

# 添加脚本目录到 sys.path，以便 import 同目录模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from semantic_classes import FREE, GENERAL_OBJECT, PERSON, UNOBSERVED, lookup_class_id
from voxel_grid import VoxelGrid

# ===========================================================================
# SC132GS 相机参数
# ===========================================================================
CAM_W, CAM_H = 1280, 1080
PIXEL_SIZE_UM = 2.7
FOCAL_LENGTH_MM = 1.75
DIAG_FOV_DEG = 157.2
BASELINE_M = 0.08  # 80mm

fx = FOCAL_LENGTH_MM / PIXEL_SIZE_UM * 1000  # 648.15 px
cx = CAM_W / 2.0
cy = CAM_H / 2.0
K1_EQUIDISTANT = 1.0 / fx

print(f"[Capture] SC132GS x2, baseline={BASELINE_M*1000:.0f}mm, FOV={DIAG_FOV_DEG}deg")

# ===========================================================================
# Assets
# ===========================================================================
LOCAL_ASSETS_PATH = "D:/code/IsaacLab/Assets/Isaac/5.1"
if not os.path.isdir(LOCAL_ASSETS_PATH):
    try:
        from isaacsim.storage.native import get_assets_root_path
        LOCAL_ASSETS_PATH = get_assets_root_path()
    except Exception:
        pass
if not LOCAL_ASSETS_PATH or not os.path.isdir(LOCAL_ASSETS_PATH):
    print("[Capture] ERROR: Cannot find Isaac assets.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)

ASSETS_ROOT = LOCAL_ASSETS_PATH.replace("\\", "/")
SCENE_USD = f"{ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

# ===========================================================================
# NPC character models
# ===========================================================================
CHARACTER_MODELS = [
    "Isaac/People/Characters/F_Business_02/F_Business_02.usd",
    "Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd",
    "Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd",
    "Isaac/People/Characters/M_Medical_01/M_Medical_01.usd",
    "Isaac/People/Characters/male_adult_construction_03/male_adult_construction_03.usd",
    "Isaac/People/Characters/female_adult_police_03_new/female_adult_police_03_new.usd",
]
AVAILABLE_MODELS = [m for m in CHARACTER_MODELS if os.path.isfile(f"{ASSETS_ROOT}/{m}")]


# ===========================================================================
# NPC config generation (same as test_stereo_pair.py)
# ===========================================================================
def generate_npc_command_file(num_characters, walk_distance, num_loops=100):
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
    temp_dir = tempfile.gettempdir()
    cmd_file = os.path.join(temp_dir, "npc_walk_commands.txt")
    with open(cmd_file, "w") as f:
        f.write("\n".join(lines))
    return "npc_walk_commands.txt"


def generate_npc_config_file(num_characters, command_file):
    char_folder = f"{ASSETS_ROOT}/Isaac/People/Characters/"
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
        f"    command_file: {command_file}\n"
        f"    num: {num_characters}\n"
        "  replicator:\n"
        "    writer: IRABasicWriter\n"
        "    parameters:\n"
        "      rgb: false\n"
    )
    temp_dir = tempfile.gettempdir()
    config_file = os.path.join(temp_dir, "npc_walk_config.yaml")
    with open(config_file, "w") as f:
        f.write(config)
    return config_file


# ===========================================================================
# Helper: set fisheye lens on USD prim
# ===========================================================================
def set_fisheye_on_prim(stage, cam_prim_path):
    prim = stage.GetPrimAtPath(cam_prim_path)
    if not prim.IsValid():
        print(f"[Capture] WARNING: prim {cam_prim_path} not valid, skip fisheye setup")
        return
    prim.ApplyAPI("OmniLensDistortionFthetaAPI")
    prim.GetAttribute("omni:lensdistortion:model").Set("ftheta")
    prim.GetAttribute("omni:lensdistortion:ftheta:nominalWidth").Set(float(CAM_W))
    prim.GetAttribute("omni:lensdistortion:ftheta:nominalHeight").Set(float(CAM_H))
    prim.GetAttribute("omni:lensdistortion:ftheta:opticalCenter").Set((float(cx), float(cy)))
    prim.GetAttribute("omni:lensdistortion:ftheta:maxFov").Set(float(DIAG_FOV_DEG))
    prim.GetAttribute("omni:lensdistortion:ftheta:k0").Set(0.0)
    prim.GetAttribute("omni:lensdistortion:ftheta:k1").Set(float(K1_EQUIDISTANT))
    prim.GetAttribute("omni:lensdistortion:ftheta:k2").Set(0.0)
    prim.GetAttribute("omni:lensdistortion:ftheta:k3").Set(0.0)
    prim.GetAttribute("omni:lensdistortion:ftheta:k4").Set(0.0)
    prim.GetAttribute("fStop").Set(0.0)


# ===========================================================================
# Helper: camera visual cone
# ===========================================================================
def create_camera_visual(stage, parent_path, name, color_rgb):
    cone_path = f"{parent_path}/{name}_visual"
    cone_prim = UsdGeom.Cone.Define(stage, cone_path)
    cone_prim.GetRadiusAttr().Set(0.03)
    cone_prim.GetHeightAttr().Set(0.06)
    cone_prim.GetAxisAttr().Set("X")
    cone_prim.GetDisplayColorAttr().Set([Gf.Vec3f(*[c / 255.0 for c in color_rgb])])


# ===========================================================================
# Async save pool
# ===========================================================================
_save_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="saver")
_pending_saves = []


def async_save_image(path, rgb_array):
    def _save(p, arr):
        bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(p, bgr)
    future = _save_pool.submit(_save, path, rgb_array.copy())
    _pending_saves.append(future)


def async_save_voxel(path_prefix, semantic, instance):
    def _save(pp, sem, ins):
        os.makedirs(os.path.dirname(pp), exist_ok=True)
        np.savez_compressed(f"{pp}_semantic.npz", data=sem)
        np.savez_compressed(f"{pp}_instance.npz", data=ins)
    future = _save_pool.submit(_save, path_prefix, semantic.copy(), instance.copy())
    _pending_saves.append(future)


def async_save_json(path, data):
    def _save(p, d):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(d, f, indent=2)
    future = _save_pool.submit(_save, path, data)
    _pending_saves.append(future)


def wait_pending_saves():
    for f in _pending_saves:
        f.result()
    _pending_saves.clear()


# ===========================================================================
# PhysX overlap query helpers
# ===========================================================================
def get_object_type_from_prim_path(stage, prim_path: str) -> str:
    """从 PhysX hit 的 rigid_body prim path 提取物体类型名。

    策略：向上遍历找最近的有意义 Xform/Mesh，提取名字。
    特殊处理 NPC 角色路径。
    """
    # NPC 角色路径通常包含 Character / Character_01 等
    if "/Character" in prim_path:
        return "NPC:person"

    # 向上遍历找有意义的 prim name
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return ""

    current = prim
    while current.IsValid():
        name = current.GetName()
        # 跳过内部细节（Collider, Mesh, Shape 等）
        if name in ("Collider", "CollisionMesh", "CollisionPlane", "Mesh",
                     "Shape", "Physics", "Collision", "collision"):
            current = current.GetParent()
            continue
        # 跳过根和 World 节点
        if name in ("", "World", "Environment", "Warehouse"):
            current = current.GetParent()
            continue
        # 跳过纯 Scope/Xform 容器
        if current.GetTypeName() in ("Scope",):
            current = current.GetParent()
            continue
        return name

    # 回退：取 prim path 最后一段
    return prim_path.rsplit("/", 1)[-1] if "/" in prim_path else prim_path


# Instance ID 管理
_instance_map = {}  # prim_path → instance_id
_next_instance_id = 1


def get_instance_id(prim_path: str) -> int:
    global _next_instance_id
    if prim_path not in _instance_map:
        _instance_map[prim_path] = _next_instance_id
        _next_instance_id += 1
    return _instance_map[prim_path]


def fill_voxel_grid(stage, voxel_grid, world_centers_flat, physx_sqi):
    """用 PhysX overlap_box 填充体素网格。

    采用粗查+细查两阶段策略：
      1. 粗查：5x5x5 体素块(0.5m³)做一次 overlap，无 hit 则整块标 FREE
      2. 细查：有 hit 的块内逐个体素查询

    Args:
        stage: USD stage
        voxel_grid: VoxelGrid 实例（会被修改）
        world_centers_flat: (N, 3) 世界坐标
        physx_sqi: PhysX scene query interface
    """
    NX, NY, NZ = voxel_grid.NX, voxel_grid.NY, voxel_grid.NZ
    VOXEL_SIZE = voxel_grid.VOXEL_SIZE
    world_centers = world_centers_flat.reshape(NX, NY, NZ, 3)

    COARSE = 5  # 粗查块大小
    coarse_half = COARSE * VOXEL_SIZE / 2.0  # 0.25m
    fine_half = VOXEL_SIZE / 2.0  # 0.05m
    identity_rot = carb.Float4(0.0, 0.0, 0.0, 1.0)

    total_voxels = NX * NY * NZ
    coarse_skipped = 0
    occupied_count = 0

    # 粗查遍历
    for ci in range(0, NX, COARSE):
        for cj in range(0, NY, COARSE):
            for ck in range(0, NZ, COARSE):
                # 粗查块的世界中心
                ei = min(ci + COARSE, NX)
                ej = min(cj + COARSE, NY)
                ek = min(ck + COARSE, NZ)
                block_center = world_centers[ci:ei, cj:ej, ck:ek].mean(axis=(0, 1, 2))

                # 粗查 half extent（适配边界块可能不足 5x5x5）
                bh_x = (ei - ci) * VOXEL_SIZE / 2.0
                bh_y = (ej - cj) * VOXEL_SIZE / 2.0
                bh_z = (ek - ck) * VOXEL_SIZE / 2.0

                # 粗查
                coarse_hits = []

                def on_coarse_hit(hit):
                    coarse_hits.append(hit.rigid_body)
                    return True

                physx_sqi.overlap_box(
                    carb.Float3(bh_x, bh_y, bh_z),
                    carb.Float3(float(block_center[0]), float(block_center[1]), float(block_center[2])),
                    identity_rot,
                    on_coarse_hit,
                    False,
                )

                if not coarse_hits:
                    # 整块标 FREE
                    voxel_grid.semantic[ci:ei, cj:ej, ck:ek] = FREE
                    coarse_skipped += (ei - ci) * (ej - cj) * (ek - ck)
                    continue

                # 细查：逐个体素
                for i in range(ci, ei):
                    for j in range(cj, ej):
                        for k in range(ck, ek):
                            center = world_centers[i, j, k]
                            fine_hits = []

                            def on_fine_hit(hit):
                                fine_hits.append(hit.rigid_body)
                                return True

                            physx_sqi.overlap_box(
                                carb.Float3(fine_half, fine_half, fine_half),
                                carb.Float3(float(center[0]), float(center[1]), float(center[2])),
                                identity_rot,
                                on_fine_hit,
                                False,
                            )

                            if not fine_hits:
                                voxel_grid.semantic[i, j, k] = FREE
                                voxel_grid.instance[i, j, k] = 0
                            else:
                                # 取第一个 hit 的物体类型
                                hit_path = fine_hits[0]
                                obj_type = get_object_type_from_prim_path(stage, hit_path)
                                class_id = lookup_class_id(obj_type)
                                voxel_grid.semantic[i, j, k] = class_id
                                voxel_grid.instance[i, j, k] = get_instance_id(hit_path)
                                occupied_count += 1

    free_count = np.sum(voxel_grid.semantic == FREE)
    unobs_count = np.sum(voxel_grid.semantic == UNOBSERVED)
    print(f"    Voxel fill: {occupied_count} occupied, {free_count} free, "
          f"{unobs_count} unobserved, {coarse_skipped} coarse-skipped")


# ============================================================================
# PHASE 1: 场景 + NPC 初始化
# ============================================================================
from isaacsim.core.utils.extensions import enable_extension

npc_ready = False
num_chars = 0
use_npc = not args.no_npc and len(AVAILABLE_MODELS) > 0

if use_npc:
    num_chars = min(args.num_characters, len(AVAILABLE_MODELS))
    print(f"[Capture] Found {len(AVAILABLE_MODELS)} NPC model(s), using {num_chars}")

    NPC_EXTENSIONS = [
        "omni.anim.timeline", "omni.anim.graph.bundle", "omni.anim.graph.core",
        "omni.anim.retarget.core", "omni.anim.navigation.core",
        "omni.anim.navigation.bundle", "omni.anim.people", "omni.kit.scripting",
    ]
    print("[Capture] Enabling NPC extensions...")
    for ext in NPC_EXTENSIONS:
        enable_extension(ext)
        simulation_app.update()

    # 手动打开场景，等待 ASSETS_LOADED
    print(f"[Capture] Pre-loading scene: {SCENE_USD}")
    scene_loaded = [False]

    def on_scene_loaded(e):
        scene_loaded[0] = True

    _handle = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.ASSETS_LOADED),
        on_event=on_scene_loaded,
        observer_name="capture/on_scene_preload",
    )

    import omni.kit.window.file
    old_ignore = carb.settings.get_settings().get("/app/file/ignoreUnsavedStage")
    carb.settings.get_settings().set("/app/file/ignoreUnsavedStage", True)
    omni.kit.window.file.open_stage(SCENE_USD, omni.usd.UsdContextInitialLoadSet.LOAD_ALL)
    carb.settings.get_settings().set("/app/file/ignoreUnsavedStage",
                                      old_ignore if old_ignore is not None else False)

    tick = 0
    while not scene_loaded[0] and not simulation_app.is_exiting() and tick < 1500:
        simulation_app.update()
        tick += 1
        if tick % 300 == 0:
            print(f"[Capture] Waiting for scene... tick={tick}")
    _handle = None
    print(f"[Capture] Scene loaded after {tick} ticks")

    for _ in range(30):
        simulation_app.update()

    # IRA setup
    print(f"[Capture] Setting up {num_chars} NPC(s) via IRA...")
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
    config_file = generate_npc_config_file(num_chars, cmd_file)

    can_load = sim_manager.load_config_file(config_file)
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
                print(f"[Capture] Waiting for IRA setup... tick={tick}")

        if setup_done[0]:
            npc_ready = True
            print(f"[Capture] {num_chars} NPC(s) loaded after {tick} ticks!")
        else:
            print(f"[Capture] WARNING: IRA timeout at {tick} ticks (NPC may still work)")
    else:
        print("[Capture] WARNING: Failed to load NPC config")

    for _ in range(30):
        simulation_app.update()
else:
    if args.no_npc:
        print("[Capture] NPC disabled (--no_npc)")
    else:
        print("[Capture] No NPC models found")
    print(f"[Capture] Loading scene: {SCENE_USD}")
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().subLayerPaths.append(SCENE_USD)
    for _ in range(30):
        simulation_app.update()

# ============================================================================
# PHASE 2: 创建双目相机
# ============================================================================
print("[Capture] Creating stereo cameras...")
stage = omni.usd.get_context().get_stage()
for _ in range(5):
    simulation_app.update()

stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
stage_scale = 1.0 / stage_mpu
print(f"[Capture] Stage metersPerUnit={stage_mpu}, scale={stage_scale}")

# StereoRig Xform
height_m = args.camera_height
rig_path = "/World/StereoRig"
rig_xform = UsdGeom.Xform.Define(stage, rig_path)
rig_pos = Gf.Vec3d(
    float(args.camera_x * stage_scale),
    float(args.camera_y * stage_scale),
    float(height_m * stage_scale),
)
rig_xform.AddTranslateOp().Set(rig_pos)

half_baseline = (BASELINE_M / 2.0) * stage_scale

left_cam_path = f"{rig_path}/Left"
right_cam_path = f"{rig_path}/Right"

# 相机朝向：+X → -Z (俯拍), euler (0, 0, 90)
import isaacsim.core.utils.numpy.rotations as rot_utils
cam_euler = np.array([0.0, 0.0, 90.0])
cam_quat = rot_utils.euler_angles_to_quats(cam_euler, degrees=True)

for cam_path, y_offset in [(left_cam_path, half_baseline), (right_cam_path, -half_baseline)]:
    cam_prim = UsdGeom.Camera.Define(stage, cam_path)
    xf = UsdGeom.Xformable(cam_prim.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, float(y_offset), 0.0))
    xf.AddOrientOp().Set(Gf.Quatf(
        float(cam_quat[0]),
        Gf.Vec3f(float(cam_quat[1]), float(cam_quat[2]), float(cam_quat[3])),
    ))
    cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(float(0.1 * stage_scale), float(100.0 * stage_scale)))
    set_fisheye_on_prim(stage, cam_path)

create_camera_visual(stage, left_cam_path, "left_cam", [255, 50, 50])
create_camera_visual(stage, right_cam_path, "right_cam", [50, 80, 255])

bar_prim = UsdGeom.Cube.Define(stage, f"{rig_path}/baseline_bar")
bar_prim.GetSizeAttr().Set(1.0)
bar_prim.AddScaleOp().Set(Gf.Vec3f(0.01, float(BASELINE_M * stage_scale), 0.01))
bar_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.8, 0.2)])

for _ in range(5):
    simulation_app.update()

print(f"[Capture] StereoRig at ({args.camera_x}, {args.camera_y}, {height_m})m, baseline={BASELINE_M*1000:.0f}mm")

# ============================================================================
# PHASE 3: Replicator annotators + 体素系统初始化
# ============================================================================
print("[Capture] Setting up replicator annotators...")
rp_left = rep.create.render_product(left_cam_path, resolution=(CAM_W, CAM_H))
rp_right = rep.create.render_product(right_cam_path, resolution=(CAM_W, CAM_H))

annot_left = rep.AnnotatorRegistry.get_annotator("rgb")
annot_left.attach([rp_left.path])
annot_right = rep.AnnotatorRegistry.get_annotator("rgb")
annot_right.attach([rp_right.path])

for _ in range(5):
    simulation_app.update()

print("[Capture] Initializing voxel grid...")
voxel_template = VoxelGrid()
print(f"[Capture] VoxelGrid: {voxel_template.NX}x{voxel_template.NY}x{voxel_template.NZ} "
      f"= {voxel_template.NX * voxel_template.NY * voxel_template.NZ:,} voxels")

# PhysX scene query interface
from omni.physx import get_physx_scene_query_interface
physx_sqi = get_physx_scene_query_interface()

# ============================================================================
# Output directories
# ============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
LEFT_DIR = os.path.join(OUTPUT_DIR, "left")
RIGHT_DIR = os.path.join(OUTPUT_DIR, "right")
VOXEL_DIR = os.path.join(OUTPUT_DIR, "voxel")
META_DIR = os.path.join(OUTPUT_DIR, "meta")
for d in [LEFT_DIR, RIGHT_DIR, VOXEL_DIR, META_DIR]:
    os.makedirs(d, exist_ok=True)

# 清空输出目录，每次运行从头开始
import shutil
for d in [LEFT_DIR, RIGHT_DIR, VOXEL_DIR, META_DIR]:
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
# 清理根目录下的旧配置文件
for old_file in ["calibration.json", "voxel_config.json", "instance_map.json"]:
    old_path = os.path.join(OUTPUT_DIR, old_file)
    if os.path.isfile(old_path):
        os.remove(old_path)
start_frame_id = 0
print(f"[Capture] Output directory cleaned: {OUTPUT_DIR}")

# 保存标定信息
calib_path = os.path.join(OUTPUT_DIR, "calibration.json")
with open(calib_path, "w") as f:
    json.dump({
        "sensor": "SC132GS (simulated)",
        "projection": "ftheta_equidistant",
        "resolution": [CAM_W, CAM_H],
        "fx": fx, "fy": fx, "cx": cx, "cy": cy,
        "k1": K1_EQUIDISTANT,
        "max_fov_deg": DIAG_FOV_DEG,
        "baseline_m": BASELINE_M,
        "baseline_direction": "Y-axis (left=+Y, right=-Y)",
    }, f, indent=2)

# 保存体素配置
voxel_config_path = os.path.join(OUTPUT_DIR, "voxel_config.json")
with open(voxel_config_path, "w") as f:
    json.dump(VoxelGrid.get_config(), f, indent=2)

print(f"[Capture] Output: {OUTPUT_DIR}")


# ============================================================================
# 读取相机世界位姿
# ============================================================================
def get_rig_world_pose(stage, rig_path):
    """获取 StereoRig 的世界位置和 yaw 角。

    Returns:
        cam_pos: (3,) 世界坐标 [x, y, z]（米）
        cam_yaw: float, 绕 Z 轴偏航角（弧度）
    """
    rig_prim = stage.GetPrimAtPath(rig_path)
    if not rig_prim.IsValid():
        return np.array([0.0, 0.0, args.camera_height]), 0.0

    xformable = UsdGeom.Xformable(rig_prim)
    world_xform = xformable.ComputeLocalToWorldTransform(0.0)
    translate = world_xform.ExtractTranslation()

    # 转换为米
    cam_pos = np.array([
        translate[0] * stage_mpu,
        translate[1] * stage_mpu,
        translate[2] * stage_mpu,
    ])

    # 提取 yaw（绕 Z 轴旋转）
    rotation = world_xform.ExtractRotationMatrix()
    # yaw = atan2(R[1][0], R[0][0])
    cam_yaw = np.arctan2(rotation[1][0], rotation[0][0])

    return cam_pos, cam_yaw


# ============================================================================
# 同步采集函数
# ============================================================================
def capture_frame(frame_id):
    """冻结 → 拍照 + 体素查询 → 恢复。

    Returns:
        bool: 是否成功采集
    """
    timeline = omni.timeline.get_timeline_interface()

    # 1. 冻结世界
    timeline.pause()

    # 2. 刷新渲染管线（pause 后仍需渲染当前帧）
    for _ in range(3):
        simulation_app.update()

    # 3. 采集双目图像
    data_l = annot_left.get_data()
    data_r = annot_right.get_data()

    if data_l is None or data_r is None:
        timeline.play()
        return False

    if not isinstance(data_l, np.ndarray) or not isinstance(data_r, np.ndarray):
        timeline.play()
        return False

    if data_l.size == 0 or data_r.size == 0:
        timeline.play()
        return False

    rgb_l = data_l[:, :, :3] if data_l.ndim == 3 else data_l
    rgb_r = data_r[:, :, :3] if data_r.ndim == 3 else data_r

    if rgb_l.mean() < 1.0 or rgb_r.mean() < 1.0:
        timeline.play()
        return False

    # 4. 读取相机世界位姿
    cam_pos, cam_yaw = get_rig_world_pose(stage, rig_path)

    # 5. 体素填充
    vg = VoxelGrid()
    world_centers_flat = vg.get_world_centers_flat(cam_pos, cam_yaw)
    fill_voxel_grid(stage, vg, world_centers_flat, physx_sqi)

    # 6. 异步保存
    frame_str = f"frame_{frame_id:06d}"
    async_save_image(os.path.join(LEFT_DIR, f"{frame_str}.png"), rgb_l)
    async_save_image(os.path.join(RIGHT_DIR, f"{frame_str}.png"), rgb_r)
    async_save_voxel(os.path.join(VOXEL_DIR, frame_str), vg.semantic, vg.instance)

    # 帧元数据
    meta = {
        "frame_id": frame_id,
        "camera_pos": cam_pos.tolist(),
        "camera_yaw_rad": float(cam_yaw),
        "camera_height_m": float(cam_pos[2]),
        "voxel_origin_world": [float(cam_pos[0]), float(cam_pos[1]), 0.0],
        "voxel_size": VoxelGrid.VOXEL_SIZE,
        "voxel_shape": [VoxelGrid.NX, VoxelGrid.NY, VoxelGrid.NZ],
        "z_ground_index": VoxelGrid.Z_GROUND_INDEX,
        "stats": vg.stats() if frame_id % 10 == 0 else {},
    }
    async_save_json(os.path.join(META_DIR, f"{frame_str}.json"), meta)

    if frame_id % 5 == 0:
        occ = int(np.sum((vg.semantic > 0) & (vg.semantic < UNOBSERVED)))
        print(f"  Frame {frame_id}: L_mean={rgb_l.mean():.1f}, R_mean={rgb_r.mean():.1f}, "
              f"occupied={occ}, pos=({cam_pos[0]:.1f},{cam_pos[1]:.1f},{cam_pos[2]:.1f})")

    # 7. 恢复世界
    timeline.play()
    return True


# ============================================================================
# PHASE 4: 主循环
# ============================================================================
frame_id = start_frame_id
sim_step = 0
warmup_steps = 30

timeline = omni.timeline.get_timeline_interface()

if args.headless:
    # === Headless 模式 ===
    print(f"[Capture] Headless: capturing {args.num_frames} frames, interval={args.capture_interval}...")
    timeline.play()

    for _ in range(warmup_steps):
        simulation_app.update()

    captured = 0
    max_steps = args.num_frames * args.capture_interval * 3
    step = 0

    while captured < args.num_frames and step < max_steps and not simulation_app.is_exiting():
        simulation_app.update()
        step += 1

        if step % args.capture_interval == 0:
            if capture_frame(frame_id):
                frame_id += 1
                captured += 1

    timeline.stop()
    wait_pending_saves()
    print(f"[Capture] Headless done: {captured} frames captured.")

else:
    # === GUI 模式 ===
    recording = False
    was_playing = False

    print("\n" + "=" * 60)
    print("[Capture] GUI mode")
    print(f"  - StereoRig: /World/StereoRig (draggable)")
    print(f"  - Camera: {height_m}m height, {BASELINE_M*1000:.0f}mm baseline")
    if npc_ready:
        print(f"  - {num_chars} NPC(s) ready")
    print(f"  - Capture every {args.capture_interval} steps with timeline freeze")
    print(f"  - Output: {OUTPUT_DIR}")
    print("  - Play to start | Stop to pause")
    print("=" * 60 + "\n")

    while simulation_app.is_running():
        simulation_app.update()

        is_playing = timeline.is_playing()

        if is_playing and not was_playing:
            recording = True
            sim_step = 0
            print(f"[Capture] Recording started (frame_id={frame_id})")

        if not is_playing and was_playing:
            recording = False
            wait_pending_saves()
            print(f"[Capture] Recording paused (captured to frame_id={frame_id})")

        was_playing = is_playing

        if not recording:
            continue

        sim_step += 1

        if sim_step <= warmup_steps:
            continue

        if sim_step % args.capture_interval == 0:
            capture_frame(frame_id)
            frame_id += 1

# ============================================================================
# 收尾
# ============================================================================
wait_pending_saves()

# 保存 instance 映射
instance_map_path = os.path.join(OUTPUT_DIR, "instance_map.json")
with open(instance_map_path, "w") as f:
    json.dump({str(v): k for k, v in _instance_map.items()}, f, indent=2)

total = frame_id - start_frame_id
print(f"\n[Capture] Done! {total} frames saved (frame {start_frame_id}~{frame_id - 1})")
print(f"  Left:     {LEFT_DIR}")
print(f"  Right:    {RIGHT_DIR}")
print(f"  Voxel:    {VOXEL_DIR}")
print(f"  Meta:     {META_DIR}")
print(f"  Calib:    {calib_path}")
print(f"  Voxel cfg: {voxel_config_path}")
print(f"  Instance:  {instance_map_path}")

_save_pool.shutdown(wait=True)
simulation_app.close()
