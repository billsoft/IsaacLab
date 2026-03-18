"""双目鱼眼相机 + IRA NPC 数据采集
====================================
使用 IRA 官方 async API 管理 NPC，使用 omni.replicator annotator 异步采集双目图像。

关键设计：
  1. IRA 管 NPC 行走动画（通过 config YAML + async API）
  2. 相机采集用 omni.replicator 的 render_product + annotator（不用 Camera.get_rgb）
  3. 图像存储用异步线程池，不阻塞渲染主线程
  4. annotator.get_data() 在渲染完成后才返回有效数据，天然同步

为什么不用 Camera.get_rgb()：
  Camera 类的 _stage_open_callback_fn 在 StageEventType.OPENED 事件到达时
  会清除 _acquisition_callback = None，导致 open_stage 后创建的相机返回全黑。
  直接用 replicator annotator 绕过这个问题。

运行：
    isaaclab.bat -p projects/stereo_voxel/scripts/test_stereo_pair.py
    isaaclab.bat -p projects/stereo_voxel/scripts/test_stereo_pair.py --headless --num_frames 50
"""

import argparse
import glob
import json
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Stereo fisheye camera + NPC data generation")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--num_frames", type=int, default=50, help="Number of frame pairs to capture (headless mode)")
parser.add_argument("--camera_height", type=float, default=3.0, help="Camera height (m)")
parser.add_argument("--num_characters", type=int, default=3, help="Number of NPC characters")
parser.add_argument("--walk_distance", type=float, default=8.0, help="NPC walk distance (m)")
parser.add_argument("--capture_interval", type=int, default=3, help="Capture every N simulation steps")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 1080})

import carb
import cv2
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, UsdGeom

# ---------------------------------------------------------------------------
# SC132GS 参数
# ---------------------------------------------------------------------------
CAM_W, CAM_H = 1280, 1080
PIXEL_SIZE_UM = 2.7
FOCAL_LENGTH_MM = 1.75
DIAG_FOV_DEG = 157.2
BASELINE_M = 0.08  # 80mm

fx = FOCAL_LENGTH_MM / PIXEL_SIZE_UM * 1000  # 648.15 px
cx = CAM_W / 2.0
cy = CAM_H / 2.0
K1_EQUIDISTANT = 1.0 / fx

print(f"[Stereo] SC132GS x2, baseline={BASELINE_M*1000:.0f}mm")
print(f"  Resolution: {CAM_W}x{CAM_H}, fx={fx:.2f}px, FOV={DIAG_FOV_DEG}deg")

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
LOCAL_ASSETS_PATH = "D:/code/IsaacLab/Assets/Isaac/5.1"
if not os.path.isdir(LOCAL_ASSETS_PATH):
    try:
        from isaacsim.storage.native import get_assets_root_path
        LOCAL_ASSETS_PATH = get_assets_root_path()
    except Exception:
        pass
if not LOCAL_ASSETS_PATH or not os.path.isdir(LOCAL_ASSETS_PATH):
    print("[Stereo] ERROR: Cannot find Isaac assets.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)

ASSETS_ROOT = LOCAL_ASSETS_PATH.replace("\\", "/")
SCENE_USD = f"{ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

# ---------------------------------------------------------------------------
# NPC character models
# ---------------------------------------------------------------------------
CHARACTER_MODELS = [
    "Isaac/People/Characters/F_Business_02/F_Business_02.usd",
    "Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd",
    "Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd",
    "Isaac/People/Characters/M_Medical_01/M_Medical_01.usd",
    "Isaac/People/Characters/male_adult_construction_03/male_adult_construction_03.usd",
    "Isaac/People/Characters/female_adult_police_03_new/female_adult_police_03_new.usd",
]
AVAILABLE_MODELS = [m for m in CHARACTER_MODELS if os.path.isfile(f"{ASSETS_ROOT}/{m}")]
print(f"[Stereo] Found {len(AVAILABLE_MODELS)} NPC character model(s).")


# ---------------------------------------------------------------------------
# NPC config generation
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helper: set fisheye lens properties on a camera prim
# ---------------------------------------------------------------------------
def set_fisheye_on_prim(stage, cam_prim_path):
    """在 USD prim 上正确应用 ftheta 鱼眼 API schema 并设置参数。

    关键：必须先 ApplyAPI 注册 schema，渲染器才会识别 ftheta 属性。
    属性名必须用 k0-k4（不是 p0-p4），opticalCenter（不是 opticalCentreX/Y）。
    这与 Camera.set_ftheta_properties() 内部实现一致。
    """
    prim = stage.GetPrimAtPath(cam_prim_path)
    if not prim.IsValid():
        print(f"[Stereo] WARNING: prim {cam_prim_path} not valid, skip fisheye setup")
        return

    # 第1步：应用 OmniLensDistortion API Schema（关键！没有这步渲染器不识别）
    prim.ApplyAPI("OmniLensDistortionFthetaAPI")

    # 第2步：设置模型类型
    prim.GetAttribute("omni:lensdistortion:model").Set("ftheta")

    # 第3步：设置 ftheta 参数（属性由 schema 定义，ApplyAPI 后自动存在）
    prim.GetAttribute("omni:lensdistortion:ftheta:nominalWidth").Set(float(CAM_W))
    prim.GetAttribute("omni:lensdistortion:ftheta:nominalHeight").Set(float(CAM_H))
    prim.GetAttribute("omni:lensdistortion:ftheta:opticalCenter").Set((float(cx), float(cy)))
    prim.GetAttribute("omni:lensdistortion:ftheta:maxFov").Set(float(DIAG_FOV_DEG))

    # 第4步：等距投影畸变系数 k0-k4（不是 p0-p4！）
    # 等距模型：theta = k0 + k1*r + k2*r^2 + k3*r^3 + k4*r^4
    # 纯等距：k1 = 1/fx，其余为 0
    prim.GetAttribute("omni:lensdistortion:ftheta:k0").Set(0.0)
    prim.GetAttribute("omni:lensdistortion:ftheta:k1").Set(float(K1_EQUIDISTANT))
    prim.GetAttribute("omni:lensdistortion:ftheta:k2").Set(0.0)
    prim.GetAttribute("omni:lensdistortion:ftheta:k3").Set(0.0)
    prim.GetAttribute("omni:lensdistortion:ftheta:k4").Set(0.0)

    # 第5步：禁用景深模糊
    prim.GetAttribute("fStop").Set(0.0)


# ---------------------------------------------------------------------------
# Helper: 可视化锥体
# ---------------------------------------------------------------------------
def create_camera_visual(stage, parent_path, name, color_rgb):
    cone_path = f"{parent_path}/{name}_visual"
    cone_prim = UsdGeom.Cone.Define(stage, cone_path)
    cone_prim.GetRadiusAttr().Set(0.03)
    cone_prim.GetHeightAttr().Set(0.06)
    cone_prim.GetAxisAttr().Set("X")
    cone_prim.GetDisplayColorAttr().Set([Gf.Vec3f(*[c / 255.0 for c in color_rgb])])


# ---------------------------------------------------------------------------
# Async image saver (non-blocking disk IO)
# ---------------------------------------------------------------------------
_save_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="img_saver")
_pending_saves = []


def async_save_image(path, rgb_array):
    """将 RGB numpy array 异步保存为 PNG（不阻塞渲染线程）。"""
    def _save(p, arr):
        bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(p, bgr)
    future = _save_pool.submit(_save, path, rgb_array.copy())
    _pending_saves.append(future)


def wait_pending_saves():
    """等待所有异步保存完成。"""
    for f in _pending_saves:
        f.result()
    _pending_saves.clear()


# ===========================================================================
# PHASE 1: IRA 加载 NPC
# ===========================================================================
from isaacsim.core.utils.extensions import enable_extension

npc_ready = False
num_chars = 0

if AVAILABLE_MODELS:
    num_chars = min(args.num_characters, len(AVAILABLE_MODELS))

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
    print("[Stereo] Enabling NPC extensions...")
    for ext in NPC_EXTENSIONS:
        enable_extension(ext)
        simulation_app.update()

    # 手动打开场景，等待 ASSETS_LOADED（让 IRA 跳过 open_stage）
    print(f"[Stereo] Pre-loading scene: {SCENE_USD}")
    scene_loaded = [False]

    def on_scene_loaded(e):
        scene_loaded[0] = True

    _handle = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.ASSETS_LOADED),
        on_event=on_scene_loaded,
        observer_name="stereo/on_scene_preload",
    )

    import omni.kit.window.file
    old_ignore = carb.settings.get_settings().get("/app/file/ignoreUnsavedStage")
    carb.settings.get_settings().set("/app/file/ignoreUnsavedStage", True)
    omni.kit.window.file.open_stage(SCENE_USD, omni.usd.UsdContextInitialLoadSet.LOAD_ALL)
    carb.settings.get_settings().set("/app/file/ignoreUnsavedStage", old_ignore if old_ignore is not None else False)

    tick = 0
    while not scene_loaded[0] and not simulation_app.is_exiting() and tick < 1500:
        simulation_app.update()
        tick += 1
        if tick % 300 == 0:
            print(f"[Stereo] Waiting for scene... tick={tick}")
    _handle = None
    print(f"[Stereo] Scene loaded after {tick} ticks")

    for _ in range(30):
        simulation_app.update()

    # IRA setup
    print(f"[Stereo] Setting up {num_chars} NPC(s) via IRA...")
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
                print(f"[Stereo] Waiting for IRA setup... tick={tick}")

        if setup_done[0]:
            npc_ready = True
            print(f"[Stereo] {num_chars} NPC(s) loaded after {tick} ticks!")
        else:
            print(f"[Stereo] WARNING: IRA timeout at {tick} ticks")
    else:
        print("[Stereo] WARNING: Failed to load NPC config")

    for _ in range(30):
        simulation_app.update()
else:
    print("[Stereo] No NPC models found, loading scene without NPC...")
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().subLayerPaths.append(SCENE_USD)
    for _ in range(30):
        simulation_app.update()

# ===========================================================================
# PHASE 2: 创建双目相机 (直接操作 USD prim + replicator annotator)
# ===========================================================================
print("[Stereo] Creating stereo cameras...")
stage = omni.usd.get_context().get_stage()
for _ in range(5):
    simulation_app.update()

stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
stage_scale = 1.0 / stage_mpu
print(f"[Stereo] Stage metersPerUnit={stage_mpu}, scale={stage_scale}")

# StereoRig Xform
height_m = args.camera_height
rig_path = "/World/StereoRig"
rig_xform = UsdGeom.Xform.Define(stage, rig_path)
rig_translate_z = float(height_m * stage_scale)
rig_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, rig_translate_z))

half_baseline = (BASELINE_M / 2.0) * stage_scale

# 创建左右相机 prim (UsdGeom.Camera)
left_cam_path = f"{rig_path}/Left"
right_cam_path = f"{rig_path}/Right"

# 相机朝向：+X → -Z (俯拍)
# euler (0, 0, 90) → 使光轴从 +X 转到 -Z（向下看）
import isaacsim.core.utils.numpy.rotations as rot_utils
cam_euler = np.array([0.0, 0.0, 90.0])
cam_quat = rot_utils.euler_angles_to_quats(cam_euler, degrees=True)  # [w, x, y, z]

for cam_path, y_offset in [(left_cam_path, half_baseline), (right_cam_path, -half_baseline)]:
    cam_prim = UsdGeom.Camera.Define(stage, cam_path)
    # 设置 xform
    xf = UsdGeom.Xformable(cam_prim.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, float(y_offset), 0.0))
    xf.AddOrientOp().Set(Gf.Quatf(float(cam_quat[0]),
                                    Gf.Vec3f(float(cam_quat[1]), float(cam_quat[2]), float(cam_quat[3]))))
    # 基本相机属性
    cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(float(0.1 * stage_scale), float(100.0 * stage_scale)))
    # 鱼眼属性
    set_fisheye_on_prim(stage, cam_path)

# 可视化
create_camera_visual(stage, left_cam_path, "left_cam", [255, 50, 50])   # 左眼=红
create_camera_visual(stage, right_cam_path, "right_cam", [50, 80, 255])  # 右眼=蓝

# 基线标识条
bar_prim = UsdGeom.Cube.Define(stage, f"{rig_path}/baseline_bar")
bar_prim.GetSizeAttr().Set(1.0)
bar_prim.AddScaleOp().Set(Gf.Vec3f(0.01, float(BASELINE_M * stage_scale), 0.01))
bar_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.8, 0.2)])

for _ in range(5):
    simulation_app.update()

print(f"[Stereo] StereoRig at z={height_m}m, baseline={BASELINE_M*1000:.0f}mm")

# ===========================================================================
# PHASE 3: 用 omni.replicator 创建 render product + annotator（异步采集）
# ===========================================================================
# 这是关键！不用 Camera.get_rgb()，而是直接用 replicator 的 annotator。
# annotator.get_data() 在新帧渲染完成后才返回有效数据，天然同步。
print("[Stereo] Setting up replicator render products and annotators...")

rp_left = rep.create.render_product(left_cam_path, resolution=(CAM_W, CAM_H))
rp_right = rep.create.render_product(right_cam_path, resolution=(CAM_W, CAM_H))

annot_left = rep.AnnotatorRegistry.get_annotator("rgb")
annot_left.attach([rp_left.path])

annot_right = rep.AnnotatorRegistry.get_annotator("rgb")
annot_right.attach([rp_right.path])

for _ in range(5):
    simulation_app.update()

print("[Stereo] Render products and annotators attached.")

# ===========================================================================
# Output dirs
# ===========================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
LEFT_DIR = os.path.join(OUTPUT_DIR, "left")
RIGHT_DIR = os.path.join(OUTPUT_DIR, "right")
os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

# 帧序号续接
existing = glob.glob(os.path.join(LEFT_DIR, "frame_*.png")) + glob.glob(os.path.join(RIGHT_DIR, "frame_*.png"))
start_frame_id = 0
for f in existing:
    try:
        num = int(os.path.splitext(os.path.basename(f))[0].split("_")[1])
        start_frame_id = max(start_frame_id, num + 1)
    except (IndexError, ValueError):
        pass
if start_frame_id > 0:
    print(f"[Stereo] Continuing from frame_id={start_frame_id}")

# 保存标定
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

# ===========================================================================
# Main loop
# ===========================================================================
frame_id = start_frame_id
sim_step = 0
warmup_steps = 30
recording = False
was_playing = False


def try_capture():
    """尝试从 annotator 获取数据并异步保存。返回是否成功。"""
    global frame_id
    data_l = annot_left.get_data()
    data_r = annot_right.get_data()

    if data_l is None or data_r is None:
        return False

    # annotator 返回 RGBA (H, W, 4) 或 RGB (H, W, 3)
    if isinstance(data_l, np.ndarray) and isinstance(data_r, np.ndarray):
        if data_l.size == 0 or data_r.size == 0:
            return False
        # 取 RGB 通道
        rgb_l = data_l[:, :, :3] if data_l.ndim == 3 else data_l
        rgb_r = data_r[:, :, :3] if data_r.ndim == 3 else data_r
        # 跳过全黑帧
        if rgb_l.mean() < 1.0 or rgb_r.mean() < 1.0:
            return False

        # 异步保存（不阻塞渲染线程）
        async_save_image(os.path.join(LEFT_DIR, f"frame_{frame_id:06d}.png"), rgb_l)
        async_save_image(os.path.join(RIGHT_DIR, f"frame_{frame_id:06d}.png"), rgb_r)

        if frame_id % 10 == 0:
            print(f"  Pair {frame_id}: L={rgb_l.shape} mean={rgb_l.mean():.1f}, "
                  f"R={rgb_r.shape} mean={rgb_r.mean():.1f}")
        frame_id += 1
        return True
    return False


if args.headless:
    # Headless 模式
    print(f"[Stereo] Headless: capturing {args.num_frames} pairs...")
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # warmup
    for _ in range(warmup_steps):
        simulation_app.update()

    captured = 0
    max_attempts = args.num_frames * args.capture_interval * 3
    attempt = 0
    while captured < args.num_frames and attempt < max_attempts and not simulation_app.is_exiting():
        simulation_app.update()
        attempt += 1
        if attempt % args.capture_interval == 0:
            if try_capture():
                captured += 1

    timeline.stop()
    wait_pending_saves()
    print(f"[Stereo] Headless done: {captured} pairs captured.")

else:
    # GUI 模式
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()

    print("\n" + "=" * 60)
    print("[Stereo] GUI 交互模式")
    print("  - /World/StereoRig 可拖拽调整位姿")
    print(f"  - 相机高度: {height_m}m, 基线: {BASELINE_M*1000:.0f}mm")
    if npc_ready:
        print(f"  - {num_chars} 个 NPC 就绪")
    print(f"  - 每 {args.capture_interval} 步采集一帧（异步存储）")
    print("  - Play ▶ 开始 | Stop ⏹ 暂停")
    print("=" * 60 + "\n")

    while simulation_app.is_running():
        simulation_app.update()

        is_playing = timeline.is_playing()

        # Play/Stop 状态切换
        if is_playing and not was_playing:
            recording = True
            sim_step = 0
            print(f"[Stereo] ▶ Recording started (frame_id={frame_id})")

        if not is_playing and was_playing:
            recording = False
            wait_pending_saves()
            print(f"[Stereo] ⏹ Recording paused (captured to frame_id={frame_id})")

        was_playing = is_playing

        if not recording:
            continue

        sim_step += 1

        # warmup
        if sim_step <= warmup_steps:
            continue

        # 按间隔采集
        if sim_step % args.capture_interval == 0:
            try_capture()

wait_pending_saves()
total = frame_id - start_frame_id
print(f"\n[Stereo] Done! {total} pairs saved (frame {start_frame_id}~{frame_id - 1})")
print(f"  Left:  {LEFT_DIR}")
print(f"  Right: {RIGHT_DIR}")
print(f"  Calib: {calib_path}")

_save_pool.shutdown(wait=True)
simulation_app.close()
