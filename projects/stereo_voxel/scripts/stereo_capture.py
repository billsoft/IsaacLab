"""
双目鱼眼相机 NPC 数据采集脚本
================================
在仓库场景中创建双目鱼眼相机（SC132GS × 2），俯视拍摄 NPC 行人。
以 5 FPS 保存左右眼图像 + 时间戳 + 标定文件。

功能：
  - IRA SimulationManager 管理 NPC 行走
  - 双目鱼眼相机 (80mm 基线, 2.5m 高度, 向下看)
  - 5 FPS 定时采集 RGB 图像
  - 可选深度图采集
  - 输出 timestamps.csv + calibration.json

运行方式：
    isaaclab.bat -p projects/stereo_voxel/scripts/stereo_capture.py
    isaaclab.bat -p projects/stereo_voxel/scripts/stereo_capture.py --headless
    isaaclab.bat -p projects/stereo_voxel/scripts/stereo_capture.py --num_characters 5 --duration 60 --save_depth
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import time

parser = argparse.ArgumentParser(description="Stereo fisheye camera NPC capture")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--num_characters", type=int, default=3, help="Number of NPCs")
parser.add_argument("--walk_distance", type=float, default=8.0, help="Walk distance (m)")
parser.add_argument("--duration", type=float, default=120.0, help="Simulation duration (s)")
parser.add_argument("--capture_fps", type=float, default=5.0, help="Image capture FPS")
parser.add_argument("--camera_height", type=float, default=2.5, help="Camera height (m)")
parser.add_argument("--camera_x", type=float, default=0.0, help="Camera X position")
parser.add_argument("--camera_y", type=float, default=3.0, help="Camera Y position (center of NPC area)")
parser.add_argument("--save_depth", action="store_true", help="Also save depth maps")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# 1. SimulationApp
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1920,
    "height": 1080,
})

# ---------------------------------------------------------------------------
# 2. Enable required extensions
# ---------------------------------------------------------------------------
import carb
import omni.kit.app
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

print("[Stereo Capture] Enabling extensions...")
for ext in REQUIRED_EXTENSIONS:
    enable_extension(ext)
    simulation_app.update()

simulation_app.update()
simulation_app.update()

# ---------------------------------------------------------------------------
# 3. Imports (after SimulationApp)
# ---------------------------------------------------------------------------
import cv2
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
from isaacsim.sensors.camera import Camera

# ---------------------------------------------------------------------------
# 4. Asset paths
# ---------------------------------------------------------------------------
LOCAL_ASSETS_PATH = "D:/code/IsaacLab/Assets/Isaac/5.1"
if not os.path.isdir(LOCAL_ASSETS_PATH):
    try:
        from isaacsim.storage.native import get_assets_root_path
        LOCAL_ASSETS_PATH = get_assets_root_path()
    except Exception:
        pass

if not LOCAL_ASSETS_PATH or not os.path.isdir(LOCAL_ASSETS_PATH):
    print("[Stereo Capture] ERROR: Cannot find Isaac assets.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)

ASSETS_ROOT = LOCAL_ASSETS_PATH.replace("\\", "/")
SCENE_USD = f"{ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

CHARACTER_MODELS = [
    "Isaac/People/Characters/F_Business_02/F_Business_02.usd",
    "Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd",
    "Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd",
    "Isaac/People/Characters/M_Medical_01/M_Medical_01.usd",
    "Isaac/People/Characters/male_adult_construction_03/male_adult_construction_03.usd",
    "Isaac/People/Characters/female_adult_police_03_new/female_adult_police_03_new.usd",
]

AVAILABLE_MODELS = [m for m in CHARACTER_MODELS if os.path.isfile(f"{ASSETS_ROOT}/{m}")]
if not AVAILABLE_MODELS:
    print("[Stereo Capture] ERROR: No character models found.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)

print(f"[Stereo Capture] Found {len(AVAILABLE_MODELS)} character model(s)")

# ---------------------------------------------------------------------------
# 5. SC132GS camera parameters
# ---------------------------------------------------------------------------
PIXEL_SIZE_UM = 2.7
FOCAL_LENGTH_MM = 1.75
F_STOP = 2.0
CAM_WIDTH, CAM_HEIGHT = 1280, 1080
BASELINE_M = 0.08  # 80mm

pixel_size_m = PIXEL_SIZE_UM * 1e-6
focal_length_m = FOCAL_LENGTH_MM * 1e-3
horizontal_aperture_m = pixel_size_m * CAM_WIDTH
vertical_aperture_m = pixel_size_m * CAM_HEIGHT

fx = FOCAL_LENGTH_MM / PIXEL_SIZE_UM * 1000  # 648.15 px
fy = fx
cx_px = CAM_WIDTH / 2.0
cy_px = CAM_HEIGHT / 2.0
fisheye_k = [0.0, 0.0, 0.0, 0.0]

# ---------------------------------------------------------------------------
# 6. IRA command & config file generation (from npc_people_demo.py)
# ---------------------------------------------------------------------------
def generate_command_file(num_characters: int, walk_distance: float, num_loops: int = 100) -> str:
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
    return "npc_walk_commands.txt"


def generate_config_file(num_characters: int, command_file: str) -> str:
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
    return config_file


# ---------------------------------------------------------------------------
# 7. Output directories
# ---------------------------------------------------------------------------
if args.output_dir:
    OUTPUT_DIR = args.output_dir
else:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

LEFT_DIR = os.path.join(OUTPUT_DIR, "left")
RIGHT_DIR = os.path.join(OUTPUT_DIR, "right")
os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

if args.save_depth:
    DEPTH_LEFT_DIR = os.path.join(OUTPUT_DIR, "depth_left")
    DEPTH_RIGHT_DIR = os.path.join(OUTPUT_DIR, "depth_right")
    os.makedirs(DEPTH_LEFT_DIR, exist_ok=True)
    os.makedirs(DEPTH_RIGHT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def main():
    num_chars = min(args.num_characters, len(AVAILABLE_MODELS))

    # --- IRA setup ---
    cmd_file = generate_command_file(num_chars, args.walk_distance)
    config_file = generate_config_file(num_chars, cmd_file)

    enable_extension("isaacsim.replicator.agent.core")
    simulation_app.update()
    simulation_app.update()

    from isaacsim.replicator.agent.core.simulation import SimulationManager

    sim_manager = SimulationManager()

    settings = carb.settings.get_settings()
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
    settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
    settings.set("/app/omni.graph.scriptnode/enable_opt_in", False)

    can_load = sim_manager.load_config_file(config_file)
    if not can_load:
        print("[Stereo Capture] ERROR: Failed to load IRA config.", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)

    print("[Stereo Capture] Setting up NPC simulation...")

    setup_done = [False]

    def on_setup_done(e):
        setup_done[0] = True

    sim_manager.register_set_up_simulation_done_callback(on_setup_done)
    sim_manager.set_up_simulation_from_config_file()

    while not setup_done[0] and not simulation_app.is_exiting():
        simulation_app.update()

    if simulation_app.is_exiting():
        return

    print("[Stereo Capture] NPC setup complete!")

    # --- 检测场景单位 ---
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
    stage_scale = 1.0 / stage_mpu  # stage units per meter
    print(f"[Stereo Capture] Stage metersPerUnit={stage_mpu}, scale={stage_scale}")

    # --- Create stereo cameras ---
    print("[Stereo Capture] Creating stereo fisheye cameras...")

    camera_euler = np.array([0.0, 90.0, 0.0])
    orientation = rot_utils.euler_angles_to_quats(camera_euler, degrees=True)

    # 位置转换为 stage units
    cam_center = np.array([
        args.camera_x * stage_scale,
        args.camera_y * stage_scale,
        args.camera_height * stage_scale,
    ])
    baseline_su = BASELINE_M * stage_scale
    left_pos = cam_center + np.array([0.0, baseline_su / 2, 0.0])
    right_pos = cam_center - np.array([0.0, baseline_su / 2, 0.0])

    print(f"  Left:  {left_pos} (stage units)")
    print(f"  Right: {right_pos} (stage units)")

    camera_left = Camera(
        prim_path="/World/Stereo/Left",
        position=left_pos,
        frequency=30,
        resolution=(CAM_WIDTH, CAM_HEIGHT),
        orientation=orientation,
    )
    camera_right = Camera(
        prim_path="/World/Stereo/Right",
        position=right_pos,
        frequency=30,
        resolution=(CAM_WIDTH, CAM_HEIGHT),
        orientation=orientation,
    )

    # Camera needs a few updates after creation
    for _ in range(5):
        simulation_app.update()

    camera_left.initialize()
    camera_right.initialize()

    def setup_cam(cam):
        cam.set_focal_length(focal_length_m * stage_scale)
        cam.set_horizontal_aperture(horizontal_aperture_m * stage_scale)
        cam.set_lens_aperture(0.0)  # 禁用 DOF
        cam.set_clipping_range(0.1 * stage_scale, 100.0 * stage_scale)
        cam.set_focus_distance(args.camera_height * stage_scale)
        cam.set_opencv_fisheye_properties(
            cx=cx_px, cy=cy_px, fx=fx, fy=fy, fisheye=fisheye_k,
        )

    setup_cam(camera_left)
    setup_cam(camera_right)

    print("[Stereo Capture] Cameras ready.")

    # --- Save calibration ---
    calibration = {
        "sensor": "SC132GS (simulated)",
        "projection_model": "opencv_fisheye_equidistant",
        "resolution": [CAM_WIDTH, CAM_HEIGHT],
        "baseline_m": BASELINE_M,
        "baseline_direction": "Y-axis",
        "camera_height_m": args.camera_height,
        "capture_fps": args.capture_fps,
        "left": {
            "fx": fx, "fy": fy, "cx": cx_px, "cy": cy_px,
            "k1": fisheye_k[0], "k2": fisheye_k[1],
            "k3": fisheye_k[2], "k4": fisheye_k[3],
            "position_world": left_pos.tolist(),
            "position_relative_to_center": [0.0, BASELINE_M / 2, 0.0],
        },
        "right": {
            "fx": fx, "fy": fy, "cx": cx_px, "cy": cy_px,
            "k1": fisheye_k[0], "k2": fisheye_k[1],
            "k3": fisheye_k[2], "k4": fisheye_k[3],
            "position_world": right_pos.tolist(),
            "position_relative_to_center": [0.0, -BASELINE_M / 2, 0.0],
        },
    }

    calib_path = os.path.join(OUTPUT_DIR, "calibration.json")
    with open(calib_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"[Stereo Capture] Calibration saved: {calib_path}")

    # --- Start IRA data generation (NPC walking) ---
    async def run_sim():
        await sim_manager.run_data_generation_async(will_wait_until_complete=True)

    from omni.kit.async_engine import run_coroutine

    task = run_coroutine(run_sim())

    # --- Capture loop ---
    timestamps_path = os.path.join(OUTPUT_DIR, "timestamps.csv")
    ts_file = open(timestamps_path, "w", newline="")
    ts_writer = csv.writer(ts_file)
    ts_writer.writerow(["frame_id", "timestamp_sec", "cam_x", "cam_y", "cam_z"])

    capture_interval = 1.0 / args.capture_fps  # seconds between captures
    sim_dt = 1.0 / 60.0  # assume ~60 Hz sim
    steps_per_capture = max(1, int(capture_interval / sim_dt))

    frame_id = 0
    step_count = 0
    sim_time = 0.0
    capture_time_next = 0.0
    t_start = time.time()

    print(f"[Stereo Capture] Starting capture: {args.capture_fps} FPS, interval={capture_interval:.3f}s")
    print(f"[Stereo Capture] {num_chars} NPC(s), duration={args.duration}s")
    print("[Stereo Capture] Press Ctrl+C to stop early.")

    try:
        while not task.done() and not simulation_app.is_exiting():
            simulation_app.update()
            step_count += 1
            sim_time = step_count * sim_dt

            # Check if it's time to capture
            if sim_time >= capture_time_next:
                rgb_l = camera_left.get_rgb()
                rgb_r = camera_right.get_rgb()

                if rgb_l is not None and rgb_l.size > 0 and rgb_r is not None and rgb_r.size > 0:
                    # Save RGB
                    img_l = cv2.cvtColor(rgb_l.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    img_r = cv2.cvtColor(rgb_r.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(LEFT_DIR, f"frame_{frame_id:06d}.png"), img_l)
                    cv2.imwrite(os.path.join(RIGHT_DIR, f"frame_{frame_id:06d}.png"), img_r)

                    # Save depth (optional)
                    if args.save_depth:
                        depth_l = camera_left.get_depth()
                        depth_r = camera_right.get_depth()
                        if depth_l is not None:
                            np.save(os.path.join(DEPTH_LEFT_DIR, f"frame_{frame_id:06d}.npy"), depth_l)
                        if depth_r is not None:
                            np.save(os.path.join(DEPTH_RIGHT_DIR, f"frame_{frame_id:06d}.npy"), depth_r)

                    # Write timestamp
                    ts_writer.writerow([
                        frame_id,
                        f"{sim_time:.4f}",
                        f"{cam_center[0]:.4f}",
                        f"{cam_center[1]:.4f}",
                        f"{cam_center[2]:.4f}",
                    ])

                    if frame_id % 10 == 0:
                        elapsed = time.time() - t_start
                        print(f"  Frame {frame_id:04d} | sim_t={sim_time:.2f}s | wall={elapsed:.1f}s")

                    frame_id += 1
                    capture_time_next += capture_interval

            # Progress report
            if step_count % 600 == 0:
                elapsed = time.time() - t_start
                print(f"[Stereo Capture] step={step_count}, sim_t={sim_time:.1f}s, "
                      f"frames={frame_id}, wall={elapsed:.1f}s")

    except KeyboardInterrupt:
        print("\n[Stereo Capture] Interrupted by user.")
    finally:
        ts_file.close()

    elapsed = time.time() - t_start
    print(f"\n[Stereo Capture] Done!")
    print(f"  Captured {frame_id} frame pairs in {elapsed:.1f}s")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Left images:   {LEFT_DIR}")
    print(f"  Right images:  {RIGHT_DIR}")
    if args.save_depth:
        print(f"  Depth (left):  {DEPTH_LEFT_DIR}")
        print(f"  Depth (right): {DEPTH_RIGHT_DIR}")
    print(f"  Timestamps:    {timestamps_path}")
    print(f"  Calibration:   {calib_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
