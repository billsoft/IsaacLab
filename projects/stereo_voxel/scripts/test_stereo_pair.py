"""
Step 2: 双目鱼眼相机测试
========================
创建左右两个下视鱼眼相机（SC132GS × 2），80mm 基线，
验证：
  - 两个 Camera 能否同时渲染
  - 左右眼图像是否有视差
  - 基线方向正确性

运行方式：
    isaaclab.bat -p projects/stereo_voxel/scripts/test_stereo_pair.py
    isaaclab.bat -p projects/stereo_voxel/scripts/test_stereo_pair.py --headless
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Test stereo fisheye camera pair")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--num_frames", type=int, default=5, help="Number of frame pairs to capture")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 1080})

import cv2
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

# ---------------------------------------------------------------------------
# SC132GS 参数（与 test_single_camera.py 相同）
# ---------------------------------------------------------------------------
PIXEL_SIZE_UM = 2.7
FOCAL_LENGTH_MM = 1.75
F_STOP = 2.0
WIDTH, HEIGHT = 1280, 1080
CAMERA_HEIGHT = 2.5
BASELINE_MM = 80             # 双目基线 80mm
BASELINE_M = BASELINE_MM * 1e-3  # 0.08m

pixel_size_m = PIXEL_SIZE_UM * 1e-6
focal_length_m = FOCAL_LENGTH_MM * 1e-3
horizontal_aperture_m = pixel_size_m * WIDTH
vertical_aperture_m = pixel_size_m * HEIGHT

fx = FOCAL_LENGTH_MM / PIXEL_SIZE_UM * 1000
fy = fx
cx = WIDTH / 2.0
cy = HEIGHT / 2.0
fisheye_k = [0.0, 0.0, 0.0, 0.0]

# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
LOCAL_ASSETS_PATH = "D:/code/IsaacLab/Assets/Isaac/5.1"
if not os.path.isdir(LOCAL_ASSETS_PATH):
    try:
        from isaacsim.storage.native import get_assets_root_path
        LOCAL_ASSETS_PATH = get_assets_root_path()
    except Exception:
        pass

if not LOCAL_ASSETS_PATH or not os.path.isdir(LOCAL_ASSETS_PATH):
    print("[Stereo Test] ERROR: Cannot find Isaac assets.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)

ASSETS_ROOT = LOCAL_ASSETS_PATH.replace("\\", "/")
SCENE_USD = f"{ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

world = World(stage_units_in_meters=1.0)

import omni.usd
from pxr import UsdGeom
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.append(SCENE_USD)

for _ in range(5):
    simulation_app.update()

# 检测场景单位
stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
stage_scale = 1.0 / stage_mpu  # stage units per meter
print(f"[Stereo Test] Stage metersPerUnit={stage_mpu}, scale={stage_scale}")

# ---------------------------------------------------------------------------
# 创建双目相机
# ---------------------------------------------------------------------------
# 基线沿 Y 轴：左眼 +Y 偏移，右眼 -Y 偏移
# 相机朝下：euler [0, 90, 0]
camera_euler = np.array([0.0, 90.0, 0.0])
orientation = rot_utils.euler_angles_to_quats(camera_euler, degrees=True)

# 双目中心位置（转换为 stage units）
center_x, center_y = 0.0, 0.0

left_pos = np.array([center_x, center_y + BASELINE_M / 2 * stage_scale, CAMERA_HEIGHT * stage_scale])
right_pos = np.array([center_x, center_y - BASELINE_M / 2 * stage_scale, CAMERA_HEIGHT * stage_scale])

print(f"[Stereo Test] Left camera:  {left_pos}")
print(f"[Stereo Test] Right camera: {right_pos}")
print(f"[Stereo Test] Baseline: {BASELINE_MM}mm along Y-axis")

camera_left = Camera(
    prim_path="/World/Stereo/Left",
    position=left_pos,
    frequency=30,
    resolution=(WIDTH, HEIGHT),
    orientation=orientation,
)

camera_right = Camera(
    prim_path="/World/Stereo/Right",
    position=right_pos,
    frequency=30,
    resolution=(WIDTH, HEIGHT),
    orientation=orientation,
)

world.reset()
camera_left.initialize()
camera_right.initialize()


def setup_camera_params(cam):
    """设置鱼眼相机光学参数（所有值转为 stage units）"""
    cam.set_focal_length(focal_length_m * stage_scale)
    cam.set_horizontal_aperture(horizontal_aperture_m * stage_scale)
    cam.set_lens_aperture(0.0)  # 禁用 DOF
    cam.set_clipping_range(0.1 * stage_scale, 100.0 * stage_scale)
    cam.set_focus_distance(CAMERA_HEIGHT * stage_scale)
    cam.set_opencv_fisheye_properties(
        cx=cx, cy=cy, fx=fx, fy=fy,
        fisheye=fisheye_k,
    )


setup_camera_params(camera_left)
setup_camera_params(camera_right)

print("[Stereo Test] Both cameras initialized. Rendering...")

# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
LEFT_DIR = os.path.join(OUTPUT_DIR, "left")
RIGHT_DIR = os.path.join(OUTPUT_DIR, "right")
os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 渲染稳定
# ---------------------------------------------------------------------------
for i in range(10):
    world.step(render=True)

# ---------------------------------------------------------------------------
# 采集帧对
# ---------------------------------------------------------------------------
for frame_id in range(args.num_frames):
    world.step(render=True)

    rgb_left = camera_left.get_rgb()
    rgb_right = camera_right.get_rgb()

    if rgb_left is not None and rgb_left.size > 0:
        img_l = cv2.cvtColor(rgb_left.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(LEFT_DIR, f"frame_{frame_id:06d}.png"), img_l)
    else:
        print(f"  Frame {frame_id}: left camera no data")

    if rgb_right is not None and rgb_right.size > 0:
        img_r = cv2.cvtColor(rgb_right.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(RIGHT_DIR, f"frame_{frame_id:06d}.png"), img_r)
    else:
        print(f"  Frame {frame_id}: right camera no data")

    if rgb_left is not None and rgb_right is not None:
        print(f"  Frame {frame_id}: L={rgb_left.shape} R={rgb_right.shape} saved")

# ---------------------------------------------------------------------------
# 保存 calibration.json
# ---------------------------------------------------------------------------
import json

calibration = {
    "sensor": "SC132GS",
    "projection_model": "opencv_fisheye_equidistant",
    "resolution": [WIDTH, HEIGHT],
    "baseline_m": BASELINE_M,
    "baseline_direction": "Y-axis (left=+Y, right=-Y)",
    "camera_height_m": CAMERA_HEIGHT,
    "left": {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "k1": fisheye_k[0], "k2": fisheye_k[1],
        "k3": fisheye_k[2], "k4": fisheye_k[3],
        "position_world": left_pos.tolist(),
        "euler_deg": camera_euler.tolist(),
    },
    "right": {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "k1": fisheye_k[0], "k2": fisheye_k[1],
        "k3": fisheye_k[2], "k4": fisheye_k[3],
        "position_world": right_pos.tolist(),
        "euler_deg": camera_euler.tolist(),
    },
}

calib_file = os.path.join(OUTPUT_DIR, "calibration.json")
with open(calib_file, "w") as f:
    json.dump(calibration, f, indent=2)

print(f"\n[Stereo Test] Results saved to: {OUTPUT_DIR}")
print(f"  Left images:  {LEFT_DIR}")
print(f"  Right images: {RIGHT_DIR}")
print(f"  Calibration:  {calib_file}")
print(f"\n[Stereo Test] Verify:")
print(f"  1. Both left and right images show warehouse from above")
print(f"  2. Left/right images have slight horizontal shift (parallax from {BASELINE_MM}mm baseline)")
print(f"  3. Fisheye distortion visible at image edges")

simulation_app.close()
