"""
Step 1: 单目鱼眼相机下视测试
============================
使用 camera.set_ftheta_properties() (OmniLensDistortion schema)
实现原生 f-theta 鱼眼投影。

注意：cameraProjectionType 在 Isaac Sim 5.x 已废弃，
      必须用 OmniLensDistortion API（set_ftheta_properties）。

SC132GS: 1280x1080, 2.7um, 1.75mm, F2.0, 对角 157.2°

f-theta 公式：θ = k0 + k1·r + k2·r² + k3·r³ + k4·r⁴
等距投影 r = f·θ → θ = r/f → k0=0, k1=1/f

运行：
    isaaclab.bat -p projects/stereo_voxel/scripts/test_single_camera.py
    isaaclab.bat -p projects/stereo_voxel/scripts/test_single_camera.py --headless
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Test single fisheye camera looking down")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to capture")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 1080})

import cv2
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from pxr import UsdGeom

# ---------------------------------------------------------------------------
# SC132GS 参数
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 1280, 1080
PIXEL_SIZE_UM = 2.7
FOCAL_LENGTH_MM = 1.75
CAMERA_HEIGHT_M = 2.5
DIAG_FOV_DEG = 157.2

# 像素焦距
fx = FOCAL_LENGTH_MM / PIXEL_SIZE_UM * 1000   # 648.15 px
cx = WIDTH / 2.0                                # 640.0
cy = HEIGHT / 2.0                                # 540.0

# f-theta 等距系数: θ = k1 · r  (k1 = 1/fx)
K1_EQUIDISTANT = 1.0 / fx   # ≈ 0.001543

print(f"[Camera Test] SC132GS f-theta 鱼眼参数:")
print(f"  分辨率: {WIDTH}x{HEIGHT}")
print(f"  像素焦距: fx={fx:.2f} px")
print(f"  对角 FOV: {DIAG_FOV_DEG}°")
print(f"  f-theta k1 (equidistant): {K1_EQUIDISTANT:.6f}")

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
    print("[Camera Test] ERROR: Cannot find Isaac assets.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)

ASSETS_ROOT = LOCAL_ASSETS_PATH.replace("\\", "/")
SCENE_USD = f"{ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

world = World(stage_units_in_meters=1.0)

import omni.usd
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.append(SCENE_USD)

for _ in range(5):
    simulation_app.update()

# 场景单位
stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
stage_scale = 1.0 / stage_mpu
print(f"[Camera Test] Stage metersPerUnit={stage_mpu}, 1m={stage_scale} stage units")

# ---------------------------------------------------------------------------
# 创建相机
# ---------------------------------------------------------------------------
camera_position = np.array([0.0, 0.0, CAMERA_HEIGHT_M * stage_scale])
camera_euler = np.array([0.0, 90.0, 0.0])  # 光轴 +X 旋转到 -Z（向下看）

print(f"[Camera Test] Camera position: {camera_position} (stage units)")
print(f"[Camera Test] Camera height: {CAMERA_HEIGHT_M}m = {CAMERA_HEIGHT_M * stage_scale} stage units")

camera = Camera(
    prim_path="/World/TestCamera",
    position=camera_position,
    frequency=30,
    resolution=(WIDTH, HEIGHT),
    orientation=rot_utils.euler_angles_to_quats(camera_euler, degrees=True),
)

world.reset()
camera.initialize()

# ---------------------------------------------------------------------------
# 设置 f-theta 鱼眼投影 (OmniLensDistortion API)
# ---------------------------------------------------------------------------
# 这是 Isaac Sim 5.x 正确方式：
#   内部执行: prim.ApplyAPI("OmniLensDistortionFthetaAPI")
#            omni:lensdistortion:model = "ftheta"
#
# 不再使用废弃的 cameraProjectionType！
camera.set_ftheta_properties(
    nominal_width=float(WIDTH),
    nominal_height=float(HEIGHT),
    optical_center=(cx, cy),
    max_fov=DIAG_FOV_DEG,
    distortion_coefficients=[0.0, K1_EQUIDISTANT, 0.0, 0.0, 0.0],
)

# 基础相机参数（DOF 等辅助效果，f_stop=0 禁用景深模糊）
camera.set_lens_aperture(0.0)
camera.set_clipping_range(0.1 * stage_scale, 100.0 * stage_scale)

print(f"[Camera Test] set_ftheta_properties() 已调用:")
print(f"  nominal: {WIDTH}x{HEIGHT}")
print(f"  optical_center: ({cx}, {cy})")
print(f"  max_fov: {DIAG_FOV_DEG}°")
print(f"  coefficients: [0, {K1_EQUIDISTANT:.6f}, 0, 0, 0]")

# 验证属性是否设置成功
model = camera.prim.GetAttribute("omni:lensdistortion:model").Get()
print(f"  omni:lensdistortion:model = {model}")

# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "test_single")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 渲染 + 采集（解决管线延迟黑帧问题）
# ---------------------------------------------------------------------------
print("[Camera Test] 预热渲染管线（30帧）...")
for i in range(30):
    world.step(render=True)

print("[Camera Test] 开始采集...")
frame_id = 0
attempt = 0
max_attempts = args.num_frames * 4

while frame_id < args.num_frames and attempt < max_attempts:
    # 渲染 3 帧再读取，确保管线刷新
    for _ in range(3):
        world.step(render=True)
    rgb = camera.get_rgb()
    attempt += 1

    if rgb is None or rgb.size == 0:
        print(f"  attempt {attempt}: no data, skipping")
        continue

    if rgb.mean() < 1.0:
        print(f"  attempt {attempt}: black frame, skipping")
        continue

    img_bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    filename = os.path.join(OUTPUT_DIR, f"frame_{frame_id:04d}.png")
    cv2.imwrite(filename, img_bgr)
    print(f"  Saved frame {frame_id}: shape={rgb.shape}, brightness={rgb.mean():.1f}")
    frame_id += 1

print(f"[Camera Test] 采集完成: {frame_id}/{args.num_frames} 帧")

# 深度图
for _ in range(3):
    world.step(render=True)
depth = camera.get_depth()
if depth is not None and depth.size > 0:
    valid_mask = np.isfinite(depth) & (depth > 0)
    if valid_mask.any():
        d_min, d_max = depth[valid_mask].min(), depth[valid_mask].max()
        depth_vis = np.zeros_like(depth, dtype=np.uint8)
        depth_vis[valid_mask] = ((depth[valid_mask] - d_min) / max(d_max - d_min, 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "depth_sample.png"), depth_vis)
        print(f"  Depth saved: [{d_min:.2f}, {d_max:.2f}]")

# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------
print(f"\n[Camera Test] 结果: {OUTPUT_DIR}")
print(f"[Camera Test] 验证要点:")
print(f"  1. 鸟瞰图（{CAMERA_HEIGHT_M}m 高度俯视仓库地面）")
print(f"  2. 边缘有鱼眼桶形畸变（{DIAG_FOV_DEG}° 超广角）")
theta_h = (WIDTH / 2.0) / fx
theta_v = (HEIGHT / 2.0) / fx
coverage_h = 2 * CAMERA_HEIGHT_M * np.tan(theta_h)
coverage_v = 2 * CAMERA_HEIGHT_M * np.tan(theta_v)
print(f"  3. 地面覆盖约 {coverage_h:.1f}m x {coverage_v:.1f}m")

simulation_app.close()
