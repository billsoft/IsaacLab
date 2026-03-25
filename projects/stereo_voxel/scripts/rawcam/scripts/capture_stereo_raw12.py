"""双目 12-bit 伪 RAW 采集脚本
==================================
在 Isaac Lab 中创建双目立体相机（80mm 基线），
采集 RGB 并转换为 12-bit Bayer RAW (RGGB) 数据。

管线: Isaac Sim RGB(uint8) → sRGB解码 → linear(float) → Bayer CFA → 12-bit → noise → uint16

输出目录结构:
    output_dir/
    ├── left/
    │   ├── frame_000000.bin    # uint16 紧凑 Bayer RAW
    │   ├── frame_000001.bin
    │   └── ...
    ├── right/
    │   ├── frame_000000.bin
    │   └── ...
    ├── rgb_preview/             # 可选，每 N 帧保存 RGB 预览
    │   ├── frame_000000_left.png
    │   └── ...
    └── manifest.json            # 元信息

运行：
    isaaclab.bat -p projects/stereo_voxel/scripts/rawcam/scripts/capture_stereo_raw12.py \
        --headless --enable_cameras --num_frames 50
"""

# ============================================================
# Isaac Lab 启动器（必须最先执行）
# ============================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="双目 12-bit 伪 RAW 采集")
parser.add_argument("--num_frames", type=int, default=50, help="采集帧数")
parser.add_argument("--width", type=int, default=1280, help="图像宽度")
parser.add_argument("--height", type=int, default=1024, help="图像高度")
parser.add_argument("--baseline", type=float, default=0.08, help="基线距离 (m)")
parser.add_argument("--cam_height", type=float, default=3.0, help="相机高度 (m)")
parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
parser.add_argument("--noise_preset", type=str, default="sc132gs",
                    choices=["clean", "light", "sc132gs", "heavy"],
                    help="噪声预设")
parser.add_argument("--save_rgb_every", type=int, default=10,
                    help="每 N 帧保存 RGB 预览 (0=不保存)")
parser.add_argument("--warmup_frames", type=int, default=40, help="预热帧数")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================
# 主逻辑（AppLauncher 之后导入）
# ============================================================
import os
import sys
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
import omni.replicator.core as rep
import omni.usd
from pxr import UsdGeom, UsdLux, Gf

# ============================================================
# 路径设置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAWCAM_DIR = os.path.dirname(SCRIPT_DIR)

if args_cli.output_dir:
    OUTPUT_DIR = args_cli.output_dir
else:
    OUTPUT_DIR = os.path.join(RAWCAM_DIR, "output", "stereo_raw12")

LEFT_DIR = os.path.join(OUTPUT_DIR, "left")
RIGHT_DIR = os.path.join(OUTPUT_DIR, "right")
RGB_DIR = os.path.join(OUTPUT_DIR, "rgb_preview")
os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)
if args_cli.save_rgb_every > 0:
    os.makedirs(RGB_DIR, exist_ok=True)

# 添加 utils 到 path
sys.path.insert(0, RAWCAM_DIR)

W, H = args_cli.width, args_cli.height
BASELINE = args_cli.baseline
CAM_H = args_cli.cam_height

# ============================================================
# 伪 RAW 转换函数
# ============================================================


def srgb_to_linear(img_uint8):
    """sRGB uint8 → linear float [0,1]"""
    srgb = img_uint8.astype(np.float32) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return linear.astype(np.float32)


def linear_to_bayer_rggb(linear_rgb, bit_depth=12):
    """linear RGB (H,W,3) float → RGGB Bayer (H,W) uint16"""
    max_val = (1 << bit_depth) - 1
    r = np.clip(linear_rgb[:, :, 0] * max_val, 0, max_val).astype(np.uint16)
    g = np.clip(linear_rgb[:, :, 1] * max_val, 0, max_val).astype(np.uint16)
    b = np.clip(linear_rgb[:, :, 2] * max_val, 0, max_val).astype(np.uint16)

    bh, bw = linear_rgb.shape[:2]
    bayer = np.zeros((bh, bw), dtype=np.uint16)
    bayer[0::2, 0::2] = r[0::2, 0::2]   # R
    bayer[0::2, 1::2] = g[0::2, 1::2]   # Gr
    bayer[1::2, 0::2] = g[1::2, 0::2]   # Gb
    bayer[1::2, 1::2] = b[1::2, 1::2]   # B
    return bayer


def add_sensor_noise(bayer_uint16, preset="sc132gs", seed=None):
    """为 12-bit Bayer 数据添加传感器噪声。"""
    presets = {
        "light": {"read": 2.0, "shot": 0.5, "dark": 0.2, "prnu": 0.005, "row": 0.2},
        "sc132gs": {"read": 4.0, "shot": 1.0, "dark": 0.5, "prnu": 0.01, "row": 0.5},
        "heavy": {"read": 8.0, "shot": 1.5, "dark": 1.5, "prnu": 0.02, "row": 1.0},
    }
    p = presets.get(preset, presets["sc132gs"])
    rng = np.random.default_rng(seed)
    h, w = bayer_uint16.shape
    data = bayer_uint16.astype(np.float64)

    # PRNU
    if p["prnu"] > 0:
        data *= rng.normal(1.0, p["prnu"], (h, w))
    # 散粒噪声
    if p["shot"] > 0:
        data += rng.normal(0, 1, (h, w)) * np.sqrt(np.maximum(data, 0)) * p["shot"]
    # 暗电流
    if p["dark"] > 0:
        data += rng.normal(p["dark"], p["dark"] * 0.6, (h, w))
    # 读出噪声
    if p["read"] > 0:
        data += rng.normal(0, p["read"], (h, w))
    # 行噪声
    if p["row"] > 0:
        data += rng.normal(0, p["row"], (h, 1))

    return np.clip(data, 0, 4095).astype(np.uint16)


def rgb_to_raw12(rgb_uint8, noise_preset=None):
    """RGB uint8 → 12-bit Bayer RAW uint16 一站式转换。"""
    linear = srgb_to_linear(rgb_uint8)
    bayer = linear_to_bayer_rggb(linear, bit_depth=12)
    if noise_preset is not None:
        bayer = add_sensor_noise(bayer, preset=noise_preset)
    return bayer


# ============================================================
# 异步存储
# ============================================================
_save_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="raw_saver")
_pending = []


def write_dng(path, bayer_uint16, width, height):
    """写入 DNG 文件 (TIFF/CFA + 基本 DNG 标签)。"""
    from tifffile import TiffWriter
    with TiffWriter(path, bigtiff=False) as tif:
        # DNG 本质是 TIFF + CFA 元数据
        # extratags: (code, dtype, count, value)
        # dtype: 1=BYTE, 3=SHORT, 4=LONG, 5=RATIONAL, 12=DOUBLE
        tif.write(
            bayer_uint16,
            photometric=32803,        # CFA (Color Filter Array)
            compression=1,            # 无压缩
            bitspersample=16,         # tifffile 用 uint16 存储 12-bit 数据
            subfiletype=0,
            metadata=None,
            extratags=[
                # DNG 必要标签
                (50706, 1, 4, (1, 4, 0, 0)),      # DNGVersion 1.4 (BYTE)
                (50707, 1, 4, (1, 4, 0, 0)),      # DNGBackwardVersion (BYTE)
                (33421, 3, 2, (2, 2)),             # CFARepeatPatternDim (SHORT)
                (33422, 1, 4, (0, 1, 1, 2)),      # CFAPattern: RGGB (BYTE)
                (50714, 4, 1, 0),                  # BlackLevel (LONG)
                (50717, 4, 1, 4095),               # WhiteLevel (LONG)
                (50721, 12, 9, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),  # ColorMatrix1 (DOUBLE)
                (50778, 3, 1, 21),                 # CalibrationIlluminant1 = D65 (SHORT)
                (50728, 12, 3, (1.0, 1.0, 1.0)),  # AsShotNeutral (DOUBLE)
            ],
        )


def async_save_raw(path, bayer_data):
    """异步保存 RAW 二进制文件。"""
    def _save(p, d):
        d.tofile(p)
    _pending.append(_save_pool.submit(_save, path, bayer_data.copy()))


def async_save_dng(path, bayer_data, width, height):
    """异步保存 DNG 文件。"""
    def _save(p, d, w, h):
        write_dng(p, d, w, h)
    _pending.append(_save_pool.submit(_save, path, bayer_data.copy(), width, height))


def async_save_png(path, rgb_array):
    """异步保存 RGB PNG 预览。"""
    def _save(p, arr):
        from PIL import Image
        Image.fromarray(arr).save(p)
    _pending.append(_save_pool.submit(_save, path, rgb_array.copy()))


def wait_all_saves():
    for f in _pending:
        f.result()
    _pending.clear()


# ============================================================
# 场景创建
# ============================================================
print("=" * 60)
print("  双目 12-bit 伪 RAW 采集")
print("=" * 60)
print(f"  分辨率:  {W}x{H}")
print(f"  基线:    {BASELINE * 1000:.0f} mm")
print(f"  相机高度: {CAM_H} m")
print(f"  帧数:    {args_cli.num_frames}")
print(f"  噪声:    {args_cli.noise_preset}")
print(f"  输出:    {OUTPUT_DIR}")

sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 30.0)
sim = SimulationContext(sim_cfg)

# 地面
ground_cfg = sim_utils.GroundPlaneCfg()
ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

# 光源
light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
light_cfg.func("/World/DomeLight", light_cfg)

# 场景物体（彩色立方体）
cube_data = [
    ((1.0, 0.0, 0.0), (1.2, 0.0, 0.25)),
    ((0.0, 1.0, 0.0), (-1.2, 0.0, 0.25)),
    ((0.0, 0.0, 1.0), (0.0, 1.2, 0.25)),
    ((1.0, 1.0, 0.0), (0.0, -1.2, 0.25)),
    ((0.8, 0.2, 0.8), (0.6, 0.6, 0.25)),
    ((0.2, 0.8, 0.8), (-0.6, 0.6, 0.25)),
]

for i, (color, pos) in enumerate(cube_data):
    cfg = sim_utils.CuboidCfg(
        size=(0.4, 0.4, 0.4),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    cfg.func(f"/World/Obj_{i}", cfg, translation=pos)

# ============================================================
# 双目相机 (USD prim + replicator annotator)
# ============================================================
stage = omni.usd.get_context().get_stage()
half_bl = BASELINE / 2.0

left_cam_path = "/World/StereoRig/left_camera"
right_cam_path = "/World/StereoRig/right_camera"

# 创建父 Xform
from pxr import UsdGeom as _UsdGeom
_UsdGeom.Xform.Define(stage, "/World/StereoRig")

for cam_path, y_offset in [(left_cam_path, -half_bl), (right_cam_path, half_bl)]:
    cam_prim = UsdGeom.Camera.Define(stage, cam_path)
    cam_prim.GetFocalLengthAttr().Set(24.0)
    cam_prim.GetHorizontalApertureAttr().Set(20.955)
    cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

    xf = UsdGeom.Xformable(cam_prim.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, float(y_offset), float(CAM_H)))
    # USD Camera 默认光轴 -Z，即已经朝下，无需旋转

# ============================================================
# 仿真初始化
# ============================================================
sim.reset()

# Render products + annotators
rp_left = rep.create.render_product(left_cam_path, resolution=(W, H))
rp_right = rep.create.render_product(right_cam_path, resolution=(W, H))

annot_left = rep.AnnotatorRegistry.get_annotator("rgb")
annot_left.attach([rp_left])

annot_right = rep.AnnotatorRegistry.get_annotator("rgb")
annot_right.attach([rp_right])

# 预热
print(f"\n[1/3] 渲染预热 ({args_cli.warmup_frames} 帧)...")
for i in range(args_cli.warmup_frames):
    sim.step()
    if (i + 1) % 10 == 0:
        print(f"  warmup {i+1}/{args_cli.warmup_frames}")

# ============================================================
# 采集循环
# ============================================================
print(f"[2/3] 采集 {args_cli.num_frames} 帧双目 RAW12...")
start_time = time.time()
captured = 0
skipped = 0
noise_preset = args_cli.noise_preset if args_cli.noise_preset != "clean" else None

for frame_id in range(args_cli.num_frames * 2):  # 允许跳帧
    sim.step()

    data_l = annot_left.get_data()
    data_r = annot_right.get_data()

    if data_l is None or data_r is None:
        if frame_id % 5 == 0:
            print(f"  frame {frame_id}: annotator returned None (L={data_l is not None}, R={data_r is not None})")
        skipped += 1
        continue

    arr_l = np.asarray(data_l)
    arr_r = np.asarray(data_r)

    if frame_id < 3 or (frame_id % 10 == 0 and captured == 0):
        print(f"  frame {frame_id}: L shape={arr_l.shape} dtype={arr_l.dtype}, R shape={arr_r.shape} dtype={arr_r.dtype}")

    if arr_l.size == 0 or arr_r.size == 0 or arr_l.ndim < 2 or arr_r.ndim < 2:
        if frame_id % 5 == 0:
            print(f"  frame {frame_id}: bad shape L={arr_l.shape} R={arr_r.shape}")
        skipped += 1
        continue

    # RGBA → RGB
    rgb_l = arr_l[:, :, :3] if arr_l.ndim == 3 and arr_l.shape[2] >= 3 else arr_l
    rgb_r = arr_r[:, :, :3] if arr_r.ndim == 3 and arr_r.shape[2] >= 3 else arr_r

    # 跳过黑帧
    if rgb_l.mean() < 1.0 or rgb_r.mean() < 1.0:
        if frame_id % 5 == 0:
            print(f"  frame {frame_id}: dark frame L_mean={rgb_l.mean():.2f} R_mean={rgb_r.mean():.2f}")
        skipped += 1
        continue

    # RGB → 12-bit Bayer RAW
    raw_l = rgb_to_raw12(rgb_l, noise_preset=noise_preset)
    raw_r = rgb_to_raw12(rgb_r, noise_preset=noise_preset)

    # 异步保存 RAW (.bin) + DNG
    async_save_raw(os.path.join(LEFT_DIR, f"frame_{captured:06d}.bin"), raw_l)
    async_save_raw(os.path.join(RIGHT_DIR, f"frame_{captured:06d}.bin"), raw_r)
    async_save_dng(os.path.join(LEFT_DIR, f"frame_{captured:06d}.dng"), raw_l, W, H)
    async_save_dng(os.path.join(RIGHT_DIR, f"frame_{captured:06d}.dng"), raw_r, W, H)

    # RGB 预览
    if args_cli.save_rgb_every > 0 and captured % args_cli.save_rgb_every == 0:
        async_save_png(os.path.join(RGB_DIR, f"frame_{captured:06d}_left.png"), rgb_l)
        async_save_png(os.path.join(RGB_DIR, f"frame_{captured:06d}_right.png"), rgb_r)

    captured += 1
    if captured % 10 == 0 or captured == args_cli.num_frames:
        elapsed = time.time() - start_time
        fps = captured / elapsed if elapsed > 0 else 0
        print(f"  已采集 {captured}/{args_cli.num_frames} 帧 ({fps:.1f} fps, 跳过 {skipped})")

    if captured >= args_cli.num_frames:
        break

# 等待异步保存完成
print("[3/3] 等待异步保存完成...")
wait_all_saves()

# ============================================================
# 生成 manifest
# ============================================================
elapsed_total = time.time() - start_time

# 获取最后一帧的统计（仅在有采集数据时）
stats = {}
if captured > 0:
    # 读取最后一帧 bin 文件来获取统计
    last_left = os.path.join(LEFT_DIR, f"frame_{captured-1:06d}.bin")
    last_right = os.path.join(RIGHT_DIR, f"frame_{captured-1:06d}.bin")
    if os.path.exists(last_left):
        raw_last = np.fromfile(last_left, dtype=np.uint16)
        stats["left_mean"] = float(raw_last.mean())
        stats["left_max"] = int(raw_last.max())
    if os.path.exists(last_right):
        raw_last = np.fromfile(last_right, dtype=np.uint16)
        stats["right_mean"] = float(raw_last.mean())
        stats["right_max"] = int(raw_last.max())

manifest = {
    "project": "Isaac Lab Stereo Pseudo-RAW12",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "num_frames": captured,
    "image_width": W,
    "image_height": H,
    "baseline_mm": BASELINE * 1000,
    "camera_height_m": CAM_H,
    "bit_depth": 12,
    "bayer_pattern": "RGGB",
    "black_level": 0,
    "white_level": 4095,
    "noise_preset": args_cli.noise_preset,
    "file_format": "uint16 raw binary (.bin)",
    "bytes_per_frame": W * H * 2,
    "pipeline": "RGB(uint8) -> sRGB_decode -> linear(float32) -> RGGB_CFA -> 12bit_quantize -> noise -> uint16",
    "stats": stats,
    "elapsed_seconds": round(elapsed_total, 1),
}

manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

# ============================================================
# 总结
# ============================================================
total_bytes = captured * W * H * 2 * 2  # 左右各一
print(f"\n{'=' * 60}")
print(f"  采集完成！")
print(f"{'=' * 60}")
print(f"  帧数:       {captured}")
print(f"  分辨率:     {W}x{H}")
print(f"  基线:       {BASELINE * 1000:.0f} mm")
print(f"  位深:       12-bit (RGGB)")
print(f"  噪声:       {args_cli.noise_preset}")
print(f"  跳过帧:     {skipped}")
print(f"  耗时:       {elapsed_total:.1f} 秒")
print(f"  总数据量:   {total_bytes / 1024 / 1024:.1f} MB")
print(f"  左眼:       {LEFT_DIR}")
print(f"  右眼:       {RIGHT_DIR}")
print(f"  Manifest:   {manifest_path}")
print(f"{'=' * 60}")

simulation_app.close()
