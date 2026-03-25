"""测试脚本: RGB → 伪 12-bit Bayer RAW 转换验证
==================================================
从 Isaac Sim 获取标准 RGB (uint8)，在 Python 层转换为伪 12-bit Bayer RAW。

管线: RGB uint8 → linear float → Bayer CFA mosaic → 12-bit quantize → uint16

运行：
    isaaclab.bat -p projects/stereo_voxel/scripts/rawcam/scripts/test_pseudo_raw.py --enable_cameras
"""

# ============================================================
# Isaac Lab 启动器（必须最先执行）
# ============================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RGB to pseudo 12-bit RAW test")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================
# 主逻辑（AppLauncher 之后导入）
# ============================================================
import os
import sys
import json
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
import omni.replicator.core as rep
from pxr import UsdGeom, UsdLux, Gf, Sdf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "output", "test_pseudo_raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

W, H = args_cli.width, args_cli.height

print("=" * 60)
print("  RGB → 伪 12-bit RAW 转换测试")
print("=" * 60)
print(f"  分辨率: {W}x{H}")
print(f"  输出:   {OUTPUT_DIR}")

# ============================================================
# 1. 创建仿真上下文和场景
# ============================================================
sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 30.0)
sim = SimulationContext(sim_cfg)

# 地面
ground_cfg = sim_utils.GroundPlaneCfg()
ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

# 光源
light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0))
light_cfg.func("/World/DomeLight", light_cfg)

# 彩色物体
for i, (color, pos) in enumerate([
    ((1.0, 0.0, 0.0), (1.5, 0.0, 0.5)),
    ((0.0, 1.0, 0.0), (-1.5, 0.0, 0.5)),
    ((0.0, 0.0, 1.0), (0.0, 1.5, 0.5)),
    ((1.0, 1.0, 0.0), (0.0, -1.5, 0.5)),
]):
    cfg = sim_utils.CuboidCfg(
        size=(0.5, 0.5, 0.5),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    cfg.func(f"/World/Obj_{i}", cfg, translation=pos)

# ============================================================
# 2. 相机 (USD prim + replicator annotator)
# ============================================================
import omni.usd
stage = omni.usd.get_context().get_stage()
cam_path = "/World/TestCamera"
cam_prim = UsdGeom.Camera.Define(stage, cam_path)
cam_prim.GetFocalLengthAttr().Set(18.0)
cam_prim.GetHorizontalApertureAttr().Set(20.955)
cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

# 位置: 高处朝下
xf = UsdGeom.Xformable(cam_prim.GetPrim())
xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 4.0))
# 朝下看: 相机默认光轴 -Z, 让 -Z 指向世界 -Z 即正面朝下
# 绕X轴转0度（已经朝下），但需要一定角度才能看到物体
# 从高处俯视: 不旋转即可（USD Camera -Z 就是朝下）
# 不设置 orient 则默认朝下

# ============================================================
# 3. 初始化仿真 + 渲染
# ============================================================
sim.reset()

# 创建 render product 和 annotator
rp = rep.create.render_product(cam_path, resolution=(W, H))
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach([rp])

print("\n[Step 1] 渲染预热 (40帧)...")
for i in range(40):
    sim.step()
    if (i + 1) % 10 == 0:
        print(f"  warmup frame {i+1}/40")

# ============================================================
# 4. 采集 RGB
# ============================================================
print("[Step 2] 采集 RGB...")
rgb_data = None
for attempt in range(30):
    sim.step()
    data = annot.get_data()
    if data is not None:
        arr = np.asarray(data)
        if arr.size > 0 and arr.ndim >= 2:
            if arr.mean() > 1.0:
                rgb_data = arr
                print(f"  采集成功 (attempt {attempt+1}), shape={arr.shape}, dtype={arr.dtype}, mean={arr.mean():.1f}")
                break
            elif attempt % 5 == 4:
                print(f"  attempt {attempt+1}: got data but mean={arr.mean():.2f} (dark frame)")
        elif attempt % 5 == 4:
            print(f"  attempt {attempt+1}: shape={arr.shape}, size={arr.size}")
    elif attempt % 5 == 4:
        print(f"  attempt {attempt+1}: annotator returned None")

if rgb_data is None or rgb_data.size == 0:
    print("[ERROR] 无法获取 RGB 数据！")
    simulation_app.close()
    sys.exit(1)

# annotator 返回 RGBA，取 RGB
if rgb_data.ndim == 3 and rgb_data.shape[2] == 4:
    rgb = rgb_data[:, :, :3].copy()
else:
    rgb = rgb_data.copy()

print(f"  RGB: shape={rgb.shape}, dtype={rgb.dtype}, range=[{rgb.min()}, {rgb.max()}], mean={rgb.mean():.1f}")

# ============================================================
# 5. RGB → 伪 12-bit Bayer RAW
# ============================================================
print("\n[Step 3] RGB → 伪 12-bit Bayer RAW 转换...")


def srgb_to_linear(img_uint8):
    """sRGB uint8 → linear float [0,1]"""
    srgb = img_uint8.astype(np.float64) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return linear


def linear_to_bayer_rggb(linear_rgb, bit_depth=12):
    """linear RGB → RGGB Bayer CFA → 量化到指定位深。"""
    H, W, _ = linear_rgb.shape
    max_val = (1 << bit_depth) - 1  # 4095

    r = np.clip(linear_rgb[:, :, 0] * max_val, 0, max_val).astype(np.uint16)
    g = np.clip(linear_rgb[:, :, 1] * max_val, 0, max_val).astype(np.uint16)
    b = np.clip(linear_rgb[:, :, 2] * max_val, 0, max_val).astype(np.uint16)

    bayer = np.zeros((H, W), dtype=np.uint16)
    bayer[0::2, 0::2] = r[0::2, 0::2]   # R
    bayer[0::2, 1::2] = g[0::2, 1::2]   # Gr
    bayer[1::2, 0::2] = g[1::2, 0::2]   # Gb
    bayer[1::2, 1::2] = b[1::2, 1::2]   # B

    return bayer


def add_sensor_noise(bayer, read_noise_sigma=2.0, shot_noise_gain=0.5, seed=42):
    """模拟传感器噪声（读出噪声 + 散粒噪声）。"""
    rng = np.random.default_rng(seed)
    data = bayer.astype(np.float32)

    shot_sigma = np.sqrt(np.maximum(data, 0)) * shot_noise_gain
    shot = rng.normal(0, 1, data.shape).astype(np.float32) * shot_sigma

    read = rng.normal(0, read_noise_sigma, data.shape).astype(np.float32)

    noisy = data + shot + read
    return np.clip(noisy, 0, 4095).astype(np.uint16)


# 执行转换
linear = srgb_to_linear(rgb)
print(f"  Linear: range=[{linear.min():.4f}, {linear.max():.4f}], mean={linear.mean():.4f}")

bayer_clean = linear_to_bayer_rggb(linear, bit_depth=12)
print(f"  Bayer (clean): range=[{bayer_clean.min()}, {bayer_clean.max()}], mean={bayer_clean.mean():.1f}")

bayer_noisy = add_sensor_noise(bayer_clean, read_noise_sigma=3.0, shot_noise_gain=0.3)
print(f"  Bayer (noisy): range=[{bayer_noisy.min()}, {bayer_noisy.max()}], mean={bayer_noisy.mean():.1f}")

# Bayer 子通道统计
for name, sl in [("R", (slice(0, None, 2), slice(0, None, 2))),
                 ("Gr", (slice(0, None, 2), slice(1, None, 2))),
                 ("Gb", (slice(1, None, 2), slice(0, None, 2))),
                 ("B", (slice(1, None, 2), slice(1, None, 2)))]:
    ch = bayer_noisy[sl]
    print(f"    {name:2s}: mean={ch.mean():.1f}, std={ch.std():.1f}, "
          f"min={ch.min()}, max={ch.max()}")

# ============================================================
# 6. 保存
# ============================================================
print("\n[Step 4] 保存...")

np.save(os.path.join(OUTPUT_DIR, "rgb_uint8.npy"), rgb)
print(f"  rgb_uint8.npy ({rgb.nbytes:,} bytes)")

np.save(os.path.join(OUTPUT_DIR, "linear_float32.npy"), linear.astype(np.float32))
print(f"  linear_float32.npy")

np.save(os.path.join(OUTPUT_DIR, "bayer_12bit_clean.npy"), bayer_clean)
print(f"  bayer_12bit_clean.npy ({bayer_clean.nbytes:,} bytes)")

np.save(os.path.join(OUTPUT_DIR, "bayer_12bit_noisy.npy"), bayer_noisy)
print(f"  bayer_12bit_noisy.npy ({bayer_noisy.nbytes:,} bytes)")

bayer_noisy.tofile(os.path.join(OUTPUT_DIR, "raw12.bin"))
print(f"  raw12.bin ({bayer_noisy.nbytes:,} bytes, uint16 packed)")

meta = {
    "width": W, "height": H,
    "bit_depth": 12,
    "bayer_pattern": "RGGB",
    "black_level": 0,
    "white_level": 4095,
    "noise_model": {"read_sigma": 3.0, "shot_gain": 0.3},
    "source": "isaac_sim_rgb_to_pseudo_raw",
    "rgb_mean": float(rgb.mean()),
    "bayer_mean": float(bayer_noisy.mean()),
}
with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"  meta.json")

# ============================================================
# 7. 验证: 简单 demosaic 还原
# ============================================================
print("\n[Step 5] 验证: 简单 demosaic...")


def simple_demosaic_rggb(bayer, bit_depth=12):
    """最简单的 demosaic：2x2 块取 RGGB 平均 → (H/2, W/2, 3)"""
    r = bayer[0::2, 0::2].astype(np.float32)
    gr = bayer[0::2, 1::2].astype(np.float32)
    gb = bayer[1::2, 0::2].astype(np.float32)
    b = bayer[1::2, 1::2].astype(np.float32)
    g = (gr + gb) / 2.0
    max_val = (1 << bit_depth) - 1
    rgb_out = np.stack([r, g, b], axis=-1) / max_val
    return (np.clip(rgb_out, 0, 1) * 255).astype(np.uint8)


demosaiced = simple_demosaic_rggb(bayer_noisy)
print(f"  Demosaiced: shape={demosaiced.shape}, mean={demosaiced.mean():.1f}")

orig_ds = rgb[0::2, 0::2, :]
psnr_approx = 10 * np.log10(255**2 / max(np.mean((demosaiced.astype(float) - orig_ds.astype(float))**2), 1e-10))
print(f"  PSNR (vs 原始下采样): {psnr_approx:.1f} dB")

np.save(os.path.join(OUTPUT_DIR, "demosaiced_uint8.npy"), demosaiced)

# ============================================================
# 8. 总结
# ============================================================
print("\n" + "=" * 60)
print("  测试结果")
print("=" * 60)
files = os.listdir(OUTPUT_DIR)
total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in files)
print(f"  输出文件: {len(files)} 个, 总大小: {total_size:,} bytes")
for f in sorted(files):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"    {f:<30s} {size:>10,} bytes")

print(f"\n  转换管线: RGB(uint8) → sRGB解码 → linear(float) → RGGB CFA → 12-bit → noise → uint16")
print(f"  有效位数: 12 bit (max={bayer_noisy.max()}, 理论max=4095)")
print(f"  后续: raw12.bin 可直接送入网络，或用 dng_writer.py 封装为 DNG")

print("\n[Done]")
simulation_app.close()
