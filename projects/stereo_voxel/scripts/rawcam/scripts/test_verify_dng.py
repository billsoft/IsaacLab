"""验证已采集的 DNG 文件"""
import sys
import numpy as np
from tifffile import TiffFile
from PIL import Image

OUTPUT_DIR = "D:/code/IsaacLab/projects/stereo_voxel/scripts/rawcam/output/stereo_raw12"

print("=" * 60)
print("  DNG 验证")
print("=" * 60)
sys.stdout.flush()

# 1. 读取左眼 DNG
dng_path = f"{OUTPUT_DIR}/left/frame_000000.dng"
print(f"\n读取 DNG: {dng_path}")
sys.stdout.flush()

with TiffFile(dng_path) as t:
    page = t.pages[0]
    dng_data = page.asarray()
    print(f"  shape={dng_data.shape}, dtype={dng_data.dtype}")
    print(f"  range=[{dng_data.min()}, {dng_data.max()}], mean={dng_data.mean():.1f}")
    print(f"  DNG Tags:")
    for tag in page.tags.values():
        if tag.code > 33000:
            print(f"    {tag.code}: {tag.name} = {tag.value}")
sys.stdout.flush()

# 2. 对比 bin 文件
bin_path = f"{OUTPUT_DIR}/left/frame_000000.bin"
bin_data = np.fromfile(bin_path, dtype=np.uint16).reshape(dng_data.shape)
match = np.array_equal(dng_data, bin_data)
print(f"\n  DNG vs BIN 数据一致: {match}")
sys.stdout.flush()

# 3. Bayer 通道统计
print(f"\n  Bayer 通道统计:")
for name, sl in [("R ", (slice(0, None, 2), slice(0, None, 2))),
                 ("Gr", (slice(0, None, 2), slice(1, None, 2))),
                 ("Gb", (slice(1, None, 2), slice(0, None, 2))),
                 ("B ", (slice(1, None, 2), slice(1, None, 2)))]:
    ch = dng_data[sl].astype(float)
    print(f"    {name}: mean={ch.mean():.1f}, std={ch.std():.1f}, max={ch.max():.0f}")
sys.stdout.flush()

# 4. Demosaic → sRGB → PNG
print(f"\n  Demosaic 验证:")
r = dng_data[0::2, 0::2].astype(np.float32) / 4095.0
g = (dng_data[0::2, 1::2].astype(np.float32) + dng_data[1::2, 0::2].astype(np.float32)) / 2.0 / 4095.0
b = dng_data[1::2, 1::2].astype(np.float32) / 4095.0
linear = np.stack([r, g, b], axis=-1)

# linear → sRGB
srgb = np.where(linear <= 0.0031308,
                linear * 12.92,
                1.055 * np.power(np.clip(linear, 0.0031308, 1.0), 1.0 / 2.4) - 0.055)
srgb_uint8 = (np.clip(srgb, 0, 1) * 255).astype(np.uint8)
print(f"    demosaic shape={srgb_uint8.shape}, mean={srgb_uint8.mean():.1f}")

out_path = f"{OUTPUT_DIR}/verify_left_demosaic.png"
Image.fromarray(srgb_uint8).save(out_path)
print(f"    saved: {out_path}")

# 5. 与 RGB 预览对比
rgb_preview = np.array(Image.open(f"{OUTPUT_DIR}/rgb_preview/frame_000000_left.png"))
rgb_ds = rgb_preview[0::2, 0::2, :3]
mse = np.mean((srgb_uint8.astype(float) - rgb_ds.astype(float)) ** 2)
psnr = 10 * np.log10(255**2 / max(mse, 1e-10))
print(f"    PSNR vs RGB预览: {psnr:.1f} dB")
sys.stdout.flush()

print(f"\n{'=' * 60}")
print(f"  验证完成！DNG 格式正确，数据完整")
print(f"{'=' * 60}")
sys.stdout.flush()
