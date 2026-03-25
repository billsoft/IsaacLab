"""验证脚本: 12-bit Bayer RAW → RGB 还原验证
=============================================
读取 capture_stereo_raw12.py 生成的 .bin 文件，
demosaic 还原为 RGB 并与 RGB 预览对比，验证转换正确性。

运行（不需要 Isaac Sim，普通 Python 即可）：
    python scripts/verify_raw12.py --input_dir output/stereo_raw12
"""

import argparse
import os
import sys
import json
import glob
import numpy as np


def demosaic_rggb_bilinear(bayer, bit_depth=12):
    """双线性插值 demosaic (RGGB)，输出全分辨率 RGB。"""
    H, W = bayer.shape
    max_val = float((1 << bit_depth) - 1)
    data = bayer.astype(np.float32) / max_val  # → [0, 1]

    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # Red channel
    r = np.zeros((H, W), dtype=np.float32)
    r[0::2, 0::2] = data[0::2, 0::2]
    # 水平插值 (偶数行,奇数列)
    r[0::2, 1::2] = (data[0::2, 0::2][:, :-1] + data[0::2, 0::2][:, 1:]) / 2.0 if W > 2 else data[0::2, 0::2]
    # 如果宽度不够，复制边缘
    if W > 2:
        r[0::2, 1::2] = np.pad(
            (data[0::2, 0::2][:, :-1] + data[0::2, 0::2][:, 1:]) / 2.0,
            ((0, 0), (0, 1 if data[0::2, 0::2].shape[1] > r[0::2, 1::2].shape[1] else 0)),
            mode='edge'
        )[:, :r[0::2, 1::2].shape[1]]
    # 垂直插值 (奇数行)
    r[1::2, :] = (r[0::2, :][:r[1::2, :].shape[0], :] +
                   np.roll(r[0::2, :], -1, axis=0)[:r[1::2, :].shape[0], :]) / 2.0

    # Green channel (两个绿色位置取平均更简单的方式)
    g = np.zeros((H, W), dtype=np.float32)
    g[0::2, 1::2] = data[0::2, 1::2]  # Gr
    g[1::2, 0::2] = data[1::2, 0::2]  # Gb
    # 红蓝位置：取四邻域绿色平均
    for y in range(0, H, 2):
        for x in range(0, W, 2):
            vals = []
            if x > 0: vals.append(data[y, x - 1] if y % 2 == 0 else 0)
            if x < W - 1: vals.append(data[y, x + 1])
            if y > 0: vals.append(data[y - 1, x])
            if y < H - 1: vals.append(data[y + 1, x])
            if vals:
                g[y, x] = np.mean(vals)
    for y in range(1, H, 2):
        for x in range(1, W, 2):
            vals = []
            if x > 0: vals.append(data[y, x - 1])
            if x < W - 1: vals.append(data[y, x + 1])
            if y > 0: vals.append(data[y - 1, x])
            if y < H - 1: vals.append(data[y + 1, x])
            if vals:
                g[y, x] = np.mean(vals)

    # Blue channel
    b = np.zeros((H, W), dtype=np.float32)
    b[1::2, 1::2] = data[1::2, 1::2]
    b[1::2, 0::2] = (data[1::2, 1::2][:, :b[1::2, 0::2].shape[1]] +
                      np.roll(data[1::2, 1::2], 1, axis=1)[:, :b[1::2, 0::2].shape[1]]) / 2.0
    b[0::2, :] = (b[1::2, :][:b[0::2, :].shape[0], :] +
                   np.roll(b[1::2, :], 1, axis=0)[:b[0::2, :].shape[0], :]) / 2.0

    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb


def simple_demosaic_rggb(bayer, bit_depth=12):
    """简单 2x2 块 demosaic，半分辨率但快速准确。"""
    r = bayer[0::2, 0::2].astype(np.float32)
    gr = bayer[0::2, 1::2].astype(np.float32)
    gb = bayer[1::2, 0::2].astype(np.float32)
    b = bayer[1::2, 1::2].astype(np.float32)
    g = (gr + gb) / 2.0
    max_val = (1 << bit_depth) - 1
    rgb = np.stack([r, g, b], axis=-1) / max_val
    return rgb


def linear_to_srgb(linear):
    """linear float [0,1] → sRGB uint8"""
    srgb = np.where(linear <= 0.0031308,
                    linear * 12.92,
                    1.055 * np.power(np.clip(linear, 0.0031308, 1.0), 1.0 / 2.4) - 0.055)
    return (np.clip(srgb, 0, 1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="验证 12-bit Bayer RAW 数据")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="输入目录 (含 left/, right/, manifest.json)")
    parser.add_argument("--num_verify", type=int, default=3, help="验证帧数")
    parser.add_argument("--save_demosaic", action="store_true", default=True,
                        help="保存 demosaic 结果为 PNG")
    args = parser.parse_args()

    if args.input_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.input_dir = os.path.join(os.path.dirname(script_dir), "output", "stereo_raw12")

    manifest_path = os.path.join(args.input_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[ERROR] 未找到 manifest.json: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    W = manifest["image_width"]
    H = manifest["image_height"]
    bit_depth = manifest.get("bit_depth", 12)

    print("=" * 60)
    print("  12-bit Bayer RAW 验证")
    print("=" * 60)
    print(f"  输入:    {args.input_dir}")
    print(f"  分辨率:  {W}x{H}")
    print(f"  位深:    {bit_depth}")
    print(f"  CFA:     {manifest.get('bayer_pattern', 'RGGB')}")
    print(f"  噪声:    {manifest.get('noise_preset', 'unknown')}")

    verify_dir = os.path.join(args.input_dir, "verify")
    os.makedirs(verify_dir, exist_ok=True)

    for side in ["left", "right"]:
        side_dir = os.path.join(args.input_dir, side)
        bin_files = sorted(glob.glob(os.path.join(side_dir, "*.bin")))
        if not bin_files:
            print(f"  [{side}] 无 .bin 文件")
            continue

        print(f"\n  [{side}] 共 {len(bin_files)} 帧")

        for i, bin_path in enumerate(bin_files[:args.num_verify]):
            fname = os.path.basename(bin_path)
            frame_id = fname.replace(".bin", "")

            # 读取 RAW
            raw = np.fromfile(bin_path, dtype=np.uint16).reshape(H, W)
            print(f"\n    {fname}:")
            print(f"      shape={raw.shape}, dtype={raw.dtype}")
            print(f"      range=[{raw.min()}, {raw.max()}], mean={raw.mean():.1f}")

            # Bayer 子通道统计
            for name, sl in [("R ", (slice(0, None, 2), slice(0, None, 2))),
                             ("Gr", (slice(0, None, 2), slice(1, None, 2))),
                             ("Gb", (slice(1, None, 2), slice(0, None, 2))),
                             ("B ", (slice(1, None, 2), slice(1, None, 2)))]:
                ch = raw[sl]
                print(f"      {name}: mean={ch.mean():.1f}, std={ch.std():.1f}")

            # Demosaic → sRGB
            linear_rgb = simple_demosaic_rggb(raw, bit_depth=bit_depth)
            srgb = linear_to_srgb(linear_rgb)
            print(f"      demosaic: shape={srgb.shape}, mean={srgb.mean():.1f}")

            # 与 RGB 预览对比
            rgb_preview_path = os.path.join(
                args.input_dir, "rgb_preview",
                f"{frame_id}_{side}.png"
            )
            if os.path.exists(rgb_preview_path):
                from PIL import Image
                rgb_ref = np.array(Image.open(rgb_preview_path))
                # 下采样到半分辨率对比
                rgb_ref_ds = rgb_ref[0::2, 0::2, :3]
                mse = np.mean((srgb.astype(float) - rgb_ref_ds.astype(float)) ** 2)
                psnr = 10 * np.log10(255**2 / max(mse, 1e-10))
                print(f"      vs RGB预览 PSNR: {psnr:.1f} dB")
            else:
                print(f"      (无 RGB 预览可对比)")

            # 保存 demosaic 结果
            if args.save_demosaic:
                from PIL import Image
                out_path = os.path.join(verify_dir, f"{frame_id}_{side}_demosaic.png")
                Image.fromarray(srgb).save(out_path)
                print(f"      saved: {out_path}")

    # 验证 12-bit 范围
    print(f"\n{'=' * 60}")
    print(f"  验证总结")
    print(f"{'=' * 60}")
    print(f"  输出目录:   {verify_dir}")
    print(f"  demosaic PNG 可直接打开查看，确认图像内容正确")
    print(f"  PSNR > 20 dB 说明 RAW→RGB 往返转换基本正确")
    print(f"  PSNR 在 10-20 dB 是因为 sRGB 非线性 + 噪声 + Bayer 采样损失")


if __name__ == "__main__":
    main()
