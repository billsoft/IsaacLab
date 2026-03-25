"""RAW 数据验证工具

demosaic 还原 + PSNR 对比 + DNG 完整性检查（纯 tifffile，不依赖 exiftool）。
"""

import numpy as np
from typing import Optional

from .bayer_mosaic import BayerPattern, simple_demosaic_half
from .raw_converter import RawConverter


def verify_raw_roundtrip(
    bayer_uint16: np.ndarray,
    rgb_reference: np.ndarray,
    bit_depth: int = 12,
    bayer_pattern: str = "RGGB",
) -> dict:
    """验证 RAW 数据质量：简单 demosaic → 与原始 RGB 对比。

    Args:
        bayer_uint16: (H, W) uint16 Bayer 数据
        rgb_reference: (H, W, 3) uint8 原始 RGB
        bit_depth: 位深
        bayer_pattern: CFA 排列

    Returns:
        {"psnr": float, "mse": float, "bayer_stats": dict}
    """
    pattern = BayerPattern(bayer_pattern)

    # 半分辨率 demosaic
    demosaiced = simple_demosaic_half(bayer_uint16, pattern=pattern)
    srgb = RawConverter.linear_to_srgb(demosaiced)

    # 下采样参考 RGB 到半分辨率
    ref_ds = rgb_reference[0::2, 0::2, :3]

    # 确保尺寸匹配
    min_h = min(srgb.shape[0], ref_ds.shape[0])
    min_w = min(srgb.shape[1], ref_ds.shape[1])
    srgb = srgb[:min_h, :min_w]
    ref_ds = ref_ds[:min_h, :min_w]

    # MSE 和 PSNR
    mse = float(np.mean((srgb.astype(float) - ref_ds.astype(float)) ** 2))
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-10))

    # Bayer 子通道统计
    bayer_stats = {}
    for name, sl in [
        ("R", (slice(0, None, 2), slice(0, None, 2))),
        ("Gr", (slice(0, None, 2), slice(1, None, 2))),
        ("Gb", (slice(1, None, 2), slice(0, None, 2))),
        ("B", (slice(1, None, 2), slice(1, None, 2))),
    ]:
        ch = bayer_uint16[sl].astype(float)
        bayer_stats[name] = {
            "mean": float(ch.mean()),
            "std": float(ch.std()),
            "min": int(ch.min()),
            "max": int(ch.max()),
        }

    return {
        "psnr": float(psnr),
        "mse": mse,
        "demosaic_shape": srgb.shape,
        "demosaic_mean": float(srgb.mean()),
        "bayer_stats": bayer_stats,
    }


def verify_dng(dng_path: str) -> dict:
    """读取 DNG 文件并验证完整性。

    Args:
        dng_path: DNG 文件路径

    Returns:
        {"shape": tuple, "dtype": str, "range": [min, max], "dng_tags": dict}
    """
    from tifffile import TiffFile

    with TiffFile(dng_path) as t:
        page = t.pages[0]
        data = page.asarray()

        dng_tags = {}
        for tag in page.tags.values():
            if tag.code > 33000:
                dng_tags[tag.code] = {"name": tag.name, "value": tag.value}

    # 检查必要 DNG 标签
    required_tags = {
        50706: "DNGVersion",
        33421: "CFARepeatPatternDim",
        33422: "CFAPattern",
        50714: "BlackLevel",
        50717: "WhiteLevel",
    }
    missing = []
    for code, name in required_tags.items():
        if code not in dng_tags:
            missing.append(f"{code} ({name})")

    return {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "range": [int(data.min()), int(data.max())],
        "mean": float(data.mean()),
        "dng_tags": dng_tags,
        "missing_tags": missing,
        "valid": len(missing) == 0,
    }
