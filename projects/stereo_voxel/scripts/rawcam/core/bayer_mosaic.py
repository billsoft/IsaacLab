"""Bayer CFA 马赛克操作模块

将 RGB 图像转换为单通道 Bayer CFA 数据（模拟传感器的 CFA 采样），
以及简单的去马赛克（demosaic）和可视化。
"""

import numpy as np
from enum import Enum


class BayerPattern(Enum):
    """Bayer CFA 排列模式。

    2x2 重复单元:
        RGGB: R G / G B
        GRBG: G R / B G
        GBRG: G B / R G
        BGGR: B G / G R
    """
    RGGB = "RGGB"
    GRBG = "GRBG"
    GBRG = "GBRG"
    BGGR = "BGGR"


# channel: 0=R, 1=G, 2=B
_PATTERN_MAP = {
    BayerPattern.RGGB: np.array([[0, 1], [1, 2]]),
    BayerPattern.GRBG: np.array([[1, 0], [2, 1]]),
    BayerPattern.GBRG: np.array([[1, 2], [0, 1]]),
    BayerPattern.BGGR: np.array([[2, 1], [1, 0]]),
}


def rgb_to_bayer(
    rgb: np.ndarray,
    pattern: BayerPattern = BayerPattern.RGGB,
    output_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """RGB 图像 → 单通道 Bayer CFA 数据。

    Args:
        rgb: (H, W, 3)，uint8 [0,255] 或 float [0,1]
        pattern: Bayer 排列模式
        output_dtype: 输出数据类型

    Returns:
        (H, W) 单通道 Bayer 数据
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"输入必须是 (H, W, 3) 的 RGB 图像，当前 shape: {rgb.shape}")

    h, w = rgb.shape[:2]
    if h % 2 != 0 or w % 2 != 0:
        h = h - (h % 2)
        w = w - (w % 2)
        rgb = rgb[:h, :w, :]

    if rgb.dtype == np.uint8:
        rgb_float = rgb.astype(np.float32) / 255.0
    else:
        rgb_float = rgb.astype(np.float32)

    bayer = np.zeros((h, w), dtype=np.float32)
    channel_map = _PATTERN_MAP[pattern]

    for dy in range(2):
        for dx in range(2):
            ch = channel_map[dy, dx]
            bayer[dy::2, dx::2] = rgb_float[dy::2, dx::2, ch]

    return bayer.astype(output_dtype)


def bayer_to_rgb_nearest(
    bayer: np.ndarray,
    pattern: BayerPattern = BayerPattern.RGGB,
) -> np.ndarray:
    """最近邻去马赛克（仅用于快速预览）。

    Args:
        bayer: (H, W) 单通道 Bayer 数据
        pattern: Bayer 排列模式

    Returns:
        (H, W, 3) float32 RGB
    """
    h, w = bayer.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    channel_map = _PATTERN_MAP[pattern]

    for dy in range(2):
        for dx in range(2):
            ch = channel_map[dy, dx]
            rgb[dy::2, dx::2, ch] = bayer[dy::2, dx::2]

    for ch in range(3):
        layer = rgb[:, :, ch]
        for dy in range(2):
            for dx in range(2):
                if channel_map[dy, dx] != ch:
                    if dx == 0:
                        layer[dy::2, 0::2] = layer[dy::2, 1::2]
                    else:
                        layer[dy::2, 1::2] = layer[dy::2, 0::2]
        for row in range(h):
            for col in range(w):
                if layer[row, col] == 0:
                    if row > 0:
                        layer[row, col] = layer[row - 1, col]

    return rgb


def simple_demosaic_half(
    bayer: np.ndarray,
    pattern: BayerPattern = BayerPattern.RGGB,
) -> np.ndarray:
    """2x2 块取平均的半分辨率 demosaic，快速准确。

    Args:
        bayer: (H, W) Bayer 数据（float 或 uint16）
        pattern: Bayer 排列模式

    Returns:
        (H/2, W/2, 3) float32 RGB [0,1] 范围
    """
    channel_map = _PATTERN_MAP[pattern]

    # 找到各通道在 2x2 块中的位置
    r_pos = g_pos = b_pos = []
    for dy in range(2):
        for dx in range(2):
            ch = channel_map[dy, dx]
            if ch == 0:
                r_pos.append((dy, dx))
            elif ch == 1:
                g_pos.append((dy, dx))
            else:
                b_pos.append((dy, dx))

    data = bayer.astype(np.float32)
    max_val = float(data.max()) if data.max() > 1.0 else 1.0

    # 红色通道（通常 1 个位置）
    r = data[r_pos[0][0]::2, r_pos[0][1]::2]
    # 蓝色通道（通常 1 个位置）
    b = data[b_pos[0][0]::2, b_pos[0][1]::2]
    # 绿色通道（通常 2 个位置，取平均）
    g = (data[g_pos[0][0]::2, g_pos[0][1]::2] + data[g_pos[1][0]::2, g_pos[1][1]::2]) / 2.0

    rgb = np.stack([r, g, b], axis=-1) / max_val
    return rgb


def visualize_bayer_pattern(
    bayer: np.ndarray,
    pattern: BayerPattern = BayerPattern.RGGB,
    output_path: str = None,
) -> np.ndarray:
    """将 Bayer 数据可视化为彩色图像。

    Args:
        bayer: (H, W) 单通道 Bayer 数据
        pattern: Bayer 排列模式
        output_path: 可选，保存路径

    Returns:
        (H, W, 3) uint8 彩色可视化
    """
    h, w = bayer.shape
    channel_map = _PATTERN_MAP[pattern]

    if bayer.max() > 1.0:
        bayer_norm = (bayer / bayer.max() * 255).astype(np.uint8)
    else:
        bayer_norm = (bayer * 255).astype(np.uint8)

    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for dy in range(2):
        for dx in range(2):
            ch = channel_map[dy, dx]
            vis[dy::2, dx::2, ch] = bayer_norm[dy::2, dx::2]

    if output_path:
        from PIL import Image
        Image.fromarray(vis).save(output_path)

    return vis
