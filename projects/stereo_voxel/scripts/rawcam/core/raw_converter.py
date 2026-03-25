"""RGB → 12-bit Bayer RAW 转换引擎

管线: sRGB uint8 → linear float → Bayer CFA mosaic → 量化 → 噪声 → uint16
"""

import numpy as np
from typing import Optional

from ..configs.dataclasses import SensorConfig, NoiseConfig
from .bayer_mosaic import BayerPattern, rgb_to_bayer
from .noise_model import NoiseModel, NOISE_PRESETS


class RawConverter:
    """RGB 到 Bayer RAW 的转换器"""

    def __init__(
        self,
        sensor_cfg: SensorConfig,
        noise_cfg: Optional[NoiseConfig] = None,
    ):
        self._sensor_cfg = sensor_cfg
        self._bit_depth = sensor_cfg.bit_depth
        self._max_val = sensor_cfg.white_level
        self._black_level = sensor_cfg.black_level
        self._pattern = BayerPattern(sensor_cfg.bayer_pattern)

        # 噪声模型
        self._noise_model: Optional[NoiseModel] = None
        if noise_cfg and noise_cfg.enabled:
            if noise_cfg.preset == "custom":
                from .noise_model import SensorNoiseParams
                params = SensorNoiseParams(
                    read_noise_std=noise_cfg.read_noise_std,
                    shot_noise_gain=noise_cfg.shot_noise_gain,
                    dark_current_mean=noise_cfg.dark_current_mean,
                    dark_current_std=noise_cfg.dark_current_std,
                    prnu_std=noise_cfg.prnu_std,
                    row_noise_std=noise_cfg.row_noise_std,
                    seed=noise_cfg.seed,
                )
                self._noise_model = NoiseModel(params=params)
            else:
                self._noise_model = NoiseModel(preset=noise_cfg.preset)

    @property
    def pattern(self) -> BayerPattern:
        return self._pattern

    @property
    def noise_model(self) -> Optional[NoiseModel]:
        return self._noise_model

    def rgb_to_raw(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """RGB uint8 (H,W,3) → Bayer uint16 (H,W)。

        完整管线: sRGB 解码 → linear → CFA mosaic → 量化 → 噪声

        Args:
            rgb_uint8: (H, W, 3) uint8 sRGB 图像

        Returns:
            (H, W) uint16 Bayer RAW 数据
        """
        linear = self.srgb_to_linear(rgb_uint8)
        bayer = self.linear_to_bayer(linear)
        if self._noise_model:
            bayer = self._noise_model.apply(bayer, white_level=self._max_val)
        return bayer

    def srgb_to_linear(self, img_uint8: np.ndarray) -> np.ndarray:
        """sRGB uint8 [0,255] → linear float32 [0,1]

        Args:
            img_uint8: (H, W, 3) 或 (H, W) uint8

        Returns:
            同 shape 的 float32 [0,1]
        """
        srgb = img_uint8.astype(np.float32) / 255.0
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            ((srgb + 0.055) / 1.055) ** 2.4,
        )
        return linear.astype(np.float32)

    def linear_to_bayer(self, linear_rgb: np.ndarray) -> np.ndarray:
        """linear RGB float (H,W,3) → Bayer uint16 (H,W)。

        先做 CFA 采样，再量化到 bit_depth。

        Args:
            linear_rgb: (H, W, 3) float32 [0,1]

        Returns:
            (H, W) uint16，值域 [black_level, white_level]
        """
        # CFA 采样 → 单通道 float [0,1]
        cfa_float = rgb_to_bayer(linear_rgb, pattern=self._pattern, output_dtype=np.float32)

        # 量化到 [black_level, white_level]
        dynamic_range = self._max_val - self._black_level
        bayer = np.clip(
            cfa_float * dynamic_range + self._black_level,
            self._black_level,
            self._max_val,
        ).astype(np.uint16)

        return bayer

    def add_noise(self, bayer: np.ndarray) -> np.ndarray:
        """为 Bayer 数据添加噪声。

        Args:
            bayer: (H, W) uint16

        Returns:
            (H, W) uint16
        """
        if self._noise_model:
            return self._noise_model.apply(bayer, white_level=self._max_val)
        return bayer

    @staticmethod
    def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
        """linear float [0,1] → sRGB uint8 [0,255]（验证用）

        Args:
            linear: float 数组 [0,1]

        Returns:
            uint8 数组 [0,255]
        """
        srgb = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(np.clip(linear, 0.0031308, 1.0), 1.0 / 2.4) - 0.055,
        )
        return (np.clip(srgb, 0, 1) * 255).astype(np.uint8)
