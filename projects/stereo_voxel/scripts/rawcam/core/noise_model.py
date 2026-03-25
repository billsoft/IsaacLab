"""传感器噪声模型

为仿真 RAW 数据添加真实传感器噪声：
1. PRNU（乘性，像素级固定增益差异）
2. 光子散粒噪声（泊松近似，信号相关）
3. 暗电流（加性，温度/曝光相关）
4. 读出噪声（加性高斯白噪声）
5. 行噪声（每行共同偏移）
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SensorNoiseParams:
    """传感器噪声参数（12-bit DN 单位）"""

    read_noise_std: float = 4.0
    dark_current_mean: float = 0.5
    dark_current_std: float = 0.3
    prnu_std: float = 0.01
    shot_noise_gain: float = 1.0
    row_noise_std: float = 0.5
    seed: Optional[int] = None


NOISE_PRESETS = {
    "clean": SensorNoiseParams(
        read_noise_std=0, dark_current_mean=0, dark_current_std=0,
        prnu_std=0, shot_noise_gain=0, row_noise_std=0,
    ),
    "light": SensorNoiseParams(
        read_noise_std=2.0, dark_current_mean=0.2, dark_current_std=0.1,
        prnu_std=0.005, shot_noise_gain=0.5, row_noise_std=0.2,
    ),
    "sc132gs": SensorNoiseParams(
        read_noise_std=4.0, dark_current_mean=0.5, dark_current_std=0.3,
        prnu_std=0.01, shot_noise_gain=1.0, row_noise_std=0.5,
    ),
    "heavy": SensorNoiseParams(
        read_noise_std=8.0, dark_current_mean=1.5, dark_current_std=0.8,
        prnu_std=0.02, shot_noise_gain=1.5, row_noise_std=1.0,
    ),
}


class NoiseModel:
    """传感器噪声模型（支持 PRNU 缓存）。

    PRNU map 在首次 apply() 时生成并缓存，同一传感器实例
    的多帧采集共享固定的像素级增益差异（符合真实物理）。
    """

    def __init__(
        self,
        params: Optional[SensorNoiseParams] = None,
        preset: str = "sc132gs",
    ):
        if params is not None:
            self._params = params
        elif preset in NOISE_PRESETS:
            self._params = NOISE_PRESETS[preset]
        else:
            raise ValueError(f"未知预设 '{preset}'，可选: {list(NOISE_PRESETS.keys())}")

        self._rng = np.random.default_rng(self._params.seed)
        self._prnu_map: Optional[np.ndarray] = None
        self._prnu_shape: Optional[tuple] = None

    @property
    def params(self) -> SensorNoiseParams:
        return self._params

    def _get_prnu_map(self, h: int, w: int) -> np.ndarray:
        """获取或创建 PRNU map（像素级固定增益偏差）"""
        if self._prnu_map is None or self._prnu_shape != (h, w):
            if self._params.prnu_std > 0:
                self._prnu_map = self._rng.normal(1.0, self._params.prnu_std, (h, w))
            else:
                self._prnu_map = np.ones((h, w), dtype=np.float64)
            self._prnu_shape = (h, w)
        return self._prnu_map

    def apply(self, raw_12bit: np.ndarray, white_level: int = 4095) -> np.ndarray:
        """为 12-bit RAW 数据添加传感器噪声。

        Args:
            raw_12bit: (H, W) uint16 或 float
            white_level: 白电平

        Returns:
            (H, W) uint16
        """
        p = self._params
        h, w = raw_12bit.shape
        noisy = raw_12bit.astype(np.float64)

        # 1. PRNU（乘性）
        if p.prnu_std > 0:
            noisy *= self._get_prnu_map(h, w)

        # 2. 散粒噪声
        if p.shot_noise_gain > 0:
            signal = np.maximum(noisy, 0)
            shot_std = np.sqrt(signal) * p.shot_noise_gain
            noisy += self._rng.normal(0, 1, (h, w)) * shot_std

        # 3. 暗电流
        if p.dark_current_mean > 0 or p.dark_current_std > 0:
            noisy += self._rng.normal(p.dark_current_mean, p.dark_current_std, (h, w))

        # 4. 读出噪声
        if p.read_noise_std > 0:
            noisy += self._rng.normal(0, p.read_noise_std, (h, w))

        # 5. 行噪声
        if p.row_noise_std > 0:
            noisy += self._rng.normal(0, p.row_noise_std, (h, 1))

        return np.clip(noisy, 0, white_level).astype(np.uint16)

    def reset_prnu(self):
        """重置 PRNU 缓存（模拟更换传感器）"""
        self._prnu_map = None
        self._prnu_shape = None


def add_sensor_noise(
    raw_12bit: np.ndarray,
    params: Optional[SensorNoiseParams] = None,
    preset: str = "sc132gs",
    white_level: int = 4095,
) -> np.ndarray:
    """快捷函数：为 12-bit RAW 添加噪声。

    Args:
        raw_12bit: (H, W) uint16 或 float
        params: 噪声参数（优先于 preset）
        preset: 预设名称
        white_level: 白电平

    Returns:
        (H, W) uint16
    """
    model = NoiseModel(params=params, preset=preset)
    return model.apply(raw_12bit, white_level=white_level)


def estimate_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """估计信噪比 (dB)"""
    clean_f = clean.astype(np.float64)
    noisy_f = noisy.astype(np.float64)
    signal_power = np.mean(clean_f ** 2)
    noise_power = np.mean((clean_f - noisy_f) ** 2)
    if noise_power < 1e-10:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)
