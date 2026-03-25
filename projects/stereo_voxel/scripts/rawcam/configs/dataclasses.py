"""RawCam 核心配置 dataclass"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SensorConfig:
    """传感器硬件参数"""

    # 分辨率
    width: int = 1280
    height: int = 1080

    # 像素物理参数
    pixel_size_um: float = 2.7
    focal_length_mm: float = 1.75

    # RAW 参数
    bit_depth: int = 12
    black_level: int = 0
    white_level: int = 4095
    bayer_pattern: str = "RGGB"

    # 鱼眼参数
    fisheye_enabled: bool = True
    diagonal_fov_deg: float = 157.2
    fisheye_model: str = "ftheta"
    ftheta_coeffs: Optional[Tuple[float, ...]] = None  # None=自动计算等距投影

    # 光学
    f_stop: float = 0.0
    clipping_near: float = 0.01
    clipping_far: float = 100.0

    # 镜头参数（用于 USD Camera prim）
    # 默认 = pixel_size_um * width / 1000 = 2.7 * 1280 / 1000 = 3.456 mm (SC132GS)
    horizontal_aperture_mm: float = 3.456

    @property
    def fx(self) -> float:
        """像素焦距 (pixels)"""
        return self.focal_length_mm / self.pixel_size_um * 1000.0

    @property
    def cx(self) -> float:
        """光心 X (pixels)"""
        return self.width / 2.0

    @property
    def cy(self) -> float:
        """光心 Y (pixels)"""
        return self.height / 2.0

    @property
    def k1_equidistant(self) -> float:
        """等距投影 ftheta k1 系数: k1 = 1/fx"""
        return 1.0 / self.fx

    @property
    def max_val(self) -> int:
        """最大像素值"""
        return (1 << self.bit_depth) - 1

    @classmethod
    def from_preset(cls, name: str) -> "SensorConfig":
        """从预设名称创建配置"""
        from .sensor_presets import PRESETS
        if name not in PRESETS:
            raise ValueError(f"未知预设 '{name}'，可选: {list(PRESETS.keys())}")
        return PRESETS[name]


@dataclass
class NoiseConfig:
    """传感器噪声参数"""

    preset: str = "sc132gs"
    enabled: bool = True

    # 自定义参数（preset="custom" 时生效）
    read_noise_std: float = 4.0
    shot_noise_gain: float = 1.0
    dark_current_mean: float = 0.5
    dark_current_std: float = 0.3
    prnu_std: float = 0.01
    row_noise_std: float = 0.5

    seed: Optional[int] = None


@dataclass
class StereoConfig:
    """双目相机参数"""

    baseline_m: float = 0.08
    rig_prim_path: str = "/World/StereoRig"

    # 安装朝向 (match stereo_voxel_capture.py)
    mount_euler_deg: Tuple[float, float, float] = (0.0, 0.0, 90.0)
    mount_height_m: float = 3.0


@dataclass
class OutputConfig:
    """输出控制参数"""

    output_dir: str = "./output/stereo_raw12"

    # 输出格式
    save_dng: bool = True
    save_bin: bool = False
    save_rgb_preview: bool = True
    save_npy: bool = False

    # DNG 元数据
    dng_camera_make: str = "NVIDIA"
    dng_camera_model: str = "IsaacSim_SC132GS_Virtual"
    dng_color_matrix: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    dng_calibration_illuminant: int = 21  # D65

    # 采集控制
    capture_interval: int = 1
    warmup_frames: int = 30
    max_frames: int = -1

    # RGB 预览间隔 (0=每帧都保存, N=每N帧保存)
    rgb_preview_interval: int = 1

    # 异步 IO
    async_save: bool = True
    save_workers: int = 2
