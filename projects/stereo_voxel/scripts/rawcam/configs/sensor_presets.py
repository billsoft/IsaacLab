"""传感器预设库"""

from .dataclasses import SensorConfig


SC132GS = SensorConfig(
    width=1280,
    height=1080,
    pixel_size_um=2.7,
    focal_length_mm=1.75,
    diagonal_fov_deg=157.2,
    bit_depth=12,
    bayer_pattern="RGGB",
    fisheye_enabled=True,
    fisheye_model="ftheta",
    horizontal_aperture_mm=20.955,
)

IMX678 = SensorConfig(
    width=3840,
    height=2160,
    pixel_size_um=1.12,
    focal_length_mm=3.6,
    diagonal_fov_deg=120.0,
    bit_depth=12,
    bayer_pattern="RGGB",
    fisheye_enabled=True,
    fisheye_model="ftheta",
)

OV9282 = SensorConfig(
    width=1280,
    height=800,
    pixel_size_um=3.0,
    focal_length_mm=2.1,
    diagonal_fov_deg=110.0,
    bit_depth=10,
    black_level=0,
    white_level=1023,
    bayer_pattern="RGGB",
    fisheye_enabled=False,
)

PRESETS = {
    "sc132gs": SC132GS,
    "imx678": IMX678,
    "ov9282": OV9282,
}
