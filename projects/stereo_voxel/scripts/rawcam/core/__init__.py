"""RawCam 核心层（不依赖 Isaac Sim）"""

from .raw_converter import RawConverter
from .dng_writer import DngWriter
from .noise_model import NoiseModel, SensorNoiseParams, NOISE_PRESETS, add_sensor_noise, estimate_snr
from .bayer_mosaic import BayerPattern, rgb_to_bayer, bayer_to_rgb_nearest, simple_demosaic_half
from .verify import verify_raw_roundtrip, verify_dng
