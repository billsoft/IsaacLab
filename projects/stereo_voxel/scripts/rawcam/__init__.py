"""RawCam — 可复用的 12-bit RAW 相机仿真组件包

核心层 (core/): 不依赖 Isaac Sim，可在普通 Python 环境中使用
仿真层 (sim/):  封装 USD Camera + replicator annotator，需要 Isaac Sim

用法 (核心层):
    from rawcam.core import RawConverter, DngWriter, BayerPattern
    from rawcam.configs import SensorConfig, NoiseConfig

    converter = RawConverter(SensorConfig.from_preset("sc132gs"), NoiseConfig())
    raw = converter.rgb_to_raw(rgb_uint8)

用法 (仿真层, 需在 Isaac Sim 环境中):
    from rawcam.sim import StereoRawRig
    from rawcam.configs import SensorConfig, StereoConfig, NoiseConfig, OutputConfig

    rig = StereoRawRig(sensor_cfg, stereo_cfg, noise_cfg, output_cfg)
    rig.create(stage)
    rig.attach_annotators()
"""

# 核心层 — 始终可导入
from .core.raw_converter import RawConverter
from .core.dng_writer import DngWriter
from .core.noise_model import NoiseModel
from .core.bayer_mosaic import BayerPattern

# 配置 — 始终可导入
from .configs.dataclasses import SensorConfig, NoiseConfig, StereoConfig, OutputConfig
from .configs.sensor_presets import PRESETS as SENSOR_PRESETS
