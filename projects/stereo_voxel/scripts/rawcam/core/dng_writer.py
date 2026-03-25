"""DNG 文件写入器

纯 tifffile 实现，不依赖 exiftool。
DNG = TIFF 6.0 容器 + CFA photometric + DNG 扩展标签。

extratags dtype 码: 1=BYTE, 2=ASCII, 3=SHORT, 4=LONG, 5=RATIONAL, 12=DOUBLE
"""

import os
import json
import time
import numpy as np
from typing import Optional, Tuple

from ..configs.dataclasses import SensorConfig, OutputConfig


# CFA pattern → DNG CFAPattern tag 字节
_CFA_BYTES = {
    "RGGB": (0, 1, 1, 2),
    "GRBG": (1, 0, 2, 1),
    "GBRG": (1, 2, 0, 1),
    "BGGR": (2, 1, 1, 0),
}


class DngWriter:
    """DNG 文件写入器（纯 tifffile，不依赖 exiftool）"""

    def __init__(
        self,
        sensor_cfg: SensorConfig,
        output_cfg: Optional[OutputConfig] = None,
    ):
        self._sensor_cfg = sensor_cfg
        self._output_cfg = output_cfg or OutputConfig()
        self._extratags = self._build_extratags()

    def _build_extratags(self) -> list:
        """构建 DNG 扩展标签列表"""
        cfg = self._sensor_cfg
        out = self._output_cfg

        cfa_bytes = _CFA_BYTES.get(cfg.bayer_pattern, (0, 1, 1, 2))

        return [
            # DNGVersion 1.4
            (50706, 1, 4, (1, 4, 0, 0)),
            # DNGBackwardVersion 1.4
            (50707, 1, 4, (1, 4, 0, 0)),
            # CFARepeatPatternDim: 2x2
            (33421, 3, 2, (2, 2)),
            # CFAPattern
            (33422, 1, 4, cfa_bytes),
            # BlackLevel
            (50714, 4, 1, cfg.black_level),
            # WhiteLevel
            (50717, 4, 1, cfg.white_level),
            # ColorMatrix1 (identity)
            (50721, 12, 9, out.dng_color_matrix),
            # CalibrationIlluminant1
            (50778, 3, 1, out.dng_calibration_illuminant),
            # AsShotNeutral
            (50728, 12, 3, (1.0, 1.0, 1.0)),
        ]

    def write(
        self,
        bayer_uint16: np.ndarray,
        path: str,
        frame_meta: Optional[dict] = None,
    ):
        """写入单个 DNG 文件。

        Args:
            bayer_uint16: (H, W) uint16 Bayer 数据
            path: 输出路径
            frame_meta: 可选的额外帧元数据
        """
        from tifffile import TiffWriter

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with TiffWriter(path, bigtiff=False) as tif:
            tif.write(
                bayer_uint16,
                photometric=32803,    # CFA
                compression=1,        # 无压缩
                bitspersample=16,
                subfiletype=0,
                metadata=None,
                extratags=self._extratags,
            )

    def write_stereo_pair(
        self,
        left: np.ndarray,
        right: np.ndarray,
        frame_id: int,
        output_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """写入一对双目 DNG。

        Args:
            left: (H, W) uint16 左眼 Bayer 数据
            right: (H, W) uint16 右眼 Bayer 数据
            frame_id: 帧编号
            output_dir: 输出目录（默认用 OutputConfig）

        Returns:
            (left_path, right_path)
        """
        base = output_dir or self._output_cfg.output_dir
        left_dir = os.path.join(base, "left")
        right_dir = os.path.join(base, "right")

        left_path = os.path.join(left_dir, f"frame_{frame_id:06d}.dng")
        right_path = os.path.join(right_dir, f"frame_{frame_id:06d}.dng")

        self.write(left, left_path)
        self.write(right, right_path)

        return left_path, right_path

    def write_manifest(
        self,
        output_dir: str,
        total_frames: int,
        extra: Optional[dict] = None,
    ):
        """写入 manifest.json 元数据文件。

        Args:
            output_dir: 输出目录
            total_frames: 总帧数
            extra: 额外元数据字典
        """
        cfg = self._sensor_cfg
        manifest = {
            "project": "RawCam Stereo Pseudo-RAW",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_frames": total_frames,
            "image_width": cfg.width,
            "image_height": cfg.height,
            "bit_depth": cfg.bit_depth,
            "bayer_pattern": cfg.bayer_pattern,
            "black_level": cfg.black_level,
            "white_level": cfg.white_level,
            "pixel_size_um": cfg.pixel_size_um,
            "focal_length_mm": cfg.focal_length_mm,
            "pipeline": "RGB(uint8) -> sRGB_decode -> linear(float32) -> CFA -> quantize -> noise -> uint16",
        }
        if extra:
            manifest.update(extra)

        path = os.path.join(output_dir, "manifest.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
