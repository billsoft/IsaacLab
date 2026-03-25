"""RawCamera: 封装 USD Camera prim + replicator annotator + RAW 转换

不使用 isaacsim.sensors.camera.Camera（避免 IRA stage 事件回调冲突），
直接操作 USD prim + omni.replicator annotator。
"""

import numpy as np
from typing import Optional, Tuple

from ..configs.dataclasses import SensorConfig, NoiseConfig
from ..core.raw_converter import RawConverter
from .fisheye_setup import setup_ftheta_fisheye


class RawCamera:
    """单目 RAW 相机

    创建 USD Camera prim，挂接 replicator annotator，
    支持 RGB 采集和 RGB→RAW 转换。

    用法:
        cam = RawCamera("/World/Camera", sensor_cfg, noise_cfg)
        cam.create_prim(stage, translation=(0,0,3), orientation_quat=(1,0,0,0))
        cam.attach_annotator()
        # 仿真循环中:
        raw = cam.capture_raw()  # → (H,W) uint16 或 None
    """

    def __init__(
        self,
        prim_path: str,
        sensor_cfg: SensorConfig,
        noise_cfg: Optional[NoiseConfig] = None,
    ):
        self._prim_path = prim_path
        self._sensor_cfg = sensor_cfg
        self._converter = RawConverter(sensor_cfg, noise_cfg)

        self._stage = None
        self._stage_scale = 1.0
        self._annotator = None
        self._render_product = None

    @property
    def prim_path(self) -> str:
        return self._prim_path

    @property
    def converter(self) -> RawConverter:
        return self._converter

    def create_prim(
        self,
        stage,
        translation: Tuple[float, float, float] = (0.0, 0.0, 3.0),
        orientation_quat: Optional[Tuple[float, float, float, float]] = None,
    ):
        """在 USD stage 上创建相机 prim 并设置属性。

        自动处理 stage 单位适配（cm/m 场景都能用）。

        Args:
            stage: USD stage
            translation: 位置 (x, y, z)，单位：米
            orientation_quat: 朝向四元数 (w, x, y, z)，None=默认朝下
        """
        from pxr import UsdGeom, Gf

        self._stage = stage

        # stage 单位适配
        stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
        self._stage_scale = 1.0 / stage_mpu

        cfg = self._sensor_cfg

        # 创建 Camera prim
        cam_prim = UsdGeom.Camera.Define(stage, self._prim_path)
        cam_prim.GetFocalLengthAttr().Set(cfg.focal_length_mm)
        cam_prim.GetHorizontalApertureAttr().Set(cfg.horizontal_aperture_mm)
        cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(
            cfg.clipping_near * self._stage_scale,
            cfg.clipping_far * self._stage_scale,
        ))

        # fStop=0 禁用景深模糊
        from pxr import Sdf
        prim = cam_prim.GetPrim()
        fstop_attr = prim.GetAttribute("fStop")
        if fstop_attr and fstop_attr.IsValid():
            fstop_attr.Set(cfg.f_stop)
        else:
            prim.CreateAttribute("fStop", Sdf.ValueTypeNames.Float).Set(cfg.f_stop)

        # 设置位置和朝向
        xf = UsdGeom.Xformable(cam_prim.GetPrim())
        tx, ty, tz = translation
        xf.AddTranslateOp().Set(Gf.Vec3d(
            tx * self._stage_scale,
            ty * self._stage_scale,
            tz * self._stage_scale,
        ))

        if orientation_quat is not None:
            w, x, y, z = orientation_quat
            xf.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))

        # 鱼眼设置（仅在 fisheye_enabled=True 时应用）
        if cfg.fisheye_enabled and cfg.fisheye_model == "ftheta":
            setup_ftheta_fisheye(stage, self._prim_path, cfg)

    def attach_annotator(self):
        """创建 render_product 并附加 rgb annotator。

        必须在 sim.reset() 之后调用。
        """
        import omni.replicator.core as rep

        cfg = self._sensor_cfg
        self._render_product = rep.create.render_product(
            self._prim_path,
            resolution=(cfg.width, cfg.height),
        )
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annotator.attach([self._render_product])

    def get_rgb(self, skip_black_check: bool = False) -> Optional[np.ndarray]:
        """从 annotator 获取 RGB 数据。

        Args:
            skip_black_check: 跳过黑帧检测

        Returns:
            (H, W, 3) uint8 或 None（数据不可用或黑帧）
        """
        if self._annotator is None:
            return None

        try:
            data = self._annotator.get_data()
        except Exception:
            return None
        if data is None:
            return None

        # annotator 可能返回 dict（含 "data" 键）或直接是 ndarray
        if isinstance(data, dict):
            data = data.get("data", None)
            if data is None:
                return None

        arr = np.asarray(data)
        if arr.size == 0 or arr.ndim < 2:
            return None

        # RGBA → RGB
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[:, :, :3].copy()
        else:
            rgb = arr

        # 确保 uint8
        if rgb.dtype != np.uint8:
            if rgb.max() > 255:
                rgb = np.clip(rgb, 0, 255)
            rgb = rgb.astype(np.uint8)

        # 黑帧检测
        if not skip_black_check and rgb.mean() < 1.0:
            return None

        return rgb

    def capture_raw(self) -> Optional[np.ndarray]:
        """获取 RGB 并转换为 Bayer RAW。

        Returns:
            (H, W) uint16 或 None
        """
        rgb = self.get_rgb()
        if rgb is None:
            return None
        return self._converter.rgb_to_raw(rgb)

    def destroy(self):
        """清理资源"""
        self._annotator = None
        self._render_product = None
