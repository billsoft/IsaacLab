"""ftheta 鱼眼相机属性设置

在 USD Camera prim 上应用 OmniLensDistortionFthetaAPI schema 并设置参数。
必须先 ApplyAPI 注册 schema，渲染器才会识别 ftheta 属性。
"""

from pxr import Sdf

from ..configs.dataclasses import SensorConfig


def _ensure_lens_distortion_extension():
    """尝试加载 omni.kit.renderer.core 以注册 OmniLensDistortionFthetaAPI schema。"""
    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.kit.renderer.core")
    except Exception:
        pass  # 如果扩展已加载或不可用，忽略


def compute_equidistant_coeffs(sensor_cfg: SensorConfig) -> tuple:
    """计算等距投影 ftheta 系数。

    等距模型: theta = k0 + k1*r + k2*r^2 + k3*r^3 + k4*r^4
    纯等距投影: k1 = 1/fx，其余为 0。

    Args:
        sensor_cfg: 传感器配置

    Returns:
        (k0, k1, k2, k3, k4)
    """
    k1 = sensor_cfg.k1_equidistant  # 1/fx
    return (0.0, k1, 0.0, 0.0, 0.0)


def setup_ftheta_fisheye(stage, prim_path: str, sensor_cfg: SensorConfig):
    """在 USD Camera prim 上设置 ftheta 鱼眼属性。

    关键步骤:
    1. ApplyAPI("OmniLensDistortionFthetaAPI") — 没有这步渲染器不识别
    2. 设置 model = "ftheta"
    3. 设置 nominalWidth/Height, opticalCenter, maxFov
    4. 设置 k0-k4 畸变系数（不是 p0-p4！）
    5. fStop=0 禁用景深模糊

    Args:
        stage: USD stage
        prim_path: 相机 prim 路径
        sensor_cfg: 传感器配置
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[RawCam] WARNING: prim {prim_path} not valid, skip fisheye setup")
        return

    # 计算系数
    if sensor_cfg.ftheta_coeffs is not None:
        coeffs = sensor_cfg.ftheta_coeffs
    else:
        coeffs = compute_equidistant_coeffs(sensor_cfg)

    cx = sensor_cfg.cx
    cy = sensor_cfg.cy

    # 1. 应用 API Schema（尝试两种方式）
    try:
        prim.ApplyAPI("OmniLensDistortionFthetaAPI")
    except Exception:
        # schema 未注册时尝试加载扩展再重试
        _ensure_lens_distortion_extension()
        try:
            prim.ApplyAPI("OmniLensDistortionFthetaAPI")
        except Exception:
            print("[RawCam] WARNING: OmniLensDistortionFthetaAPI schema not available, "
                  "creating attributes manually")

    # 辅助函数：安全设置属性（不存在则创建）
    def _set_attr(name, val, type_name=Sdf.ValueTypeNames.Float):
        attr = prim.GetAttribute(name)
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(name, type_name)
        if attr:
            attr.Set(val)

    # 2. 设置模型类型
    _set_attr("omni:lensdistortion:model", "ftheta", Sdf.ValueTypeNames.String)

    # 3. 设置尺寸和光心
    _set_attr("omni:lensdistortion:ftheta:nominalWidth", float(sensor_cfg.width))
    _set_attr("omni:lensdistortion:ftheta:nominalHeight", float(sensor_cfg.height))
    _set_attr("omni:lensdistortion:ftheta:opticalCenter",
              (float(cx), float(cy)), Sdf.ValueTypeNames.Float2)
    _set_attr("omni:lensdistortion:ftheta:maxFov", float(sensor_cfg.diagonal_fov_deg))

    # 4. k0-k4 畸变系数
    for i, k in enumerate(coeffs[:5]):
        _set_attr(f"omni:lensdistortion:ftheta:k{i}", float(k))

    # 5. 禁用景深模糊
    _set_attr("fStop", 0.0)
