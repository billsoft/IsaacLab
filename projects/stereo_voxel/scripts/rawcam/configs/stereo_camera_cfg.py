"""
双目立体相机配置模块
===================
80mm 基线，朝向地面，适配 Isaac Lab Camera / TiledCamera

使用方式:
    from configs.stereo_camera_cfg import get_stereo_camera_cfgs
    left_cfg, right_cfg = get_stereo_camera_cfgs()
"""

# ============================================================
# 核心参数（修改这里即可全局生效）
# ============================================================

# 双目基线距离 (米)
BASELINE_M = 0.08  # 80mm

# 图像分辨率
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 1024

# 镜头参数
FOCAL_LENGTH_MM = 24.0        # 焦距 (mm)
HORIZONTAL_APERTURE_MM = 20.955  # 水平孔径 (mm)
CLIPPING_NEAR = 0.01          # 最近裁剪面 (m)
CLIPPING_FAR = 100.0          # 最远裁剪面 (m)

# 数据类型
DATA_TYPES = ["rgb", "distance_to_image_plane"]

# 相机朝向: 朝下（地面方向）
# ROS 约定: 前轴 +Z, 上轴 -Y
# 要让相机光轴对准 world -Z（地面），四元数 (w, x, y, z):
DOWNWARD_QUAT = (0.5, -0.5, 0.5, -0.5)

# Prim 路径前缀
PRIM_PREFIX = "/World/Robot"


def get_stereo_camera_cfgs(
    baseline_m: float = BASELINE_M,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    prim_prefix: str = PRIM_PREFIX,
    data_types: list = None,
):
    """
    生成双目相机配置对。

    需要在 Isaac Lab 环境中导入:
        from isaaclab.sensors import CameraCfg
        import isaaclab.sim as sim_utils

    参数:
        baseline_m: 基线距离 (米)，默认 0.08 (80mm)
        width: 图像宽度
        height: 图像高度
        prim_prefix: USD prim 路径前缀
        data_types: 数据类型列表

    返回:
        (left_camera_cfg, right_camera_cfg) 元组
    """
    # 延迟导入 —— 仅在 Isaac Lab 环境中可用
    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils

    if data_types is None:
        data_types = DATA_TYPES.copy()

    half_baseline = baseline_m / 2.0

    spawn_cfg = sim_utils.PinholeCameraCfg(
        focal_length=FOCAL_LENGTH_MM,
        horizontal_aperture=HORIZONTAL_APERTURE_MM,
        clipping_range=(CLIPPING_NEAR, CLIPPING_FAR),
    )

    left_camera_cfg = CameraCfg(
        prim_path=f"{prim_prefix}/left_camera",
        spawn=spawn_cfg,
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -half_baseline, 0.0),  # Y 轴负方向 = 左眼
            rot=DOWNWARD_QUAT,
            convention="ros",
        ),
        data_types=data_types,
        width=width,
        height=height,
    )

    right_camera_cfg = CameraCfg(
        prim_path=f"{prim_prefix}/right_camera",
        spawn=spawn_cfg,
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, half_baseline, 0.0),   # Y 轴正方向 = 右眼
            rot=DOWNWARD_QUAT,
            convention="ros",
        ),
        data_types=data_types,
        width=width,
        height=height,
    )

    return left_camera_cfg, right_camera_cfg


def get_stereo_tiled_camera_cfg(
    baseline_m: float = BASELINE_M,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
):
    """
    生成 TiledCamera 配置（单实例，双目需要移动位姿模拟）。

    注意: TiledCamera 场景中只能有一个实例，
    双目需要在渲染帧之间调用 set_world_poses 切换左右眼位置。
    """
    from isaaclab.sensors import TiledCameraCfg
    import isaaclab.sim as sim_utils

    tiled_cfg = TiledCameraCfg(
        prim_path=f"{PRIM_PREFIX}/stereo_camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=FOCAL_LENGTH_MM,
            horizontal_aperture=HORIZONTAL_APERTURE_MM,
            clipping_range=(CLIPPING_NEAR, CLIPPING_FAR),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=DOWNWARD_QUAT,
            convention="ros",
        ),
        data_types=["rgb"],
        width=width,
        height=height,
    )

    return tiled_cfg, baseline_m


def print_config_summary():
    """打印配置摘要（调试用）。"""
    print("=" * 50)
    print("  双目立体相机配置摘要")
    print("=" * 50)
    print(f"  基线距离:     {BASELINE_M * 1000:.0f} mm")
    print(f"  分辨率:       {IMAGE_WIDTH} × {IMAGE_HEIGHT}")
    print(f"  焦距:         {FOCAL_LENGTH_MM} mm")
    print(f"  水平孔径:     {HORIZONTAL_APERTURE_MM} mm")
    print(f"  裁剪范围:     {CLIPPING_NEAR} ~ {CLIPPING_FAR} m")
    print(f"  朝向:         地面 (world -Z)")
    print(f"  左眼偏移:     Y = -{BASELINE_M/2*1000:.0f} mm")
    print(f"  右眼偏移:     Y = +{BASELINE_M/2*1000:.0f} mm")
    print(f"  数据类型:     {DATA_TYPES}")
    print("=" * 50)


if __name__ == "__main__":
    print_config_summary()
