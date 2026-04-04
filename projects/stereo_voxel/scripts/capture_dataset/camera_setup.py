"""双目鱼眼相机创建 + Replicator annotator 绑定
================================================
所有 omni.* / pxr.* 导入在函数内部延迟执行。
"""

import numpy as np

from .constants import (
    BASELINE_M, CAM_H, CAM_W, DIAG_FOV_DEG, K1_EQUIDISTANT, cx, cy,
)


# ===========================================================================
# ftheta 鱼眼设置
# ===========================================================================
def set_fisheye_on_prim(stage, cam_prim_path: str):
    """在 USD Camera prim 上设置 ftheta 等距鱼眼投影。"""
    prim = stage.GetPrimAtPath(cam_prim_path)
    if not prim.IsValid():
        print(f"[camera] WARNING: prim {cam_prim_path} not valid, skip fisheye setup")
        return
    prim.ApplyAPI("OmniLensDistortionFthetaAPI")
    prim.GetAttribute("omni:lensdistortion:model").Set("ftheta")
    prim.GetAttribute("omni:lensdistortion:ftheta:nominalWidth").Set(float(CAM_W))
    prim.GetAttribute("omni:lensdistortion:ftheta:nominalHeight").Set(float(CAM_H))
    prim.GetAttribute("omni:lensdistortion:ftheta:opticalCenter").Set((float(cx), float(cy)))
    prim.GetAttribute("omni:lensdistortion:ftheta:maxFov").Set(float(DIAG_FOV_DEG))
    prim.GetAttribute("omni:lensdistortion:ftheta:k0").Set(0.0)
    prim.GetAttribute("omni:lensdistortion:ftheta:k1").Set(float(K1_EQUIDISTANT))
    prim.GetAttribute("omni:lensdistortion:ftheta:k2").Set(0.0)
    prim.GetAttribute("omni:lensdistortion:ftheta:k3").Set(0.0)
    prim.GetAttribute("omni:lensdistortion:ftheta:k4").Set(0.0)
    prim.GetAttribute("fStop").Set(0.0)


def _create_camera_visual(stage, parent_path: str, name: str, color_rgb: list):
    """在相机 prim 下创建可视锥体标记。"""
    from pxr import Gf, UsdGeom
    cone_path = f"{parent_path}/{name}_visual"
    cone_prim = UsdGeom.Cone.Define(stage, cone_path)
    cone_prim.GetRadiusAttr().Set(0.03)
    cone_prim.GetHeightAttr().Set(0.06)
    cone_prim.GetAxisAttr().Set("X")
    cone_prim.GetDisplayColorAttr().Set([Gf.Vec3f(*[c / 255.0 for c in color_rgb])])


# ===========================================================================
# 双目 Rig 创建
# ===========================================================================
def create_stereo_rig(stage, simulation_app, cam_x: float, cam_y: float,
                      camera_height: float, stage_scale: float,
                      ) -> tuple[str, str, str]:
    """创建 /World/StereoRig 及左右鱼眼相机。

    Returns:
        (rig_path, left_cam_path, right_cam_path)
    """
    from pxr import Gf, UsdGeom
    import isaacsim.core.utils.numpy.rotations as rot_utils

    rig_path = "/World/StereoRig"
    rig_xform = UsdGeom.Xform.Define(stage, rig_path)
    rig_pos = Gf.Vec3d(
        float(cam_x * stage_scale),
        float(cam_y * stage_scale),
        float(camera_height * stage_scale),
    )
    rig_xform.AddTranslateOp().Set(rig_pos)

    half_baseline = (BASELINE_M / 2.0) * stage_scale
    left_cam_path = f"{rig_path}/Left"
    right_cam_path = f"{rig_path}/Right"

    # 相机朝向：USD prim 直接操作, euler (0, 0, 90) → 俯拍
    cam_euler = np.array([0.0, 0.0, 90.0])
    cam_quat = rot_utils.euler_angles_to_quats(cam_euler, degrees=True)

    for cam_path, y_offset in [(left_cam_path, half_baseline),
                                (right_cam_path, -half_baseline)]:
        cam_prim = UsdGeom.Camera.Define(stage, cam_path)
        xf = UsdGeom.Xformable(cam_prim.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, float(y_offset), 0.0))
        xf.AddOrientOp().Set(Gf.Quatf(
            float(cam_quat[0]),
            Gf.Vec3f(float(cam_quat[1]), float(cam_quat[2]), float(cam_quat[3])),
        ))
        cam_prim.GetClippingRangeAttr().Set(
            Gf.Vec2f(float(0.1 * stage_scale), float(100.0 * stage_scale)))
        set_fisheye_on_prim(stage, cam_path)

    _create_camera_visual(stage, left_cam_path, "left_cam", [255, 50, 50])
    _create_camera_visual(stage, right_cam_path, "right_cam", [50, 80, 255])

    # 基线可视化横杆
    bar_prim = UsdGeom.Cube.Define(stage, f"{rig_path}/baseline_bar")
    bar_prim.GetSizeAttr().Set(1.0)
    bar_prim.AddScaleOp().Set(Gf.Vec3f(0.01, float(BASELINE_M * stage_scale), 0.01))
    bar_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.8, 0.2)])

    for _ in range(5):
        simulation_app.update()

    print(f"[camera] StereoRig at ({cam_x:.1f}, {cam_y:.1f}, {camera_height})m, "
          f"baseline={BASELINE_M*1000:.0f}mm")
    return rig_path, left_cam_path, right_cam_path


# ===========================================================================
# Replicator annotators
# ===========================================================================
def setup_annotators(left_cam_path: str, right_cam_path: str, simulation_app):
    """创建 render_product 和 RGB annotator。

    Returns:
        (annot_left, annot_right)
    """
    import omni.replicator.core as rep

    rp_left = rep.create.render_product(left_cam_path, resolution=(CAM_W, CAM_H))
    rp_right = rep.create.render_product(right_cam_path, resolution=(CAM_W, CAM_H))

    annot_left = rep.AnnotatorRegistry.get_annotator("rgb")
    annot_left.attach([rp_left.path])
    annot_right = rep.AnnotatorRegistry.get_annotator("rgb")
    annot_right.attach([rp_right.path])

    for _ in range(5):
        simulation_app.update()

    print("[camera] Replicator annotators ready")
    return annot_left, annot_right


# ===========================================================================
# 相机世界位姿读取
# ===========================================================================
def get_rig_world_pose(stage, rig_path: str, stage_mpu: float,
                       fallback_height: float = 3.0,
                       ) -> tuple[np.ndarray, float]:
    """获取 StereoRig 的世界位置 (米) 和 yaw 角 (弧度)。

    Returns:
        (cam_pos, cam_yaw)
    """
    from pxr import UsdGeom

    rig_prim = stage.GetPrimAtPath(rig_path)
    if not rig_prim.IsValid():
        return np.array([0.0, 0.0, fallback_height]), 0.0

    xformable = UsdGeom.Xformable(rig_prim)
    world_xform = xformable.ComputeLocalToWorldTransform(0.0)
    translate = world_xform.ExtractTranslation()

    cam_pos = np.array([
        translate[0] * stage_mpu,
        translate[1] * stage_mpu,
        translate[2] * stage_mpu,
    ])

    rotation = world_xform.ExtractRotationMatrix()
    cam_yaw = np.arctan2(rotation[1][0], rotation[0][0])
    return cam_pos, cam_yaw
