"""capture_v2 常量定义
======================
相机参数、NPC 尺寸、资产路径等集中管理。
"""

import os
import sys

# ===========================================================================
# SC132GS 双目鱼眼相机参数
# ===========================================================================
CAM_W: int = 1280
CAM_H: int = 1080
PIXEL_SIZE_UM: float = 2.7
FOCAL_LENGTH_MM: float = 1.75
DIAG_FOV_DEG: float = 157.2
BASELINE_M: float = 0.08  # 80mm

# 导出像素焦距和光心
fx: float = FOCAL_LENGTH_MM / PIXEL_SIZE_UM * 1000  # ≈648.15 px
fy: float = fx
cx: float = CAM_W / 2.0  # 640.0
cy: float = CAM_H / 2.0  # 540.0
K1_EQUIDISTANT: float = 1.0 / fx

# ===========================================================================
# NPC 参数
# ===========================================================================
NPC_RADIUS_M: float = 0.25   # 人体近似半径
NPC_HEIGHT_M: float = 1.8    # 人体近似高度
SIM_FPS: float = 30.0        # 仿真帧率

# ===========================================================================
# NPC 角色模型（相对于 ASSETS_ROOT）
# ===========================================================================
CHARACTER_MODELS = [
    "Isaac/People/Characters/F_Business_02/F_Business_02.usd",
    "Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd",
    "Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd",
    "Isaac/People/Characters/M_Medical_01/M_Medical_01.usd",
    "Isaac/People/Characters/male_adult_construction_03/male_adult_construction_03.usd",
    "Isaac/People/Characters/female_adult_police_03_new/female_adult_police_03_new.usd",
]


# ===========================================================================
# 资产路径解析
# ===========================================================================
def resolve_assets_root() -> str:
    """查找 Isaac 资产根目录，找不到则退出。"""
    local = "D:/code/IsaacLab/Assets/Isaac/5.1"
    if os.path.isdir(local):
        return local.replace("\\", "/")
    try:
        from isaacsim.storage.native import get_assets_root_path
        path = get_assets_root_path()
        if path and os.path.isdir(path):
            return path.replace("\\", "/")
    except Exception:
        pass
    print("[capture_v2] ERROR: Cannot find Isaac assets.", file=sys.stderr)
    sys.exit(1)


def get_scene_usd(assets_root: str) -> str:
    return f"{assets_root}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"


def get_available_models(assets_root: str) -> list[str]:
    return [m for m in CHARACTER_MODELS if os.path.isfile(f"{assets_root}/{m}")]
