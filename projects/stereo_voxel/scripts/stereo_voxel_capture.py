"""双目鱼眼 + 语义体素 GT 同步采集
=====================================
在 Isaac Sim 仓库场景中同步采集：
  1. 双目鱼眼图像对（左/右 RGB, 1280x1080）
  2. 对应的 3D 语义体素 ground truth（72x60x32, 0.1m 分辨率）

同步策略：timeline.pause() 冻结 → 拍照 + 体素查询 → timeline.play() 恢复

运行：
    isaaclab.bat -p projects/stereo_voxel/scripts/stereo_voxel_capture.py
    isaaclab.bat -p projects/stereo_voxel/scripts/stereo_voxel_capture.py --headless --num_frames 200
    isaaclab.bat -p projects/stereo_voxel/scripts/stereo_voxel_capture.py --no_npc --headless --num_frames 50
"""

import argparse
import glob
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

# 强制 print 立即刷新，防止 subprocess 重定向时输出丢失
import builtins
_original_print = builtins.print
def _flush_print(*a, **kw):
    kw.setdefault("flush", True)
    _original_print(*a, **kw)
builtins.print = _flush_print

parser = argparse.ArgumentParser(description="Stereo fisheye + voxel GT capture")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--num_frames", type=int, default=200, help="Number of frames to capture (headless)")
parser.add_argument("--camera_height", type=float, default=3.0, help="Camera height (m)")
parser.add_argument("--camera_x", type=float, default=0.0, help="Camera X position (m)")
parser.add_argument("--camera_y", type=float, default=0.0, help="Camera Y position (m)")
parser.add_argument("--num_characters", type=int, default=3, help="Number of NPC characters")
parser.add_argument("--walk_distance", type=float, default=8.0, help="NPC walk distance (m)")
parser.add_argument("--capture_interval", type=int, default=90, help="Capture every N sim steps (~3s at 30FPS)")
parser.add_argument("--no_npc", action="store_true", help="Skip NPC loading (static scene only)")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 1080})

import carb
import cv2
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, UsdGeom

# 添加脚本目录到 sys.path，以便 import 同目录模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from semantic_classes import FREE, GENERAL_OBJECT, PERSON, UNOBSERVED, lookup_class_id
from voxel_grid import VoxelGrid


# ===========================================================================
# 场景探测：扫描关键物体位置，自动推荐相机位置
# ===========================================================================
def probe_scene_objects(stage, stage_mpu, keywords=("RackFrame", "RackShelf", "PillarA")):
    """扫描场景中名称包含关键词的顶层 Xform prim，返回世界坐标范围（米）。

    只匹配 /Root/ 下的直接子 prim（避免遍历过深）。

    Returns:
        dict: {keyword: {"positions": np.array (N,3) in meters, "bbox_min": (3,), "bbox_max": (3,)}}
    """
    from pxr import UsdGeom as _UsdGeom
    results = {k: [] for k in keywords}
    root = stage.GetPrimAtPath("/Root")
    if not root.IsValid():
        return results
    for child in root.GetChildren():
        name = child.GetName()
        for kw in keywords:
            if kw in name:
                xformable = _UsdGeom.Xformable(child)
                try:
                    xf = xformable.ComputeLocalToWorldTransform(0)
                    pos = xf.ExtractTranslation()
                    results[kw].append([pos[0] * stage_mpu, pos[1] * stage_mpu, pos[2] * stage_mpu])
                except Exception:
                    pass
                break
    out = {}
    for kw, positions in results.items():
        if positions:
            arr = np.array(positions)
            out[kw] = {
                "positions": arr,
                "bbox_min": arr.min(axis=0),
                "bbox_max": arr.max(axis=0),
                "count": len(arr),
            }
    return out


def suggest_camera_position(scene_info):
    """根据场景物体分布，推荐相机 XY 位置（米）。

    策略：找货架区域中心的通道位置。
    """
    all_positions = []
    for info in scene_info.values():
        all_positions.append(info["positions"][:, :2])  # XY only
    if not all_positions:
        return 0.0, 0.0
    all_xy = np.vstack(all_positions)
    # 货架中心
    center_x = float(np.median(all_xy[:, 0]))
    center_y = float(np.median(all_xy[:, 1]))
    return center_x, center_y


# ===========================================================================
# NPC Xform 检测：用 USD 变换替代 PhysX 检测人物
# ===========================================================================
NPC_RADIUS_M = 0.25     # 人体近似半径（米）
NPC_HEIGHT_M = 1.8       # 人体近似高度（米）


def get_npc_world_positions(stage, stage_mpu):
    """获取场景中所有 NPC Character 的世界位置（米）。

    扫描 /World/Characters/Character* 下的 SkelRoot。

    Returns:
        list of np.array (3,): 每个 NPC 脚底世界坐标 [x, y, z]
    """
    from pxr import UsdGeom as _UsdGeom
    npc_positions = []
    chars_prim = stage.GetPrimAtPath("/World/Characters")
    if not chars_prim.IsValid():
        return npc_positions

    from pxr import Usd as _Usd
    for child in chars_prim.GetChildren():
        name = child.GetName()
        if not name.startswith("Character"):
            continue
        # IRA 动画更新的是 SkelRoot 子节点，不是根 Xform
        target = child
        for p in _Usd.PrimRange(child):
            if p.GetTypeName() == "SkelRoot":
                target = p
                break
        xformable = _UsdGeom.Xformable(target)
        try:
            xf = xformable.ComputeLocalToWorldTransform(0)
            pos = xf.ExtractTranslation()
            npc_positions.append(np.array([
                pos[0] * stage_mpu,
                pos[1] * stage_mpu,
                pos[2] * stage_mpu,
            ]))
        except Exception:
            continue
    return npc_positions


def stamp_npc_voxels(voxel_grid, cam_pos, cam_yaw, npc_world_positions):
    """将 NPC 世界坐标投影到体素网格，标记为 PERSON。

    用圆柱近似：半径 NPC_RADIUS_M, 高度 NPC_HEIGHT_M。
    """
    if not npc_world_positions:
        return 0

    count = 0
    for npc_pos in npc_world_positions:
        # 世界坐标 → 体素局部坐标
        ground_proj = np.array([cam_pos[0], cam_pos[1], 0.0])
        local = npc_pos - ground_proj

        # 逆旋转（如果有 yaw）
        if abs(cam_yaw) > 1e-6:
            cos_y, sin_y = np.cos(-cam_yaw), np.sin(-cam_yaw)
            lx = local[0] * cos_y - local[1] * sin_y
            ly = local[0] * sin_y + local[1] * cos_y
            local[0], local[1] = lx, ly

        # 体素索引范围
        vx_min = int(np.floor((local[0] - NPC_RADIUS_M) / voxel_grid.VOXEL_SIZE + voxel_grid.CENTER_X))
        vx_max = int(np.ceil((local[0] + NPC_RADIUS_M) / voxel_grid.VOXEL_SIZE + voxel_grid.CENTER_X))
        vy_min = int(np.floor((local[1] - NPC_RADIUS_M) / voxel_grid.VOXEL_SIZE + voxel_grid.CENTER_Y))
        vy_max = int(np.ceil((local[1] + NPC_RADIUS_M) / voxel_grid.VOXEL_SIZE + voxel_grid.CENTER_Y))
        vz_min = int(np.floor(local[2] / voxel_grid.VOXEL_SIZE + voxel_grid.Z_GROUND_INDEX))
        vz_max = int(np.ceil((local[2] + NPC_HEIGHT_M) / voxel_grid.VOXEL_SIZE + voxel_grid.Z_GROUND_INDEX))

        # 裁剪到网格边界
        vx_min = max(0, vx_min)
        vx_max = min(voxel_grid.NX, vx_max)
        vy_min = max(0, vy_min)
        vy_max = min(voxel_grid.NY, vy_max)
        vz_min = max(0, vz_min)
        vz_max = min(voxel_grid.NZ, vz_max)

        for i in range(vx_min, vx_max):
            for j in range(vy_min, vy_max):
                # 检查是否在圆柱半径内
                cx = (i - voxel_grid.CENTER_X + 0.5) * voxel_grid.VOXEL_SIZE
                cy = (j - voxel_grid.CENTER_Y + 0.5) * voxel_grid.VOXEL_SIZE
                dist_sq = (cx - local[0])**2 + (cy - local[1])**2
                if dist_sq <= NPC_RADIUS_M**2:
                    for k in range(vz_min, vz_max):
                        voxel_grid.semantic[i, j, k] = PERSON
                        count += 1

    return count

# ===========================================================================
# SC132GS 相机参数
# ===========================================================================
CAM_W, CAM_H = 1280, 1080
PIXEL_SIZE_UM = 2.7
FOCAL_LENGTH_MM = 1.75
DIAG_FOV_DEG = 157.2
BASELINE_M = 0.08  # 80mm

fx = FOCAL_LENGTH_MM / PIXEL_SIZE_UM * 1000  # 648.15 px
cx = CAM_W / 2.0
cy = CAM_H / 2.0
K1_EQUIDISTANT = 1.0 / fx

print(f"[Capture] SC132GS x2, baseline={BASELINE_M*1000:.0f}mm, FOV={DIAG_FOV_DEG}deg")

# ===========================================================================
# Assets
# ===========================================================================
LOCAL_ASSETS_PATH = "D:/code/IsaacLab/Assets/Isaac/5.1"
if not os.path.isdir(LOCAL_ASSETS_PATH):
    try:
        from isaacsim.storage.native import get_assets_root_path
        LOCAL_ASSETS_PATH = get_assets_root_path()
    except Exception:
        pass
if not LOCAL_ASSETS_PATH or not os.path.isdir(LOCAL_ASSETS_PATH):
    print("[Capture] ERROR: Cannot find Isaac assets.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)

ASSETS_ROOT = LOCAL_ASSETS_PATH.replace("\\", "/")
SCENE_USD = f"{ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

# ===========================================================================
# NPC character models
# ===========================================================================
CHARACTER_MODELS = [
    "Isaac/People/Characters/F_Business_02/F_Business_02.usd",
    "Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd",
    "Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd",
    "Isaac/People/Characters/M_Medical_01/M_Medical_01.usd",
    "Isaac/People/Characters/male_adult_construction_03/male_adult_construction_03.usd",
    "Isaac/People/Characters/female_adult_police_03_new/female_adult_police_03_new.usd",
]
AVAILABLE_MODELS = [m for m in CHARACTER_MODELS if os.path.isfile(f"{ASSETS_ROOT}/{m}")]


# ===========================================================================
# NPC config generation (same as test_stereo_pair.py)
# ===========================================================================
def generate_npc_command_file(num_characters, walk_distance, num_loops=100):
    lines = []
    spacing = 3.0
    start_x = -walk_distance / 2.0
    for i in range(num_characters):
        if i == 0:
            name = "Character"
        elif i < 10:
            name = f"Character_0{i}"
        else:
            name = f"Character_{i}"
        y = i * spacing
        x_a, x_b = start_x, start_x + walk_distance
        for _ in range(num_loops):
            lines.append(f"{name} GoTo {x_b:.1f} {y:.1f} 0.0 0")
            lines.append(f"{name} GoTo {x_a:.1f} {y:.1f} 0.0 180")
    temp_dir = tempfile.gettempdir()
    cmd_file = os.path.join(temp_dir, "npc_walk_commands.txt")
    with open(cmd_file, "w") as f:
        f.write("\n".join(lines))
    return "npc_walk_commands.txt"


def generate_npc_config_file(num_characters, command_file):
    char_folder = f"{ASSETS_ROOT}/Isaac/People/Characters/"
    config = (
        "isaacsim.replicator.agent:\n"
        "  version: 0.7.0\n"
        "  global:\n"
        "    seed: 42\n"
        "    simulation_length: 90000\n"
        "  scene:\n"
        f"    asset_path: {SCENE_USD}\n"
        "  character:\n"
        f"    asset_path: {char_folder}\n"
        f"    command_file: {command_file}\n"
        f"    num: {num_characters}\n"
        "  replicator:\n"
        "    writer: IRABasicWriter\n"
        "    parameters:\n"
        "      rgb: false\n"
    )
    temp_dir = tempfile.gettempdir()
    config_file = os.path.join(temp_dir, "npc_walk_config.yaml")
    with open(config_file, "w") as f:
        f.write(config)
    return config_file


# ===========================================================================
# Helper: set fisheye lens on USD prim
# ===========================================================================
def set_fisheye_on_prim(stage, cam_prim_path):
    prim = stage.GetPrimAtPath(cam_prim_path)
    if not prim.IsValid():
        print(f"[Capture] WARNING: prim {cam_prim_path} not valid, skip fisheye setup")
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


# ===========================================================================
# Helper: camera visual cone
# ===========================================================================
def create_camera_visual(stage, parent_path, name, color_rgb):
    cone_path = f"{parent_path}/{name}_visual"
    cone_prim = UsdGeom.Cone.Define(stage, cone_path)
    cone_prim.GetRadiusAttr().Set(0.03)
    cone_prim.GetHeightAttr().Set(0.06)
    cone_prim.GetAxisAttr().Set("X")
    cone_prim.GetDisplayColorAttr().Set([Gf.Vec3f(*[c / 255.0 for c in color_rgb])])


# ===========================================================================
# Async save pool
# ===========================================================================
_save_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="saver")
_pending_saves = []


def async_save_image(path, rgb_array):
    def _save(p, arr):
        bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(p, bgr)
    future = _save_pool.submit(_save, path, rgb_array.copy())
    _pending_saves.append(future)


def async_save_voxel(path_prefix, semantic, instance):
    def _save(pp, sem, ins):
        os.makedirs(os.path.dirname(pp), exist_ok=True)
        np.savez_compressed(f"{pp}_semantic.npz", data=sem)
        np.savez_compressed(f"{pp}_instance.npz", data=ins)
    future = _save_pool.submit(_save, path_prefix, semantic.copy(), instance.copy())
    _pending_saves.append(future)


def async_save_json(path, data):
    def _save(p, d):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(d, f, indent=2)
    future = _save_pool.submit(_save, path, data)
    _pending_saves.append(future)


def wait_pending_saves():
    for f in _pending_saves:
        f.result()
    _pending_saves.clear()


# ===========================================================================
# PhysX overlap query helpers
# ===========================================================================
def get_object_type_from_prim_path(stage, prim_path: str) -> str:
    """从 PhysX hit 的 rigid_body prim path 提取物体类型名。

    策略：向上遍历找最近的有意义 Xform/Mesh，提取名字。
    特殊处理 NPC 角色路径。
    """
    # NPC 角色路径通常包含 Character / Character_01 等
    if "/Character" in prim_path:
        return "NPC:person"

    # 向上遍历找有意义的 prim name
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return ""

    current = prim
    while current.IsValid():
        name = current.GetName()
        # 跳过内部细节（Collider, Mesh, Shape, Section 等）
        if name in ("Collider", "CollisionMesh", "CollisionPlane", "Mesh",
                     "Shape", "Physics", "Collision", "collision") or name.startswith("Section"):
            current = current.GetParent()
            continue
        # 跳过根和 World 节点
        if name in ("", "World", "Environment", "Warehouse"):
            current = current.GetParent()
            continue
        # 跳过纯 Scope/Xform 容器
        if current.GetTypeName() in ("Scope",):
            current = current.GetParent()
            continue
        return name

    # 回退：取 prim path 最后一段
    return prim_path.rsplit("/", 1)[-1] if "/" in prim_path else prim_path


def _clean_object_name(raw_name: str) -> str:
    """清理物体名：去掉 SM_/S_ 前缀和尾部 _数字 后缀。

    SM_RackShelf_01 → RackShelf
    SM_CardBoxD_04  → CardBoxD
    S_TrafficCone   → TrafficCone
    """
    name = raw_name
    # 去前缀
    for prefix in ("SM_", "S_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # 去尾部 _数字
    import re
    name = re.sub(r"_\d+$", "", name)
    return name


# Instance ID 管理
_instance_map = {}  # prim_path → instance_id
_next_instance_id = 1


def get_instance_id(prim_path: str) -> int:
    global _next_instance_id
    if prim_path not in _instance_map:
        _instance_map[prim_path] = _next_instance_id
        _next_instance_id += 1
    return _instance_map[prim_path]


def fill_voxel_grid(stage, voxel_grid, world_centers_flat, physx_sqi, meters_to_stage):
    """用 PhysX overlap_box 填充体素网格。

    采用粗查+细查两阶段策略：
      1. 粗查：5x5x5 体素块(0.5m³)做一次 overlap，无 hit 则整块标 FREE
      2. 细查：有 hit 的块内逐个体素查询

    关键：PhysX 在 stage 单位（通常是厘米）下工作，
         体素世界坐标是米单位，传入 PhysX 前必须乘以 meters_to_stage。

    Args:
        stage: USD stage
        voxel_grid: VoxelGrid 实例（会被修改）
        world_centers_flat: (N, 3) 世界坐标（米）
        physx_sqi: PhysX scene query interface
        meters_to_stage: float, 米→stage 单位的换算系数（= 1/metersPerUnit）
    """
    NX, NY, NZ = voxel_grid.NX, voxel_grid.NY, voxel_grid.NZ
    VOXEL_SIZE = voxel_grid.VOXEL_SIZE

    # 将世界坐标从米转为 stage 单位（PhysX 的工作单位）
    world_centers_stage = (world_centers_flat * meters_to_stage).reshape(NX, NY, NZ, 3)

    # 修复地面层边界效应：z=Z_GROUND_INDEX 的体素中心在 z=+0.05m，
    # fine_half=0.05m → PhysX box 下边界恰好在 z=0（地面平面），
    # 导致碰撞检测不稳定，约 50% 的地面体素 miss。
    # 解决：将地面层 z 坐标下移 2mm，确保 box 与地面明确重叠。
    Z_GI = voxel_grid.Z_GROUND_INDEX
    GROUND_Z_NUDGE = 0.002 * meters_to_stage  # 2mm in stage units
    world_centers_stage[:, :, Z_GI, 2] -= GROUND_Z_NUDGE

    COARSE = 5  # 粗查块大小
    # half extent 也需要换算为 stage 单位
    fine_half = (VOXEL_SIZE / 2.0) * meters_to_stage
    identity_rot = carb.Float4(0.0, 0.0, 0.0, 1.0)

    total_voxels = NX * NY * NZ
    coarse_skipped = 0
    occupied_count = 0

    # 粗查遍历
    for ci in range(0, NX, COARSE):
        for cj in range(0, NY, COARSE):
            for ck in range(0, NZ, COARSE):
                ei = min(ci + COARSE, NX)
                ej = min(cj + COARSE, NY)
                ek = min(ck + COARSE, NZ)
                block_center = world_centers_stage[ci:ei, cj:ej, ck:ek].mean(axis=(0, 1, 2))

                # 粗查 half extent（stage 单位）
                bh_x = (ei - ci) * VOXEL_SIZE * meters_to_stage / 2.0
                bh_y = (ej - cj) * VOXEL_SIZE * meters_to_stage / 2.0
                bh_z = (ek - ck) * VOXEL_SIZE * meters_to_stage / 2.0

                # 粗查
                coarse_hits = []

                def on_coarse_hit(hit):
                    coarse_hits.append(hit.rigid_body)
                    return True

                physx_sqi.overlap_box(
                    carb.Float3(bh_x, bh_y, bh_z),
                    carb.Float3(float(block_center[0]), float(block_center[1]), float(block_center[2])),
                    identity_rot,
                    on_coarse_hit,
                    False,
                )

                if not coarse_hits:
                    voxel_grid.semantic[ci:ei, cj:ej, ck:ek] = FREE
                    coarse_skipped += (ei - ci) * (ej - cj) * (ek - ck)
                    continue

                # 细查：逐个体素
                for i in range(ci, ei):
                    for j in range(cj, ej):
                        for k in range(ck, ek):
                            center = world_centers_stage[i, j, k]
                            fine_hits = []

                            def on_fine_hit(hit):
                                fine_hits.append(hit.rigid_body)
                                return True

                            physx_sqi.overlap_box(
                                carb.Float3(fine_half, fine_half, fine_half),
                                carb.Float3(float(center[0]), float(center[1]), float(center[2])),
                                identity_rot,
                                on_fine_hit,
                                False,
                            )

                            if not fine_hits:
                                voxel_grid.semantic[i, j, k] = FREE
                                voxel_grid.instance[i, j, k] = 0
                            else:
                                hit_path = fine_hits[0]
                                obj_type = get_object_type_from_prim_path(stage, hit_path)
                                clean_name = _clean_object_name(obj_type)
                                class_id = lookup_class_id(clean_name)
                                voxel_grid.semantic[i, j, k] = class_id
                                voxel_grid.instance[i, j, k] = get_instance_id(hit_path)
                                occupied_count += 1

    # 后处理：地面层兜底填充
    # 如果 z=Z_GI 是 free 但 z=Z_GI-1（地下层）是 occupied，说明边界检测 miss，
    # 用地下层的类别填充地面层。
    from semantic_classes import OTHER_GROUND
    if Z_GI > 0:
        underground = voxel_grid.semantic[:, :, Z_GI - 1]
        ground = voxel_grid.semantic[:, :, Z_GI]
        miss_mask = (ground == FREE) & (underground > 0) & (underground != UNOBSERVED)
        patched = int(miss_mask.sum())
        if patched > 0:
            voxel_grid.semantic[:, :, Z_GI][miss_mask] = OTHER_GROUND
            occupied_count += patched
            print(f"    Ground layer patch: filled {patched} missing ground voxels (z={Z_GI})")

    free_count = np.sum(voxel_grid.semantic == FREE)
    unobs_count = np.sum(voxel_grid.semantic == UNOBSERVED)
    print(f"    Voxel fill: {occupied_count} occupied, {free_count} free, "
          f"{unobs_count} unobserved, {coarse_skipped} coarse-skipped")

    # 诊断：打印检测到的唯一 prim paths 和对应类别
    unique_paths = set()
    for path, iid in _instance_map.items():
        unique_paths.add(path)
    if unique_paths:
        print(f"    Detected {len(unique_paths)} unique collision prims:")
        for p in sorted(unique_paths):
            obj_type = get_object_type_from_prim_path(stage, p)
            clean_name = _clean_object_name(obj_type)
            cid = lookup_class_id(clean_name)
            print(f"      [{cid:2d}] {clean_name:<25s} (raw={obj_type}) ← {p}")


# ============================================================================
# PHASE 1: 场景 + NPC 初始化
# ============================================================================
from isaacsim.core.utils.extensions import enable_extension

npc_ready = False
num_chars = 0
use_npc = not args.no_npc and len(AVAILABLE_MODELS) > 0

if use_npc:
    num_chars = min(args.num_characters, len(AVAILABLE_MODELS))
    print(f"[Capture] Found {len(AVAILABLE_MODELS)} NPC model(s), using {num_chars}")

    NPC_EXTENSIONS = [
        "omni.anim.timeline", "omni.anim.graph.bundle", "omni.anim.graph.core",
        "omni.anim.retarget.core", "omni.anim.navigation.core",
        "omni.anim.navigation.bundle", "omni.anim.people", "omni.kit.scripting",
    ]
    print("[Capture] Enabling NPC extensions...")
    for ext in NPC_EXTENSIONS:
        enable_extension(ext)
        simulation_app.update()

    # 手动打开场景，等待 ASSETS_LOADED
    print(f"[Capture] Pre-loading scene: {SCENE_USD}")
    scene_loaded = [False]

    def on_scene_loaded(e):
        scene_loaded[0] = True

    _handle = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.ASSETS_LOADED),
        on_event=on_scene_loaded,
        observer_name="capture/on_scene_preload",
    )

    import omni.kit.window.file
    old_ignore = carb.settings.get_settings().get("/app/file/ignoreUnsavedStage")
    carb.settings.get_settings().set("/app/file/ignoreUnsavedStage", True)
    omni.kit.window.file.open_stage(SCENE_USD, omni.usd.UsdContextInitialLoadSet.LOAD_ALL)
    carb.settings.get_settings().set("/app/file/ignoreUnsavedStage",
                                      old_ignore if old_ignore is not None else False)

    tick = 0
    while not scene_loaded[0] and not simulation_app.is_exiting() and tick < 1500:
        simulation_app.update()
        tick += 1
        if tick % 300 == 0:
            print(f"[Capture] Waiting for scene... tick={tick}")
    _handle = None
    print(f"[Capture] Scene loaded after {tick} ticks")

    for _ in range(30):
        simulation_app.update()

    # IRA setup
    print(f"[Capture] Setting up {num_chars} NPC(s) via IRA...")
    enable_extension("isaacsim.replicator.agent.core")
    simulation_app.update()
    simulation_app.update()

    from isaacsim.replicator.agent.core.simulation import SimulationManager

    sim_manager = SimulationManager()
    settings = carb.settings.get_settings()
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
    settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
    settings.set("/app/omni.graph.scriptnode/enable_opt_in", False)

    cmd_file = generate_npc_command_file(num_chars, args.walk_distance)
    config_file = generate_npc_config_file(num_chars, cmd_file)

    can_load = sim_manager.load_config_file(config_file)
    if can_load:
        setup_done = [False]

        def on_setup_done(e):
            setup_done[0] = True

        sim_manager.register_set_up_simulation_done_callback(on_setup_done)
        sim_manager.set_up_simulation_from_config_file()

        tick = 0
        while not setup_done[0] and not simulation_app.is_exiting() and tick < 3000:
            simulation_app.update()
            tick += 1
            if tick % 300 == 0:
                print(f"[Capture] Waiting for IRA setup... tick={tick}")

        if setup_done[0]:
            npc_ready = True
            print(f"[Capture] {num_chars} NPC(s) loaded after {tick} ticks!")
        else:
            print(f"[Capture] WARNING: IRA timeout at {tick} ticks (NPC may still work)")
    else:
        print("[Capture] WARNING: Failed to load NPC config")

    for _ in range(30):
        simulation_app.update()

    # 关键修复：必须启动 IRA 数据生成才能激活 NPC GoTo 命令执行
    if npc_ready:
        async def _run_ira():
            await sim_manager.run_data_generation_async(will_wait_until_complete=True)
        from omni.kit.async_engine import run_coroutine
        _ira_task = run_coroutine(_run_ira())
        print("[Capture] IRA data generation started (NPC GoTo commands active)")
    else:
        _ira_task = None
else:
    _ira_task = None
    if args.no_npc:
        print("[Capture] NPC disabled (--no_npc)")
    else:
        print("[Capture] No NPC models found")
    print(f"[Capture] Loading scene: {SCENE_USD}")
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().subLayerPaths.append(SCENE_USD)
    for _ in range(30):
        simulation_app.update()

# ============================================================================
# PHASE 2: 场景探测 + 创建双目相机
# ============================================================================
print("[Capture] Creating stereo cameras...")
stage = omni.usd.get_context().get_stage()
for _ in range(5):
    simulation_app.update()

stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
stage_scale = 1.0 / stage_mpu
print(f"[Capture] Stage metersPerUnit={stage_mpu}, scale={stage_scale}")

# --- 场景探测：扫描货架位置，自动推荐相机坐标 ---
scene_info = probe_scene_objects(stage, stage_mpu)
for prefix, info in scene_info.items():
    print(f"[Probe] {prefix}: {info['count']} prims, "
          f"X=[{info['bbox_min'][0]:.1f}, {info['bbox_max'][0]:.1f}]m, "
          f"Y=[{info['bbox_min'][1]:.1f}, {info['bbox_max'][1]:.1f}]m, "
          f"Z=[{info['bbox_min'][2]:.1f}, {info['bbox_max'][2]:.1f}]m")

# 相机位置：默认 (0, 0)，与 test_stereo_pair.py 一致
# 仅在用户显式传入 --camera_x/--camera_y 时才覆盖
cam_x = args.camera_x
cam_y = args.camera_y
if scene_info:
    suggested_x, suggested_y = suggest_camera_position(scene_info)
    print(f"[Probe] Suggested position: ({suggested_x:.1f}, {suggested_y:.1f})m (use --camera_x/--camera_y to override)")

# StereoRig Xform
height_m = args.camera_height
rig_path = "/World/StereoRig"
rig_xform = UsdGeom.Xform.Define(stage, rig_path)
rig_pos = Gf.Vec3d(
    float(cam_x * stage_scale),
    float(cam_y * stage_scale),
    float(height_m * stage_scale),
)
rig_xform.AddTranslateOp().Set(rig_pos)

half_baseline = (BASELINE_M / 2.0) * stage_scale

left_cam_path = f"{rig_path}/Left"
right_cam_path = f"{rig_path}/Right"

# 相机朝向：+X → -Z (俯拍), euler (0, 0, 90)
import isaacsim.core.utils.numpy.rotations as rot_utils
cam_euler = np.array([0.0, 0.0, 90.0])
cam_quat = rot_utils.euler_angles_to_quats(cam_euler, degrees=True)

for cam_path, y_offset in [(left_cam_path, half_baseline), (right_cam_path, -half_baseline)]:
    cam_prim = UsdGeom.Camera.Define(stage, cam_path)
    xf = UsdGeom.Xformable(cam_prim.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, float(y_offset), 0.0))
    xf.AddOrientOp().Set(Gf.Quatf(
        float(cam_quat[0]),
        Gf.Vec3f(float(cam_quat[1]), float(cam_quat[2]), float(cam_quat[3])),
    ))
    cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(float(0.1 * stage_scale), float(100.0 * stage_scale)))
    set_fisheye_on_prim(stage, cam_path)

create_camera_visual(stage, left_cam_path, "left_cam", [255, 50, 50])
create_camera_visual(stage, right_cam_path, "right_cam", [50, 80, 255])

bar_prim = UsdGeom.Cube.Define(stage, f"{rig_path}/baseline_bar")
bar_prim.GetSizeAttr().Set(1.0)
bar_prim.AddScaleOp().Set(Gf.Vec3f(0.01, float(BASELINE_M * stage_scale), 0.01))
bar_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.8, 0.2)])

for _ in range(5):
    simulation_app.update()

print(f"[Capture] StereoRig at ({cam_x:.1f}, {cam_y:.1f}, {height_m})m, baseline={BASELINE_M*1000:.0f}mm")

# ============================================================================
# PHASE 3: Replicator annotators + 体素系统初始化
# ============================================================================
print("[Capture] Setting up replicator annotators...")
rp_left = rep.create.render_product(left_cam_path, resolution=(CAM_W, CAM_H))
rp_right = rep.create.render_product(right_cam_path, resolution=(CAM_W, CAM_H))

annot_left = rep.AnnotatorRegistry.get_annotator("rgb")
annot_left.attach([rp_left.path])
annot_right = rep.AnnotatorRegistry.get_annotator("rgb")
annot_right.attach([rp_right.path])

for _ in range(5):
    simulation_app.update()

print("[Capture] Initializing voxel grid...")
voxel_template = VoxelGrid()
print(f"[Capture] VoxelGrid: {voxel_template.NX}x{voxel_template.NY}x{voxel_template.NZ} "
      f"= {voxel_template.NX * voxel_template.NY * voxel_template.NZ:,} voxels")

# PhysX scene query interface
from omni.physx import get_physx_scene_query_interface
physx_sqi = get_physx_scene_query_interface()

# ============================================================================
# Output directories
# ============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
LEFT_DIR = os.path.join(OUTPUT_DIR, "left")
RIGHT_DIR = os.path.join(OUTPUT_DIR, "right")
VOXEL_DIR = os.path.join(OUTPUT_DIR, "voxel")
META_DIR = os.path.join(OUTPUT_DIR, "meta")
for d in [LEFT_DIR, RIGHT_DIR, VOXEL_DIR, META_DIR]:
    os.makedirs(d, exist_ok=True)

# 清空输出目录，每次运行从头开始
import shutil
for d in [LEFT_DIR, RIGHT_DIR, VOXEL_DIR, META_DIR]:
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
# 清理根目录下的旧配置文件
for old_file in ["calibration.json", "voxel_config.json", "instance_map.json"]:
    old_path = os.path.join(OUTPUT_DIR, old_file)
    if os.path.isfile(old_path):
        os.remove(old_path)
start_frame_id = 0
print(f"[Capture] Output directory cleaned: {OUTPUT_DIR}")

# 保存标定信息
calib_path = os.path.join(OUTPUT_DIR, "calibration.json")
with open(calib_path, "w") as f:
    json.dump({
        "sensor": "SC132GS (simulated)",
        "projection": "ftheta_equidistant",
        "resolution": [CAM_W, CAM_H],
        "fx": fx, "fy": fx, "cx": cx, "cy": cy,
        "k1": K1_EQUIDISTANT,
        "max_fov_deg": DIAG_FOV_DEG,
        "baseline_m": BASELINE_M,
        "baseline_direction": "Y-axis (left=+Y, right=-Y)",
    }, f, indent=2)

# 保存体素配置
voxel_config_path = os.path.join(OUTPUT_DIR, "voxel_config.json")
with open(voxel_config_path, "w") as f:
    json.dump(VoxelGrid.get_config(), f, indent=2)

print(f"[Capture] Output: {OUTPUT_DIR}")


# ============================================================================
# 读取相机世界位姿
# ============================================================================
def get_rig_world_pose(stage, rig_path):
    """获取 StereoRig 的世界位置和 yaw 角。

    Returns:
        cam_pos: (3,) 世界坐标 [x, y, z]（米）
        cam_yaw: float, 绕 Z 轴偏航角（弧度）
    """
    rig_prim = stage.GetPrimAtPath(rig_path)
    if not rig_prim.IsValid():
        return np.array([0.0, 0.0, args.camera_height]), 0.0

    xformable = UsdGeom.Xformable(rig_prim)
    world_xform = xformable.ComputeLocalToWorldTransform(0.0)
    translate = world_xform.ExtractTranslation()

    # 转换为米
    cam_pos = np.array([
        translate[0] * stage_mpu,
        translate[1] * stage_mpu,
        translate[2] * stage_mpu,
    ])

    # 提取 yaw（绕 Z 轴旋转）
    rotation = world_xform.ExtractRotationMatrix()
    # yaw = atan2(R[1][0], R[0][0])
    cam_yaw = np.arctan2(rotation[1][0], rotation[0][0])

    return cam_pos, cam_yaw


# ============================================================================
# 同步采集函数
# ============================================================================
def capture_frame(frame_id):
    """冻结 → 拍照 + 体素查询 → 恢复。

    Returns:
        bool: 是否成功采集
    """
    timeline = omni.timeline.get_timeline_interface()

    # 1. 冻结世界
    timeline.pause()

    # 2. 刷新渲染管线（pause 后仍需渲染当前帧）
    for _ in range(3):
        simulation_app.update()

    # 3. 采集双目图像
    data_l = annot_left.get_data()
    data_r = annot_right.get_data()

    if data_l is None or data_r is None:
        timeline.play()
        return False

    if not isinstance(data_l, np.ndarray) or not isinstance(data_r, np.ndarray):
        timeline.play()
        return False

    if data_l.size == 0 or data_r.size == 0:
        timeline.play()
        return False

    rgb_l = data_l[:, :, :3] if data_l.ndim == 3 else data_l
    rgb_r = data_r[:, :, :3] if data_r.ndim == 3 else data_r

    if rgb_l.mean() < 1.0 or rgb_r.mean() < 1.0:
        timeline.play()
        return False

    # 4. 读取相机世界位姿
    cam_pos, cam_yaw = get_rig_world_pose(stage, rig_path)

    # 5. 体素填充（注意：PhysX 在 stage 单位下工作，需要 meters→stage 换算）
    vg = VoxelGrid()
    world_centers_flat = vg.get_world_centers_flat(cam_pos, cam_yaw)
    fill_voxel_grid(stage, vg, world_centers_flat, physx_sqi, stage_scale)

    # 5b. NPC 人物检测（IRA 动画角色无 PhysX 碰撞体，用 USD Xform 补充）
    npc_positions = get_npc_world_positions(stage, stage_mpu)
    npc_voxel_count = stamp_npc_voxels(vg, cam_pos, cam_yaw, npc_positions)

    # 6. 异步保存
    frame_str = f"frame_{frame_id:06d}"
    async_save_image(os.path.join(LEFT_DIR, f"{frame_str}.png"), rgb_l)
    async_save_image(os.path.join(RIGHT_DIR, f"{frame_str}.png"), rgb_r)
    async_save_voxel(os.path.join(VOXEL_DIR, frame_str), vg.semantic, vg.instance)

    # 帧元数据
    meta = {
        "frame_id": frame_id,
        "camera_pos": cam_pos.tolist(),
        "camera_yaw_rad": float(cam_yaw),
        "camera_height_m": float(cam_pos[2]),
        "voxel_origin_world": [float(cam_pos[0]), float(cam_pos[1]), 0.0],
        "voxel_size": VoxelGrid.VOXEL_SIZE,
        "voxel_shape": [VoxelGrid.NX, VoxelGrid.NY, VoxelGrid.NZ],
        "z_ground_index": VoxelGrid.Z_GROUND_INDEX,
        "stats": vg.stats() if frame_id % 10 == 0 else {},
    }
    async_save_json(os.path.join(META_DIR, f"{frame_str}.json"), meta)

    if frame_id % 5 == 0:
        occ = int(np.sum((vg.semantic > 0) & (vg.semantic < UNOBSERVED)))
        person_ct = int(np.sum(vg.semantic == PERSON))
        print(f"  Frame {frame_id}: L_mean={rgb_l.mean():.1f}, R_mean={rgb_r.mean():.1f}, "
              f"occupied={occ}, person={person_ct}, npc_pos={len(npc_positions)}, "
              f"pos=({cam_pos[0]:.1f},{cam_pos[1]:.1f},{cam_pos[2]:.1f})")

    # 7. 恢复世界
    timeline.play()
    return True


# ============================================================================
# PHASE 4: 主循环
# ============================================================================
frame_id = start_frame_id
sim_step = 0
warmup_steps = 30

timeline = omni.timeline.get_timeline_interface()

if args.headless:
    # === Headless 模式 ===
    print(f"[Capture] Headless: capturing {args.num_frames} frames, interval={args.capture_interval}...")
    timeline.play()

    for _ in range(warmup_steps):
        simulation_app.update()

    captured = 0
    max_steps = args.num_frames * args.capture_interval * 3
    step = 0

    while captured < args.num_frames and step < max_steps and not simulation_app.is_exiting():
        simulation_app.update()
        step += 1

        if step % args.capture_interval == 0:
            if capture_frame(frame_id):
                frame_id += 1
                captured += 1

    timeline.stop()
    wait_pending_saves()
    print(f"[Capture] Headless done: {captured} frames captured.")

else:
    # === GUI 模式（自动 play，不需要手动点）===
    print("\n" + "=" * 60)
    print("[Capture] GUI mode (auto-play)")
    print(f"  - StereoRig: /World/StereoRig (draggable)")
    print(f"  - Camera: ({cam_x:.1f}, {cam_y:.1f}, {height_m})m, {BASELINE_M*1000:.0f}mm baseline")
    if npc_ready:
        print(f"  - {num_chars} NPC(s) ready")
    print(f"  - Capture every {args.capture_interval} steps with timeline freeze")
    print(f"  - Target: {args.num_frames} frames")
    print(f"  - Output: {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    timeline.play()

    for _ in range(warmup_steps):
        simulation_app.update()

    captured = 0
    sim_step = 0

    while simulation_app.is_running():
        simulation_app.update()
        sim_step += 1

        if captured >= args.num_frames:
            continue  # 达标后仍保持 GUI 运行，但不再采集

        if sim_step % args.capture_interval == 0:
            if capture_frame(frame_id):
                frame_id += 1
                captured += 1

# ============================================================================
# 收尾
# ============================================================================
wait_pending_saves()

# 保存 instance 映射
instance_map_path = os.path.join(OUTPUT_DIR, "instance_map.json")
with open(instance_map_path, "w") as f:
    json.dump({str(v): k for k, v in _instance_map.items()}, f, indent=2)

total = frame_id - start_frame_id
print(f"\n[Capture] Done! {total} frames saved (frame {start_frame_id}~{frame_id - 1})")
print(f"  Left:     {LEFT_DIR}")
print(f"  Right:    {RIGHT_DIR}")
print(f"  Voxel:    {VOXEL_DIR}")
print(f"  Meta:     {META_DIR}")
print(f"  Calib:    {calib_path}")
print(f"  Voxel cfg: {voxel_config_path}")
print(f"  Instance:  {instance_map_path}")

_save_pool.shutdown(wait=True)
simulation_app.close()
