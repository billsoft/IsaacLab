"""PhysX 体素查询 + NPC 圆柱标记
=================================
- fill_voxel_grid(): 粗查+细查两阶段 overlap_box
- get_npc_world_positions(): 从 USD 读取 NPC 位置
- stamp_npc_voxels(): 圆柱近似标记 PERSON 体素，返回占用索引映射
"""

import os
import re
import sys

import numpy as np

from .constants import NPC_HEIGHT_M, NPC_RADIUS_M

# 复用现有 semantic_classes
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from semantic_classes import FREE, OTHER_GROUND, PERSON, UNOBSERVED, lookup_class_id


# ===========================================================================
# prim 名称提取
# ===========================================================================
def get_object_type_from_prim_path(stage, prim_path: str) -> str:
    """从 PhysX hit 的 rigid_body prim path 提取物体类型名。"""
    if "/Character" in prim_path:
        return "NPC:person"

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return ""

    current = prim
    while current.IsValid():
        name = current.GetName()
        if name in ("Collider", "CollisionMesh", "CollisionPlane", "Mesh",
                     "Shape", "Physics", "Collision", "collision") or name.startswith("Section"):
            current = current.GetParent()
            continue
        if name in ("", "World", "Environment", "Warehouse"):
            current = current.GetParent()
            continue
        if current.GetTypeName() in ("Scope",):
            current = current.GetParent()
            continue
        return name

    return prim_path.rsplit("/", 1)[-1] if "/" in prim_path else prim_path


def clean_object_name(raw_name: str) -> str:
    """清理物体名：去掉 SM_/S_ 前缀和尾部 _数字 后缀。"""
    name = raw_name
    for prefix in ("SM_", "S_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = re.sub(r"_\d+$", "", name)
    return name


# ===========================================================================
# PhysX 体素填充
# ===========================================================================
def fill_voxel_grid(stage, voxel_grid, world_centers_flat, physx_sqi,
                    meters_to_stage: float, instance_registry=None,
                    frame_id: str = ""):
    """用 PhysX overlap_box 填充体素网格。

    粗查+细查两阶段策略：
      1. 粗查：5x5x5 体素块做一次 overlap，无 hit 则整块标 FREE
      2. 细查：有 hit 的块内逐个体素查询

    Args:
        stage: USD stage
        voxel_grid: VoxelGridV2 实例（会被修改）
        world_centers_flat: (N, 3) 世界坐标（米）
        physx_sqi: PhysX scene query interface
        meters_to_stage: 米→stage 单位换算系数
        instance_registry: InstanceRegistry 或 None（None 时退化为旧行为）
        frame_id: 当前帧 ID（用于 instance_registry 追踪）
    """
    import carb

    NX, NY, NZ = voxel_grid.NX, voxel_grid.NY, voxel_grid.NZ
    VOXEL_SIZE = voxel_grid.VOXEL_SIZE

    world_centers_stage = (world_centers_flat * meters_to_stage).reshape(NX, NY, NZ, 3)

    # 地面层边界效应修复
    Z_GI = voxel_grid.Z_GROUND_INDEX
    GROUND_Z_NUDGE = 0.002 * meters_to_stage
    world_centers_stage[:, :, Z_GI, 2] -= GROUND_Z_NUDGE

    COARSE = 5
    fine_half = (VOXEL_SIZE / 2.0) * meters_to_stage
    identity_rot = carb.Float4(0.0, 0.0, 0.0, 1.0)

    coarse_skipped = 0
    occupied_count = 0
    _detected_paths = set()

    for ci in range(0, NX, COARSE):
        for cj in range(0, NY, COARSE):
            for ck in range(0, NZ, COARSE):
                ei = min(ci + COARSE, NX)
                ej = min(cj + COARSE, NY)
                ek = min(ck + COARSE, NZ)
                block_center = world_centers_stage[ci:ei, cj:ej, ck:ek].mean(axis=(0, 1, 2))

                bh_x = (ei - ci) * VOXEL_SIZE * meters_to_stage / 2.0
                bh_y = (ej - cj) * VOXEL_SIZE * meters_to_stage / 2.0
                bh_z = (ek - ck) * VOXEL_SIZE * meters_to_stage / 2.0

                coarse_hits = []

                def on_coarse_hit(hit):
                    coarse_hits.append(hit.rigid_body)
                    return True

                physx_sqi.overlap_box(
                    carb.Float3(bh_x, bh_y, bh_z),
                    carb.Float3(float(block_center[0]), float(block_center[1]),
                                float(block_center[2])),
                    identity_rot, on_coarse_hit, False,
                )

                if not coarse_hits:
                    voxel_grid.semantic[ci:ei, cj:ej, ck:ek] = FREE
                    coarse_skipped += (ei - ci) * (ej - cj) * (ek - ck)
                    continue

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
                                carb.Float3(float(center[0]), float(center[1]),
                                            float(center[2])),
                                identity_rot, on_fine_hit, False,
                            )

                            if not fine_hits:
                                voxel_grid.semantic[i, j, k] = FREE
                                voxel_grid.instance[i, j, k] = 0
                            else:
                                hit_path = fine_hits[0]
                                obj_type = get_object_type_from_prim_path(stage, hit_path)
                                cname = clean_object_name(obj_type)
                                class_id = lookup_class_id(cname)
                                voxel_grid.semantic[i, j, k] = class_id

                                if instance_registry is not None:
                                    voxel_grid.instance[i, j, k] = \
                                        instance_registry.get_or_assign(
                                            hit_path, class_id, frame_id)
                                _detected_paths.add(hit_path)
                                occupied_count += 1

    # 地面层兜底
    if Z_GI > 0:
        underground = voxel_grid.semantic[:, :, Z_GI - 1]
        ground = voxel_grid.semantic[:, :, Z_GI]
        miss_mask = (ground == FREE) & (underground > 0) & (underground != UNOBSERVED)
        patched = int(miss_mask.sum())
        if patched > 0:
            voxel_grid.semantic[:, :, Z_GI][miss_mask] = OTHER_GROUND
            occupied_count += patched
            print(f"    Ground layer patch: filled {patched} missing ground voxels (z={Z_GI})")

    free_count = int(np.sum(voxel_grid.semantic == FREE))
    unobs_count = int(np.sum(voxel_grid.semantic == UNOBSERVED))
    print(f"    Voxel fill: {occupied_count} occupied, {free_count} free, "
          f"{unobs_count} unobserved, {coarse_skipped} coarse-skipped")

    if _detected_paths:
        print(f"    Detected {len(_detected_paths)} unique collision prims:")
        for p in sorted(_detected_paths):
            obj_type = get_object_type_from_prim_path(stage, p)
            cname = clean_object_name(obj_type)
            cid = lookup_class_id(cname)
            print(f"      [{cid:2d}] {cname:<25s} (raw={obj_type}) <- {p}")


# ===========================================================================
# NPC 位置检测
# ===========================================================================
def _find_skelroot(character_prim):
    """在 Character 根 prim 下查找 SkelRoot 子节点。

    IRA NPC 结构:
      /World/Characters/Character_XX  (根 Xform - 静态生成位置，不更新)
        └── <model_name>              (SkelRoot - 由 omni.anim.people 动画驱动，位置持续更新)

    Returns:
        SkelRoot prim（若找到），否则返回原始 prim 作为回退。
    """
    from pxr import Usd
    for prim in Usd.PrimRange(character_prim):
        if prim.GetTypeName() == "SkelRoot":
            return prim
    return character_prim


def get_npc_world_positions(stage, stage_mpu: float) -> list[np.ndarray]:
    """获取所有 NPC Character 的世界位置（米）。

    IRA 动画 NPC 无 PhysX 刚体，只能从 USD 读取。
    尝试多种方式获取位置以提高兼容性。

    Returns:
        list of np.array(3,): 每个 NPC 脚底世界坐标 [x, y, z]
    """
    from pxr import UsdGeom, Gf, Usd

    positions = []
    chars_prim = stage.GetPrimAtPath("/World/Characters")
    if not chars_prim.IsValid():
        chars_prim = stage.GetPrimAtPath("/Root")
        if not chars_prim.IsValid():
            return positions

    time_code = Usd.TimeCode.Default()

    for child in chars_prim.GetChildren():
        name = child.GetName()
        if not name.startswith("Character"):
            continue
        # IRA 动画更新的是 SkelRoot 子节点，不是根 Xform
        target_prim = _find_skelroot(child)
        xformable = UsdGeom.Xformable(target_prim)
        try:
            # 方法 1: ComputeLocalToWorldTransform (标准方式)
            xf = xformable.ComputeLocalToWorldTransform(time_code)
            pos = xf.ExtractTranslation()
        except Exception:
            # 方法 2: 使用 BBoxCache 获取世界包围盒 (更准确，包含骨骼)
            try:
                bbox_cache = UsdGeom.BBoxCache(
                    time_code,
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                    False
                )
                bounds = xformable.ComputeWorldBound(time_code, UsdGeom.Tokens.default_)
                range_vec = bounds.GetRange().GetSize()
                pos = Gf.Vec3d(
                    bounds.GetRange().GetMin()[0] + range_vec[0] / 2,
                    bounds.GetRange().GetMin()[1] + range_vec[1] / 2,
                    bounds.GetRange().GetMin()[2]
                )
            except Exception as e:
                print(f"[voxel] Warning: 无法获取 {name} 位置: {e}")
                continue

        positions.append(np.array([
            pos[0] * stage_mpu,
            pos[1] * stage_mpu,
            pos[2] * stage_mpu,
        ]))
    return positions


def get_npc_world_orientations(stage, stage_mpu: float) -> list[np.ndarray]:
    """获取所有 NPC Character 的世界系四元数朝向。

    Returns:
        list of np.array(4,): 每个 NPC 的四元数 (x, y, z, w)
    """
    from pxr import UsdGeom

    orientations = []
    chars_prim = stage.GetPrimAtPath("/World/Characters")
    if not chars_prim.IsValid():
        return orientations

    for child in chars_prim.GetChildren():
        name = child.GetName()
        if not name.startswith("Character"):
            continue
        target_prim = _find_skelroot(child)
        xformable = UsdGeom.Xformable(target_prim)
        try:
            xf = xformable.ComputeLocalToWorldTransform(0)
            rot = xf.ExtractRotationQuat()
            real = rot.GetReal()
            imag = rot.GetImaginary()
            orientations.append(np.array([imag[0], imag[1], imag[2], real]))
        except Exception:
            orientations.append(np.array([0.0, 0.0, 0.0, 1.0]))
    return orientations


def _get_npc_name(index: int) -> str:
    """IRA NPC 命名规则。"""
    if index == 0:
        return "Character"
    elif index < 10:
        return f"Character_0{index}"
    else:
        return f"Character_{index}"


def _get_npc_prim_path(index: int) -> str:
    """NPC prim 路径。"""
    return f"/World/Characters/{_get_npc_name(index)}"


# ===========================================================================
# NPC 圆柱体素标记
# ===========================================================================
def stamp_npc_voxels(voxel_grid, cam_pos: np.ndarray, cam_yaw: float,
                     npc_world_positions: list[np.ndarray],
                     instance_registry=None,
                     frame_id: str = "",
                     ) -> dict[int, list[tuple[int, int, int]]]:
    """将 NPC 世界坐标投影到体素网格，标记为 PERSON。

    用圆柱近似：半径 NPC_RADIUS_M, 高度 NPC_HEIGHT_M。

    Returns:
        {npc_index: [(i, j, k), ...]} 每个 NPC 占据的体素索引列表
    """
    if not npc_world_positions:
        return {}

    occupied_map: dict[int, list[tuple[int, int, int]]] = {}

    for idx, npc_pos in enumerate(npc_world_positions):
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
        vx_min = max(0, int(np.floor(
            (local[0] - NPC_RADIUS_M) / voxel_grid.VOXEL_SIZE + voxel_grid.CENTER_X)))
        vx_max = min(voxel_grid.NX, int(np.ceil(
            (local[0] + NPC_RADIUS_M) / voxel_grid.VOXEL_SIZE + voxel_grid.CENTER_X)))
        vy_min = max(0, int(np.floor(
            (local[1] - NPC_RADIUS_M) / voxel_grid.VOXEL_SIZE + voxel_grid.CENTER_Y)))
        vy_max = min(voxel_grid.NY, int(np.ceil(
            (local[1] + NPC_RADIUS_M) / voxel_grid.VOXEL_SIZE + voxel_grid.CENTER_Y)))
        vz_min = max(0, int(np.floor(
            local[2] / voxel_grid.VOXEL_SIZE + voxel_grid.Z_GROUND_INDEX)))
        vz_max = min(voxel_grid.NZ, int(np.ceil(
            (local[2] + NPC_HEIGHT_M) / voxel_grid.VOXEL_SIZE + voxel_grid.Z_GROUND_INDEX)))

        # 实例 ID
        npc_prim = _get_npc_prim_path(idx)
        if instance_registry is not None:
            npc_instance_id = instance_registry.get_or_assign(
                npc_prim, PERSON, frame_id)
        else:
            npc_instance_id = 0

        indices = []
        for i in range(vx_min, vx_max):
            for j in range(vy_min, vy_max):
                cx_v = (i - voxel_grid.CENTER_X + 0.5) * voxel_grid.VOXEL_SIZE
                cy_v = (j - voxel_grid.CENTER_Y + 0.5) * voxel_grid.VOXEL_SIZE
                dist_sq = (cx_v - local[0]) ** 2 + (cy_v - local[1]) ** 2
                if dist_sq <= NPC_RADIUS_M ** 2:
                    for k in range(vz_min, vz_max):
                        voxel_grid.semantic[i, j, k] = PERSON
                        voxel_grid.instance[i, j, k] = npc_instance_id
                        indices.append((i, j, k))

        occupied_map[idx] = indices

    return occupied_map
