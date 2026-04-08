"""Flow 网格生成
================
将 NPC 速度、航向角、角速度填入体素网格字段。

flow:        (72,60,32,2) float16 — 体素局部坐标系下的 (vx, vy)
flow_mask:   (72,60,32)   uint8   — 1=动态物体（参与 loss），0=静态/空气
orientation: (72,60,32)   float16 — NPC 航向角（局部坐标系，弧度，[-π,π]）
angular_vel: (72,60,32)   float16 — NPC 角速度 ωz（rad/s）
"""

import math

import numpy as np

from .npc_tracker import NPCTracker


def fill_flow(voxel_grid, npc_tracker: NPCTracker, cam_yaw: float,
              occupied_map: dict[int, list[tuple[int, int, int]]],
              sim_step: int, capture_interval: int):
    """将 NPC 速度/方向/角速度写入 voxel_grid 各字段。

    Args:
        voxel_grid: VoxelGridV2 实例（修改 .flow/.flow_mask/.orientation/.angular_vel）
        npc_tracker: NPCTracker 实例
        cam_yaw: 相机偏航角（弧度）
        occupied_map: stamp_npc_voxels 返回的 {npc_idx: [(i,j,k),...]}
        sim_step: 当前仿真步
        capture_interval: 采集间隔
    """
    for npc_idx, indices in occupied_map.items():
        if not indices:
            continue

        # 世界系速度 → 体素局部系
        v_world = npc_tracker.get_velocity(npc_idx, sim_step, capture_interval)
        vx_local, vy_local = _rotate_velocity_to_local(
            v_world[0], v_world[1], cam_yaw)

        # 航向角：世界系 → 体素局部系（减去相机偏航，归一化到 [-π,π]）
        heading_world = npc_tracker.get_heading_rad(npc_idx)
        heading_local = heading_world - cam_yaw
        heading_local = math.atan2(math.sin(heading_local), math.cos(heading_local))

        # 角速度 ωz（绕 Z 轴，世界系与局部系相同，因为 yaw 只影响 XY）
        ang_vel_world = npc_tracker.get_angular_velocity(npc_idx)
        omega_z = float(ang_vel_world[2]) if len(ang_vel_world) > 2 else 0.0

        # 转 float16 常量
        vx_f16 = np.float16(vx_local)
        vy_f16 = np.float16(vy_local)
        orient_f16 = np.float16(heading_local)
        angvel_f16 = np.float16(omega_z)

        # 写入所有占据体素
        for (i, j, k) in indices:
            voxel_grid.flow[i, j, k, 0] = vx_f16
            voxel_grid.flow[i, j, k, 1] = vy_f16
            voxel_grid.flow_mask[i, j, k] = 1
            voxel_grid.orientation[i, j, k] = orient_f16
            voxel_grid.angular_vel[i, j, k] = angvel_f16


def _rotate_velocity_to_local(vx_world: float, vy_world: float,
                               cam_yaw: float) -> tuple[float, float]:
    """世界系速度 → 体素局部坐标系速度。

    v_local = R_z(-yaw) @ v_world
    """
    if abs(cam_yaw) < 1e-6:
        return vx_world, vy_world

    cos_y = np.cos(-cam_yaw)
    sin_y = np.sin(-cam_yaw)
    vx_local = vx_world * cos_y - vy_world * sin_y
    vy_local = vx_world * sin_y + vy_world * cos_y
    return float(vx_local), float(vy_local)
