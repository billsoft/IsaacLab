"""Flow 网格生成
================
将 NPC 速度填入体素网格的 flow 和 flow_mask 字段。

flow: (72,60,32,2) float16 — 体素局部坐标系下的 (vx, vy)
flow_mask: (72,60,32) uint8  — 1=动态物体（参与 loss），0=静态/空气
"""

import numpy as np

from .npc_tracker import NPCTracker


def fill_flow(voxel_grid, npc_tracker: NPCTracker, cam_yaw: float,
              occupied_map: dict[int, list[tuple[int, int, int]]],
              sim_step: int, capture_interval: int):
    """将 NPC 速度写入 voxel_grid.flow 和 flow_mask。

    Args:
        voxel_grid: VoxelGridV2 实例（会修改 .flow 和 .flow_mask）
        npc_tracker: NPCTracker 实例
        cam_yaw: 相机偏航角（弧度）
        occupied_map: stamp_npc_voxels 返回的 {npc_idx: [(i,j,k),...]}
        sim_step: 当前仿真步
        capture_interval: 采集间隔
    """
    for npc_idx, indices in occupied_map.items():
        if not indices:
            continue

        # 世界系速度
        v_world = npc_tracker.get_velocity(npc_idx, sim_step, capture_interval)

        # 旋转到体素局部坐标系
        vx_local, vy_local = _rotate_velocity_to_local(
            v_world[0], v_world[1], cam_yaw)

        # 写入所有占据体素
        vx_f16 = np.float16(vx_local)
        vy_f16 = np.float16(vy_local)
        for (i, j, k) in indices:
            voxel_grid.flow[i, j, k, 0] = vx_f16
            voxel_grid.flow[i, j, k, 1] = vy_f16
            voxel_grid.flow_mask[i, j, k] = 1


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
