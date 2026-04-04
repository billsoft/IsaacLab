"""NPC 位置/朝向历史追踪 + 速度/角速度计算
==============================================
IRA 动画 NPC 无 PhysX 刚体，无法用 get_linear_velocity() / get_angular_velocity()。
通过记录每个采集帧的 NPC 世界位置和四元数朝向，用差分计算线速度和角速度。
"""

from __future__ import annotations

import json
import os
from collections import deque

import numpy as np

from .constants import SIM_FPS
from .voxel_filling import get_npc_world_orientations, get_npc_world_positions


# ===========================================================================
# 四元数辅助函数（(x, y, z, w) 格式）
# ===========================================================================

def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """四元数共轭：(x,y,z,w) → (-x,-y,-z,w)"""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton 四元数乘法，(x,y,z,w) 格式。"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def _angular_velocity_from_quats(q_curr: np.ndarray, q_prev: np.ndarray,
                                  dt: float) -> np.ndarray:
    """从连续两帧四元数差分计算角速度 (rad/s)。

    dq = q_curr * conj(q_prev)
    ω = 2 * dq.xyz / dt  （小角度近似）

    Returns:
        (3,) 角速度向量 [ωx, ωy, ωz]，世界坐标系
    """
    dq = _quat_multiply(q_curr, _quat_conjugate(q_prev))
    # 规范化双重覆盖：q 和 -q 表示相同旋转，取 w > 0 的一侧
    if dq[3] < 0:
        dq = -dq
    return 2.0 * dq[:3] / dt


class NPCTracker:
    """跟踪 NPC 位置和朝向历史，计算速度和角速度。"""

    def __init__(self, sim_fps: float = SIM_FPS, max_history: int = 200):
        self._sim_fps = sim_fps
        self._max_history = max_history
        # {npc_index: deque of {"sim_step": int, "position": np.array(3), "timestamp_sec": float}}
        self._history: dict[int, deque] = {}
        # {npc_index: deque of {"sim_step": int, "quaternion": np.array(4), "timestamp_sec": float}}
        self._orientations: dict[int, deque] = {}
        # 全局记录（用于后处理轨迹）
        self._global_log: list[dict] = []

    def update(self, stage, stage_mpu: float, sim_step: int):
        """读取所有 NPC 世界位置和朝向并存入历史。

        在每个采集帧调用。

        Returns:
            (positions, orientations): 位置列表和四元数列表
        """
        positions = get_npc_world_positions(stage, stage_mpu)
        orientations = get_npc_world_orientations(stage, stage_mpu)
        timestamp = sim_step / self._sim_fps

        record = {
            "sim_step": sim_step,
            "timestamp_sec": timestamp,
            "positions": [],
            "orientations": [],
        }

        for idx, pos in enumerate(positions):
            # 位置历史
            if idx not in self._history:
                self._history[idx] = deque(maxlen=self._max_history)
            self._history[idx].append({
                "sim_step": sim_step,
                "position": pos.copy(),
                "timestamp_sec": timestamp,
            })
            record["positions"].append(pos.tolist())

            # 朝向历史
            quat = orientations[idx] if idx < len(orientations) else np.array([0.0, 0.0, 0.0, 1.0])
            if idx not in self._orientations:
                self._orientations[idx] = deque(maxlen=self._max_history)
            self._orientations[idx].append({
                "sim_step": sim_step,
                "quaternion": quat.copy(),
                "timestamp_sec": timestamp,
            })
            record["orientations"].append(quat.tolist())

        self._global_log.append(record)
        return positions, orientations

    def get_velocity(self, npc_index: int, current_step: int,
                     capture_interval: int) -> np.ndarray:
        """计算 NPC 线速度（世界坐标系 m/s）。

        v = (pos[current] - pos[previous]) / dt
        """
        history = self._history.get(npc_index)
        if not history or len(history) < 2:
            return np.zeros(3)

        current = history[-1]
        prev = history[-2]

        dt = (current["sim_step"] - prev["sim_step"]) / self._sim_fps
        if dt < 1e-6:
            return np.zeros(3)

        return (current["position"] - prev["position"]) / dt

    def get_angular_velocity(self, npc_index: int) -> np.ndarray:
        """计算 NPC 角速度（世界坐标系 rad/s）。

        从连续两帧四元数差分：ω = 2 * (q_curr * conj(q_prev)).xyz / dt

        Returns:
            (3,) 角速度向量 [ωx, ωy, ωz]
        """
        orient_history = self._orientations.get(npc_index)
        if not orient_history or len(orient_history) < 2:
            return np.zeros(3)

        current = orient_history[-1]
        prev = orient_history[-2]

        dt = (current["sim_step"] - prev["sim_step"]) / self._sim_fps
        if dt < 1e-6:
            return np.zeros(3)

        return _angular_velocity_from_quats(current["quaternion"], prev["quaternion"], dt)

    def get_heading_rad(self, npc_index: int) -> float:
        """从最近两个位置计算运动朝向角（弧度，XY 平面）。

        heading = atan2(dy, dx), 0 = +X 方向
        """
        history = self._history.get(npc_index)
        if not history or len(history) < 2:
            return 0.0

        current = history[-1]["position"]
        prev = history[-2]["position"]
        dx = current[0] - prev[0]
        dy = current[1] - prev[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        return float(np.arctan2(dy, dx))

    def get_current_positions(self) -> dict[int, np.ndarray]:
        """返回所有 NPC 当前位置。"""
        result = {}
        for idx, history in self._history.items():
            if history:
                result[idx] = history[-1]["position"].copy()
        return result

    @property
    def num_tracked(self) -> int:
        return len(self._history)

    def save_history(self, path: str):
        """保存完整位置和朝向历史（供后处理轨迹脚本使用）。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "sim_fps": self._sim_fps,
                "num_npcs": len(self._history),
                "records": self._global_log,
            }, f, indent=2)
        print(f"[npc_tracker] Saved {len(self._global_log)} records → {path}")

    @staticmethod
    def load_history(path: str) -> dict:
        """加载位置历史（用于后处理）。"""
        with open(path) as f:
            return json.load(f)
