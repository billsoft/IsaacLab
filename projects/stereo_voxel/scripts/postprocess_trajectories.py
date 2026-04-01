"""轨迹后处理：从 NPC 位置历史回填 past/future positions
=========================================================
不需要 Isaac Sim，用任意 Python 运行：
    python postprocess_trajectories.py --data_dir output_v2
    C:\\ProgramData\\anaconda3\\envs\\carla\\python.exe postprocess_trajectories.py --data_dir output_v2
"""

import argparse
import json
import os
import sys

import numpy as np


def load_npc_history(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_meta(meta_dir: str) -> dict[str, dict]:
    """加载所有帧元数据，按 frame_str 索引。"""
    metas = {}
    for fname in sorted(os.listdir(meta_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(meta_dir, fname)
        with open(fpath) as f:
            data = json.load(f)
        key = data.get("frame_str", fname.replace(".json", ""))
        metas[key] = data
    return metas


def load_instance_meta(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def build_npc_timelines(history: dict) -> dict[int, list[dict]]:
    """从 npc_history.json 构建每个 NPC 的时间线。

    Returns:
        {npc_index: [{"timestamp_sec": float, "position": [x,y,z],
                       "quaternion": [x,y,z,w] | None}, ...]}
    """
    records = history.get("records", [])
    timelines: dict[int, list[dict]] = {}
    has_orientations = any("orientations" in rec for rec in records)

    for rec in records:
        t = rec["timestamp_sec"]
        orientations = rec.get("orientations", [])
        for idx, pos in enumerate(rec["positions"]):
            if idx not in timelines:
                timelines[idx] = []
            entry = {
                "timestamp_sec": t,
                "position": pos,
            }
            if has_orientations and idx < len(orientations):
                entry["quaternion"] = orientations[idx]
            timelines[idx].append(entry)

    # 按时间排序
    for idx in timelines:
        timelines[idx].sort(key=lambda x: x["timestamp_sec"])

    return timelines


def find_positions_at_offsets(timeline: list[dict], current_t: float,
                              offsets: list[float]) -> list[dict]:
    """在 NPC 时间线中查找给定时间偏移的位置。

    Args:
        timeline: 排序的时间线
        current_t: 当前帧时间
        offsets: 时间偏移列表（正=未来，负=过去）

    Returns:
        [{"t": offset, "pos": [x,y,z]}, ...]
    """
    if not timeline:
        return []

    results = []
    for offset in offsets:
        target_t = current_t + offset
        # 二分查找最近时间点
        best_idx = 0
        best_dt = abs(timeline[0]["timestamp_sec"] - target_t)
        for i, entry in enumerate(timeline):
            dt = abs(entry["timestamp_sec"] - target_t)
            if dt < best_dt:
                best_dt = dt
                best_idx = i

        # 只有在时间差合理时才包含（<0.5s 容差）
        if best_dt < 0.5:
            results.append({
                "t": round(offset, 3),
                "pos": timeline[best_idx]["position"],
            })

    return results


def compute_velocity(timeline: list[dict], current_t: float) -> list[float]:
    """从相邻位置计算速度。"""
    if len(timeline) < 2:
        return [0.0, 0.0, 0.0]

    # 找当前时间最近的索引
    best_idx = 0
    best_dt = abs(timeline[0]["timestamp_sec"] - current_t)
    for i, entry in enumerate(timeline):
        dt = abs(entry["timestamp_sec"] - current_t)
        if dt < best_dt:
            best_dt = dt
            best_idx = i

    if best_idx == 0:
        prev, curr = timeline[0], timeline[1]
    else:
        prev, curr = timeline[best_idx - 1], timeline[best_idx]

    dt = curr["timestamp_sec"] - prev["timestamp_sec"]
    if dt < 1e-6:
        return [0.0, 0.0, 0.0]

    return [
        (curr["position"][d] - prev["position"][d]) / dt
        for d in range(3)
    ]


def compute_heading(velocity: list[float]) -> float:
    """从速度计算朝向角。"""
    if abs(velocity[0]) < 1e-6 and abs(velocity[1]) < 1e-6:
        return 0.0
    return float(np.arctan2(velocity[1], velocity[0]))


def compute_angular_velocity(timeline: list[dict], current_t: float) -> list[float]:
    """从相邻两帧四元数差分计算角速度 (rad/s)。

    向后兼容：如果时间线中无 quaternion 字段，返回 [0,0,0]。
    """
    if len(timeline) < 2:
        return [0.0, 0.0, 0.0]

    # 检查是否有四元数数据
    if "quaternion" not in timeline[0]:
        return [0.0, 0.0, 0.0]

    # 找当前时间最近的索引
    best_idx = 0
    best_dt = abs(timeline[0]["timestamp_sec"] - current_t)
    for i, entry in enumerate(timeline):
        dt = abs(entry["timestamp_sec"] - current_t)
        if dt < best_dt:
            best_dt = dt
            best_idx = i

    if best_idx == 0:
        prev_idx, curr_idx = 0, 1
    else:
        prev_idx, curr_idx = best_idx - 1, best_idx

    prev_entry = timeline[prev_idx]
    curr_entry = timeline[curr_idx]

    if "quaternion" not in prev_entry or "quaternion" not in curr_entry:
        return [0.0, 0.0, 0.0]

    dt = curr_entry["timestamp_sec"] - prev_entry["timestamp_sec"]
    if dt < 1e-6:
        return [0.0, 0.0, 0.0]

    q_curr = np.array(curr_entry["quaternion"])  # (x, y, z, w)
    q_prev = np.array(prev_entry["quaternion"])

    # dq = q_curr * conj(q_prev)
    q_prev_conj = np.array([-q_prev[0], -q_prev[1], -q_prev[2], q_prev[3]])
    x1, y1, z1, w1 = q_curr
    x2, y2, z2, w2 = q_prev_conj
    dq = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])

    # 规范化双重覆盖
    if dq[3] < 0:
        dq = -dq

    # ω = 2 * dq.xyz / dt
    omega = 2.0 * dq[:3] / dt
    return omega.tolist()


def generate_trajectory_file(frame_str: str, meta: dict,
                              npc_timelines: dict[int, list[dict]],
                              instance_meta: dict) -> dict:
    """为单帧生成轨迹数据。"""
    current_t = meta.get("timestamp_sec", 0.0)

    # 过去时间偏移
    past_offsets = [-0.033, -0.067, -0.100]
    # 未来时间偏移（参见 方向预测.md §3.4）
    future_offsets = [0.033, 0.067, 0.100, 0.500, 1.000, 2.000, 3.000]

    objects = []
    instances = instance_meta.get("instances", {})

    for npc_idx, timeline in npc_timelines.items():
        if not timeline:
            continue

        # 查找该 NPC 的实例 ID
        # NPC prim path 格式: /World/Characters/Character 或 Character_0N
        if npc_idx == 0:
            npc_name = "Character"
        elif npc_idx < 10:
            npc_name = f"Character_0{npc_idx}"
        else:
            npc_name = f"Character_{npc_idx}"
        npc_prim = f"/World/Characters/{npc_name}"

        instance_id = None
        for iid_str, info in instances.items():
            if info.get("prim_path") == npc_prim:
                instance_id = int(iid_str)
                break
        if instance_id is None:
            instance_id = 1000 + npc_idx

        velocity = compute_velocity(timeline, current_t)
        angular_vel = compute_angular_velocity(timeline, current_t)
        heading = compute_heading(velocity)
        speed = float(np.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2))

        # 当前位置
        best_pos = timeline[0]["position"]
        best_dt = abs(timeline[0]["timestamp_sec"] - current_t)
        for entry in timeline:
            dt = abs(entry["timestamp_sec"] - current_t)
            if dt < best_dt:
                best_dt = dt
                best_pos = entry["position"]

        past_positions = find_positions_at_offsets(timeline, current_t, past_offsets)
        future_positions = find_positions_at_offsets(timeline, current_t, future_offsets)

        objects.append({
            "instance_id": instance_id,
            "prim_path": npc_prim,
            "semantic_class": 6,  # PERSON
            "semantic_name": "person",
            "bbox_center": best_pos,
            "bbox_size": [0.5, 0.5, 1.8],
            "heading_rad": heading,
            "velocity": velocity,
            "angular_velocity": angular_vel,
            "past_positions": past_positions,
            "future_positions": future_positions,
            "is_static": speed < 0.05,
            "occlusion_ratio": 0.0,
        })

    return {
        "frame_id": frame_str,
        "timestamp_sec": current_t,
        "objects": objects,
    }


def main():
    parser = argparse.ArgumentParser(description="Post-process: generate trajectory files")
    parser.add_argument("--data_dir", required=True, help="Dataset directory (output_v2)")
    args = parser.parse_args()

    data_dir = args.data_dir
    meta_dir = os.path.join(data_dir, "meta")
    traj_dir = os.path.join(data_dir, "trajectory")
    history_path = os.path.join(data_dir, "npc_history.json")
    instance_path = os.path.join(data_dir, "instance_meta.json")

    if not os.path.isfile(history_path):
        print(f"ERROR: {history_path} not found. Run capture_v2/main.py first.")
        sys.exit(1)

    os.makedirs(traj_dir, exist_ok=True)

    print(f"Loading NPC history from {history_path}...")
    history = load_npc_history(history_path)
    npc_timelines = build_npc_timelines(history)
    print(f"  {len(npc_timelines)} NPC(s), "
          f"{len(history.get('records', []))} time records")

    print(f"Loading frame metadata from {meta_dir}...")
    metas = load_meta(meta_dir)
    print(f"  {len(metas)} frames")

    instance_meta = load_instance_meta(instance_path)

    print(f"Generating trajectory files in {traj_dir}...")
    count = 0
    for frame_str, meta in sorted(metas.items()):
        traj_data = generate_trajectory_file(
            frame_str, meta, npc_timelines, instance_meta)

        traj_path = os.path.join(traj_dir, f"{frame_str}_traj.json")
        with open(traj_path, "w") as f:
            json.dump(traj_data, f, indent=2)
        count += 1

    print(f"Done! {count} trajectory files generated.")

    # 更新 instance_meta.json 中的 first_frame / last_frame
    if instance_meta and metas:
        frame_list = sorted(metas.keys())
        print(f"  Frame range: {frame_list[0]} ~ {frame_list[-1]}")


if __name__ == "__main__":
    main()
