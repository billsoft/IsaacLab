"""
NPC 轨迹数据采集脚本 —— 自动驾驶训练数据集格式
================================================
在 IRA NPC 仿真运行的同时，采集每个 NPC 的：
  • 当前位置 (x, y, z)
  • 朝向 (yaw, 单位：度)
  • 瞬时速度 (speed, vx, vy)
  • 未来轨迹预测（ground truth：未来 N 秒的位置序列）

输出格式：JSON Lines (.jsonl)，每行一个时刻的所有 NPC 状态。
可直接用于轨迹预测模型训练（类似 nuScenes / Argoverse 格式）。

运行方式：
    isaaclab.bat -p scripts/npc/collect_npc_trajectory.py
    isaaclab.bat -p scripts/npc/collect_npc_trajectory.py --headless --duration 60 --sample_hz 10
    isaaclab.bat -p scripts/npc/collect_npc_trajectory.py --output_dir D:/datasets/npc_traj
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict

# ---------------------------------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="NPC Trajectory Data Collector")
parser.add_argument("--num_characters", type=int, default=3, help="Number of NPC characters")
parser.add_argument("--walk_distance", type=float, default=8.0, help="One-way walk distance in meters")
parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration in seconds")
parser.add_argument("--sample_hz", type=float, default=10.0, help="Sampling frequency in Hz")
parser.add_argument("--future_secs", type=float, default=5.0, help="Future trajectory horizon in seconds")
parser.add_argument("--future_step", type=float, default=0.5, help="Future trajectory time step in seconds")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for trajectory data")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# 2. SimulationApp (NOT AppLauncher — need full Isaac Sim config)
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "width": 1920, "height": 1080})

# ---------------------------------------------------------------------------
# 3. Enable extensions
# ---------------------------------------------------------------------------
import carb
import omni.kit.app
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, UsdGeom

REQUIRED_EXTENSIONS = [
    "omni.anim.timeline",
    "omni.anim.graph.bundle",
    "omni.anim.graph.core",
    "omni.anim.retarget.core",
    "omni.anim.navigation.core",
    "omni.anim.navigation.bundle",
    "omni.anim.people",
    "omni.kit.scripting",
]

print("[Traj] Enabling extensions...")
for ext in REQUIRED_EXTENSIONS:
    enable_extension(ext)
    simulation_app.update()

# ---------------------------------------------------------------------------
# 4. Asset paths
# ---------------------------------------------------------------------------
ASSETS_ROOT = "D:/code/IsaacLab/Assets/Isaac/5.1"
if not os.path.isdir(ASSETS_ROOT):
    try:
        from isaacsim.storage.native import get_assets_root_path
        ASSETS_ROOT = get_assets_root_path()
    except Exception:
        pass

if not ASSETS_ROOT or not os.path.isdir(ASSETS_ROOT):
    print("[Traj] ERROR: Cannot find Isaac assets.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)

ASSETS_ROOT = ASSETS_ROOT.replace("\\", "/")
SCENE_USD = f"{ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

CHARACTER_MODELS = [
    "Isaac/People/Characters/F_Business_02/F_Business_02.usd",
    "Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd",
    "Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd",
    "Isaac/People/Characters/M_Medical_01/M_Medical_01.usd",
    "Isaac/People/Characters/male_adult_construction_03/male_adult_construction_03.usd",
    "Isaac/People/Characters/female_adult_police_03_new/female_adult_police_03_new.usd",
]

AVAILABLE_MODELS = [m for m in CHARACTER_MODELS if os.path.isfile(f"{ASSETS_ROOT}/{m}")]
if not AVAILABLE_MODELS:
    print("[Traj] ERROR: No character models found.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)


# ---------------------------------------------------------------------------
# 5. IRA config/command generation (same as npc_people_demo.py)
# ---------------------------------------------------------------------------
def generate_command_file(num_characters, walk_distance, num_loops=200):
    lines = []
    spacing = 3.0
    start_x = -walk_distance / 2.0
    for i in range(num_characters):
        name = "Character" if i == 0 else (f"Character_0{i}" if i < 10 else f"Character_{i}")
        y = i * spacing
        x_a, x_b = start_x, start_x + walk_distance
        for _ in range(num_loops):
            lines.append(f"{name} GoTo {x_b:.1f} {y:.1f} 0.0 0")
            lines.append(f"{name} GoTo {x_a:.1f} {y:.1f} 0.0 180")

    temp_dir = tempfile.gettempdir()
    cmd_file = os.path.join(temp_dir, "npc_traj_commands.txt")
    with open(cmd_file, "w") as f:
        f.write("\n".join(lines))
    return "npc_traj_commands.txt"


def generate_config_file(num_characters, command_file):
    char_folder = f"{ASSETS_ROOT}/Isaac/People/Characters/"
    config = (
        "isaacsim.replicator.agent:\n"
        "  version: 0.7.0\n"
        "  global:\n"
        "    seed: 42\n"
        f"    simulation_length: {int(args.duration * 30 + 300)}\n"
        "  scene:\n"
        f"    asset_path: {SCENE_USD}\n"
        "  character:\n"
        f"    asset_path: {char_folder}\n"
        f"    command_file: {command_file}\n"
        f"    num: {num_characters}\n"
        "  robot:\n"
        "    nova_carter_num: 0\n"
        "    iw_hub_num: 0\n"
        '    command_file: ""\n'
        "  sensor:\n"
        "    camera_num: 0\n"
        "  replicator:\n"
        "    writer: IRABasicWriter\n"
        "    parameters:\n"
        "      rgb: false\n"
    )
    temp_dir = tempfile.gettempdir()
    config_file = os.path.join(temp_dir, "npc_traj_config.yaml")
    with open(config_file, "w") as f:
        f.write(config)
    return config_file


# ---------------------------------------------------------------------------
# 6. Trajectory sampler
# ---------------------------------------------------------------------------
class TrajectorySampler:
    """
    从 USD Stage 采样 NPC 位置，计算速度/朝向，
    记录完整轨迹历史，最终生成含未来轨迹的训练数据。
    """

    def __init__(self, char_prim_paths: list, sample_hz: float,
                 future_secs: float, future_step: float):
        self.char_paths = char_prim_paths
        self.sample_interval = 1.0 / sample_hz
        self.future_secs = future_secs
        self.future_step = future_step

        # 每个角色的采样历史：[{t, x, y, z, yaw, speed, vx, vy}, ...]
        self.history = defaultdict(list)
        # 上一次采样的位置（用于计算速度）
        self._prev_pos = {}
        self._prev_time = {}
        self._last_sample_time = 0.0

    def _get_char_xform(self, stage, prim_path):
        """从 USD Stage 读取角色世界坐标变换"""
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None, None

        xform = UsdGeom.Xformable(prim)
        mat = xform.ComputeLocalToWorldTransform(0)
        translation = mat.ExtractTranslation()

        # 从变换矩阵提取 yaw（绕 Z 轴旋转角）
        # 矩阵列 0 = local X 轴方向，列 1 = local Y 轴方向
        # Z-up 场景中，yaw = atan2(mat[0][1], mat[0][0])
        yaw_rad = math.atan2(mat[0][1], mat[0][0])
        yaw_deg = math.degrees(yaw_rad)

        pos = (translation[0], translation[1], translation[2])
        return pos, yaw_deg

    def sample(self, stage, sim_time: float):
        """在指定仿真时间采样所有 NPC 状态"""
        if sim_time - self._last_sample_time < self.sample_interval:
            return False  # 未到采样间隔

        self._last_sample_time = sim_time

        for path in self.char_paths:
            pos, yaw = self._get_char_xform(stage, path)
            if pos is None:
                continue

            # 计算速度
            speed, vx, vy = 0.0, 0.0, 0.0
            agent_id = path.split("/")[-1]

            if agent_id in self._prev_pos:
                prev = self._prev_pos[agent_id]
                dt = sim_time - self._prev_time[agent_id]
                if dt > 1e-6:
                    dx = pos[0] - prev[0]
                    dy = pos[1] - prev[1]
                    vx = dx / dt
                    vy = dy / dt
                    speed = math.sqrt(vx * vx + vy * vy)

            self._prev_pos[agent_id] = pos
            self._prev_time[agent_id] = sim_time

            self.history[agent_id].append({
                "t": round(sim_time, 4),
                "x": round(pos[0], 4),
                "y": round(pos[1], 4),
                "z": round(pos[2], 4),
                "yaw": round(yaw, 2),
                "speed": round(speed, 4),
                "vx": round(vx, 4),
                "vy": round(vy, 4),
            })

        return True

    def build_dataset(self) -> list:
        """
        构建训练数据集：为每个采样点附加未来轨迹 ground truth。

        输出格式（每条记录）：
        {
            "timestamp": 3.5,
            "agent_id": "Character",
            "position": [x, y, z],
            "heading": yaw_deg,
            "velocity": [vx, vy],
            "speed": 1.2,
            "future_trajectory": [
                {"dt": 0.5, "x": ..., "y": ...},
                {"dt": 1.0, "x": ..., "y": ...},
                ...
                {"dt": 5.0, "x": ..., "y": ...}
            ]
        }
        """
        dataset = []
        future_steps = []
        dt = self.future_step
        while dt <= self.future_secs + 1e-6:
            future_steps.append(round(dt, 2))
            dt += self.future_step

        for agent_id, samples in self.history.items():
            # 建立时间→索引的查找表
            times = [s["t"] for s in samples]

            for i, sample in enumerate(samples):
                t_now = sample["t"]

                # 查找未来各时间步的位置（从历史中查找最接近的采样点）
                future_traj = []
                for f_dt in future_steps:
                    t_future = t_now + f_dt
                    # 二分查找最近的采样点
                    best_idx = self._find_nearest(times, t_future, i)
                    if best_idx is not None and abs(times[best_idx] - t_future) < self.sample_interval * 1.5:
                        fs = samples[best_idx]
                        future_traj.append({
                            "dt": f_dt,
                            "x": fs["x"],
                            "y": fs["y"],
                        })

                # 只输出有完整未来轨迹的样本（跳过尾部不完整的）
                if len(future_traj) < len(future_steps) * 0.8:
                    continue

                dataset.append({
                    "timestamp": sample["t"],
                    "agent_id": agent_id,
                    "position": [sample["x"], sample["y"], sample["z"]],
                    "heading": sample["yaw"],
                    "velocity": [sample["vx"], sample["vy"]],
                    "speed": sample["speed"],
                    "future_trajectory": future_traj,
                })

        return dataset

    @staticmethod
    def _find_nearest(sorted_times, target, start_idx):
        """在排序时间列表中查找最接近 target 的索引"""
        best = None
        best_diff = float("inf")
        for j in range(start_idx, len(sorted_times)):
            diff = abs(sorted_times[j] - target)
            if diff < best_diff:
                best_diff = diff
                best = j
            elif sorted_times[j] > target + 1.0:
                break  # 已经远超目标时间
        return best


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main():
    num_chars = min(args.num_characters, len(AVAILABLE_MODELS))

    # Generate IRA files
    cmd_file = generate_command_file(num_chars, args.walk_distance)
    config_file = generate_config_file(num_chars, cmd_file)

    # Enable IRA
    enable_extension("isaacsim.replicator.agent.core")
    simulation_app.update()
    simulation_app.update()

    from isaacsim.replicator.agent.core.simulation import SimulationManager

    sim_manager = SimulationManager()
    settings = carb.settings.get_settings()
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
    settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
    settings.set("/app/omni.graph.scriptnode/enable_opt_in", False)

    can_load = sim_manager.load_config_file(config_file)
    if not can_load:
        print("[Traj] ERROR: Failed to load config.", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)

    print("[Traj] Setting up simulation...")
    setup_done = [False]

    def on_setup_done(e):
        setup_done[0] = True

    sim_manager.register_set_up_simulation_done_callback(on_setup_done)
    sim_manager.set_up_simulation_from_config_file()

    while not setup_done[0] and not simulation_app.is_exiting():
        simulation_app.update()

    if simulation_app.is_exiting():
        return

    print("[Traj] Simulation ready!")

    # 发现场景中的 NPC prim 路径
    stage = omni.usd.get_context().get_stage()
    char_root = "/World/Characters"
    char_paths = []
    root_prim = stage.GetPrimAtPath(char_root)
    if root_prim.IsValid():
        for child in root_prim.GetChildren():
            if child.IsValid():
                char_paths.append(str(child.GetPath()))

    if not char_paths:
        print("[Traj] WARNING: No character prims found under /World/Characters")
        # 尝试 IRA 默认路径
        for i in range(num_chars):
            name = "Character" if i == 0 else (f"Character_0{i}" if i < 10 else f"Character_{i}")
            p = f"/World/{name}"
            if stage.GetPrimAtPath(p).IsValid():
                char_paths.append(p)

    print(f"[Traj] Tracking {len(char_paths)} character(s): {char_paths}")

    # 创建采样器
    sampler = TrajectorySampler(
        char_prim_paths=char_paths,
        sample_hz=args.sample_hz,
        future_secs=args.future_secs,
        future_step=args.future_step,
    )

    # 启动仿真
    async def run_sim():
        await sim_manager.run_data_generation_async(will_wait_until_complete=True)

    from omni.kit.async_engine import run_coroutine
    task = run_coroutine(run_sim())

    print(f"[Traj] Collecting data for {args.duration}s at {args.sample_hz} Hz...")
    print(f"[Traj] Future horizon: {args.future_secs}s, step: {args.future_step}s")
    print("-" * 60)

    frame = 0
    sim_start = time.time()
    sample_count = 0

    try:
        while not task.done() and not simulation_app.is_exiting():
            simulation_app.update()
            frame += 1

            sim_time = time.time() - sim_start

            if sim_time > args.duration:
                print(f"\n[Traj] Duration {args.duration}s reached, stopping.")
                break

            if sampler.sample(stage, sim_time):
                sample_count += 1

            if frame % 300 == 0:
                total_points = sum(len(v) for v in sampler.history.values())
                print(f"[Traj] t={sim_time:.1f}s  samples={total_points}  "
                      f"agents={len(sampler.history)}")
                # 打印当前 NPC 状态
                for agent_id, samples in sampler.history.items():
                    if samples:
                        s = samples[-1]
                        print(f"  {agent_id:15s}: pos=({s['x']:6.2f}, {s['y']:5.2f})  "
                              f"yaw={s['yaw']:6.1f}  speed={s['speed']:.2f} m/s")

    except KeyboardInterrupt:
        print("\n[Traj] Interrupted.")

    # ---------------------------------------------------------------------------
    # 8. 构建数据集并保存
    # ---------------------------------------------------------------------------
    print("\n[Traj] Building dataset with future trajectory ground truth...")
    dataset = sampler.build_dataset()

    # 输出目录
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "trajectory_data")
    os.makedirs(output_dir, exist_ok=True)

    # 保存完整数据集（JSON Lines 格式，每行一条记录）
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"npc_trajectory_{timestamp_str}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"[Traj] Data collection complete!")
    print(f"  Total samples:     {len(dataset)}")
    print(f"  Agents tracked:    {len(sampler.history)}")
    print(f"  Duration:          {args.duration:.1f}s")
    print(f"  Sample rate:       {args.sample_hz} Hz")
    print(f"  Future horizon:    {args.future_secs}s (step={args.future_step}s)")
    print(f"  Output file:       {output_file}")
    print(f"  File size:         {os.path.getsize(output_file) / 1024:.1f} KB")
    print(f"{'='*60}")

    # 打印数据集前 3 条作为示例
    print("\n[Traj] Sample records:")
    for rec in dataset[:3]:
        print(json.dumps(rec, indent=2, ensure_ascii=False))

    # 同时保存一份统计摘要
    summary = {
        "total_samples": len(dataset),
        "agents": list(sampler.history.keys()),
        "duration_sec": args.duration,
        "sample_hz": args.sample_hz,
        "future_horizon_sec": args.future_secs,
        "future_step_sec": args.future_step,
        "num_characters": num_chars,
        "walk_distance": args.walk_distance,
        "scene": SCENE_USD,
    }
    summary_file = os.path.join(output_dir, f"npc_trajectory_{timestamp_str}_meta.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Traj] Metadata:     {summary_file}")

    simulation_app.close()


if __name__ == "__main__":
    main()
