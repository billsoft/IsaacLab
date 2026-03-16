"""
体素数据采集脚本（配合 Isaac People GUI 配置使用）
====================================================
前提：
  已在 Isaac Sim GUI 中完成：
    1. 加载场景
    2. 烘焙 NavMesh
    3. 添加角色 + 挂载 CharacterBehavior
    4. File → Save As → 保存为配置好的 USD

运行（不加 --headless，保留 GUI 让动画系统正常工作）：
    isaaclab.bat -p scripts/npc/collect_voxel_data.py

采集结果保存到：
    D:/code/IsaacLab/outputs/voxel_data/
"""

import argparse
import os
import time
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Voxel data collection with Isaac People")
parser.add_argument(
    "--scene",
    default="D:/code/IsaacLab/Assets/Isaac/5.1/Isaac/Environments/"
            "Simple_Warehouse/full_warehouse.usd",
    help="已配置好 NPC 的场景 USD 路径",
)
parser.add_argument("--cmd_file",
    default="D:/code/IsaacLab/scripts/npc/commands/patrol_commands.txt",
    help="omni.anim.people 命令文件路径",
)
parser.add_argument("--voxel_size",  type=float, default=0.5,  help="体素边长（米）")
parser.add_argument("--collect_sec", type=float, default=30.0, help="采集总时长（秒）")
parser.add_argument("--interval_sec",type=float, default=0.5,  help="采集间隔（秒）")
parser.add_argument("--output_dir",
    default="D:/code/IsaacLab/outputs/voxel_data",
    help="采集数据输出目录",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── AppLauncher 后导入 ────────────────────────────────────────────────────
import carb
import carb.settings
import omni.kit.app
import omni.usd
import omni.timeline

# 动态加入项目路径，使 scene_voxel_query 可被 import
import sys
sys.path.insert(0, "D:/code/IsaacLab")
from scripts.tools.scene_voxel_query import SceneVoxelQuery, VoxelGrid

# ── 配置 ──────────────────────────────────────────────────────────────────
# 仓库场景体素范围（根据实际场景调整）
VOXEL_GRID_ORIGIN = (-10.0,  0.0, -5.0)
VOXEL_GRID_SIZE   = ( 20.0,  3.0, 10.0)


def enable_anim_people():
    """启用 omni.anim.people 及依赖扩展"""
    mgr = omni.kit.app.get_app().get_extension_manager()
    for ext in [
        "omni.anim.graph.core",
        "omni.anim.retarget.core",
        "omni.anim.navigation.core",
        "omni.kit.scripting",
        "omni.anim.people",
    ]:
        try:
            if not mgr.is_extension_enabled(ext):
                mgr.set_extension_enabled_immediate(ext, True)
                carb.log_info(f"[Collect] Enabled: {ext}")
        except Exception as e:
            carb.log_warn(f"[Collect] Could not enable {ext}: {e}")


def configure_people(cmd_file: str):
    """配置 omni.anim.people 参数"""
    s = carb.settings.get_settings()
    path = cmd_file.replace("\\", "/")
    s.set("/exts/omni.anim.people/command_settings/command_file_path", path)
    s.set("/exts/omni.anim.people/command_settings/number_of_loop", "inf")
    s.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
    s.set("/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", False)
    s.set("/persistent/exts/omni.anim.people/character_prim_path", "/World/Characters")
    carb.log_info(f"[Collect] Command file: {path}")


def load_scene(scene_usd: str):
    ctx = omni.usd.get_context()
    carb.log_info(f"[Collect] Opening: {scene_usd}")
    ctx.open_stage(scene_usd)
    for _ in range(400):
        simulation_app.update()
        if ctx.get_stage().GetPrimAtPath("/World").IsValid():
            break
        time.sleep(0.05)
    return ctx.get_stage()


def collect_frame(vq: SceneVoxelQuery, grid: VoxelGrid, timestamp: float) -> dict:
    """采集当前帧的体素数据"""
    occupied = vq.scan_grid(grid, skip_empty=True)
    frame_data = {
        "timestamp": round(timestamp, 3),
        "voxels": []
    }
    for cell in occupied:
        frame_data["voxels"].append({
            "idx":    list(cell.voxel_idx),
            "center": [round(v, 3) for v in cell.world_center],
            "actors": [
                {
                    "id":   a.prim_path,
                    "type": a.actor_type,
                    "pos":  [round(v, 3) for v in a.world_pos],
                }
                for a in cell.actors
            ],
        })
    return frame_data


def main():
    print("\n" + "="*60)
    print("[Collect] 体素数据采集  (Isaac People + SceneVoxelQuery)")
    print("="*60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: 启用扩展
    print("[Step 1] 启用动画扩展...")
    enable_anim_people()
    simulation_app.update()

    # Step 2: 配置命令文件
    print("[Step 2] 配置 omni.anim.people...")
    configure_people(args.cmd_file)

    # Step 3: 加载场景
    print("[Step 3] 加载场景...")
    stage = load_scene(args.scene)

    # Step 4: 构建体素查询器
    print("[Step 4] 构建体素空间索引...")
    vq   = SceneVoxelQuery(voxel_size=args.voxel_size)
    grid = VoxelGrid(
        origin     = VOXEL_GRID_ORIGIN,
        size       = VOXEL_GRID_SIZE,
        voxel_size = args.voxel_size,
    )
    n = vq.build_static_index(stage)
    print(f"  静态索引: {n} prim")
    print(f"  网格: {grid}")

    # Step 5: 启动仿真
    print("[Step 5] 启动仿真时间轴...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_start_time(0)
    timeline.set_end_time(1e9)
    timeline.play()
    # 等角色完成初始化（CharacterBehavior 需要几帧才激活）
    for _ in range(120):
        simulation_app.update()
    print("  NPC 已激活，开始采集...")

    # Step 6: 采集循环
    total_frames  = int(args.collect_sec / args.interval_sec)
    interval_steps = max(1, int(args.interval_sec * 60))  # 约 60fps
    dataset       = {"config": vars(args), "frames": []}
    frame_idx     = 0

    print(f"\n[Step 6] 开始采集 {args.collect_sec:.0f}s  "
          f"间隔 {args.interval_sec:.1f}s  预计 {total_frames} 帧")
    print("-"*60)

    try:
        while simulation_app.is_running() and frame_idx < total_frames:
            # 推进 N 帧
            for _ in range(interval_steps):
                simulation_app.update()

            timestamp   = frame_idx * args.interval_sec
            frame_data  = collect_frame(vq, grid, timestamp)
            dataset["frames"].append(frame_data)
            frame_idx  += 1

            n_occ = len(frame_data["voxels"])
            n_act = sum(len(v["actors"]) for v in frame_data["voxels"])
            print(f"  t={timestamp:6.1f}s  占用体素={n_occ:4d}  actors={n_act}")

    except KeyboardInterrupt:
        print("\n[Collect] 用户中断.")

    # Step 7: 保存数据
    out_file = os.path.join(
        args.output_dir,
        f"voxel_{int(time.time())}.json"
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n[Collect] 完成！共采集 {frame_idx} 帧")
    print(f"[Collect] 数据保存至：{out_file}")

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
