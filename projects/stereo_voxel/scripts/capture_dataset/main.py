"""双目鱼眼 + 语义体素 + Flow + 实例 ID 同步采集
====================================================
相比 stereo_voxel_capture_dng.py 的改进：
  - 模块化拆分（constants / scene / camera / voxel / npc / flow / io）
  - 新增 flow.npz: 逐体素运动向量 (vx, vy) + mask
  - 新增 instance ID 分段范围（行人 1000+, 车辆 1+）
  - 新增 NPC 位置历史（供后处理生成轨迹）

运行：
    isaaclab.bat -p projects/stereo_voxel/scripts/capture_dataset/main.py
    isaaclab.bat -p projects/stereo_voxel/scripts/capture_dataset/main.py --headless --num_frames 200
    isaaclab.bat -p projects/stereo_voxel/scripts/capture_dataset/main.py --no_npc --headless --num_frames 50
"""

import argparse
import json
import os
import shutil
import sys

# 强制 print 立即刷新
import builtins
_original_print = builtins.print
def _flush_print(*a, **kw):
    kw.setdefault("flush", True)
    _original_print(*a, **kw)
builtins.print = _flush_print

# ===========================================================================
# 参数解析（必须在 SimulationApp 之前）
# ===========================================================================
parser = argparse.ArgumentParser(description="Stereo fisheye + voxel + flow capture (v2)")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--num_frames", type=int, default=200, help="Number of frames to capture")
parser.add_argument("--camera_height", type=float, default=3.0, help="Camera height (m)")
parser.add_argument("--camera_x", type=float, default=0.0, help="Camera X position (m)")
parser.add_argument("--camera_y", type=float, default=0.0, help="Camera Y position (m)")
parser.add_argument("--num_characters", type=int, default=3, help="Number of NPC characters")
parser.add_argument("--walk_distance", type=float, default=8.0, help="NPC walk distance (m)")
parser.add_argument("--capture_interval", type=int, default=90,
                    help="Capture every N sim steps (~3s at 30FPS)")
parser.add_argument("--no_npc", action="store_true", help="Skip NPC loading")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory override")
args, _ = parser.parse_known_args()

# ===========================================================================
# Isaac Sim 初始化（必须在所有 omni import 之前）
# ===========================================================================
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 1080})

# --- 下面可以安全地 import omni / pxr ---
import numpy as np
import omni.timeline
import omni.usd
from pxr import UsdGeom

# 添加 scripts 目录到 path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# capture_dataset 子模块（延迟到 SimulationApp 之后）
from capture_dataset.constants import (
    BASELINE_M, CAM_H, CAM_W, DIAG_FOV_DEG, K1_EQUIDISTANT,
    PIXEL_SIZE_UM, FOCAL_LENGTH_MM, SIM_FPS, fx, cx, cy,
    resolve_assets_root,
)
from capture_dataset.scene_setup import (
    probe_scene_objects, setup_scene, suggest_camera_position,
)
from capture_dataset.camera_setup import (
    create_stereo_rig, get_rig_world_pose, setup_annotators,
)
from capture_dataset.voxel_grid_v2 import VoxelGridV2
from capture_dataset.voxel_filling import (
    fill_voxel_grid, get_npc_world_positions, stamp_npc_voxels,
)
from capture_dataset.instance_registry import InstanceRegistry
from capture_dataset.npc_tracker import NPCTracker
from capture_dataset.flow_generator import fill_flow
from capture_dataset import async_io

from semantic_classes import PERSON, UNOBSERVED
from rawcam import RawConverter, DngWriter, SensorConfig, NoiseConfig, OutputConfig

# ===========================================================================
# 资产路径
# ===========================================================================
ASSETS_ROOT = resolve_assets_root()
print(f"[v2] Assets: {ASSETS_ROOT}")
print(f"[v2] SC132GS x2, baseline={BASELINE_M*1000:.0f}mm, FOV={DIAG_FOV_DEG}deg")

# ===========================================================================
# PHASE 1: 场景 + NPC
# ===========================================================================
npc_ready, num_chars, sim_manager = setup_scene(simulation_app, args, ASSETS_ROOT)

# 关键修复：必须在 setup_scene 返回后立即启动 IRA 数据生成协程！
# 参考 stereo_voxel_capture.py 的正确做法：run_coroutine 在相机创建之前调用。
# 如果延迟到 PHASE 4 才启动，NPC 会因为初始化时序问题而冻结不动。
_ira_task = None
if npc_ready and sim_manager is not None:
    async def _run_ira_data_gen():
        await sim_manager.run_data_generation_async(will_wait_until_complete=True)
    from omni.kit.async_engine import run_coroutine
    _ira_task = run_coroutine(_run_ira_data_gen())
    print("[v2] ✅ IRA data generation started immediately after setup (NPC GoTo commands active)")

# ===========================================================================
# PHASE 2: 双目相机
# ===========================================================================
print("[v2] Creating stereo cameras...")
stage = omni.usd.get_context().get_stage()
for _ in range(5):
    simulation_app.update()

stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
stage_scale = 1.0 / stage_mpu
print(f"[v2] Stage metersPerUnit={stage_mpu}, scale={stage_scale}")

# 场景探测
scene_info = probe_scene_objects(stage, stage_mpu)
for prefix, info in scene_info.items():
    print(f"[v2:probe] {prefix}: {info['count']} prims, "
          f"X=[{info['bbox_min'][0]:.1f}, {info['bbox_max'][0]:.1f}]m, "
          f"Y=[{info['bbox_min'][1]:.1f}, {info['bbox_max'][1]:.1f}]m")

cam_x, cam_y = args.camera_x, args.camera_y
if scene_info:
    sx, sy = suggest_camera_position(scene_info)
    print(f"[v2:probe] Suggested: ({sx:.1f}, {sy:.1f})m")

rig_path, left_cam_path, right_cam_path = create_stereo_rig(
    stage, simulation_app, cam_x, cam_y, args.camera_height, stage_scale)
annot_left, annot_right = setup_annotators(
    left_cam_path, right_cam_path, simulation_app)

# ===========================================================================
# PHASE 3: 体素 + PhysX + rawcam 管线
# ===========================================================================
print("[v2] Initializing systems...")
voxel_template = VoxelGridV2()
print(f"[v2] VoxelGridV2: {voxel_template.NX}x{voxel_template.NY}x{voxel_template.NZ}")

from omni.physx import get_physx_scene_query_interface
physx_sqi = get_physx_scene_query_interface()

instance_registry = InstanceRegistry()
npc_tracker = NPCTracker(sim_fps=SIM_FPS)

# rawcam DNG 管线
OUTPUT_DIR = args.output_dir or os.path.join(
    os.path.dirname(SCRIPTS_DIR), "output_dng")
LEFT_DIR = os.path.join(OUTPUT_DIR, "left")
RIGHT_DIR = os.path.join(OUTPUT_DIR, "right")
LEFT_DNG_DIR = os.path.join(OUTPUT_DIR, "left_dng")
RIGHT_DNG_DIR = os.path.join(OUTPUT_DIR, "right_dng")
VOXEL_DIR = os.path.join(OUTPUT_DIR, "voxel")
META_DIR = os.path.join(OUTPUT_DIR, "meta")
TRAJ_DIR = os.path.join(OUTPUT_DIR, "trajectory")
ALL_DIRS = [LEFT_DIR, RIGHT_DIR, LEFT_DNG_DIR, RIGHT_DNG_DIR,
            VOXEL_DIR, META_DIR, TRAJ_DIR]

# 清空输出目录
for d in ALL_DIRS:
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
for old_file in ["calibration.json", "voxel_config.json", "instance_meta.json"]:
    old_path = os.path.join(OUTPUT_DIR, old_file)
    if os.path.isfile(old_path):
        os.remove(old_path)
print(f"[v2] Output: {OUTPUT_DIR}")

# 保存标定信息
_sensor_cfg = SensorConfig(
    width=CAM_W, height=CAM_H,
    pixel_size_um=PIXEL_SIZE_UM, focal_length_mm=FOCAL_LENGTH_MM,
    horizontal_aperture_mm=PIXEL_SIZE_UM * CAM_W / 1000.0,
    fisheye_enabled=True, diagonal_fov_deg=DIAG_FOV_DEG,
    bit_depth=12, bayer_pattern="RGGB", black_level=0, white_level=4095,
)
_noise_cfg = NoiseConfig(preset="sc132gs", enabled=True)
_output_cfg = OutputConfig(output_dir=OUTPUT_DIR)
_raw_converter = RawConverter(_sensor_cfg, _noise_cfg)
_dng_writer = DngWriter(_sensor_cfg, _output_cfg)
print(f"[v2] RawCam: {_sensor_cfg.bit_depth}-bit {_sensor_cfg.bayer_pattern}, "
      f"noise={_noise_cfg.preset}")

with open(os.path.join(OUTPUT_DIR, "calibration.json"), "w") as f:
    json.dump({
        "sensor": "SC132GS (simulated)",
        "projection": "ftheta_equidistant",
        "resolution": [CAM_W, CAM_H],
        "fx": fx, "fy": fx, "cx": cx, "cy": cy,
        "k1": K1_EQUIDISTANT,
        "max_fov_deg": DIAG_FOV_DEG,
        "baseline_m": BASELINE_M,
        "baseline_direction": "Y-axis (left=+Y, right=-Y)",
        "raw_pipeline": {
            "format": "DNG (12-bit Bayer RGGB)",
            "bit_depth": 12, "bayer_pattern": "RGGB",
            "black_level": 0, "white_level": 4095,
            "noise_preset": "sc132gs",
            "pipeline": "RGB(uint8) -> sRGB_decode -> linear -> RGGB_CFA -> 12bit -> noise -> DNG",
        },
    }, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "voxel_config.json"), "w") as f:
    json.dump(VoxelGridV2.get_config(), f, indent=2)


# ===========================================================================
# 同步采集函数
# ===========================================================================
def capture_frame(frame_id: int, sim_step: int) -> bool:
    """冻结 → 拍照 + 体素查询 + flow 计算 → 恢复。"""
    timeline = omni.timeline.get_timeline_interface()
    frame_str = f"frame_{frame_id:06d}"

    # 1. 冻结虚拟世界（图像/位置/体素精确同步）
    timeline.pause()
    for _ in range(3):
        simulation_app.update()

    # 诊断：对比 pause 前（通过 update 获取）和 pause 后的 NPC Xform
    from pxr import UsdGeom
    _diag_before = []
    for _i in range(3):
        _p = stage.GetPrimAtPath(f"/Root/Character_{_i}")
        if _p.IsValid():
            _m = UsdGeom.Xformable(_p).ComputeLocalToWorldTransform(0)
            _t = _m.ExtractTranslation()
            _diag_before.append((_t[0] * stage_mpu, _t[1] * stage_mpu))

    # 2. 双目图像（与冻结瞬间同步）
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

    # 3. 相机位姿
    cam_pos, cam_yaw = get_rig_world_pose(
        stage, rig_path, stage_mpu, args.camera_height)

    # 4. NPC 位置 + 朝向追踪
    npc_positions, npc_orientations = npc_tracker.update(stage, stage_mpu, sim_step)

    # 诊断：对比 pause 后立即读取 vs get_npc_world_positions 读取的位置
    if frame_id % 10 == 0 and _diag_before:
        _diag_after = [(p[0], p[1]) for p in npc_positions]
        for _i in range(min(len(_diag_before), len(_diag_after))):
            _dx = _diag_after[_i][0] - _diag_before[_i][0]
            _dy = _diag_after[_i][1] - _diag_before[_i][1]
            _same = abs(_dx) < 0.001 and abs(_dy) < 0.001
            print(f"  [DIAG] NPC{_i}: before=({_diag_before[_i][0]:.2f},{_diag_before[_i][1]:.2f}) "
                  f"after=({_diag_after[_i][0]:.2f},{_diag_after[_i][1]:.2f}) "
                  f"Δ=({_dx:.4f},{_dy:.4f}) {'SAME' if _same else 'MOVED'}")

    # 5. 体素填充（PhysX 静态物体）
    vg = VoxelGridV2()
    world_centers_flat = vg.get_world_centers_flat(cam_pos, cam_yaw)
    fill_voxel_grid(stage, vg, world_centers_flat, physx_sqi, stage_scale,
                    instance_registry, frame_str)

    # 6. NPC 体素标记 → 获取占用索引映射
    occupied_map = stamp_npc_voxels(
        vg, cam_pos, cam_yaw, npc_positions,
        instance_registry, frame_str)

    # 7. Flow 计算
    fill_flow(vg, npc_tracker, cam_yaw, occupied_map,
              sim_step, args.capture_interval)

    # 8. 异步保存
    async_io.async_save_image(
        os.path.join(LEFT_DIR, f"{frame_str}.png"), rgb_l)
    async_io.async_save_image(
        os.path.join(RIGHT_DIR, f"{frame_str}.png"), rgb_r)
    async_io.async_save_dng(
        os.path.join(LEFT_DNG_DIR, f"{frame_str}.dng"),
        rgb_l, _raw_converter, _dng_writer)
    async_io.async_save_dng(
        os.path.join(RIGHT_DNG_DIR, f"{frame_str}.dng"),
        rgb_r, _raw_converter, _dng_writer)
    async_io.async_save_voxel(
        os.path.join(VOXEL_DIR, frame_str),
        vg.semantic, vg.instance, vg.flow, vg.flow_mask,
        vg.orientation, vg.angular_vel)

    # 9. 帧元数据（扩展）
    timestamp_sec = sim_step / SIM_FPS
    num_dynamic = sum(1 for idx in occupied_map if occupied_map[idx])
    meta = {
        "frame_id": frame_id,
        "frame_str": frame_str,
        "camera_pos": cam_pos.tolist(),
        "camera_yaw_rad": float(cam_yaw),
        "camera_height_m": float(cam_pos[2]),
        "voxel_origin_world": [float(cam_pos[0]), float(cam_pos[1]), 0.0],
        "voxel_size": VoxelGridV2.VOXEL_SIZE,
        "voxel_shape": [VoxelGridV2.NX, VoxelGridV2.NY, VoxelGridV2.NZ],
        "z_ground_index": VoxelGridV2.Z_GROUND_INDEX,
        "timestamp_sec": timestamp_sec,
        "sim_step": sim_step,
        "sim_step_sec": 1.0 / SIM_FPS,
        "num_dynamic_objects": num_dynamic,
        "num_npc_positions": len(npc_positions),
        "npc_angular_velocities": [
            npc_tracker.get_angular_velocity(idx).tolist()
            for idx in range(len(npc_positions))
        ],
    }
    if frame_id % 10 == 0:
        meta["voxel_stats"] = vg.stats()
        meta["flow_stats"] = vg.flow_stats()
    async_io.async_save_json(
        os.path.join(META_DIR, f"{frame_str}.json"), meta)

    # 10. 日志（不刷屏：第 0 帧详细，之后每 10 帧一行摘要）
    occ = int(np.sum((vg.semantic > 0) & (vg.semantic < UNOBSERVED)))
    person_ct = int(np.sum(vg.semantic == PERSON))
    flow_ct = int(vg.flow_mask.sum())

    if frame_id == 0:
        # 首帧：详细输出 NPC 位置，帮助诊断
        print(f"  Frame 0: L={rgb_l.mean():.0f} R={rgb_r.mean():.0f} "
              f"occ={occ} person={person_ct} flow={flow_ct}")
        for ni, np_ in enumerate(npc_positions):
            rel = np_ - np.array([cam_pos[0], cam_pos[1], 0.0])
            in_range = abs(rel[0]) <= 3.6 and abs(rel[1]) <= 3.0
            tag = "OK" if in_range else "OUT"
            print(f"    NPC{ni}: world=({np_[0]:.2f},{np_[1]:.2f}) "
                  f"rel=({rel[0]:.2f},{rel[1]:.2f}) [{tag}]")
        # 记录首帧语义哈希用于后续对比
        capture_frame._prev_sem_hash = hash(vg.semantic.tobytes())
    elif frame_id % 10 == 0:
        # 每 10 帧：摘要 + 体素是否有变化
        cur_hash = hash(vg.semantic.tobytes())
        changed = cur_hash != getattr(capture_frame, '_prev_sem_hash', None)
        capture_frame._prev_sem_hash = cur_hash
        npc_pos_str = " ".join(f"({p[0]:.1f},{p[1]:.1f})" for p in npc_positions[:3])
        print(f"  Frame {frame_id}: occ={occ} person={person_ct} "
              f"voxΔ={'YES' if changed else 'NO'} "
              f"npc=[{npc_pos_str}] t={timestamp_sec:.1f}s")

    # 11. 恢复
    timeline.play()
    return True


# ===========================================================================
# PHASE 4: 主循环 (IRA 协程已在 PHASE 1 后启动)
# ===========================================================================

frame_id = 0
warmup_steps = 30
timeline = omni.timeline.get_timeline_interface()

if args.headless:
    print(f"[v2] Headless: {args.num_frames} frames, interval={args.capture_interval}")
    timeline.play()
    for _ in range(warmup_steps):
        simulation_app.update()
    # warmup 后诊断：IRA 是否激活，NPC 初始位置
    if _ira_task is not None:
        print(f"[v2] IRA task status after warmup: done={_ira_task.done()}")
    from capture_dataset.voxel_filling import get_npc_world_positions as _probe_npc
    _probe_pos = _probe_npc(stage, UsdGeom.GetStageMetersPerUnit(stage))
    print(f"[v2] NPC probe after warmup: {len(_probe_pos)} NPCs")
    for _i, _p in enumerate(_probe_pos):
        print(f"  NPC{_i}: ({_p[0]:.2f}, {_p[1]:.2f}, {_p[2]:.2f})m")

    captured = 0
    max_steps = args.num_frames * args.capture_interval * 3
    step = 0

    while captured < args.num_frames and step < max_steps and not simulation_app.is_exiting():
        simulation_app.update()
        step += 1
        if step % args.capture_interval == 0:
            if capture_frame(frame_id, step):
                frame_id += 1
                captured += 1

    timeline.stop()
    async_io.wait_pending_saves()
    print(f"[v2] Headless done: {captured} frames.")
else:
    print(f"\n{'=' * 60}")
    print(f"[v2] GUI mode (auto-play)")
    print(f"  Camera: ({cam_x:.1f}, {cam_y:.1f}, {args.camera_height})m")
    if npc_ready:
        print(f"  {num_chars} NPC(s) ready")
    print(f"  Interval: {args.capture_interval} steps, target: {args.num_frames} frames")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    timeline.play()
    for _ in range(warmup_steps):
        simulation_app.update()

    captured = 0
    sim_step = 0
    while simulation_app.is_running():
        simulation_app.update()
        sim_step += 1
        if captured >= args.num_frames:
            continue
        # 检查 IRA 协程是否已结束
        if _ira_task is not None and _ira_task.done() and captured < args.num_frames:
            print(f"[v2] IRA finished after {captured} frames (target: {args.num_frames})")
        if sim_step % args.capture_interval == 0:
            if capture_frame(frame_id, sim_step):
                frame_id += 1
                captured += 1

# ===========================================================================
# 收尾
# ===========================================================================
async_io.wait_pending_saves()

# 保存实例注册表
instance_registry.save(os.path.join(OUTPUT_DIR, "instance_meta.json"))

# 保存 NPC 位置历史（供后处理轨迹脚本使用）
npc_tracker.save_history(os.path.join(OUTPUT_DIR, "npc_history.json"))

total = frame_id
print(f"\n[v2] Done! {total} frames (0~{total - 1})")
print(f"  Left PNG:    {LEFT_DIR}")
print(f"  Right PNG:   {RIGHT_DIR}")
print(f"  Left DNG:    {LEFT_DNG_DIR}")
print(f"  Right DNG:   {RIGHT_DNG_DIR}")
print(f"  Voxel:       {VOXEL_DIR}")
print(f"  Meta:        {META_DIR}")
print(f"  Trajectory:  {TRAJ_DIR} (run postprocess_trajectories.py)")
print(f"  Instance:    {os.path.join(OUTPUT_DIR, 'instance_meta.json')}")
print(f"  NPC history: {os.path.join(OUTPUT_DIR, 'npc_history.json')}")

async_io.shutdown()
simulation_app.close()
