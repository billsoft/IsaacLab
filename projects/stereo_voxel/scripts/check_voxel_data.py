"""体素数据集验证脚本
====================
5项关键检查：
1. 体素 MD5 唯一性 → 帧间是否完全克隆
2. 逐帧内容统计 + Δ 差异指示
3. NPC 位置变化 + 是否超出体素范围
4. Flow 速度数据是否全零
5. 左右眼图像亮度抽查

用法：
    C:\\ProgramData\\anaconda3\\envs\\carla\\python.exe projects/stereo_voxel/scripts/check_voxel_data.py
    C:\\ProgramData\\anaconda3\\envs\\carla\\python.exe projects/stereo_voxel/scripts/check_voxel_data.py --data_dir output_dng
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np


def load_npz(frame_str: str, data_dir: str, dtype: str = "semantic") -> np.ndarray:
    """加载单帧 .npz 数据"""
    path = os.path.join(data_dir, "voxel", f"{frame_str}_{dtype}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data["data"] if "data" in data else data[list(data.keys())[0]]


def load_meta(frame_str: str, data_dir: str) -> dict:
    """加载帧元数据"""
    path = os.path.join(data_dir, "meta", f"{frame_str}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_npc_history(data_dir: str) -> dict:
    """加载 NPC 历史记录"""
    path = os.path.join(data_dir, "npc_history.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def compute_md5(arr: np.ndarray) -> str:
    """计算数组的 MD5 哈希"""
    return hashlib.md5(arr.tobytes()).hexdigest()


def check1_voxel_md5_uniqueness(data_dir: str, frames: list[str]) -> tuple[bool, str]:
    """检查1：体素 MD5 唯一性 → 帧间是否完全克隆"""
    print("\n" + "=" * 70)
    print("📊 检查 1/5：体素语义数据 MD5 唯一性")
    print("=" * 70)

    md5_map = {}
    for f in frames:
        vox = load_npz(f, data_dir, "semantic")
        if vox is None:
            continue
        md5_val = compute_md5(vox)
        if md5_val not in md5_map:
            md5_map[md5_val] = []
        md5_map[md5_val].append(f)

    unique_md5_count = len(md5_map)
    total_frames = len(frames)

    print(f"总帧数: {total_frames}")
    print(f"唯一 MD5 数量: {unique_md5_count}")

    if unique_md5_count == 1:
        print("❌ 结论: 所有帧的体素数据完全相同 (NPC 未移动)")
        all_same_frames = list(md5_map.values())[0]
        print(f"   所有 {len(all_same_frames)} 帧 → 同一个 MD5: {list(md5_map.keys())[0][:16]}...")
        return False, "所有帧完全克隆"
    elif unique_md5_count < total_frames * 0.8:
        print(f"⚠️ 结论: 大部分帧重复 ({unique_md5_count}/{total_frames} 唯一)")
        for i, (md5_val, frame_list) in enumerate(list(md5_map.items())[:5]):
            print(f"   组{i+1}: {len(frame_list)} 帧 → {frame_list[0]}..{frame_list[-1]}")
        return False, f"部分帧重复 ({unique_md5_count}/{total_frames})"
    else:
        print("✅ 结论: 每帧体素数据都不同 (NPC 正常移动)")
        return True, f"所有帧独立 ({unique_md5_count}/{total_frames})"


def check2_frame_content_stats(data_dir: str, frames: list[str]) -> tuple[bool, str]:
    """检查2：逐帧内容统计 + Δ 差异指示"""
    print("\n" + "=" * 70)
    print("📊 检查 2/5：逐帧体素内容统计")
    print("=" * 70)

    stats_list = []
    prev_semantic = None
    change_count = 0

    for f in frames:
        vox = load_npz(f, data_dir, "semantic")
        if vox is None:
            continue

        unique, counts = np.unique(vox, return_counts=True)
        stat = dict(zip(unique.tolist(), counts.tolist()))
        stat["_frame"] = f

        if prev_semantic is not None:
            diff = np.sum(vox != prev_semantic)
            total = vox.size
            pct = 100.0 * diff / total
            stat["_changed_voxels"] = int(diff)
            stat["_change_pct"] = round(pct, 3)
            if diff > 0:
                change_count += 1

        stats_list.append(stat)
        prev_semantic = vox

    # 输出摘要（每10帧一行）
    print(f"{'帧':<12} {'占用':<10} {'空闲':<10} {'人物':<8} {'变化体素':<12} {'变化%':<8}")
    print("-" * 60)
    for s in stats_list:
        if int(s.get("_frame", "").split("_")[-1]) % 10 == 0 or len(stats_list) <= 20:
            changed = s.get("_changed_voxels", "N/A")
            cpct = s.get("_change_pct", "N/A")
            print(f"{s['_frame']:<12} "
                  f"{s.get(12, 0):<10} "
                  f"{s.get(0, 0):<10} "
                  f"{s.get(6, 0):<8} "
                  f"{str(changed):<12} "
                  f"{str(cpct):<8}")

    total_compared = len(stats_list) - 1
    if total_compared > 0:
        change_rate = 100.0 * change_count / total_compared
        print(f"\n相邻帧对比: {change_count}/{total_compared} 有变化 ({change_rate:.1f}%)")

        if change_count == 0:
            return False, "无任何帧间变化"
        elif change_rate < 50:
            return False, f"变化率低 ({change_rate:.1f}%)"
        else:
            return True, f"正常变化 ({change_rate:.1f}%)"

    return False, "数据不足"


def check3_npc_position_change(data_dir: str, frames: list[str]) -> tuple[bool, str]:
    """检查3：NPC 位置变化 + 是否超出体素范围"""
    print("\n" + "=" * 70)
    print("📊 检查 3/5：NPC 位置变化分析")
    print("=" * 70)

    npc_history = load_npc_history(data_dir)
    if not npc_history or "records" not in npc_history or len(npc_history["records"]) < 2:
        print("❌ 无 NPC 历史数据或记录不足")
        return False, "无 NPC 数据"

    records = npc_history["records"]
    num_npcs = npc_history.get("num_npcs", 0)
    print(f"NPC 数量: {num_npcs}, 记录数: {len(records)}")

    # 加载第一帧 meta 获取相机位置
    first_meta = load_meta(frames[0], data_dir)
    cam_pos = first_meta.get("camera_pos", [0, 0, 3])
    voxel_origin = first_meta.get("voxel_origin_world", [cam_pos[0], cam_pos[1], 0])
    print(f"相机位置: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})m")

    # 体素网格范围估算（基于 voxel_config）
    voxel_config_path = os.path.join(data_dir, "voxel_config.json")
    voxel_range_x = 36.0  # 默认值，后续可从配置读取
    voxel_range_y = 30.0
    if os.path.exists(voxel_config_path):
        with open(voxel_config_path) as vf:
            vc = json.load(vf)
            voxel_range_x = vc.get("NX", 72) * vc.get("voxel_size", 0.2) / 2
            voxel_range_y = vc.get("NY", 60) * vc.get("voxel_size", 0.2) / 2

    print(f"体素覆盖范围: X±{voxel_range_x:.1f}m, Y±{voxel_range_y:.1f}m (以相机为中心)")

    # 分析每个 NPC 的位置变化
    npc_moved = []
    npc_in_range_count = [0] * num_npcs

    for npc_idx in range(num_npcs):
        positions = [r["positions"][npc_idx] for r in records]
        positions_arr = np.array(positions)

        # 计算移动距离
        first_pos = positions_arr[0]
        last_pos = positions_arr[-1]
        total_distance = np.linalg.norm(last_pos - first_pos)

        # 计算最大位移（任意两帧之间）
        max_displacement = 0
        for i in range(len(positions_arr)):
            for j in range(i + 1, len(positions_arr)):
                d = np.linalg.norm(positions_arr[j] - positions_arr[i])
                max_displacement = max(max_displacement, d)

        # 检查是否在体素范围内
        in_range_any = False
        for pos in positions:
            rel_x = pos[0] - voxel_origin[0]
            rel_y = pos[1] - voxel_origin[1]
            if abs(rel_x) <= voxel_range_x and abs(rel_y) <= voxel_range_y:
                in_range_any = True
                break
        if in_range_any:
            npc_in_range_count[npc_idx] = 1

        moved = total_distance > 0.01  # 超过 1cm 视为移动
        npc_moved.append(moved)

        status = "✅ 移动" if moved else "❌ 冻结"
        range_tag = "[范围内]" if in_range_any else "[范围外]"
        print(f"  NPC{npc_idx}: {status} 总行程={total_distance:.3f}m "
              f"最大位移={max_displacement:.3f}m {range_tag}")
        print(f"    首帧: ({first_pos[0]:.2f}, {first_pos[1]:.2f}) → "
              f"末帧: ({last_pos[0]:.2f}, {last_pos[1]:.2f})")

    moved_count = sum(npc_moved)
    in_range_count = sum(npc_in_range_count)

    if moved_count == 0:
        print(f"\n❌ 结论: 所有 {num_npcs} 个 NPC 完全未移动")
        return False, f"全部冻结 ({num_npcs}/{num_npcs})"
    elif moved_count < num_npcs:
        print(f"\n⚠️ 结论: {moved_count}/{num_npcs} 个 NPC 移动")
        return False, f"部分冻结 ({moved_count}/{num_npcs})"
    else:
        print(f"\n✅ 结论: 所有 {num_npcs} 个 NPC 都在移动")
        if in_range_count < num_npcs:
            print(f"   注意: {num_npcs - in_range_count} 个 NPC 在体素范围外")
        return True, f"全部移动 ({moved_count}/{num_npcs})"


def check4_flow_velocity(data_dir: str, frames: list[str]) -> tuple[bool, str]:
    """检查4：Flow 速度数据是否全零"""
    print("\n" + "=" * 70)
    print("📊 检查 4/5：Flow 速度数据分析")
    print("=" * 70)

    zero_flow_count = 0
    non_zero_flow_count = 0
    sampled_frames = frames[::max(1, len(frames) // 10)]  # 抽样 10 帧

    for f in sampled_frames:
        flow = load_npz(f, data_dir, "flow")
        flow_mask = load_npz(f, data_dir, "flow_mask")

        if flow is None or flow_mask is None:
            continue

        mask_bool = flow_mask.astype(bool)
        if not mask_bool.any():
            zero_flow_count += 1
            continue

        flow_values = flow[mask_bool]  # shape: (N, 2) vx, vy
        magnitudes = np.linalg.norm(flow_values, axis=1)
        mean_vel = magnitudes.mean()
        max_vel = magnitudes.max()

        if max_vel < 1e-6:
            zero_flow_count += 1
            print(f"  {f}: Flow 全零 (mask={mask_bool.sum()} 体素)")
        else:
            non_zero_flow_count += 1
            print(f"  {f}: Flow 有效 → 均速={mean_vel:.4f}m/s, "
                  f"最高={max_vel:.4f}m/s, 动态体素={mask_bool.sum()}")

    total_checked = zero_flow_count + non_zero_flow_count
    if total_checked == 0:
        print("❌ 无 Flow 数据文件")
        return False, "无 Flow 数据"

    print(f"\n抽样检查: {non_zero_flow_count}/{total_checked} 帧有非零 Flow")

    if non_zero_flow_count == 0:
        print("❌ 结论: 所有帧 Flow 全零 (NPC 未移动或 Flow 计算错误)")
        return False, "Flow 全零"
    elif non_zero_flow_count < total_checked * 0.5:
        print(f"⚠️ 结论: 大部分帧 Flow 为零")
        return False, f"大部分为零 ({non_zero_flow_count}/{total_checked})"
    else:
        print("✅ 结论: Flow 数据正常 (检测到运动)")
        return True, f"Flow 正常 ({non_zero_flow_count}/{total_checked})"


def check5_image_brightness(data_dir: str, frames: list[str]) -> tuple[bool, str]:
    """检查5：左右眼图像亮度抽查"""
    print("\n" + "=" * 70)
    print("📊 检查 5/5：图像亮度抽查")
    print("=" * 70)

    from PIL import Image

    sample_indices = [0, len(frames) // 2, len(frames) - 1]
    issues = []

    for idx in sample_indices:
        if idx >= len(frames):
            continue
        f = frames[idx]

        left_path = os.path.join(data_dir, "left", f"{f}.png")
        right_path = os.path.join(data_dir, "right", f"{f}.png")

        for side, path in [("左眼 L", left_path), ("右眼 R", right_path)]:
            if not os.path.exists(path):
                issues.append(f"{f} {side}: 文件不存在")
                continue

            img = Image.open(path).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            brightness = arr.mean()
            min_val = arr.min()
            max_val = arr.max()

            status = "✅" if brightness > 10 else "❌"
            print(f"  {status} {f} {side}: 亮度={brightness:.1f} "
                  f"(min={min_val:.0f}, max={max_val:.0f})")

            if brightness <= 10:
                issues.append(f"{f} {side}: 过暗 (brightness={brightness:.1f})")

    if issues:
        print(f"\n⚠️ 发现 {len(issues)} 个问题:")
        for issue in issues[:5]:
            print(f"   - {issue}")
        return False, f"{len(issues)} 个亮度异常"
    else:
        print("\n✅ 结论: 图像亮度正常")
        return True, "亮度正常"


def main():
    parser = argparse.ArgumentParser(description="体素数据集验证工具")
    parser.add_argument("--data_dir",
                        default=os.path.join(os.path.dirname(__file__),
                                            "..", "output_dng"),
                        help="数据集目录")
    args = parser.parse_args()

    data_dir = os.path.normpath(args.data_dir)
    print("=" * 70)
    print("🔍 体素数据集验证工具")
    print(f"📁 数据目录: {data_dir}")
    print("=" * 70)

    if not os.path.isdir(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        sys.exit(1)

    # 扫描帧列表
    voxel_dir = os.path.join(data_dir, "voxel")
    if not os.path.isdir(voxel_dir):
        print(f"❌ 体素目录不存在: {voxel_dir}")
        sys.exit(1)

    frames = sorted([f.replace("_semantic.npz", "")
                     for f in os.listdir(voxel_dir)
                     if f.endswith("_semantic.npz")])

    if not frames:
        print("❌ 未找到任何体素数据")
        sys.exit(1)

    print(f"发现 {len(frames)} 帧: {frames[0]} ~ {frames[-1]}")

    # 执行 5 项检查
    results = {}

    results["MD5唯一性"], msg1 = check1_voxel_md5_uniqueness(data_dir, frames)
    results["帧间差异"], msg2 = check2_frame_content_stats(data_dir, frames)
    results["NPC移动"], msg3 = check3_npc_position_change(data_dir, frames)
    results["Flow速度"], msg4 = check4_flow_velocity(data_dir, frames)
    results["图像亮度"], msg5 = check5_image_brightness(data_dir, frames)

    # 最终总结
    print("\n" + "=" * 70)
    print("📋 验证总结")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, (ok, msg) in zip(results.keys(), [msg1, msg2, msg3, msg4, msg5]):
        icon = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {icon}  {name:<10}: {msg}")

    print("-" * 70)
    overall_icon = "✅ 通过" if passed == total else "❌ 不通过"
    print(f"  {overall_icon}  总计: {passed}/{total} 项通过")

    if passed < total:
        print("\n💡 可能原因:")
        if not results["MD5唯一性"]:
            print("   • NPC 未移动 → 缺少 run_data_generation_async() 或 IRA 配置错误")
        if not results["NPC移动"]:
            print("   • NPC GoTo 命令未执行 → 检查 scene_setup.py 的 IRA 初始化")
        if not results["Flow速度"]:
            print("   • Flow 计算依赖位置差分，NPC 不动则全零")
        if not results["图像亮度"]:
            print("   • 图像可能未正确渲染或相机设置问题")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
