"""数据集完整性验证
===================
检查 capture_dataset 输出的所有数据是否符合规范（方向预测.md §6）。

不需要 Isaac Sim，用任意 Python 运行：
    python validate_dataset_v2.py --data_dir output_v2
"""

import argparse
import json
import os
import sys

import numpy as np


class DatasetValidator:
    """v2 数据集验证器。"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.stats: dict = {}

    def error(self, msg: str):
        self.errors.append(msg)
        print(f"  [ERROR] {msg}")

    def warn(self, msg: str):
        self.warnings.append(msg)
        print(f"  [WARN]  {msg}")

    def check_directory_structure(self):
        """检查目录和配置文件存在性。"""
        print("\n=== 目录结构 ===")
        required_dirs = ["left_dng", "right_dng", "voxel", "meta"]
        optional_dirs = ["left", "right", "trajectory"]
        required_files = ["calibration.json", "voxel_config.json"]
        optional_files = ["instance_meta.json", "npc_history.json"]

        for d in required_dirs:
            path = os.path.join(self.data_dir, d)
            if not os.path.isdir(path):
                self.error(f"Missing required directory: {d}/")
            else:
                count = len([f for f in os.listdir(path) if not f.startswith(".")])
                print(f"  {d}/: {count} files")

        for d in optional_dirs:
            path = os.path.join(self.data_dir, d)
            if os.path.isdir(path):
                count = len([f for f in os.listdir(path) if not f.startswith(".")])
                print(f"  {d}/: {count} files")

        for f in required_files:
            if not os.path.isfile(os.path.join(self.data_dir, f)):
                self.error(f"Missing required file: {f}")

        for f in optional_files:
            path = os.path.join(self.data_dir, f)
            if os.path.isfile(path):
                print(f"  {f}: OK")

    def discover_frames(self) -> list[str]:
        """发现所有帧 ID。"""
        voxel_dir = os.path.join(self.data_dir, "voxel")
        if not os.path.isdir(voxel_dir):
            return []
        frames = set()
        for f in os.listdir(voxel_dir):
            if f.endswith("_semantic.npz"):
                frames.add(f.replace("_semantic.npz", ""))
        return sorted(frames)

    def check_frame_completeness(self, frames: list[str]):
        """检查每帧文件齐全。"""
        print(f"\n=== 帧完整性 ({len(frames)} 帧) ===")
        missing_count = 0

        for frame_str in frames:
            required = [
                f"voxel/{frame_str}_semantic.npz",
                f"voxel/{frame_str}_instance.npz",
                f"meta/{frame_str}.json",
            ]
            # DNG 文件
            for side in ["left_dng", "right_dng"]:
                required.append(f"{side}/{frame_str}.dng")

            for path in required:
                full = os.path.join(self.data_dir, path)
                if not os.path.isfile(full):
                    self.error(f"Missing: {path}")
                    missing_count += 1

            # flow 是 v2 新增，检查存在性
            flow_path = os.path.join(self.data_dir, f"voxel/{frame_str}_flow.npz")
            if not os.path.isfile(flow_path):
                self.warn(f"Missing flow: voxel/{frame_str}_flow.npz")

        if missing_count == 0:
            print(f"  All {len(frames)} frames complete")

    def check_voxel_shapes(self, frames: list[str]):
        """检查体素数据形状。"""
        print(f"\n=== 体素形状 ===")
        expected_shape = (72, 60, 32)

        checked = 0
        for frame_str in frames[:10]:  # 抽查前 10 帧
            prefix = os.path.join(self.data_dir, "voxel", frame_str)

            sem = np.load(f"{prefix}_semantic.npz")["data"]
            if sem.shape != expected_shape:
                self.error(f"{frame_str} semantic shape {sem.shape} != {expected_shape}")
            if sem.dtype != np.uint8:
                self.error(f"{frame_str} semantic dtype {sem.dtype} != uint8")

            inst_path = f"{prefix}_instance.npz"
            if os.path.isfile(inst_path):
                inst = np.load(inst_path)["data"]
                if inst.shape != expected_shape:
                    self.error(f"{frame_str} instance shape {inst.shape} != {expected_shape}")
                if inst.dtype not in (np.uint16, np.int32):
                    self.warn(f"{frame_str} instance dtype {inst.dtype} (expected uint16)")

            flow_path = f"{prefix}_flow.npz"
            if os.path.isfile(flow_path):
                data = np.load(flow_path)
                flow = data["flow"]
                flow_mask = data["flow_mask"]
                if flow.shape != (*expected_shape, 2):
                    self.error(f"{frame_str} flow shape {flow.shape} != {(*expected_shape, 2)}")
                if flow_mask.shape != expected_shape:
                    self.error(f"{frame_str} flow_mask shape {flow_mask.shape}")

            checked += 1

        print(f"  Checked {checked} frames, shapes OK"
              if not any("shape" in e for e in self.errors) else "")

    def check_flow_values(self, frames: list[str]):
        """检查 flow 值合理性。"""
        print(f"\n=== Flow 统计 ===")

        all_speeds = []
        dynamic_voxel_counts = []

        for frame_str in frames:
            flow_path = os.path.join(self.data_dir, "voxel", f"{frame_str}_flow.npz")
            if not os.path.isfile(flow_path):
                continue
            data = np.load(flow_path)
            flow = data["flow"].astype(np.float32)
            mask = data["flow_mask"]

            dynamic_count = int(mask.sum())
            dynamic_voxel_counts.append(dynamic_count)

            if dynamic_count > 0:
                speeds = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                masked_speeds = speeds[mask == 1]
                all_speeds.append(masked_speeds)

                max_speed = float(masked_speeds.max())
                if max_speed > 30.0:
                    self.warn(f"{frame_str}: max flow speed {max_speed:.1f} m/s (>30)")

        if all_speeds:
            combined = np.concatenate(all_speeds)
            print(f"  Dynamic voxels/frame: mean={np.mean(dynamic_voxel_counts):.0f}, "
                  f"max={max(dynamic_voxel_counts)}")
            print(f"  Speed distribution:")
            print(f"    |v|=0 (stationary): {(combined < 0.01).mean()*100:.1f}%")
            print(f"    |v|<1 m/s: {(combined < 1.0).mean()*100:.1f}%")
            print(f"    |v| 1~3 m/s: {((combined >= 1.0) & (combined < 3.0)).mean()*100:.1f}%")
            print(f"    |v|>3 m/s: {(combined >= 3.0).mean()*100:.1f}%")
            print(f"    max: {combined.max():.2f} m/s")
        else:
            print(f"  No flow data found")

    def check_instance_consistency(self, frames: list[str]):
        """检查实例 ID 一致性。"""
        print(f"\n=== 实例 ID ===")
        instance_meta_path = os.path.join(self.data_dir, "instance_meta.json")

        if not os.path.isfile(instance_meta_path):
            self.warn("instance_meta.json not found, skipping instance checks")
            return

        with open(instance_meta_path) as f:
            meta = json.load(f)

        registered_ids = set(int(k) for k in meta.get("instances", {}).keys())
        print(f"  Registered instances: {len(registered_ids)}")

        # 抽查帧中的实例 ID
        for frame_str in frames[:10]:
            inst_path = os.path.join(self.data_dir, "voxel", f"{frame_str}_instance.npz")
            if not os.path.isfile(inst_path):
                continue
            inst = np.load(inst_path)["data"]
            unique_ids = set(int(x) for x in np.unique(inst))
            unique_ids.discard(0)       # 背景
            unique_ids.discard(65535)    # 未观测

            unregistered = unique_ids - registered_ids
            if unregistered:
                self.warn(f"{frame_str}: unregistered instance IDs: {unregistered}")

    def check_meta_timestamps(self, frames: list[str]):
        """检查 meta 时间戳单调递增。"""
        print(f"\n=== 时间戳 ===")
        timestamps = []
        for frame_str in frames:
            meta_path = os.path.join(self.data_dir, "meta", f"{frame_str}.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            t = meta.get("timestamp_sec")
            if t is not None:
                timestamps.append((frame_str, t))

        if len(timestamps) < 2:
            self.warn("Not enough timestamps to check monotonicity")
            return

        non_monotonic = 0
        for i in range(1, len(timestamps)):
            if timestamps[i][1] <= timestamps[i-1][1]:
                self.error(f"Non-monotonic timestamp: {timestamps[i-1][0]} "
                           f"({timestamps[i-1][1]:.3f}) >= {timestamps[i][0]} "
                           f"({timestamps[i][1]:.3f})")
                non_monotonic += 1

        if non_monotonic == 0:
            dt_values = [timestamps[i][1] - timestamps[i-1][1]
                        for i in range(1, len(timestamps))]
            mean_dt = np.mean(dt_values)
            std_dt = np.std(dt_values)
            print(f"  Time range: {timestamps[0][1]:.3f}s ~ {timestamps[-1][1]:.3f}s")
            print(f"  Frame interval: {mean_dt:.3f}s +/- {std_dt:.4f}s")
            print(f"  Monotonic: OK")

    def print_summary(self):
        """打印验证总结。"""
        print(f"\n{'=' * 50}")
        print(f"验证总结")
        print(f"{'=' * 50}")
        print(f"  错误 (ERROR): {len(self.errors)}")
        print(f"  警告 (WARN):  {len(self.warnings)}")

        if self.errors:
            print(f"\n致命错误列表:")
            for e in self.errors[:20]:
                print(f"  - {e}")
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more")

        if not self.errors:
            print(f"\n  数据集验证通过!")
        else:
            print(f"\n  数据集存在问题，请检查上述错误。")


def main():
    parser = argparse.ArgumentParser(description="Validate v2 dataset")
    parser.add_argument("--data_dir", required=True, help="Dataset directory")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: {args.data_dir} not found")
        sys.exit(1)

    validator = DatasetValidator(args.data_dir)

    validator.check_directory_structure()
    frames = validator.discover_frames()

    if not frames:
        print("\nNo frames found, nothing to validate.")
        sys.exit(1)

    print(f"\nDiscovered {len(frames)} frames: {frames[0]} ~ {frames[-1]}")

    validator.check_frame_completeness(frames)
    validator.check_voxel_shapes(frames)
    validator.check_flow_values(frames)
    validator.check_instance_consistency(frames)
    validator.check_meta_timestamps(frames)
    validator.print_summary()

    sys.exit(1 if validator.errors else 0)


if __name__ == "__main__":
    main()
