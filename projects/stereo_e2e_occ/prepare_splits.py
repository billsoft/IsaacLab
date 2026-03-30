"""
数据集分割工具

从 stereo_voxel_capture_dng 输出目录生成 train.txt / val.txt 分割文件

用法:
  python prepare_splits.py --data_root D:/path/to/output_dng --val_ratio 0.2
"""
import argparse
import os
import glob
import random


def main():
    parser = argparse.ArgumentParser(description='生成训练/验证分割文件')
    parser.add_argument('--data_root', type=str, required=True, help='stereo_voxel_capture_dng 输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    # 发现所有有体素标签的帧
    voxel_dir = os.path.join(args.data_root, 'voxel')
    if not os.path.isdir(voxel_dir):
        print(f"错误: 找不到体素目录 {voxel_dir}")
        return

    pattern = os.path.join(voxel_dir, 'frame_*_semantic.npz')
    files = sorted(glob.glob(pattern))
    frame_ids = [os.path.basename(f).replace('_semantic.npz', '') for f in files]

    if not frame_ids:
        print("错误: 未找到任何体素文件")
        return

    # 验证对应的 DNG 文件存在
    valid_frames = []
    for fid in frame_ids:
        left = os.path.join(args.data_root, 'left_dng', f'{fid}.dng')
        right = os.path.join(args.data_root, 'right_dng', f'{fid}.dng')
        if os.path.exists(left) and os.path.exists(right):
            valid_frames.append(fid)
        else:
            print(f"[跳过] {fid}: 缺少 DNG 文件")

    print(f"有效帧: {len(valid_frames)} / {len(frame_ids)}")

    # 分割
    random.seed(args.seed)
    random.shuffle(valid_frames)
    n_val = max(1, int(len(valid_frames) * args.val_ratio))
    val_frames = sorted(valid_frames[:n_val])
    train_frames = sorted(valid_frames[n_val:])

    # 写入
    train_path = os.path.join(args.data_root, 'train.txt')
    val_path = os.path.join(args.data_root, 'val.txt')

    with open(train_path, 'w') as f:
        f.write('\n'.join(train_frames))
    with open(val_path, 'w') as f:
        f.write('\n'.join(val_frames))

    print(f"训练集: {len(train_frames)} 帧 → {train_path}")
    print(f"验证集: {len(val_frames)} 帧 → {val_path}")


if __name__ == '__main__':
    main()
