"""
网络验证: 前向传播 + 反向传播 + 显存测试

用法:
  python test.py
  python test.py --device cpu
"""
import torch
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import StereoOccConfig
from e2e_occ_net import build_model
from loss import OccupancyLoss


def test_forward(device_str='cuda'):
    print("=" * 60)
    print("StereoOccNet 验证 (72x60x32 输出)")
    print("=" * 60)

    config = StereoOccConfig()
    print(f"\n配置:")
    print(f"  图像: {config.image_size} (H, W)")
    print(f"  特征图: {config.feat_size}")
    print(f"  相机数: {config.num_cameras}")
    print(f"  粗查询: {config.coarse_size} ({config.num_coarse_queries} queries)")
    print(f"  精查询: {config.fine_size} ({config.num_fine_queries} queries)")
    print(f"  体素: {config.voxel_size}")
    print(f"  感知范围: {config.voxel_range}")
    print(f"  分辨率: {config.voxel_resolution}m")
    print(f"  类别数: {config.num_classes}")
    print(f"  embed_dim: {config.embed_dim}")

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    model = build_model(config).to(device)
    num_params = model.get_num_params()
    print(f"参数量: {num_params / 1e6:.2f}M")

    # 输入
    B = 1
    images = torch.randn(B, 2, 1, *config.image_size).to(device)
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, 2, 1, 1).to(device)
    intrinsics[:, :, 0, 0] = config.focal_length_px
    intrinsics[:, :, 1, 1] = config.focal_length_px
    intrinsics[:, :, 0, 2] = config.cx
    intrinsics[:, :, 1, 2] = config.cy
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, 2, 1, 1).to(device)

    print(f"\n输入: {images.shape}")

    # 推理
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        outputs = model(images, intrinsics, extrinsics)

    logits = outputs['semantic']
    print(f"输出: {logits.shape}")
    expected = (B, config.num_classes, *config.voxel_size)
    assert logits.shape == expected, f"期望 {expected}, 得到 {logits.shape}"

    if device.type == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"推理显存峰值: {mem:.2f} GB")

    # 训练 (反向传播)
    target = torch.randint(0, config.num_classes, (B, *config.voxel_size)).to(device)
    criterion = OccupancyLoss(num_classes=config.num_classes)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    model.train()
    outputs = model(images, intrinsics, extrinsics)
    losses = criterion(outputs['semantic'], target)
    losses['total'].backward()

    if device.type == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"训练显存峰值: {mem:.2f} GB")

    print(f"\nLoss: {losses['total'].item():.4f}")
    print(f"CE: {losses['ce'].item():.4f}")
    print(f"Lovasz: {losses['lovasz'].item():.4f}")

    print("\n" + "=" * 60)
    print("验证通过!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    test_forward(args.device)
