"""
推理脚本

用法:
  python inference.py --checkpoint checkpoints/best_model.pth --data_root D:/path/to/output_dng
  python inference.py --checkpoint checkpoints/best_model.pth --benchmark
"""
import torch
import numpy as np
import argparse
import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import StereoOccConfig
from e2e_occ_net import build_model
from dataset import StereoOccDataset


class StereoOccInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = ckpt.get('config', StereoOccConfig())
        self.model = build_model(self.config).to(self.device)
        missing, unexpected = self.model.load_state_dict(ckpt['model'], strict=False)
        if missing:
            print(f'[推理] 缺失键 (使用默认值): {missing}')
        if unexpected:
            print(f'[推理] 多余键 (忽略): {unexpected}')
        self.model.eval()
        print(f'[推理] 模型加载: {checkpoint_path}')
        print(f'[推理] 设备: {self.device}, 参数: {self.model.get_num_params() / 1e6:.2f}M')

    @torch.no_grad()
    def predict(self, images, intrinsics, extrinsics):
        """
        images: [B, 2, 1, H, W]
        returns: [B, 72, 60, 32] uint8
        """
        images = images.to(self.device)
        intrinsics = intrinsics.to(self.device)
        extrinsics = extrinsics.to(self.device)

        with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
            outputs = self.model(images, intrinsics, extrinsics)

        logits = outputs['semantic']  # [B, 18, 72, 60, 32]
        pred = logits.argmax(dim=1)   # [B, 72, 60, 32]
        return pred.cpu().numpy().astype(np.uint8)

    def benchmark(self, num_runs=100, warmup=10):
        cfg = self.config
        dummy_img = torch.randn(1, 2, 1, *cfg.image_size).to(self.device)
        dummy_K = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1).to(self.device)
        dummy_K[:, :, 0, 0] = cfg.focal_length_px
        dummy_K[:, :, 1, 1] = cfg.focal_length_px
        dummy_K[:, :, 0, 2] = cfg.cx
        dummy_K[:, :, 1, 2] = cfg.cy
        dummy_E = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1).to(self.device)

        for _ in range(warmup):
            with torch.no_grad():
                self.model(dummy_img, dummy_K, dummy_E)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                self.model(dummy_img, dummy_K, dummy_E)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        fps = num_runs / elapsed
        print(f'FPS: {fps:.2f}, Latency: {1000 / fps:.2f}ms')
        return fps


def run_inference_on_dataset(checkpoint_path, data_root, output_dir, num_samples=100, device='cuda'):
    engine = StereoOccInference(checkpoint_path, device)
    config = engine.config

    # 单帧推理
    config_infer = StereoOccConfig()
    config_infer.use_temporal = False
    dataset = StereoOccDataset(data_root, split='train', config=config_infer)

    total = min(num_samples, len(dataset))
    print(f'[推理] 数据集: {data_root}, 帧数: {len(dataset)}, 推理: {total}')

    out_dir = Path(output_dir)
    voxel_dir = out_dir / 'voxel'
    voxel_dir.mkdir(parents=True, exist_ok=True)

    written_ids = []
    for i in range(total):
        t0 = time.time()
        sample = dataset[i]
        images = sample['images'].unsqueeze(0)       # [1, 2, 1, H, W]
        intrinsics = sample['intrinsics'].unsqueeze(0)
        extrinsics = sample['extrinsics'].unsqueeze(0)

        pred = engine.predict(images, intrinsics, extrinsics)[0]  # [72, 60, 32]

        frame_id = sample['frame_id']
        np.savez_compressed(str(voxel_dir / f'{frame_id}_semantic.npz'), data=pred)
        written_ids.append(frame_id)

        non_empty = int(np.count_nonzero(pred))
        elapsed = time.time() - t0
        print(f'[{i + 1:4d}/{total}] {frame_id} non_empty={non_empty:,} {elapsed * 1000:.0f}ms', flush=True)

    # 写索引
    with open(out_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(written_ids))

    # 复制配置文件
    src_calib = Path(data_root) / 'calibration.json'
    src_voxel_cfg = Path(data_root) / 'voxel_config.json'
    if src_calib.exists():
        import shutil
        shutil.copy2(src_calib, out_dir / 'calibration.json')
    if src_voxel_cfg.exists():
        import shutil
        shutil.copy2(src_voxel_cfg, out_dir / 'voxel_config.json')

    print(f'\n[推理] 完成. 结果保存到: {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='双目体素占用推理')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--output', type=str, default='./inference_results')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()

    if args.benchmark:
        engine = StereoOccInference(args.checkpoint, args.device)
        engine.benchmark()
        return

    if args.data_root is None:
        print("错误: --data_root 必须指定")
        sys.exit(1)

    run_inference_on_dataset(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output,
        num_samples=args.num_samples,
        device=args.device,
    )


if __name__ == '__main__':
    main()
