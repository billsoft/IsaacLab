"""
训练脚本

用法:
  python train.py --data_root D:/path/to/output_dng --epochs 100 --amp
  python train.py --data_root D:/path/to/output_dng --resume checkpoints/best_model.pth --amp
"""
import torch
import torch.nn as nn
import argparse
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 当作独立脚本运行时, 确保包目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import StereoOccConfig
from e2e_occ_net import build_model
from loss import OccupancyLoss
from dataset import get_dataloader


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                use_amp=False, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    start = time.time()
    optimizer.zero_grad()

    TBPTT_CHUNK_SIZE = 2

    for i, batch in enumerate(loader):
        images = batch['images'].to(device)
        voxels = batch['voxels'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)

        is_sequence = images.dim() == 6  # [B, T, N, C, H, W]

        # 回归标签（可选，无则为 None）
        flow_target   = batch.get('flow',        None)
        flow_mask     = batch.get('flow_mask',   None)
        orient_target = batch.get('orientation', None)
        angvel_target = batch.get('angular_vel', None)
        if flow_target   is not None: flow_target   = flow_target.to(device)
        if flow_mask     is not None: flow_mask     = flow_mask.to(device)
        if orient_target is not None: orient_target = orient_target.to(device)
        if angvel_target is not None: angvel_target = angvel_target.to(device)

        loss_val = 0.0
        ce_loss_val = 0.0

        if is_sequence:
            B, T, N, C, H, W = images.shape
            memory = None
            seq_loss = 0.0
            seq_ce = 0.0

            for t_start in range(0, T, TBPTT_CHUNK_SIZE):
                t_end = min(t_start + TBPTT_CHUNK_SIZE, T)
                if memory is not None:
                    memory = memory.detach()

                chunk_loss = torch.tensor(0.0, device=device)
                total_weight = 0.0

                for t in range(t_start, t_end):
                    img_t = images[:, t]
                    vox_t = voxels[:, t]
                    ext_t = extrinsics[:, t]
                    fl_t  = flow_target[:,   t] if flow_target   is not None else None
                    fm_t  = flow_mask[:,     t] if flow_mask     is not None else None
                    or_t  = orient_target[:, t] if orient_target is not None else None
                    av_t  = angvel_target[:, t] if angvel_target is not None else None

                    ego_motion = None
                    if t > 0:
                        ext_prev = extrinsics[:, t - 1]
                        pose_t = ext_t[:, 0]
                        pose_prev = ext_prev[:, 0]
                        ego_motion = torch.linalg.inv(pose_t) @ pose_prev

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = model(img_t, intrinsics, ext_t, memory=memory, ego_motion=ego_motion)
                        loss_dict = criterion(outputs, vox_t, fl_t, fm_t, or_t, av_t)
                        time_weight = 1.0 + (t / max(1, T - 1))
                        chunk_loss = chunk_loss + loss_dict['total'] * time_weight
                        total_weight += time_weight

                    seq_loss += loss_dict['total'].item()
                    seq_ce += loss_dict['ce'].item()
                    memory = outputs['memory']

                chunk_loss_norm = chunk_loss / (total_weight * grad_accum_steps)
                scaler.scale(chunk_loss_norm).backward()

            loss_val = seq_loss / T
            ce_loss_val = seq_ce / T
        else:
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images, intrinsics, extrinsics)
                losses = criterion(outputs, voxels,
                                   flow_target, flow_mask, orient_target, angvel_target)
                loss = losses['total'] / grad_accum_steps
                loss_val = losses['total'].item()
                ce_loss_val = losses['ce'].item()
            scaler.scale(loss).backward()

        if (i + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss_val * grad_accum_steps
        elapsed = time.time() - start
        print(f'Epoch {epoch} [{i + 1}/{len(loader)}] Loss: {loss_val:.4f} CE: {ce_loss_val:.4f} Time: {elapsed:.1f}s',
              flush=True)

    # 处理尾部未凑满 grad_accum_steps 的残余梯度
    if len(loader) % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(len(loader), 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            voxels = batch['voxels'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)

            flow_target   = batch.get('flow',        None)
            flow_mask     = batch.get('flow_mask',   None)
            orient_target = batch.get('orientation', None)
            angvel_target = batch.get('angular_vel', None)
            if flow_target   is not None: flow_target   = flow_target.to(device)
            if flow_mask     is not None: flow_mask     = flow_mask.to(device)
            if orient_target is not None: orient_target = orient_target.to(device)
            if angvel_target is not None: angvel_target = angvel_target.to(device)

            is_sequence = images.dim() == 6
            if is_sequence:
                B, T, N, C, H, W = images.shape
                memory = None
                seq_loss = 0.0
                for t in range(T):
                    img_t = images[:, t]
                    vox_t = voxels[:, t]
                    ext_t = extrinsics[:, t]
                    fl_t  = flow_target[:,   t] if flow_target   is not None else None
                    fm_t  = flow_mask[:,     t] if flow_mask     is not None else None
                    or_t  = orient_target[:, t] if orient_target is not None else None
                    av_t  = angvel_target[:, t] if angvel_target is not None else None
                    ego_motion = None
                    if t > 0:
                        pose_t = ext_t[:, 0]
                        pose_prev = extrinsics[:, t - 1][:, 0]
                        ego_motion = torch.linalg.inv(pose_t) @ pose_prev
                    outputs = model(img_t, intrinsics, ext_t, memory=memory, ego_motion=ego_motion)
                    loss_dict = criterion(outputs, vox_t, fl_t, fm_t, or_t, av_t)
                    seq_loss += loss_dict['total'].item()
                    memory = outputs['memory']
                total_loss += seq_loss / T
            else:
                outputs = model(images, intrinsics, extrinsics)
                losses = criterion(outputs, voxels,
                                   flow_target, flow_mask, orient_target, angvel_target)
                total_loss += losses['total'].item()

    torch.cuda.empty_cache()
    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description='双目 RAW12 体素占用网络训练')
    parser.add_argument('--data_root', type=str, required=True, help='stereo_voxel_capture_dng 的输出目录')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (Windows 默认 0)')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的 checkpoint 路径')
    parser.add_argument('--amp', action='store_true', help='启用混合精度训练')
    parser.add_argument('--grad_accum', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--temporal', action='store_true', help='启用时序融合')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = StereoOccConfig()
    if args.temporal:
        config.use_temporal = True
    start_epoch = 0

    if args.resume:
        print(f"从 {args.resume} 恢复...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if 'config' in ckpt:
            config = ckpt['config']
            print("已从 checkpoint 加载配置")

    model = build_model(config).to(device)
    criterion = OccupancyLoss(num_classes=config.num_classes, ignore_index=config.ignore_index)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    if args.resume:
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        if missing:
            print(f'缺失键 (使用默认值): {missing}')
        if unexpected:
            print(f'多余键 (忽略): {unexpected}')
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
            print("已恢复 scheduler 状态")
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
            print("已恢复 scaler 状态")
        start_epoch = ckpt['epoch'] + 1
        print(f"从 epoch {start_epoch - 1} 恢复成功")

    num_params = model.get_num_params()
    print(f'模型参数量: {num_params / 1e6:.2f}M')
    print(f'设备: {device}')
    print(f'体素: {config.voxel_size}, 范围: {config.voxel_range}')
    print(f'粗查询: {config.coarse_size} ({config.num_coarse_queries})')
    print(f'精查询: {config.fine_size} ({config.num_fine_queries})')
    print(f'AMP={args.amp}, GradAccum={args.grad_accum}, Temporal={config.use_temporal}')

    train_loader = get_dataloader(args.data_root, 'train', args.batch_size, args.num_workers, config)
    val_loader = get_dataloader(args.data_root, 'val', args.batch_size, args.num_workers, config)

    print(f'训练集: {len(train_loader.dataset)} 帧, 验证集: {len(val_loader.dataset)} 帧')

    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            use_amp=args.amp, grad_accum_steps=args.grad_accum,
        )
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}', flush=True)

        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'loss': val_loss,
                'config': config,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'保存最佳模型 epoch {epoch}', flush=True)


if __name__ == '__main__':
    main()
