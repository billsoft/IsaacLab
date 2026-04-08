"""
逐阶段 shape 验证脚本

逐层跟踪张量 shape, 打印每个模块的输入输出维度,
确保整条管线 (输入 → PatchEmbed → Encoder → Decoder → Head → 输出) 维度正确对齐。

用法:
  python verify_shapes.py
  python verify_shapes.py --device cpu
"""
import torch
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import StereoOccConfig
from raw_embed import StereoPatchEmbed
from image_encoder import ImageEncoder
from occ_decoder import OccupancyDecoder
from voxel_head import VoxelHead
from loss import OccupancyLoss


def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def info(tag, tensor):
    if tensor is None:
        print(f"  {tag:40s} = None")
    else:
        shape_str = str(list(tensor.shape))
        print(f"  {tag:40s} = {shape_str:30s}  dtype={tensor.dtype}  dev={tensor.device}")


def gpu_mem(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  [GPU] {tag:20s} alloc={alloc:.0f}MB  peak={peak:.0f}MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config = StereoOccConfig()

    # =====================================================================
    sep("0. 配置信息")
    # =====================================================================
    print(f"  num_cameras       = {config.num_cameras}")
    print(f"  image_size (H,W)  = {config.image_size}")
    print(f"  feat_size  (H,W)  = {config.feat_size}")
    print(f"  embed_dim         = {config.embed_dim}")
    print(f"  num_heads         = {config.num_heads}")
    print(f"  coarse_size       = {config.coarse_size}  ({config.num_coarse_queries} queries)")
    print(f"  fine_size         = {config.fine_size}  ({config.num_fine_queries} queries)")
    print(f"  voxel_size        = {config.voxel_size}")
    print(f"  voxel_range       = {config.voxel_range}")
    print(f"  num_classes       = {config.num_classes}")
    print(f"  focal_length_px   = {config.focal_length_px}")
    print(f"  baseline_m        = {config.baseline_m}")
    print(f"  device            = {device}")

    # =====================================================================
    sep("1. 构建各阶段模块")
    # =====================================================================
    patch_embed = StereoPatchEmbed(config).to(device)
    encoder = ImageEncoder(config).to(device)
    decoder = OccupancyDecoder(config).to(device)
    head = VoxelHead(config).to(device)
    criterion = OccupancyLoss(num_classes=config.num_classes)

    def count_params(m, name):
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name:25s} params = {n:>10,d}  ({n/1e6:.2f}M)")

    count_params(patch_embed, "StereoPatchEmbed")
    count_params(encoder, "ImageEncoder")
    count_params(decoder, "OccupancyDecoder")
    count_params(head, "VoxelHead")
    total = sum(sum(p.numel() for p in m.parameters()) for m in [patch_embed, encoder, decoder, head])
    print(f"  {'TOTAL':25s} params = {total:>10,d}  ({total/1e6:.2f}M)")

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    gpu_mem("after model init")

    # =====================================================================
    sep("2. 构造输入张量")
    # =====================================================================
    B = 1
    N = config.num_cameras  # 2
    C_in = config.raw_channels  # 1
    H, W = config.image_size  # 1080, 1280

    images = torch.randn(B, N, C_in, H, W, device=device)
    info("images (输入)", images)

    # 内参
    intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
    intrinsics[:, :, 0, 0] = config.focal_length_px
    intrinsics[:, :, 1, 1] = config.focal_length_px
    intrinsics[:, :, 0, 2] = config.cx
    intrinsics[:, :, 1, 2] = config.cy
    info("intrinsics", intrinsics)

    # 外参
    extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
    extrinsics[0, 0, 1, 3] = config.baseline_m / 2   # 左眼
    extrinsics[0, 1, 1, 3] = -config.baseline_m / 2  # 右眼
    info("extrinsics", extrinsics)

    # =====================================================================
    sep("3. StereoPatchEmbed: 图像 → 特征图")
    # =====================================================================
    print(f"  管线: [B,N,1,H,W] → RGGB_conv(stride=2) → stem(stride=8) → LayerNorm")
    print(f"  总 stride = 2 * 2 * 2 * 2 = 16")
    print(f"  期望输出: [B, N, {config.embed_dim}, {H//16}, {W//16}]")
    print()

    with torch.no_grad():
        feats = patch_embed(images)
    info("patch_embed 输出", feats)
    gpu_mem("after patch_embed")

    Hf, Wf = feats.shape[3], feats.shape[4]
    assert feats.shape == (B, N, config.embed_dim, Hf, Wf), f"shape 不匹配!"
    print(f"  -> 特征图尺寸: Hf={Hf}, Wf={Wf} (期望 {config.feat_size})")

    # =====================================================================
    sep("4. ImageEncoder: 窗口注意力 + 射线方向编码")
    # =====================================================================
    print(f"  管线: 逐相机处理 × {N} cameras")
    print(f"    - FisheyeRayEncoding: f-theta 等距投影 → 正弦编码 → MLP")
    print(f"    - EncoderBlock × {config.encoder_layers}: WindowAttn(ws=7) + MLP")
    print(f"    - LayerNorm")
    print()

    with torch.no_grad():
        enc_feats = encoder(feats, intrinsics, extrinsics)
    info("encoder 输出", enc_feats)
    gpu_mem("after encoder")

    assert enc_feats.shape == feats.shape, "encoder 应保持 shape 不变!"

    # =====================================================================
    sep("5. OccupancyDecoder: 粗→精两阶段")
    # =====================================================================
    cx, cy, cz = config.coarse_size
    fx, fy, fz = config.fine_size

    print(f"  --- 5a. 粗阶段 ---")
    print(f"    可学习 query: [1, {config.num_coarse_queries}, {config.embed_dim}]")
    print(f"    + 3D 正弦位置编码: ({cx}, {cy}, {cz}) → [{cx*cy*cz}, {config.embed_dim}]")
    print(f"    参考点: [{cx*cy*cz}, 3] in [0,1] → 世界坐标 → f-theta 鱼眼投影 → 图像坐标")
    print(f"    DeformableDecoderLayer × {config.decoder_layers}:")
    print(f"      self_attn={config.use_self_attention}, cross_attn(N={N}, pts={config.num_sample_points})")
    print(f"    输出: [{cx}, {cy}, {cz}, {config.embed_dim}] → permute → [C, {cx}, {cy}, {cz}]")
    print()
    print(f"  --- 5b. 精阶段 ---")
    print(f"    trilinear: [{cx},{cy},{cz}] → [{fx},{fy},{fz}]")
    print(f"    coarse_to_fine MLP: {config.embed_dim} → {config.embed_dim*2} → {config.embed_dim}")
    print(f"    + 3D 正弦位置编码: ({fx}, {fy}, {fz})")
    print(f"    DeformableDecoderLayer × {config.decoder_layers}:")
    print(f"      self_attn={config.use_fine_self_attention}")
    print(f"    + depthwise Conv3d(k=3, groups={config.embed_dim})")
    print(f"    输出: [B, {fx}, {fy}, {fz}, {config.embed_dim}]")
    print()

    with torch.no_grad():
        voxel_feats, new_memory = decoder(enc_feats, intrinsics, extrinsics)
    info("decoder voxel_feats", voxel_feats)
    info("decoder memory", new_memory)
    gpu_mem("after decoder")

    assert voxel_feats.shape == (B, fx, fy, fz, config.embed_dim), \
        f"期望 [{B},{fx},{fy},{fz},{config.embed_dim}], 得到 {list(voxel_feats.shape)}"

    # =====================================================================
    sep("6. VoxelHead: 精特征 → 体素 logits")
    # =====================================================================
    vx, vy, vz = config.voxel_size
    mid_ch = config.embed_dim // 4
    print(f"  管线:")
    print(f"    permute → [B, {config.embed_dim}, {fx}, {fy}, {fz}]")
    print(f"    reduce:  {config.embed_dim} → {config.embed_dim//2} → {mid_ch}  (Conv3d k=3)")
    print(f"    trilinear: [{fx},{fy},{fz}] → [{vx},{vy},{vz}]")
    print(f"    refine + skip: {mid_ch} → {mid_ch}  (Conv3d k=3 + Conv3d k=1)")
    print(f"    cls_head + cls_skip: {mid_ch} → {config.num_classes}")
    print(f"  期望输出: [B, {config.num_classes}, {vx}, {vy}, {vz}]")
    print()

    with torch.no_grad():
        outputs = head(voxel_feats)
    info("head semantic", outputs['semantic'])
    if 'flow' in outputs:
        info("head flow", outputs['flow'])
    if 'orientation' in outputs:
        info("head orientation", outputs['orientation'])
    if 'angular_vel' in outputs:
        info("head angular_vel", outputs['angular_vel'])
    gpu_mem("after head")

    expected_logits = (B, config.num_classes, vx, vy, vz)
    assert outputs['semantic'].shape == expected_logits, f"期望 {expected_logits}, 得到 {list(outputs['semantic'].shape)}"

    # =====================================================================
    sep("7. Loss 计算")
    # =====================================================================
    target = torch.randint(0, config.num_classes, (B, vx, vy, vz), device=device)
    # 随机标记 10% 为 UNOBSERVED
    mask = torch.rand(B, vx, vy, vz, device=device) < 0.1
    target[mask] = config.ignore_index
    info("target (含 ignore)", target)

    flow_target   = torch.randn(B, 2, vx, vy, vz, device=device) if config.predict_flow else None
    flow_mask     = torch.randint(0, 2, (B, vx, vy, vz), device=device) if config.predict_flow else None
    orient_target = torch.randn(B, 1, vx, vy, vz, device=device) if config.predict_orientation else None
    angvel_target = torch.randn(B, 1, vx, vy, vz, device=device) if config.predict_angular_vel else None

    losses = criterion(outputs, target, flow_target, flow_mask, orient_target, angvel_target)
    print(f"  total  = {losses['total'].item():.4f}")
    print(f"  ce     = {losses['ce'].item():.4f}")
    if 'flow' in losses:
        print(f"  flow   = {losses['flow'].item():.4f}")
    if 'orient' in losses:
        print(f"  orient = {losses['orient'].item():.4f}")
    if 'angvel' in losses:
        print(f"  angvel = {losses['angvel'].item():.4f}")

    # =====================================================================
    sep("8. 反向传播测试")
    # =====================================================================
    # 重新前向 (需要梯度)
    patch_embed.train()
    encoder.train()
    decoder.train()
    head.train()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    feats2 = patch_embed(images)
    enc2 = encoder(feats2, intrinsics, extrinsics)
    vox2, _ = decoder(enc2, intrinsics, extrinsics)
    outputs2 = head(vox2)
    losses2 = criterion(outputs2, target, flow_target, flow_mask, orient_target, angvel_target)
    losses2['total'].backward()

    gpu_mem("after backward")

    # 检查梯度
    has_grad = 0
    no_grad = 0
    for name, p in list(patch_embed.named_parameters()) + list(encoder.named_parameters()) + \
                     list(decoder.named_parameters()) + list(head.named_parameters()):
        if p.requires_grad:
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad += 1
            else:
                no_grad += 1
    print(f"  有梯度的参数: {has_grad}")
    print(f"  无梯度的参数: {no_grad}")

    # =====================================================================
    sep("9. 完整管线 shape 汇总")
    # =====================================================================
    print(f"""
  输入:
    images      [B, 2, 1, {H}, {W}]          双目 RAW12, 归一化 [0,1]
    intrinsics  [B, 2, 3, 3]                  f-theta 鱼眼内参
    extrinsics  [B, 2, 4, 4]                  Camera→World

  StereoPatchEmbed (stride=16):
    → feats     [B, 2, {config.embed_dim}, {Hf}, {Wf}]

  ImageEncoder ({config.encoder_layers}× WindowAttn + FisheyeRayEncoding):
    → enc_feats [B, 2, {config.embed_dim}, {Hf}, {Wf}]

  OccupancyDecoder:
    粗阶段:     {config.num_coarse_queries} queries ({cx}×{cy}×{cz})
      → coarse  [B, {config.embed_dim}, {cx}, {cy}, {cz}]
    精阶段:     {config.num_fine_queries} queries ({fx}×{fy}×{fz})
      → fine    [B, {fx}, {fy}, {fz}, {config.embed_dim}]

  VoxelHead (trilinear upsample + Conv3d):
    → semantic  [B, {config.num_classes}, {vx}, {vy}, {vz}]
    → flow      [B,  2, {vx}, {vy}, {vz}]  (可选)
    → orient    [B,  1, {vx}, {vy}, {vz}]  (可选)
    → angvel    [B,  1, {vx}, {vy}, {vz}]  (可选)

  Loss (CE + Smooth-L1):
    → scalar

  输出:
    pred        [B, {vx}, {vy}, {vz}]          argmax, uint8 [0-17]
""")

    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  GPU 峰值显存: {peak:.2f} GB")

    print("=" * 70)
    print("  所有阶段 shape 验证通过!")
    print("=" * 70)


if __name__ == '__main__':
    main()
