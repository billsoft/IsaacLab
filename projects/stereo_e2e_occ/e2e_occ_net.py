"""
双目 RAW12 端到端占用网络 (主模型)

Pipeline: StereoPatchEmbed → ImageEncoder → OccupancyDecoder → VoxelHead
"""
import torch
import torch.nn as nn

try:
    from .config import StereoOccConfig
    from .raw_embed import StereoPatchEmbed
    from .image_encoder import ImageEncoder
    from .occ_decoder import OccupancyDecoder
    from .voxel_head import VoxelHead
except ImportError:
    from config import StereoOccConfig
    from raw_embed import StereoPatchEmbed
    from image_encoder import ImageEncoder
    from occ_decoder import OccupancyDecoder
    from voxel_head import VoxelHead


class StereoOccNet(nn.Module):
    def __init__(self, config: StereoOccConfig = None):
        super().__init__()
        self.config = config or StereoOccConfig()
        self.patch_embed = StereoPatchEmbed(self.config)
        self.encoder = ImageEncoder(self.config)
        self.decoder = OccupancyDecoder(self.config)
        self.head = VoxelHead(self.config)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images, intrinsics=None, extrinsics=None, memory=None, ego_motion=None):
        """
        images: [B, 2, 1, H, W]  双目 RAW12
        intrinsics: [B, 2, 3, 3]
        extrinsics: [B, 2, 4, 4]
        memory: optional [B, Q, C]
        ego_motion: optional [B, 4, 4]

        returns: {
            'semantic': [B, 18, 72, 60, 32],  logits
            'memory': [B, Q, C] or None,
        }
        """
        feats = self.patch_embed(images)                                    # [B, 2, C, Hf, Wf]
        feats = self.encoder(feats, intrinsics, extrinsics)                 # [B, 2, C, Hf, Wf]
        voxel_feats, new_memory = self.decoder(                             # [B, fx, fy, fz, C]
            feats, intrinsics, extrinsics, memory, ego_motion,
        )
        head_out = self.head(voxel_feats)                                   # dict of tensors
        head_out['memory'] = new_memory
        return head_out

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config=None):
    config = config or StereoOccConfig()
    return StereoOccNet(config)
