"""
数据集加载器

直接对齐 stereo_voxel_capture_dng.py 的输出目录结构:
  output_dng/
  ├── calibration.json          ← 鱼眼内参
  ├── voxel_config.json         ← 体素网格参数
  ├── left_dng/frame_XXXXXX.dng ← 左眼 12-bit Bayer RAW
  ├── right_dng/frame_XXXXXX.dng← 右眼 12-bit Bayer RAW
  ├── voxel/frame_XXXXXX_semantic.npz ← (72,60,32) uint8
  └── meta/frame_XXXXXX.json    ← 帧级元数据 (cam_pos, cam_yaw_rad)
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import glob


class StereoOccDataset(Dataset):
    """双目 RAW12 → 体素占用数据集"""

    def __init__(self, data_root, split='train', config=None):
        self.data_root = data_root
        self.split = split
        self.config = config
        self.image_size = config.image_size if config else (1080, 1280)
        self.voxel_size = config.voxel_size if config else (72, 60, 32)

        # 时序
        self.sequence_length = 1
        if config and config.use_temporal:
            self.sequence_length = config.temporal_frames

        # 加载帧列表
        self.frames = self._discover_frames()

        # 加载标定
        self.calibration = self._load_calibration()
        self.intrinsics, self.extrinsics = self._build_camera_matrices()

    def _discover_frames(self):
        """发现所有可用帧 (按帧号排序)"""
        # 优先使用 split 文件
        split_file = os.path.join(self.data_root, f'{self.split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]

        # 否则扫描 voxel 目录
        voxel_dir = os.path.join(self.data_root, 'voxel')
        if not os.path.isdir(voxel_dir):
            return []
        pattern = os.path.join(voxel_dir, 'frame_*_semantic.npz')
        files = sorted(glob.glob(pattern))
        # frame_000000_semantic.npz → frame_000000
        return [os.path.basename(f).replace('_semantic.npz', '') for f in files]

    def _load_calibration(self):
        """加载 calibration.json"""
        calib_path = os.path.join(self.data_root, 'calibration.json')
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                return json.load(f)
        return {}

    @staticmethod
    def _camera_down_rotation(cam_yaw=0.0):
        """
        构建朝下相机的 Camera→World 旋转矩阵

        推导过程:
          1. USD 相机默认光轴 = -Z (Z-up 场景中即朝下)
          2. 数据采集 prim euler=[0,0,90] → R_z(90°) 绕 Z 旋转
          3. f-theta 投影约定: 光轴 = +Z, X=图像右, Y=图像下 (OpenCV 风格)
          4. USD→f-theta: diag(1, -1, -1)
          5. R_cam2world = R_z(yaw) @ R_z(90°) @ diag(1, -1, -1)

        结果 (yaw=0):
          f-theta +Z (光轴) → World -Z (向下)  ✓
          f-theta +X (图像右) → World +Y (左)
          f-theta +Y (图像下) → World +X (前)
        """
        cos_y = np.cos(cam_yaw)
        sin_y = np.sin(cam_yaw)
        # R = R_z(yaw + 90°) @ diag(1, -1, -1), 化简后:
        R = torch.tensor([
            [-sin_y, cos_y, 0.0],
            [cos_y, sin_y, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=torch.float32)
        return R

    def _build_camera_matrices(self):
        """
        从 calibration.json 构建内参和外参矩阵

        内参 [2, 3, 3]: 两个相机相同 (fx, fy, cx, cy)
        外参 [2, 4, 4]: 左眼 +baseline/2, 右眼 -baseline/2 (Y 轴)
        """
        calib = self.calibration
        fx = calib.get('fx', 648.148)
        fy = calib.get('fy', 648.148)
        cx = calib.get('cx', 640.0)
        cy = calib.get('cy', 540.0)
        baseline = calib.get('baseline_m', 0.08)

        # 内参 (鱼眼, fx=fy=焦距像素)
        K = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)  # [2, 3, 3]
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy

        # 外参: 相机 → 世界 (体素局部坐标系)
        # 相机朝下 (USD prim euler=[0,0,90]), 位于 z=camera_height
        # 左眼在 Y=+baseline/2, 右眼在 Y=-baseline/2
        R = self._camera_down_rotation(cam_yaw=0.0)
        E = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)  # [2, 4, 4]
        E[:, :3, :3] = R
        E[0, 1, 3] = baseline / 2   # 左眼
        E[1, 1, 3] = -baseline / 2  # 右眼

        return K, E

    def _load_dng(self, path):
        """加载 12-bit DNG 文件, 归一化到 [0, 1]"""
        try:
            import tifffile
            raw = tifffile.imread(path)  # (H, W) uint16
            img = raw.astype(np.float32) / 4095.0
            return torch.from_numpy(img[np.newaxis, :, :])  # [1, H, W]
        except ImportError:
            pass

        try:
            import rawpy
            with rawpy.imread(path) as raw:
                img = raw.raw_image_visible.astype(np.float32)
                img = img / 4095.0
                return torch.from_numpy(img[np.newaxis, :, :])
        except Exception as e:
            print(f"[WARN] Failed to load DNG {path}: {e}")

        return torch.zeros(1, *self.image_size)

    def _load_voxel(self, frame_id):
        """加载体素语义标签 (72, 60, 32) uint8"""
        npz_path = os.path.join(self.data_root, 'voxel', f'{frame_id}_semantic.npz')
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            return torch.from_numpy(data['data']).long()

        # fallback: npy
        npy_path = os.path.join(self.data_root, 'voxel', f'{frame_id}_semantic.npy')
        if os.path.exists(npy_path):
            return torch.from_numpy(np.load(npy_path)).long()

        return torch.zeros(self.voxel_size, dtype=torch.long)

    def _load_meta(self, frame_id):
        """加载帧级元数据"""
        meta_path = os.path.join(self.data_root, 'meta', f'{frame_id}.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                return json.load(f)
        return {}

    def _get_extrinsics_for_frame(self, frame_id):
        """
        根据帧元数据计算当前帧的绝对外参

        stereo_voxel_capture_dng.py 中:
        - 相机在世界坐标 (cam_x, cam_y, cam_height) 位置
        - 相机朝下: USD prim euler = [0, 0, 90], 光轴指向 -Z (地面)
        - 体素局部原点 = 相机地面投影点

        外参 = Camera→World 变换 [R | t]:
          R: 朝下相机旋转 (含 yaw)
          t: 相机在体素局部坐标系中的位置
        """
        meta = self._load_meta(frame_id)
        cam_pos = meta.get('camera_pos', [0.0, 0.0, 3.0])
        cam_yaw = meta.get('camera_yaw_rad', 0.0)
        baseline = self.calibration.get('baseline_m', 0.08)

        # Camera→World 旋转 (包含朝下 + yaw)
        R = self._camera_down_rotation(cam_yaw)

        E = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)  # [2, 4, 4]
        E[:, :3, :3] = R

        # 相机高度
        E[:, 2, 3] = cam_pos[2]  # z = height

        # 双目基线偏移 (Y 方向)
        E[0, 1, 3] = baseline / 2   # 左眼
        E[1, 1, 3] = -baseline / 2  # 右眼

        return E

    def _load_single_frame(self, idx):
        frame_id = self.frames[idx]

        # 1. 双目图像 [2, 1, H, W]
        left_path = os.path.join(self.data_root, 'left_dng', f'{frame_id}.dng')
        right_path = os.path.join(self.data_root, 'right_dng', f'{frame_id}.dng')
        left_img = self._load_dng(left_path)
        right_img = self._load_dng(right_path)
        images = torch.stack([left_img, right_img], dim=0)  # [2, 1, H, W]

        # 2. 体素标签 [X, Y, Z]
        voxels = self._load_voxel(frame_id)

        # 3. 相机参数
        intrinsics = self.intrinsics.clone()  # [2, 3, 3]
        extrinsics = self._get_extrinsics_for_frame(frame_id)  # [2, 4, 4]

        return {
            'images': images.float(),
            'voxels': voxels,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'frame_id': frame_id,
        }

    def __len__(self):
        if self.sequence_length > 1:
            return max(0, len(self.frames) - self.sequence_length + 1)
        return len(self.frames)

    def __getitem__(self, idx):
        if self.sequence_length == 1:
            return self._load_single_frame(idx)

        frames = [self._load_single_frame(idx + t) for t in range(self.sequence_length)]
        return {
            'images': torch.stack([f['images'] for f in frames]),         # [T, 2, 1, H, W]
            'voxels': torch.stack([f['voxels'] for f in frames]),         # [T, X, Y, Z]
            'intrinsics': frames[0]['intrinsics'],                        # [2, 3, 3]
            'extrinsics': torch.stack([f['extrinsics'] for f in frames]), # [T, 2, 4, 4]
            'frame_id': frames[0]['frame_id'],
        }


def get_dataloader(data_root, split='train', batch_size=1, num_workers=0, config=None):
    dataset = StereoOccDataset(data_root, split, config)
    shuffle = split == 'train'
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=shuffle,
    )
