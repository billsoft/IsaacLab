"""StereoRawRig: 双目 RAW 相机对

管理左右 RawCamera 实例，自动处理:
- USD StereoRig Xform 父节点创建
- 基线偏移和安装朝向
- 按间隔采集 + 黑帧跳过
- DNG/bin/PNG 异步保存
- manifest.json 输出
"""

import os
import time
import numpy as np
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from ..configs.dataclasses import SensorConfig, NoiseConfig, StereoConfig, OutputConfig
from ..core.raw_converter import RawConverter
from ..core.dng_writer import DngWriter
from .raw_camera import RawCamera


def _euler_to_quat(euler_deg: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    """欧拉角 (度) → 四元数 (w, x, y, z)。

    使用 Gf.Rotation 的 XYZ 旋转。
    Isaac Sim Camera 默认光轴 +X。
    euler=(0,0,90) 使 +X 指向世界 -Z（朝下）。
    """
    from pxr import Gf
    rx, ry, rz = euler_deg
    rot = Gf.Rotation(Gf.Vec3d(1, 0, 0), rx)
    rot = rot * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry)
    rot = rot * Gf.Rotation(Gf.Vec3d(0, 0, 1), rz)
    q = rot.GetQuat()
    real = q.GetReal()
    img = q.GetImaginary()
    return (real, img[0], img[1], img[2])


class StereoRawRig:
    """双目 RAW 相机对

    用法:
        rig = StereoRawRig(sensor_cfg, stereo_cfg, noise_cfg, output_cfg)
        rig.create(stage)
        rig.attach_annotators()

        for step in range(num_steps):
            sim.step()
            rig.try_capture(step)

        rig.finalize()
    """

    def __init__(
        self,
        sensor_cfg: SensorConfig,
        stereo_cfg: Optional[StereoConfig] = None,
        noise_cfg: Optional[NoiseConfig] = None,
        output_cfg: Optional[OutputConfig] = None,
    ):
        self._sensor_cfg = sensor_cfg
        self._stereo_cfg = stereo_cfg or StereoConfig()
        self._noise_cfg = noise_cfg
        self._output_cfg = output_cfg or OutputConfig()

        self.left: Optional[RawCamera] = None
        self.right: Optional[RawCamera] = None
        self._writer = DngWriter(sensor_cfg, self._output_cfg)

        self._frame_count = 0
        self._skipped = 0
        self._start_time = None

        # 异步 IO
        self._save_pool: Optional[ThreadPoolExecutor] = None
        self._pending = []

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def create(self, stage, parent_xform_path: Optional[str] = None):
        """创建双目相机 USD prims。

        Args:
            stage: USD stage
            parent_xform_path: 自定义父节点路径（默认用 StereoConfig.rig_prim_path）
        """
        from pxr import UsdGeom

        sc = self._stereo_cfg
        rig_path = parent_xform_path or sc.rig_prim_path
        half_bl = sc.baseline_m / 2.0

        # 创建父 Xform
        UsdGeom.Xform.Define(stage, rig_path)

        # 计算朝向四元数
        quat = _euler_to_quat(sc.mount_euler_deg)

        # 左眼: Y-  右眼: Y+
        left_path = f"{rig_path}/left_camera"
        right_path = f"{rig_path}/right_camera"

        self.left = RawCamera(left_path, self._sensor_cfg, self._noise_cfg)
        self.right = RawCamera(right_path, self._sensor_cfg, self._noise_cfg)

        self.left.create_prim(
            stage,
            translation=(0.0, -half_bl, sc.mount_height_m),
            orientation_quat=quat,
        )
        self.right.create_prim(
            stage,
            translation=(0.0, half_bl, sc.mount_height_m),
            orientation_quat=quat,
        )

        # 准备输出目录
        base = self._output_cfg.output_dir
        for sub in ["left", "right"]:
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        if self._output_cfg.save_rgb_preview:
            os.makedirs(os.path.join(base, "rgb_preview"), exist_ok=True)

        # 初始化异步保存
        if self._output_cfg.async_save:
            self._save_pool = ThreadPoolExecutor(
                max_workers=self._output_cfg.save_workers,
                thread_name_prefix="raw_saver",
            )

    def attach_annotators(self):
        """为左右相机创建 render_product 并挂接 annotator。

        必须在 sim.reset() 之后调用。
        """
        self.left.attach_annotator()
        self.right.attach_annotator()
        self._start_time = time.time()

    def try_capture(self, step: int) -> bool:
        """尝试采集一帧。

        根据 warmup_frames 和 capture_interval 决定是否采集。
        自动处理黑帧检测、RGB→RAW 转换、异步保存。

        Args:
            step: 当前仿真步数

        Returns:
            True 如果成功采集
        """
        oc = self._output_cfg

        # 预热阶段跳过
        if step < oc.warmup_frames:
            return False

        # 采集间隔
        effective_step = step - oc.warmup_frames
        if oc.capture_interval > 1 and effective_step % oc.capture_interval != 0:
            return False

        # 最大帧数
        if oc.max_frames > 0 and self._frame_count >= oc.max_frames:
            return False

        # 获取 RGB
        rgb_l = self.left.get_rgb()
        rgb_r = self.right.get_rgb()

        if rgb_l is None or rgb_r is None:
            self._skipped += 1
            return False

        # RGB → RAW
        raw_l = self.left.converter.rgb_to_raw(rgb_l)
        raw_r = self.right.converter.rgb_to_raw(rgb_r)

        frame_id = self._frame_count
        base = oc.output_dir

        # 保存 DNG
        if oc.save_dng:
            lp = os.path.join(base, "left", f"frame_{frame_id:06d}.dng")
            rp = os.path.join(base, "right", f"frame_{frame_id:06d}.dng")
            self._async_submit(self._writer.write, raw_l, lp)
            self._async_submit(self._writer.write, raw_r, rp)

        # 保存 bin
        if oc.save_bin:
            lp = os.path.join(base, "left", f"frame_{frame_id:06d}.bin")
            rp = os.path.join(base, "right", f"frame_{frame_id:06d}.bin")
            self._async_submit(lambda d, p: d.tofile(p), raw_l.copy(), lp)
            self._async_submit(lambda d, p: d.tofile(p), raw_r.copy(), rp)

        # 保存 RGB 预览
        if oc.save_rgb_preview:
            save_preview = (oc.rgb_preview_interval <= 0 or
                            frame_id % max(oc.rgb_preview_interval, 1) == 0)
            if save_preview:
                lp = os.path.join(base, "rgb_preview", f"frame_{frame_id:06d}_left.png")
                rp = os.path.join(base, "rgb_preview", f"frame_{frame_id:06d}_right.png")
                self._async_submit(self._save_png, rgb_l.copy(), lp)
                self._async_submit(self._save_png, rgb_r.copy(), rp)

        self._frame_count += 1

        # 进度日志
        if self._frame_count % 10 == 0 or self._frame_count == 1:
            elapsed = time.time() - self._start_time if self._start_time else 0
            fps = self._frame_count / elapsed if elapsed > 0 else 0
            print(f"  [RawCam] 已采集 {self._frame_count} 帧 ({fps:.1f} fps, 跳过 {self._skipped})")

        return True

    def finalize(self):
        """等待所有异步保存完成，写入 manifest.json"""
        # 等待异步保存
        for f in self._pending:
            f.result()
        self._pending.clear()

        # 写入 manifest
        sc = self._stereo_cfg
        elapsed = time.time() - self._start_time if self._start_time else 0
        extra = {
            "baseline_mm": sc.baseline_m * 1000,
            "camera_height_m": sc.mount_height_m,
            "mount_euler_deg": list(sc.mount_euler_deg),
            "noise_preset": self._noise_cfg.preset if self._noise_cfg else "clean",
            "skipped_frames": self._skipped,
            "elapsed_seconds": round(elapsed, 1),
            "fisheye_enabled": self._sensor_cfg.fisheye_enabled,
            "diagonal_fov_deg": self._sensor_cfg.diagonal_fov_deg,
        }
        self._writer.write_manifest(
            self._output_cfg.output_dir,
            self._frame_count,
            extra=extra,
        )

        total_bytes = self._frame_count * self._sensor_cfg.width * self._sensor_cfg.height * 2 * 2
        print(f"\n  [RawCam] 采集完成: {self._frame_count} 帧, "
              f"{total_bytes / 1024 / 1024:.1f} MB, "
              f"耗时 {elapsed:.1f}s")

    def destroy(self):
        """清理所有资源"""
        if self.left:
            self.left.destroy()
        if self.right:
            self.right.destroy()
        if self._save_pool:
            self._save_pool.shutdown(wait=True)
            self._save_pool = None

    def _async_submit(self, fn, *args):
        """提交异步任务"""
        if self._save_pool:
            self._pending.append(self._save_pool.submit(fn, *args))
        else:
            fn(*args)

    @staticmethod
    def _save_png(rgb_array: np.ndarray, path: str):
        """保存 RGB 为 PNG"""
        from PIL import Image
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        Image.fromarray(rgb_array.astype(np.uint8)).save(path)
