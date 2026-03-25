# RawCam 组件包设计文档

> 可复用的 12-bit RAW 相机仿真组件，用于 Isaac Sim 双目鱼眼场景

## 1. 背景与目标

### 1.1 问题

神经网络输入需要 12-bit Bayer RAW 数据（pre-ISP），而 Isaac Sim 只能渲染 8-bit sRGB RGB。
Isaac Sim 的 pre-ISP RAW 输出接口（`IsaacSensorCreateRTXLidarScanBuffer` 等）在 5.x 版本中不可用/不稳定。

### 1.2 方案

**伪 RAW 管线**：从 Isaac Sim 获取高质量 RGB，在 Python 层逆向转换为 12-bit Bayer RAW，并添加真实传感器噪声。

```
Isaac Sim RTX 渲染
    ↓ (RGBA uint8, replicator annotator)
sRGB → linear 解码
    ↓
RGB → RGGB Bayer CFA 马赛克
    ↓
12-bit 量化 (0-4095)
    ↓
传感器噪声注入 (shot + read + dark + PRNU + row)
    ↓
DNG 封装 (TIFF/CFA + DNG 标签)
```

### 1.3 已验证结果

| 指标 | 值 |
|------|-----|
| RAW→RGB 往返 PSNR | 32.7 dB（clean）/ 14.5 dB（noisy） |
| DNG 格式合规性 | tifffile 读回正确，含完整 DNG 标签 |
| 采集吞吐 | 双目 1280×1080，3帧间隔，1607 对图像 |
| 与 IRA NPC 共存 | 已验证，无冲突 |

---

## 2. 组件架构

```
rawcam/
├── __init__.py                    # 包入口，导出核心类
├── DESIGN.md                      # 本文档
│
├── core/                          # 核心引擎（不依赖 Isaac Sim）
│   ├── __init__.py
│   ├── raw_converter.py           # RawConverter: RGB→RAW 转换核心
│   ├── dng_writer.py              # DngWriter: DNG 文件写入（纯 tifffile，不依赖 exiftool）
│   ├── noise_model.py             # NoiseModel: 传感器噪声注入
│   └── bayer_mosaic.py            # BayerMosaic: CFA 马赛克操作
│
├── sim/                           # Isaac Sim 集成层
│   ├── __init__.py
│   ├── raw_camera.py              # RawCamera: 封装 USD prim + annotator + RAW 转换
│   ├── stereo_raw_rig.py          # StereoRawRig: 双目 RAW 相机对
│   └── fisheye_setup.py           # ftheta 鱼眼属性设置工具函数
│
├── configs/                       # 配置
│   ├── __init__.py
│   ├── sensor_presets.py          # 传感器预设（SC132GS, IMX678, ...）
│   └── stereo_camera_cfg.py       # 已有：Isaac Lab CameraCfg 配置
│
├── scripts/                       # 独立脚本（已有实验脚本）
│   ├── capture_stereo_raw12.py
│   ├── test_pseudo_raw.py
│   ├── verify_raw12.py
│   └── ...
│
└── tests/                         # 单元测试
    ├── test_raw_converter.py
    ├── test_dng_writer.py
    └── test_noise_model.py
```

### 2.1 分层设计

| 层 | 目录 | 依赖 | 职责 |
|----|------|------|------|
| **核心层** | `core/` | numpy, tifffile | RAW 转换、DNG 写入、噪声模型（可脱离 Isaac Sim 独立使用） |
| **仿真层** | `sim/` | core/ + Isaac Sim + omni.replicator | USD 相机创建、annotator 采集、RAW 转换集成 |
| **配置层** | `configs/` | dataclass | 传感器参数、相机参数、噪声预设 |

核心层可在普通 Python 环境中运行（离线处理、单元测试），仿真层必须在 Isaac Sim 环境中运行。

---

## 3. 可配置参数

### 3.1 传感器参数 (`SensorConfig`)

```python
@dataclass
class SensorConfig:
    """传感器硬件参数"""
    # 分辨率
    width: int = 1280
    height: int = 1080

    # 像素物理参数
    pixel_size_um: float = 2.7          # 像素尺寸 (μm)
    focal_length_mm: float = 1.75       # 焦距 (mm)

    # RAW 参数
    bit_depth: int = 12                 # 位深 (8/10/12/14)
    black_level: int = 0                # 黑电平 (仿真=0, 真实SC132GS=64)
    white_level: int = 4095             # 白电平 (2^bit_depth - 1)
    bayer_pattern: str = "RGGB"         # CFA 排列

    # 鱼眼参数（可选）
    fisheye_enabled: bool = True
    diagonal_fov_deg: float = 157.2     # 对角线 FOV (度)
    fisheye_model: str = "ftheta"       # "ftheta" | "opencv_fisheye" | "kannala_brandt"
    ftheta_coeffs: tuple = None         # (k0,k1,k2,k3,k4), None=自动计算等距投影

    # 光学
    f_stop: float = 0.0                 # 光圈 (0=无景深模糊)
    clipping_near: float = 0.01         # 近裁剪面 (m)
    clipping_far: float = 100.0         # 远裁剪面 (m)
```

### 3.2 噪声参数 (`NoiseConfig`)

```python
@dataclass
class NoiseConfig:
    """传感器噪声参数"""
    preset: str = "sc132gs"             # 预设名: "clean"|"light"|"sc132gs"|"heavy"|"custom"
    enabled: bool = True                # 是否启用噪声

    # 自定义参数（preset="custom" 时生效）
    read_noise_std: float = 4.0         # 读出噪声 σ (DN)
    shot_noise_gain: float = 1.0        # 散粒噪声增益
    dark_current_mean: float = 0.5      # 暗电流均值 (DN)
    dark_current_std: float = 0.3       # 暗电流 σ (DN)
    prnu_std: float = 0.01              # PRNU (比例, 1%=0.01)
    row_noise_std: float = 0.5          # 行噪声 σ (DN)

    seed: int = None                    # 随机种子 (None=不固定)
```

### 3.3 双目参数 (`StereoConfig`)

```python
@dataclass
class StereoConfig:
    """双目相机参数"""
    baseline_m: float = 0.08            # 基线距离 (m), 80mm
    rig_prim_path: str = "/World/StereoRig"  # USD 父节点路径

    # 安装朝向
    mount_euler_deg: tuple = (0, 0, 90)  # 欧拉角 (度), (0,0,90)=朝下
    mount_height_m: float = 3.0          # 安装高度 (m)
```

### 3.4 输出参数 (`OutputConfig`)

```python
@dataclass
class OutputConfig:
    """输出控制参数"""
    output_dir: str = "./output/stereo_raw12"

    # 输出格式
    save_dng: bool = True               # 保存 DNG 文件
    save_bin: bool = False              # 保存 .bin (uint16 裸数据)
    save_rgb_preview: bool = True       # 保存 RGB 预览 PNG
    save_npy: bool = False              # 保存 .npy (numpy 格式)

    # DNG 元数据
    dng_camera_make: str = "NVIDIA"
    dng_camera_model: str = "IsaacSim_SC132GS_Virtual"
    dng_color_matrix: tuple = (1,0,0, 0,1,0, 0,0,1)  # 恒等矩阵
    dng_calibration_illuminant: int = 21  # D65

    # 采集控制
    capture_interval: int = 1           # 每 N 步采集一帧
    warmup_frames: int = 30             # 预热帧数
    max_frames: int = -1                # 最大帧数 (-1=无限制)

    # 异步 IO
    async_save: bool = True             # 异步保存（ThreadPoolExecutor）
    save_workers: int = 2               # 保存线程数
```

### 3.5 预设传感器 (`sensor_presets.py`)

| 预设名 | 分辨率 | 像素 | 焦距 | FOV | 用途 |
|--------|--------|------|------|-----|------|
| `sc132gs` | 1280×1080 | 2.7μm | 1.75mm | 157.2° | 当前项目默认（双目鱼眼） |
| `imx678` | 3840×2160 | 1.12μm | 3.6mm | 120° | 高分辨率广角 |
| `ov9282` | 1280×800 | 3.0μm | 2.1mm | 110° | 低成本 SLAM 相机 |
| `custom` | 自定义 | 自定义 | 自定义 | 自定义 | 用户自定义参数 |

---

## 4. 核心 API

### 4.1 `RawConverter` — RGB→RAW 转换引擎

```python
class RawConverter:
    """RGB 到 12-bit Bayer RAW 的转换器"""

    def __init__(self, sensor_cfg: SensorConfig, noise_cfg: NoiseConfig = None):
        ...

    def rgb_to_raw(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """
        RGB uint8 (H,W,3) → 12-bit Bayer uint16 (H,W)
        完整管线: sRGB解码 → linear → CFA mosaic → 量化 → 噪声
        """

    def srgb_to_linear(self, img_uint8: np.ndarray) -> np.ndarray:
        """sRGB uint8 [0,255] → linear float32 [0,1]"""

    def linear_to_bayer(self, linear_rgb: np.ndarray) -> np.ndarray:
        """linear RGB float (H,W,3) → Bayer uint16 (H,W)，含量化"""

    def add_noise(self, bayer: np.ndarray) -> np.ndarray:
        """为 Bayer 数据添加传感器噪声"""

    @staticmethod
    def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
        """linear float [0,1] → sRGB uint8 [0,255]（验证用）"""
```

### 4.2 `DngWriter` — DNG 文件写入

```python
class DngWriter:
    """纯 tifffile 实现的 DNG 写入器（不依赖 exiftool）"""

    def __init__(self, sensor_cfg: SensorConfig, output_cfg: OutputConfig = None):
        ...

    def write(self, bayer_uint16: np.ndarray, path: str, frame_meta: dict = None):
        """
        写入单个 DNG 文件。
        bayer_uint16: (H,W) uint16 Bayer 数据
        frame_meta: 可选的额外帧元数据（曝光时间、ISO 等）
        """

    def write_stereo_pair(
        self, left: np.ndarray, right: np.ndarray,
        frame_id: int, output_dir: str = None
    ) -> tuple[str, str]:
        """写入一对双目 DNG，返回 (left_path, right_path)"""

    def write_manifest(self, output_dir: str, total_frames: int, extra: dict = None):
        """写入 manifest.json 元数据文件"""
```

### 4.3 `RawCamera` — Isaac Sim 相机封装

```python
class RawCamera:
    """
    封装 USD Camera prim + replicator annotator + RAW 转换。

    不使用 isaacsim.sensors.camera.Camera（避免 IRA stage 事件回调冲突），
    直接操作 USD prim + omni.replicator annotator。
    """

    def __init__(
        self,
        prim_path: str,
        sensor_cfg: SensorConfig,
        noise_cfg: NoiseConfig = None,
        stage=None,
    ):
        ...

    def create_prim(self, translation: tuple, orientation_quat: tuple):
        """在 USD stage 上创建相机 prim 并设置属性（含鱼眼）"""

    def setup_fisheye(self):
        """为已有 prim 设置 ftheta 鱼眼属性 (ApplyAPI + 参数)"""

    def attach_annotator(self):
        """创建 render_product 并附加 rgb annotator"""

    def get_rgb(self) -> np.ndarray | None:
        """
        从 annotator 获取 RGB 数据 (H,W,3) uint8。
        返回 None 如果数据不可用或为黑帧。
        """

    def capture_raw(self) -> np.ndarray | None:
        """
        获取 RGB 并转换为 12-bit Bayer RAW。
        返回 (H,W) uint16 或 None。
        一步完成: get_rgb() → RawConverter.rgb_to_raw()
        """

    def destroy(self):
        """清理 render_product 和 annotator"""
```

### 4.4 `StereoRawRig` — 双目 RAW 相机

```python
class StereoRawRig:
    """
    双目 RAW 相机对，管理左右 RawCamera 实例。

    典型用法:
        rig = StereoRawRig(sensor_cfg, stereo_cfg, noise_cfg, output_cfg)
        rig.create(stage)          # 创建 USD prims
        rig.attach_annotators()    # 挂接 annotators

        # 在仿真循环中:
        for step in range(num_steps):
            sim.step()
            rig.try_capture(step)  # 自动按 capture_interval 采集

        rig.finalize()             # 等待异步保存完成，写入 manifest
    """

    def __init__(
        self,
        sensor_cfg: SensorConfig,
        stereo_cfg: StereoConfig,
        noise_cfg: NoiseConfig = None,
        output_cfg: OutputConfig = None,
    ):
        self.left: RawCamera       # 左眼
        self.right: RawCamera      # 右眼
        self.converter: RawConverter
        self.writer: DngWriter

    def create(self, stage, parent_xform_path: str = None):
        """创建双目相机 USD prims（含 StereoRig 父节点）"""

    def attach_annotators(self):
        """为左右相机创建 render_product 并挂接 annotator"""

    def try_capture(self, step: int) -> bool:
        """
        尝试采集一帧。根据 capture_interval 决定是否采集。
        返回 True 如果成功采集。
        自动处理:
        - 黑帧检测与跳过
        - RGB→RAW 转换
        - DNG/bin/preview 异步保存
        """

    def finalize(self):
        """等待所有异步保存完成，写入 manifest.json"""

    def get_frame_count(self) -> int:
        """返回已成功采集的帧数"""

    def destroy(self):
        """清理所有资源"""
```

### 4.5 验证工具

```python
# core/verify.py
def verify_raw_roundtrip(
    bayer_uint16: np.ndarray,
    rgb_reference: np.ndarray,
    bit_depth: int = 12,
) -> dict:
    """
    验证 RAW 数据质量：简单 demosaic → 与原始 RGB 对比。
    返回: {"psnr": float, "mse": float, "ssim": float, "bayer_stats": dict}
    """

def verify_dng(dng_path: str) -> dict:
    """
    读取 DNG 文件并验证完整性。
    返回: {"shape": tuple, "dtype": str, "dng_tags": dict, "bayer_stats": dict}
    """
```

---

## 5. 集成到生产管线

### 5.1 与 `stereo_voxel_capture.py` 集成

当前生产脚本 `stereo_voxel_capture.py` 的相机采集流程：

```python
# 现有代码 (简化)
rp_left = rep.create.render_product(left_cam_path, (CAM_W, CAM_H))
annot_left = rep.AnnotatorRegistry.get_annotator("rgb")
annot_left.attach([rp_left.path])

# 每帧采集
data_l = annot_left.get_data()
rgb_l = data_l[:, :, :3]
async_save_image(left_path, rgb_l)   # → PNG
```

集成后：

```python
from rawcam import StereoRawRig, SensorConfig, StereoConfig, NoiseConfig, OutputConfig

# 替换手动相机创建
sensor_cfg = SensorConfig.from_preset("sc132gs")
stereo_cfg = StereoConfig(baseline_m=0.08, mount_euler_deg=(0, 0, 90))
noise_cfg = NoiseConfig(preset="sc132gs")
output_cfg = OutputConfig(save_dng=True, save_rgb_preview=True, capture_interval=3)

rig = StereoRawRig(sensor_cfg, stereo_cfg, noise_cfg, output_cfg)
rig.create(stage)
rig.attach_annotators()

# 仿真循环中
for step in range(num_steps):
    sim.step()
    rig.try_capture(step)    # 自动完成 RGB→RAW→DNG + 异步保存

rig.finalize()
```

### 5.2 与 IRA NPC 共存

关键约束（已验证可行）：
1. **不使用 `Camera` 类** → 避免 `_stage_open_callback_fn` 清掉回调
2. **直接 USD prim + replicator annotator** → 天然兼容 IRA 的 timeline 控制
3. **相机创建时机**：在 IRA `open_stage()` 之后、`setup_simulation()` 之前

### 5.3 Stage 单位适配

生产场景（full_warehouse）使用厘米（`metersPerUnit=0.01`），组件需自动处理：

```python
# RawCamera.create_prim() 内部自动适配
stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
stage_scale = 1.0 / stage_mpu  # cm场景=100, m场景=1
translation_stage = (tx * stage_scale, ty * stage_scale, tz * stage_scale)
```

---

## 6. 开发计划

### Phase 1：核心层重构（`core/`）

从已有 `utils/` 代码重构为清晰的类接口：

| 任务 | 来源 | 目标 |
|------|------|------|
| `RawConverter` 类 | `test_pseudo_raw.py` 内联函数 + `bayer_mosaic.py` | `core/raw_converter.py` |
| `DngWriter` 类 | `capture_stereo_raw12.py` 的 `write_dng()` + `dng_writer.py` | `core/dng_writer.py`（去掉 exiftool 依赖） |
| `NoiseModel` 类 | `noise_model.py` | `core/noise_model.py`（保持不变，已经很好） |
| `BayerMosaic` 工具 | `bayer_mosaic.py` | `core/bayer_mosaic.py`（保持不变） |
| 配置 dataclass | 散落在各脚本中的常量 | `configs/sensor_presets.py` |

### Phase 2：仿真层封装（`sim/`）

| 任务 | 来源 | 目标 |
|------|------|------|
| `RawCamera` 类 | `capture_stereo_raw12.py` 相机创建 + annotator 逻辑 | `sim/raw_camera.py` |
| `StereoRawRig` 类 | `capture_stereo_raw12.py` + `stereo_voxel_capture.py` 双目逻辑 | `sim/stereo_raw_rig.py` |
| `fisheye_setup` | `stereo_voxel_capture.py` 的 `set_fisheye_on_prim()` | `sim/fisheye_setup.py` |

### Phase 3：集成与测试

| 任务 | 说明 |
|------|------|
| 单元测试 | `core/` 全部可在普通 Python 中测试 |
| 集成测试 | `sim/` 需要 `isaaclab.bat -p` 运行 |
| 生产集成 | 替换 `stereo_voxel_capture.py` 中的手动相机和 PNG 保存 |

---

## 7. 依赖

### 核心层依赖（`core/`）

```
numpy
tifffile >= 2024.1.0
Pillow        # 仅验证/预览用
```

### 仿真层额外依赖（`sim/`）

```
Isaac Sim 5.x (isaacsim, omni.replicator, pxr)
isaaclab      # AppLauncher, sim_utils
opencv-python # 异步图像保存
```

所有包通过 `isaaclab.bat -p -m pip install` 安装。

---

## 8. 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| RGB→RAW 而非 pre-ISP | 伪 RAW | Isaac Sim 5.x 无稳定 pre-ISP 接口 |
| DNG 写入用 tifffile | 不依赖 exiftool | Windows 环境无 exiftool，tifffile extratags 足够 |
| 不使用 `Camera` 类 | USD prim + annotator | 避免 IRA stage 事件回调冲突 |
| ftheta 鱼眼模型 | ApplyAPI + k0-k4 | 唯一支持 >180° FOV 的原生渲染方式 |
| 异步保存 | ThreadPoolExecutor | 磁盘 IO 不阻塞渲染管线 |
| core/sim 分层 | 核心不依赖 Isaac Sim | 支持离线处理、单元测试、CI |
| DNG dtype 用数字码 | 1/3/4/12 而非 'B'/'H'/'L'/'d' | tifffile extratags 仅接受数字 TIFF dtype 码 |
