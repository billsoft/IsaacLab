# Isaac Lab / Isaac Sim 5.x 双目相机 12-bit RAW 输出与 DNG 封装调研报告

> **调研日期**: 2026-03-22  
> **调研范围**: NVIDIA Isaac Sim 5.0 / 5.1、Isaac Lab (最新版)、DNG 1.6 规范  
> **文档定位**: 技术可行性分析 + 工程实施路线

---

## 一、核心结论（先说结论）

| 问题 | 结论 |
|------|------|
| Isaac Lab 能否直接输出 12-bit RAW？ | **不能**。Camera / TiledCamera API 仅支持 `torch.uint8`（RGB/RGBA）和 `torch.float32`（深度）|
| Isaac Sim 5.x 有无 RAW 相关能力？ | **有限**。5.1 新增 pre-ISP camera pipeline 示例，可获取 HDR buffer 和 raw sensor output，但为 float 格式，非 12-bit integer |
| DNG 能否封装 12-bit Bayer RAW？ | **可以**。DNG 基于 TIFF 6.0 扩展，原生支持 CFA Bayer 模式和任意 2-16 bit 深度 |
| 最终可行方案？ | 从 Isaac Sim pre-ISP 拿 float raw → 量化为 12-bit uint16 → 用 tifffile + exiftool 封装为 DNG |

---

## 二、Isaac Lab 相机传感器能力分析

### 2.1 Isaac Lab Camera 类架构

Isaac Lab 提供两种主要相机实现：

**Camera 类**：封装 `UsdGeom.Camera` prim，通过 Omniverse Replicator API 的 Annotator 机制获取各类渲染数据。每次渲染一个相机视角。

**TiledCamera 类**：面向强化学习多环境并行场景优化的向量化接口。将所有相机克隆的渲染合并到一张大图中，显著降低 GPU-CPU 数据传输开销。（Isaac Sim 4.2+ 可用）

### 2.2 支持的数据类型（完整列表）

根据官方文档，Camera / TiledCamera 支持以下 annotator 数据类型：

| 数据类型 | 维度 | 数据格式 | 说明 |
|----------|------|----------|------|
| `rgb` | (B, H, W, 3) | `torch.uint8` | 3 通道 RGB 图像 |
| `rgba` | (B, H, W, 4) | `torch.uint8` | 4 通道 RGBA 图像 |
| `distance_to_camera` | (B, H, W, 1) | `torch.float32` | 到光学中心距离 |
| `distance_to_image_plane` / `depth` | (B, H, W, 1) | `torch.float32` | 到成像平面距离 |
| `normals` | (B, H, W, 3) | `torch.float32` | 局部表面法向量 |
| `motion_vectors` | (B, H, W, 2) | `torch.float32` | 帧间像素运动向量 |
| `semantic_segmentation` | (B, H, W, 1/4) | `int32/uint8` | 语义分割 |
| `instance_segmentation_fast` | (B, H, W, 1/4) | `int32/uint8` | 实例分割 |
| `instance_id_segmentation_fast` | (B, H, W, 1/4) | `int32/uint8` | 实例 ID 分割 |

**关键发现**：

- **没有任何 RAW / Bayer 类型的 annotator**
- RGB 输出固定为 `uint8`（8-bit），无法配置为 12-bit 或 16-bit
- 所有图像数据都经过了完整的 ISP 处理（去马赛克、色彩空间转换、Tone mapping 等）
- 这是渲染引擎（RTX renderer）的固有特性——RTX 渲染的是 "完美" 的 RGB 图像，不存在 Bayer CFA 滤色器

### 2.3 Isaac Lab 中的双目相机配置

Isaac Lab 支持多相机配置，但有一个重要限制：

**Camera 类**：可以创建多个独立 Camera 实例，每个有独立的 prim_path 和 offset，天然支持双目配置。

**TiledCamera 类**：由于渲染器限制，场景中只能有一个 TiledCamera 实例。双目场景需要通过在渲染帧之间移动相机位置来模拟，官方文档给出了示例代码：

```python
# 渲染 "第一个" 相机的图像
camera_data_1 = self._tiled_camera.data.output["rgb"].clone() / 255.0

# 更新相机位姿到 "第二个" 相机位置
self._tiled_camera.set_world_poses(
    positions=pos, orientations=rot, convention="world"
)

# 步进渲染器
self.sim.render()
self._tiled_camera.update(0, force_recompute=True)

# 渲染 "第二个" 相机的图像
camera_data_2 = self._tiled_camera.data.output["rgb"].clone() / 255.0
```

**双目相机配置示例（Camera 类，推荐方案）**：

```python
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

BASELINE = 0.08  # 80mm 基线距离

# 左眼相机
left_camera_cfg = CameraCfg(
    prim_path="/World/Robot/left_camera",
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 100.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, -BASELINE/2, 0.0),  # 左移 40mm
        rot=(0.5, -0.5, 0.5, -0.5),   # 朝向地面（绕 X 轴旋转 90°）
        convention="ros",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    width=1280,
    height=1024,
)

# 右眼相机
right_camera_cfg = CameraCfg(
    prim_path="/World/Robot/right_camera",
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 100.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, BASELINE/2, 0.0),   # 右移 40mm
        rot=(0.5, -0.5, 0.5, -0.5),   # 朝向地面
        convention="ros",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    width=1280,
    height=1024,
)
```

> **注意**：`convention="ros"` 下，相机前轴为 +Z，上轴为 -Y。要让相机朝向地面（-Z world），需要适当的四元数旋转。

---

## 三、Isaac Sim 5.x pre-ISP Pipeline 分析

### 3.1 5.1 新增功能

Isaac Sim 5.1 新增了 pre-ISP camera pipeline 的 standalone 示例，这是目前最接近 "RAW 数据" 的官方途径。

**示例路径**：

```bash
./python.sh standalone_examples/api/isaacsim.sensors.camera/camera_pre_isp_pipeline.py --draw-output
```

**输出三个阶段的数据**：

| 阶段 | 内容 | 数据格式 |
|------|------|----------|
| HDR buffer | 高动态范围线性渲染缓冲区 | float32（线性光照空间）|
| Raw sensor output | 模拟传感器原始输出（含 CFA 编码和 companding） | float（非 integer，不是真正的 12-bit RAW）|
| ISP output | 经过模拟 ISP 处理的最终图像 | 标准 RGB |

### 3.2 pre-ISP 输出的本质

这里必须澄清一个关键区别：

**真实相机传感器的 12-bit RAW**：
- 光子 → 模拟信号 → ADC 量化为 12-bit 整数（0-4095）
- 单通道 Bayer 马赛克图案（RGGB/GRBG 等）
- 包含真实噪声（读出噪声、暗电流、热噪声等）
- 包含真实的像素非均匀性（PRNU）

**Isaac Sim pre-ISP raw sensor output**：
- RTX 光线追踪渲染 → HDR float buffer → 模拟 CFA 编码 + companding
- 本质是 float 类型的渲染缓冲区，不是 integer
- 噪声模型是人工添加的（如果有的话），非真实传感器噪声
- 主要用途是测试自定义 ISP 算法，而非生成真实传感器数据

### 3.3 Isaac Sim pre-ISP 数据获取后的处理路线

尽管不是"真正的" 12-bit RAW，但如果你的目标是：
- 训练 RAW-to-RGB 的 ISP 网络
- 测试 Bayer 去马赛克算法
- 模拟嵌入式相机处理管线

那么 pre-ISP 输出经过量化后，是可以作为 **伪 12-bit RAW** 使用的。

---

## 四、DNG 格式深度解析

### 4.1 DNG 的本质

DNG（Digital Negative）是 Adobe 于 2004 年推出的开放 RAW 图像格式，核心要点：

- **基于 TIFF 6.0 扩展**，兼容 TIFF/EP（ISO 12234-2）标准
- **本质是 TIFF 文件 + 额外的 DNG 专用标签**
- 支持 2-16 bit 任意位深度的数据存储
- 支持 Bayer CFA（Color Filter Array）原始数据
- 支持无损压缩（Lossless JPEG-92）和有损压缩

### 4.2 DNG 对 12-bit Bayer RAW 的封装机制

DNG 封装 12-bit Bayer RAW 数据时，文件结构如下：

```
┌─────────────────────────────────────────┐
│  TIFF Header (8 bytes)                   │
│  - Byte order (II/MM)                    │
│  - Magic number (42)                     │
│  - Offset to first IFD                   │
├─────────────────────────────────────────┤
│  IFD 0 (Main Image)                      │
│  ┌─────────────────────────────────────┐│
│  │ 标准 TIFF 标签                       ││
│  │ - ImageWidth / ImageLength           ││
│  │ - BitsPerSample = 12                 ││
│  │ - Compression = 1 (无压缩)           ││
│  │ - PhotometricInterpretation = 32803  ││
│  │   (CFA = Color Filter Array)         ││
│  │ - StripOffsets / StripByteCounts     ││
│  ├─────────────────────────────────────┤│
│  │ DNG 扩展标签                         ││
│  │ - DNGVersion = 1.4.0.0              ││
│  │ - UniqueCameraModel                  ││
│  │ - CFARepeatPatternDim = [2, 2]       ││
│  │ - CFAPattern = [0,1,1,2] (RGGB)     ││
│  │ - BlackLevel = 0                     ││
│  │ - WhiteLevel = 4095                  ││
│  │ - ColorMatrix1 (3x3)                ││
│  │ - CalibrationIlluminant1             ││
│  │ - AsShotNeutral                      ││
│  └─────────────────────────────────────┘│
├─────────────────────────────────────────┤
│  SubIFD (缩略图预览, 可选)               │
│  - JPEG 预览图                           │
├─────────────────────────────────────────┤
│  RAW Image Data                          │
│  - 12-bit 像素数据（packed 或 16-bit     │
│    容器中的 12-bit 有效位）              │
├─────────────────────────────────────────┤
│  XMP Metadata (可选)                     │
│  EXIF Metadata (可选)                    │
└─────────────────────────────────────────┘
```

### 4.3 DNG 必要标签清单（12-bit Bayer RAW）

| 标签名称 | Tag ID | 必要性 | 典型值 | 说明 |
|----------|--------|--------|--------|------|
| DNGVersion | 50706 | 必须 | 1.4.0.0 | DNG 规范版本 |
| UniqueCameraModel | 50708 | 必须 | 自定义字符串 | 相机型号标识 |
| CFARepeatPatternDim | 33421 | 必须 | [2, 2] | CFA 重复单元尺寸 |
| CFAPattern | 33422 | 必须 | [0,1,1,2] | RGGB 排列 |
| BlackLevel | 50714 | 推荐 | 0 | 黑电平 |
| WhiteLevel | 50717 | 推荐 | 4095 | 12-bit 白电平 |
| ColorMatrix1 | 50721 | 必须 | 3×3 矩阵 | CFA→XYZ 色彩矩阵 |
| CalibrationIlluminant1 | 50778 | 必须 | 21 (D65) | 校准光源 |
| PhotometricInterpretation | 262 | 必须 | 32803 | CFA 光度解释 |
| BitsPerSample | 258 | 必须 | 12 | 每样本位数 |

### 4.4 DNG vs 裸 RAW12 二进制文件

| 特性 | 裸 .raw12 | DNG |
|------|-----------|-----|
| 文件结构 | 纯像素数据 | TIFF 容器 + 元数据 |
| 元数据 | 无 | CFA 模式、黑白电平、色彩矩阵等 |
| 软件兼容性 | 需要自行指定参数 | Adobe Camera Raw、RawTherapee、darktable 等直接打开 |
| 文件大小 | 最小 | 略大（多了 metadata） |
| 标准化 | 无标准 | Adobe DNG 1.x 规范 |
| 适用场景 | 嵌入式快速存储 | 存档、后处理、交换 |

---

## 五、完整工程方案：伪 12-bit RAW → DNG

### 5.1 方案总览

```
Isaac Sim 5.1 pre-ISP Pipeline
        │
        ▼
  HDR float buffer (渲染引擎输出)
        │
        ▼ 色彩校正 + CFA 编码 + Companding
        │
  raw sensor output (float)
        │
        ▼ 步骤1: 量化为 12-bit
        │
  uint16 array [0, 4095]
        │
        ▼ 步骤2: tifffile 写入 TIFF/DNG 基础结构
        │
  proto-DNG 文件 (.dng)
        │
        ▼ 步骤3: exiftool 补全 DNG 必要标签
        │
  合规 DNG 文件 ✅
```

### 5.2 Python 实现代码

```python
import numpy as np
from tifffile import TiffWriter
import subprocess
import os


def float_raw_to_dng(
    raw_float: np.ndarray,
    width: int,
    height: int,
    output_path: str = "output.dng",
    cfa_pattern: str = "RGGB",
    camera_make: str = "NVIDIA",
    camera_model: str = "IsaacSim_PreISP_Virtual",
):
    """
    将 Isaac Sim pre-ISP float 输出转换为合规 DNG 文件。

    参数:
        raw_float: 从 pre-ISP pipeline 获取的 float 数组，shape=(H, W)
        width: 图像宽度
        height: 图像高度
        output_path: 输出 DNG 文件路径
        cfa_pattern: Bayer 排列模式
        camera_make: 相机制造商标识
        camera_model: 相机型号标识
    """

    # ===== 步骤 1: float → 12-bit uint16 量化 =====
    # 归一化到 [0, 1] 范围（如果尚未归一化）
    if raw_float.max() > 1.0:
        raw_normalized = raw_float / raw_float.max()
    else:
        raw_normalized = raw_float

    # 量化到 12-bit (0-4095)
    raw_12bit = np.clip(raw_normalized * 4095.0, 0, 4095).astype(np.uint16)

    # 确保 shape 正确
    if raw_12bit.shape != (height, width):
        raw_12bit = raw_12bit.reshape(height, width)

    # ===== 步骤 2: 用 tifffile 写入 DNG/TIFF 基础结构 =====
    with TiffWriter(output_path, bigtiff=False) as tif:
        tif.write(
            raw_12bit,
            photometric=32803,      # CFA (Color Filter Array)
            compression=1,          # 无压缩
            bitspersample=12,
            subfiletype=0,
            metadata=None,
        )

    # ===== 步骤 3: 用 exiftool 补全 DNG 必要元数据 =====
    cmd = [
        "exiftool",
        f"-DNGVersion=1.4.0.0",
        f"-DNGBackwardVersion=1.4.0.0",
        f"-CFARepeatPatternDim=2 2",
        f"-CFAPattern={cfa_pattern}",
        f"-BlackLevel=0",
        f"-WhiteLevel=4095",
        f"-BitsPerSample=12",
        f"-PhotometricInterpretation=32803",
        f"-Make={camera_make}",
        f"-Model={camera_model}",
        # 最小化色彩矩阵（恒等映射，适合仿真数据）
        "-ColorMatrix1=1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0",
        "-CalibrationIlluminant1=21",  # D65
        "-AsShotNeutral=1.0 1.0 1.0",  # 无白平衡偏移
        "-overwrite_original",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"DNG 文件生成完成: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path)} bytes")


def save_stereo_dng(
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    width: int,
    height: int,
    output_dir: str = "./stereo_dng_output",
    frame_id: int = 0,
):
    """
    保存双目 DNG 文件对。

    参数:
        left_raw: 左眼 pre-ISP float 数据
        right_raw: 右眼 pre-ISP float 数据
        width: 图像宽度
        height: 图像高度
        output_dir: 输出目录
        frame_id: 帧编号
    """
    os.makedirs(output_dir, exist_ok=True)

    left_path = os.path.join(output_dir, f"frame_{frame_id:06d}_left.dng")
    right_path = os.path.join(output_dir, f"frame_{frame_id:06d}_right.dng")

    float_raw_to_dng(left_raw, width, height, left_path,
                     camera_model="IsaacSim_Stereo_Left")
    float_raw_to_dng(right_raw, width, height, right_path,
                     camera_model="IsaacSim_Stereo_Right")

    print(f"双目 DNG 帧 {frame_id} 保存完成")
```

### 5.3 依赖安装

```bash
pip install tifffile numpy --break-system-packages
sudo apt install libimage-exiftool-perl   # exiftool
```

---

## 六、局限性与替代方案

### 6.1 当前方案的根本局限

| 局限 | 说明 | 影响程度 |
|------|------|----------|
| 非真实传感器噪声 | Isaac Sim 渲染器的噪声模型与真实 CMOS 传感器差异很大 | 高 |
| CFA 编码是后处理模拟 | RTX 渲染器先生成完美 RGB，再人工加上 Bayer 马赛克，与真实传感器的物理过程相反 | 高 |
| 无法控制 ADC 非线性 | 真实传感器的 ADC 有非线性响应，仿真中不存在 | 中 |
| pre-ISP API 非公开 | Isaac Sim 5.1 的 pre-ISP pipeline 示例可用，但 API 尚未在 Isaac Lab 层面封装暴露 | 中 |
| TiledCamera 单实例限制 | 大规模并行环境中双目场景需要额外的渲染步骤 | 低-中 |

### 6.2 如果目标是训练 RAW→RGB ISP 网络

如果你的最终目标是在嵌入式平台（如 RDK S100P）上部署 Tesla 风格的 RAW-to-AI 管线，建议考虑：

**方案 A（推荐）: Isaac Sim pre-ISP + 噪声增强**
1. 使用 Isaac Sim 5.1 pre-ISP pipeline 获取基础 raw sensor output
2. 在 Python 后处理中添加传感器噪声模型（读出噪声、暗电流、光子散粒噪声等）
3. 量化为 12-bit 并封装为 DNG
4. 优点：场景丰富、可大规模生成；缺点：噪声分布与真实传感器有差距

**方案 B: 混合数据策略**
1. 用 Isaac Sim 生成大量 GT（Ground Truth）标注数据
2. 用真实相机（SC132GS）采集少量 paired 数据
3. 用 Domain Adaptation 或 CycleGAN 方法弥合 sim-to-real gap

**方案 C: 直接在 Isaac Lab 层面获取 HDR float**
1. 如果不需要 Bayer pattern，可以直接获取 Isaac Lab 的 RGB float32 数据
2. 将 `torch.uint8` 的 RGB 转为 float32（除以 255.0）
3. 虽然精度损失（只有 8-bit 有效信息），但流程最简单

---

## 七、方案可行性评估

> 基于 NVIDIA 官方文档（Isaac Sim 5.1 Camera Sensors、Isaac Lab Camera API）+ DNG 1.6 规范

**结论**：**方案总体可行**，但属于**伪 RAW 方案**（pseudo 12-bit RAW），不是真正的硬件传感器 RAW 数据。

### 7.1 逐项验证

| 项目 | 可行性 | 说明 |
|------|--------|------|
| 从 Isaac Sim 5.1 获取 raw sensor output | ✅ 可行（有限） | 官方 standalone 示例 `camera_pre_isp_pipeline.py` 可拿到 `raw_sensor_output`（float 格式） |
| 将 float raw 量化为 12-bit uint16 | ✅ 可行 | 数学上简单，`np.clip * 4095` |
| 使用 tifffile 写入 DNG 基础结构 | ✅ 可行 | tifffile 支持 `photometric=32803`（CFA）和 `bitspersample=12` |
| 使用 exiftool 补全 DNG 元数据 | ✅ 可行 | 社区最常用、最稳定的方式 |
| 生成的 DNG 能否被 Adobe / RawTherapee 正常打开 | ✅ 可行 | 只要 CFA、BlackLevel、WhiteLevel 正确即可 |
| 是否等于真实 SC132GS 的 12-bit RAW | ❌ 否 | 缺少真实噪声、PRNU、ADC 非线性、真实黑电平等特性 |
| 双目 80mm 基线配置 | ✅ 可行 | Isaac Lab Camera 类原生支持多实例，双 CameraCfg 配置即可 |

### 7.2 总体评分：7.5 / 10

**适合**：训练 ISP 网络、测试去马赛克算法、生成大量仿真标注数据。

**不适合**：需要真实传感器噪声特性的场景（精确标定、噪声建模、最终部署验证）。

### 7.3 关键注意事项

**BlackLevel 取值**：

- **仿真数据应设为 0**。Isaac Sim 渲染器没有暗电流，pre-ISP 输出的最低值就是 0，设置非零 BlackLevel（如 64）会人为引入偏移，导致暗部信息丢失。
- **真实 SC132GS 数据用 64**（或根据实际传感器标定值）。当你切换到真实相机数据封装 DNG 时，再改为传感器的实际黑电平。

**Make/Model 标签**：

- 仿真数据建议标注为 `Make=NVIDIA`、`Model=IsaacSim_PreISP_Virtual`，明确来源是仿真。
- 不要标注为真实传感器型号（如 SC132GS/GS130W），避免在数据管理中造成混淆。
- 后续混合训练时，真实数据的 DNG 再用真实传感器的 Make/Model。

---

## 八、总结与建议

### 针对你的具体需求（双目立体 RGB → 12-bit RAW → DNG）：

1. **双目相机创建**: Isaac Lab 的 Camera 类完全支持，使用两个 CameraCfg 配置左右眼，设置 80mm baseline 和朝下方向即可。

2. **12-bit RAW 输出**: Isaac Lab 层面**不直接支持**，需要降到 Isaac Sim 5.1 的 pre-ISP pipeline 层面获取 float raw 数据，再自行量化。

3. **DNG 封装**: 完全可行，`tifffile + exiftool` 是目前最实用的 Python 方案。DNG 格式原生支持 12-bit Bayer CFA 数据的标准化封装。

4. **工程建议**: 如果项目预算和时间允许，建议先用 Isaac Lab Camera 类的标准 RGB 输出验证双目管线，待 Isaac Sim 的 pre-ISP API 更加成熟后再切换到 RAW 路线。

5. **混合数据策略**: 仿真伪 RAW 做 bulk training（大量），真实 GS130W/SC132GS 采集少量 paired 数据做 domain adaptation，是当前 sim-to-real gap 的最佳实践。

---

## 参考文献

- [Isaac Lab Camera 文档](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html)
- [Isaac Sim 5.1 Camera Sensors](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html)
- [Isaac Lab CameraCfg 源码](https://isaac-sim.github.io/IsaacLab/v2.0.0/_modules/isaaclab/sensors/camera/camera.html)
- [DNG Specification 1.6.0.0 (Adobe)](https://paulbourke.net/dataformats/dng/dng_spec_1_6_0_0.pdf)
- [Isaac Lab Tiled Rendering 文档](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html)