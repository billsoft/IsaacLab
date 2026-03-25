# Isaac Lab 双目立体相机 伪12-bit RAW → DNG 全套工具链

## 项目结构

```
isaac_stereo_raw_dng/
├── README.md                           # 本文件
├── install.sh                          # 一键安装依赖
├── configs/
│   └── stereo_camera_cfg.py            # 双目相机配置（80mm 基线，朝下）
├── scripts/
│   ├── 01_capture_stereo_rgb.py        # 步骤1: Isaac Lab 双目 RGB 采集（标准路线）
│   ├── 02_capture_pre_isp_raw.py       # 步骤2: Isaac Sim pre-ISP raw 采集（RAW 路线）
│   ├── 03_rgb_to_pseudo_raw12.py       # 步骤3A: RGB → 伪 Bayer RAW12（备用方案）
│   ├── 04_float_raw_to_dng.py          # 步骤3B: pre-ISP float → 12-bit DNG（主方案）
│   └── 05_batch_stereo_dng.py          # 步骤4: 批量双目 DNG 生成
├── utils/
│   ├── __init__.py
│   ├── dng_writer.py                   # DNG 封装核心模块
│   ├── bayer_mosaic.py                 # Bayer 马赛克模拟（RGB → 单通道 CFA）
│   ├── noise_model.py                  # 传感器噪声模型（可选增强）
│   └── verify_dng.py                   # DNG 文件验证工具
└── tests/
    └── test_dng_pipeline.py            # 端到端测试（不依赖 Isaac Sim）
```

## 使用流程

### 快速开始

```bash
# 1. 安装依赖
chmod +x install.sh && ./install.sh

# 2. 无需 Isaac Sim 的端到端测试（生成合成数据验证 DNG 管线）
python tests/test_dng_pipeline.py

# 3. 在 Isaac Lab 中采集双目 RGB（需要 Isaac Sim 环境）
./isaaclab.sh -p scripts/01_capture_stereo_rgb.py --num_envs 1 --enable_cameras

# 4. 采集 pre-ISP raw 数据（需要 Isaac Sim 5.1+）
./python.sh scripts/02_capture_pre_isp_raw.py --draw-output

# 5. 将 pre-ISP float raw 转为 DNG
python scripts/04_float_raw_to_dng.py --input ./raw_output/ --output ./dng_output/

# 6. 批量处理双目 DNG
python scripts/05_batch_stereo_dng.py --input_dir ./raw_output/ --output_dir ./stereo_dng/
```

### 两条技术路线

| 路线 | 脚本流程 | 数据质量 | 适用场景 |
|------|----------|----------|----------|
| **主路线（pre-ISP）** | 02 → 04 → 05 | 较高（含 CFA + companding） | 训练 ISP / 去马赛克网络 |
| **备用路线（RGB 降级）** | 01 → 03 → 05 | 一般（人工 Bayer 马赛克） | 快速原型 / 管线验证 |

## 关键参数

- **基线距离**: 80mm
- **分辨率**: 1280×1024（可配置）
- **RAW 位深**: 12-bit（0-4095）
- **CFA 排列**: RGGB（可配置）
- **DNG 版本**: 1.4.0.0
- **BlackLevel**: 0（仿真数据）/ 64（真实 SC132GS）
- **WhiteLevel**: 4095

## 依赖

- Python 3.8+
- numpy, tifffile, Pillow
- exiftool（系统工具）
- Isaac Sim 5.1+（仅采集脚本需要）
- Isaac Lab（仅采集脚本需要）
