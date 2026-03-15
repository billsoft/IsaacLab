# Isaac Sim 扩展开发教程 - 源码编译方式

## 概述

在 Isaac Sim 源码中开发扩展，然后通过编译部署到运行时环境，适合需要与 Isaac Sim 核心功能深度集成的扩展开发。

## 开发流程

### 1. 创建扩展目录

```bash
# 进入 Isaac Sim 源码扩展目录
cd D:\code\IsaacSim\source\extensions

# 创建你的扩展目录（命名规范：<组织>.<类别>.<扩展名>）
mkdir my.custom.fisheye_camera
cd my.custom.fisheye_camera
```

### 2. 创建扩展基本结构

```
my.custom.fisheye_camera\
├── config\
│   └── extension.toml          # 扩展配置文件
├── data\
│   ├── icon.png                # 扩展图标
│   └── preview.png             # 预览图
├── docs\
│   ├── README.md               # 文档
│   └── CHANGELOG.md            # 更新日志
├── my\
│   └── custom\
│       └── fisheye_camera\
│           ├── __init__.py     # Python 包初始化
│           ├── extension.py    # 扩展入口
│           ├── fisheye_sensor.py  # 鱼眼传感器实现
│           └── tests\
│               └── test_fisheye.py
└── premake5.lua                # 构建配置（可选）
```

### 3. 创建 `config/extension.toml`

```toml
[package]
# 扩展基本信息
version = "1.0.0"
category = "Sensors"
title = "Custom Fisheye Camera"
description = "Custom fisheye camera sensor for Isaac Sim"
authors = ["Your Name <your.email@example.com>"]
repository = ""
keywords = ["camera", "fisheye", "sensor"]
changelog = "docs/CHANGELOG.md"
readme = "docs/README.md"
icon = "data/icon.png"
preview_image = "data/preview.png"

# Python 模块路径
[python.module]
"my.custom.fisheye_camera" = "my/custom/fisheye_camera"

# 依赖项
[dependencies]
"omni.kit.uiapp" = {}
"isaacsim.sensors.camera" = {}
"isaacsim.core.utils" = {}

# 测试依赖
[[test]]
dependencies = [
    "isaacsim.test.utils",
]
```

### 4. 创建 `my/custom/fisheye_camera/__init__.py`

```python
"""Custom Fisheye Camera Extension for Isaac Sim."""

from .extension import *
from .fisheye_sensor import *
```

### 5. 创建 `my/custom/fisheye_camera/extension.py`

```python
"""Extension entry point."""

import omni.ext
import omni.ui as ui


class MyCustomFisheyeCameraExtension(omni.ext.IExt):
    """Custom Fisheye Camera Extension."""

    def on_startup(self, ext_id):
        """Called when extension starts."""
        print("[my.custom.fisheye_camera] Extension startup")
        self._window = None
        
        # 可选：创建 UI 窗口
        self._window = ui.Window("Fisheye Camera", width=300, height=300)
        with self._window.frame:
            with ui.VStack():
                ui.Label("Custom Fisheye Camera Sensor")
                ui.Button("Create Fisheye Camera", clicked_fn=self._on_create_camera)

    def on_shutdown(self):
        """Called when extension shuts down."""
        print("[my.custom.fisheye_camera] Extension shutdown")
        if self._window:
            self._window.destroy()
            self._window = None

    def _on_create_camera(self):
        """Create fisheye camera in scene."""
        print("Creating fisheye camera...")
        # 实现创建逻辑
```

### 6. 创建 `my/custom/fisheye_camera/fisheye_sensor.py`

```python
"""Fisheye camera sensor implementation."""

from typing import Optional
import numpy as np
import omni.isaac.core.utils.prims as prim_utils
from isaacsim.sensors.camera import Camera


class FisheyeCameraSensor:
    """Custom Fisheye Camera Sensor.
    
    Example:
        >>> fisheye = FisheyeCameraSensor(
        ...     prim_path="/World/Camera",
        ...     resolution=(1920, 1080),
        ...     fov=220.0
        ... )
        >>> fisheye.initialize()
        >>> rgb_data = fisheye.get_rgb()
    """
    
    def __init__(
        self,
        prim_path: str,
        resolution: tuple[int, int] = (1920, 1080),
        fov: float = 200.0,
        projection_type: str = "fisheyePolynomial"
    ):
        """Initialize fisheye camera sensor.
        
        Args:
            prim_path: USD prim path for the camera
            resolution: (width, height) in pixels
            fov: Field of view in degrees
            projection_type: Fisheye projection type
        """
        self.prim_path = prim_path
        self.resolution = resolution
        self.fov = fov
        self.projection_type = projection_type
        self._camera = None
        
    def initialize(self):
        """Initialize the camera in the scene."""
        # 实现初始化逻辑
        print(f"Initializing fisheye camera at {self.prim_path}")
        # 使用 Isaac Sim 的相机 API 创建鱼眼相机
        
    def get_rgb(self) -> Optional[np.ndarray]:
        """Get RGB image from camera.
        
        Returns:
            RGB image as numpy array (H, W, 3) or None
        """
        if self._camera is None:
            return None
        # 实现获取 RGB 数据的逻辑
        return None
        
    def get_depth(self) -> Optional[np.ndarray]:
        """Get depth image from camera.
        
        Returns:
            Depth image as numpy array (H, W) or None
        """
        if self._camera is None:
            return None
        # 实现获取深度数据的逻辑
        return None
```

### 7. 编译 Isaac Sim

```bash
# 进入 Isaac Sim 项目根目录
cd D:\code\IsaacSim

# 执行编译（会将扩展部署到 _build\windows-x86_64\release\）
.\build.bat

# 或者只编译特定扩展
.\build.bat --ext my.custom.fisheye_camera
```

### 8. 验证扩展部署

```bash
# 检查扩展是否已部署到运行时目录
dir D:\code\IsaacSim\_build\windows-x86_64\release\exts | findstr fisheye

# 应该看到：
# my.custom.fisheye_camera\
```

### 9. 在 Isaac Lab 中使用

```python
# 在 Isaac Lab 脚本中
import omni.ext

# 启用你的扩展
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled("my.custom.fisheye_camera", True)

# 使用你的传感器
from my.custom.fisheye_camera import FisheyeCameraSensor

fisheye = FisheyeCameraSensor(
    prim_path="/World/envs/env_0/Robot/FisheyeCamera",
    resolution=(1920, 1080),
    fov=220.0
)
fisheye.initialize()
```

## 编译流程详解

### 编译做了什么？

1. **复制/链接扩展**：
   ```
   D:\code\IsaacSim\source\extensions\my.custom.fisheye_camera\
   → D:\code\IsaacSim\_build\windows-x86_64\release\exts\my.custom.fisheye_camera\
   ```

2. **编译 C++ 组件**（如果有）

3. **生成 Python 字节码**（.pyc 文件）

4. **更新扩展注册表**

### 增量编译

```bash
# 只重新编译修改过的扩展
.\build.bat --incremental

# 清理后重新编译
.\build.bat --clean
```

## 调试技巧

### 1. 查看扩展日志

```bash
# 运行 Isaac Sim 并查看控制台输出
D:\code\IsaacSim\_build\windows-x86_64\release\isaac-sim.bat
```

### 2. 在 Isaac Lab 中调试

```bash
# 使用 Isaac Lab 运行你的测试脚本
isaaclab.bat -p scripts/test_fisheye.py
```

### 3. 热重载（开发模式）

在 `extension.toml` 中添加：
```toml
[settings]
exts."my.custom.fisheye_camera".dev_mode = true
```

## 优缺点

### ✅ 优点
- 与 Isaac Sim 核心深度集成
- 可以使用 C++ 加速
- 正式的扩展管理
- 可以发布到 Omniverse 扩展市场

### ❌ 缺点
- 每次修改需要重新编译
- 编译时间较长（首次编译可能需要几分钟）
- 开发迭代速度较慢

## 最佳实践

1. **先在 `extsUser` 中快速原型开发**，稳定后再迁移到源码
2. **使用版本控制**：将扩展代码提交到 Git
3. **编写单元测试**：在 `tests/` 目录下
4. **遵循命名规范**：`<组织>.<类别>.<扩展名>`
5. **文档完善**：README.md 和代码注释

## 参考资料

- [Omniverse Extensions 官方文档](https://docs.omniverse.nvidia.com/extensions/latest/index.html)
- [Isaac Sim 扩展开发指南](https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_extension.html)
- Isaac Sim 源码中的示例扩展：`D:\code\IsaacSim\source\extensions\isaacsim.sensors.camera\`
