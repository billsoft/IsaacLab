# Isaac Sim 扩展开发教程 - extsUser 快速开发方式

## 概述

在 `extsUser` 目录中直接开发扩展，**无需编译**，修改代码后重启即可生效。适合快速原型开发和迭代。

## 核心优势

- ✅ **无需编译**：修改代码后重启 Isaac Sim 即可
- ✅ **快速迭代**：开发-测试循环只需几秒
- ✅ **简单直接**：不需要配置构建系统
- ✅ **独立管理**：不影响 Isaac Sim 源码

## 开发流程

### 1. 创建扩展目录

```bash
# 进入 extsUser 目录（通过 Isaac Lab 的符号链接访问）
cd D:\code\IsaacLab\_isaac_sim\extsUser

# 创建你的扩展目录
mkdir my.custom.fisheye_camera
cd my.custom.fisheye_camera
```

### 2. 创建扩展结构

```
D:\code\IsaacLab\_isaac_sim\extsUser\my.custom.fisheye_camera\
├── config\
│   └── extension.toml          # 扩展配置文件（必需）
├── data\
│   ├── icon.png                # 扩展图标（可选）
│   └── preview.png             # 预览图（可选）
├── docs\
│   └── README.md               # 文档（推荐）
└── my\
    └── custom\
        └── fisheye_camera\
            ├── __init__.py     # Python 包初始化
            ├── extension.py    # 扩展入口点
            ├── fisheye_sensor.py  # 鱼眼传感器实现
            └── ui_window.py    # UI 窗口（可选）
```

### 3. 创建 `config/extension.toml`

```toml
[package]
# 扩展基本信息
version = "1.0.0"
category = "Sensors"
title = "Custom Fisheye Camera"
description = "Custom fisheye camera sensor for Isaac Sim - Development Version"
authors = ["Your Name"]
keywords = ["camera", "fisheye", "sensor", "custom"]

# Python 模块映射（关键配置）
[python.module]
"my.custom.fisheye_camera" = "my/custom/fisheye_camera"

# 依赖项（根据需要添加）
[dependencies]
"omni.kit.uiapp" = {}
"isaacsim.sensors.camera" = {}
"isaacsim.core.utils" = {}
"omni.isaac.core" = {}

# 设置
[settings]
# 开发模式：启用热重载
exts."my.custom.fisheye_camera".dev_mode = true
```

### 4. 创建 `my/custom/fisheye_camera/__init__.py`

```python
"""Custom Fisheye Camera Extension for Isaac Sim.

This extension provides a custom fisheye camera sensor with enhanced features.
"""

__version__ = "1.0.0"

from .extension import *
from .fisheye_sensor import FisheyeCameraSensor

__all__ = ["FisheyeCameraSensor"]
```

### 5. 创建 `my/custom/fisheye_camera/extension.py`

```python
"""Extension entry point for Custom Fisheye Camera."""

import omni.ext
import omni.ui as ui
from omni.isaac.ui.ui_utils import btn_builder
import carb


class MyCustomFisheyeCameraExtension(omni.ext.IExt):
    """Custom Fisheye Camera Extension.
    
    Provides UI and functionality for creating and managing fisheye cameras.
    """

    def on_startup(self, ext_id):
        """Called when extension starts up."""
        carb.log_info("[my.custom.fisheye_camera] Extension startup")
        
        self._ext_id = ext_id
        self._window = None
        self._camera_instance = None
        
        # 创建 UI 窗口
        self._build_ui()

    def on_shutdown(self):
        """Called when extension shuts down."""
        carb.log_info("[my.custom.fisheye_camera] Extension shutdown")
        
        if self._window:
            self._window.destroy()
            self._window = None
        
        self._camera_instance = None

    def _build_ui(self):
        """Build the extension UI window."""
        self._window = ui.Window(
            "Fisheye Camera Control",
            width=400,
            height=300,
            visible=True,
            dockPreference=ui.DockPreference.LEFT_BOTTOM
        )
        
        with self._window.frame:
            with ui.VStack(spacing=10, height=0):
                # 标题
                ui.Label(
                    "Custom Fisheye Camera Sensor",
                    alignment=ui.Alignment.CENTER,
                    style={"font_size": 18}
                )
                
                ui.Spacer(height=10)
                
                # 相机路径输入
                with ui.HStack(spacing=5):
                    ui.Label("Camera Path:", width=100)
                    self._path_field = ui.StringField()
                    self._path_field.model.set_value("/World/Camera")
                
                # FOV 滑块
                with ui.HStack(spacing=5):
                    ui.Label("FOV (degrees):", width=100)
                    self._fov_slider = ui.FloatSlider(min=120, max=360)
                    self._fov_slider.model.set_value(200)
                    self._fov_label = ui.Label("200°", width=50)
                    
                self._fov_slider.model.add_value_changed_fn(
                    lambda m: self._fov_label.set_text(f"{m.get_value_as_float():.0f}°")
                )
                
                # 分辨率选择
                with ui.HStack(spacing=5):
                    ui.Label("Resolution:", width=100)
                    self._res_combo = ui.ComboBox(
                        0,
                        "1920x1080",
                        "1280x720",
                        "640x480",
                        "3840x2160"
                    )
                
                ui.Spacer(height=10)
                
                # 按钮
                with ui.VStack(spacing=5):
                    btn_builder(
                        "Create Fisheye Camera",
                        on_clicked_fn=self._on_create_camera
                    )
                    btn_builder(
                        "Capture Image",
                        on_clicked_fn=self._on_capture_image
                    )
                    btn_builder(
                        "Remove Camera",
                        on_clicked_fn=self._on_remove_camera
                    )
                
                ui.Spacer()
                
                # 状态信息
                self._status_label = ui.Label(
                    "Ready",
                    alignment=ui.Alignment.CENTER,
                    style={"color": 0xFF00FF00}
                )

    def _on_create_camera(self):
        """Create a fisheye camera in the scene."""
        from .fisheye_sensor import FisheyeCameraSensor
        
        prim_path = self._path_field.model.get_value_as_string()
        fov = self._fov_slider.model.get_value_as_float()
        
        # 解析分辨率
        res_text = ["1920x1080", "1280x720", "640x480", "3840x2160"][
            self._res_combo.model.get_item_value_model().get_value_as_int()
        ]
        width, height = map(int, res_text.split('x'))
        
        try:
            self._camera_instance = FisheyeCameraSensor(
                prim_path=prim_path,
                resolution=(width, height),
                fov=fov
            )
            self._camera_instance.initialize()
            
            self._status_label.text = f"✓ Camera created at {prim_path}"
            self._status_label.style = {"color": 0xFF00FF00}
            carb.log_info(f"Created fisheye camera at {prim_path}")
            
        except Exception as e:
            self._status_label.text = f"✗ Error: {str(e)}"
            self._status_label.style = {"color": 0xFFFF0000}
            carb.log_error(f"Failed to create camera: {e}")

    def _on_capture_image(self):
        """Capture an image from the camera."""
        if self._camera_instance is None:
            self._status_label.text = "✗ No camera created"
            self._status_label.style = {"color": 0xFFFF0000}
            return
        
        try:
            rgb_data = self._camera_instance.get_rgb()
            if rgb_data is not None:
                self._status_label.text = f"✓ Captured {rgb_data.shape}"
                self._status_label.style = {"color": 0xFF00FF00}
                carb.log_info(f"Captured image: {rgb_data.shape}")
            else:
                self._status_label.text = "✗ No image data"
                self._status_label.style = {"color": 0xFFFF0000}
                
        except Exception as e:
            self._status_label.text = f"✗ Capture error: {str(e)}"
            self._status_label.style = {"color": 0xFFFF0000}
            carb.log_error(f"Capture failed: {e}")

    def _on_remove_camera(self):
        """Remove the camera from the scene."""
        if self._camera_instance:
            self._camera_instance.destroy()
            self._camera_instance = None
            self._status_label.text = "✓ Camera removed"
            self._status_label.style = {"color": 0xFF00FF00}
        else:
            self._status_label.text = "✗ No camera to remove"
            self._status_label.style = {"color": 0xFFFF0000}
```

### 6. 创建 `my/custom/fisheye_camera/fisheye_sensor.py`

```python
"""Fisheye camera sensor implementation."""

from typing import Optional, Tuple
import numpy as np
import carb
import omni.isaac.core.utils.prims as prim_utils
from pxr import UsdGeom, Gf
from omni.isaac.core.prims import XFormPrim
import omni.replicator.core as rep


class FisheyeCameraSensor:
    """Custom Fisheye Camera Sensor.
    
    Provides fisheye camera functionality with customizable FOV and resolution.
    
    Example:
        >>> fisheye = FisheyeCameraSensor(
        ...     prim_path="/World/Camera",
        ...     resolution=(1920, 1080),
        ...     fov=220.0
        ... )
        >>> fisheye.initialize()
        >>> rgb_data = fisheye.get_rgb()
        >>> print(rgb_data.shape)  # (1080, 1920, 3)
    """
    
    def __init__(
        self,
        prim_path: str,
        resolution: Tuple[int, int] = (1920, 1080),
        fov: float = 200.0,
        projection_type: str = "fisheyePolynomial",
        position: Tuple[float, float, float] = (0.0, 0.0, 2.0),
        orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    ):
        """Initialize fisheye camera sensor.
        
        Args:
            prim_path: USD prim path for the camera
            resolution: (width, height) in pixels
            fov: Field of view in degrees (120-360)
            projection_type: Fisheye projection type
                - "fisheyePolynomial": 360° spherical projection
                - "fisheyeSpherical": 360° full-frame projection
                - "fisheyeKannalaBrandtK3": Kannala-Brandt K3 model
            position: Camera position (x, y, z)
            orientation: Camera orientation as quaternion (w, x, y, z)
        """
        self.prim_path = prim_path
        self.resolution = resolution
        self.fov = max(120.0, min(360.0, fov))  # Clamp FOV
        self.projection_type = projection_type
        self.position = position
        self.orientation = orientation
        
        self._camera_prim = None
        self._render_product = None
        self._annotators = {}
        
        carb.log_info(f"FisheyeCameraSensor created: {prim_path}")
        
    def initialize(self):
        """Initialize the camera in the scene."""
        # 创建相机 prim
        camera = rep.create.camera(
            position=self.position,
            rotation=self.orientation,
            focus_distance=400.0,
            focal_length=24.0,
            clipping_range=(0.1, 10000.0),
        )
        
        # 获取相机 prim 并重命名
        self._camera_prim = camera.get_prim()
        
        # 设置鱼眼投影
        from pxr import UsdGeom
        camera_geom = UsdGeom.Camera(self._camera_prim)
        camera_geom.GetProjectionAttr().Set("fisheyePolynomial")
        
        # 设置 FOV
        camera_geom.GetHorizontalApertureAttr().Set(self.fov)
        
        # 创建渲染产品
        self._render_product = rep.create.render_product(
            camera,
            resolution=self.resolution
        )
        
        # 初始化标注器
        self._annotators["rgb"] = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annotators["rgb"].attach([self._render_product])
        
        self._annotators["depth"] = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        self._annotators["depth"].attach([self._render_product])
        
        carb.log_info(f"Fisheye camera initialized at {self.prim_path}")
        carb.log_info(f"  Resolution: {self.resolution}")
        carb.log_info(f"  FOV: {self.fov}°")
        
    def get_rgb(self) -> Optional[np.ndarray]:
        """Get RGB image from camera.
        
        Returns:
            RGB image as numpy array (H, W, 3) with values [0-255], or None
        """
        if "rgb" not in self._annotators:
            carb.log_warn("RGB annotator not initialized")
            return None
            
        try:
            rgb_data = self._annotators["rgb"].get_data()
            if rgb_data is not None:
                return rgb_data
        except Exception as e:
            carb.log_error(f"Failed to get RGB data: {e}")
            
        return None
        
    def get_depth(self) -> Optional[np.ndarray]:
        """Get depth image from camera.
        
        Returns:
            Depth image as numpy array (H, W) with distance values, or None
        """
        if "depth" not in self._annotators:
            carb.log_warn("Depth annotator not initialized")
            return None
            
        try:
            depth_data = self._annotators["depth"].get_data()
            if depth_data is not None:
                return depth_data
        except Exception as e:
            carb.log_error(f"Failed to get depth data: {e}")
            
        return None
    
    def set_fov(self, fov: float):
        """Update camera field of view.
        
        Args:
            fov: New FOV in degrees (120-360)
        """
        self.fov = max(120.0, min(360.0, fov))
        
        if self._camera_prim:
            camera_geom = UsdGeom.Camera(self._camera_prim)
            camera_geom.GetHorizontalApertureAttr().Set(self.fov)
            carb.log_info(f"Updated FOV to {self.fov}°")
    
    def destroy(self):
        """Clean up camera resources."""
        if self._render_product:
            self._render_product = None
        
        if self._camera_prim:
            prim_utils.delete_prim(self._camera_prim.GetPath().pathString)
            self._camera_prim = None
            
        self._annotators.clear()
        carb.log_info(f"Fisheye camera destroyed: {self.prim_path}")
```

### 7. 启用扩展

#### 方法 1：通过 Isaac Sim UI

1. 启动 Isaac Sim：
   ```bash
   D:\code\IsaacLab\_isaac_sim\isaac-sim.bat
   ```

2. 打开扩展管理器：
   - 菜单：`Window` → `Extensions`

3. 搜索你的扩展：
   - 在搜索框输入 `fisheye`
   - 找到 `Custom Fisheye Camera`

4. 启用扩展：
   - 勾选扩展旁边的复选框
   - 应该会看到 "Fisheye Camera Control" 窗口出现

#### 方法 2：通过 Isaac Lab 脚本

创建测试脚本 `d:\code\IsaacLab\scripts\test_fisheye.py`：

```python
"""Test custom fisheye camera extension."""

import argparse
from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="Test custom fisheye camera")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.ext
import carb

# 启用自定义扩展
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled("my.custom.fisheye_camera", True)

# 导入并使用传感器
from my.custom.fisheye_camera import FisheyeCameraSensor

# 创建鱼眼相机
fisheye = FisheyeCameraSensor(
    prim_path="/World/FisheyeCamera",
    resolution=(1920, 1080),
    fov=220.0,
    position=(0.0, 0.0, 3.0)
)

fisheye.initialize()

# 运行几帧
for i in range(10):
    simulation_app.update()

# 获取图像
rgb_data = fisheye.get_rgb()
if rgb_data is not None:
    carb.log_info(f"Captured RGB image: {rgb_data.shape}")
    
depth_data = fisheye.get_depth()
if depth_data is not None:
    carb.log_info(f"Captured depth image: {depth_data.shape}")

# 清理
fisheye.destroy()
simulation_app.close()
```

运行测试：
```bash
isaaclab.bat -p scripts\test_fisheye.py
```

### 8. 开发迭代流程

```bash
# 1. 修改代码
# 编辑 my/custom/fisheye_camera/fisheye_sensor.py

# 2. 重启 Isaac Sim（或 Isaac Lab 脚本）
isaaclab.bat -p scripts\test_fisheye.py

# 3. 测试新功能

# 4. 重复步骤 1-3
```

**无需编译！** 修改代码后直接重启即可。

## 在 Isaac Lab 环境中集成

### 创建场景配置

```python
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg

@configclass
class FisheyeSceneCfg(InteractiveSceneCfg):
    """Scene with custom fisheye camera."""
    
    # 使用自定义鱼眼相机
    fisheye_camera: CameraCfg = CameraCfg(
        prim_path="/World/Robot/FisheyeCamera",
        spawn=None,  # 我们手动创建
        data_types=["rgb", "depth"],
        width=1920,
        height=1080,
    )
```

### 完整示例脚本

```python
"""Isaac Lab demo with custom fisheye camera."""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

# 启用自定义扩展
import omni.ext
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled("my.custom.fisheye_camera", True)

from my.custom.fisheye_camera import FisheyeCameraSensor

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene configuration."""
    
    ground = sim_utils.GroundPlaneCfg()

# 创建场景
scene_cfg = MySceneCfg(num_envs=1, env_spacing=2.0)
scene = InteractiveScene(scene_cfg)

# 创建鱼眼相机
fisheye = FisheyeCameraSensor(
    prim_path="/World/envs/env_0/FisheyeCamera",
    resolution=(1920, 1080),
    fov=220.0,
    position=(0.0, 0.0, 2.0)
)
fisheye.initialize()

# 仿真循环
for i in range(100):
    scene.update(dt=0.01)
    
    if i % 10 == 0:
        rgb = fisheye.get_rgb()
        if rgb is not None:
            print(f"Frame {i}: RGB shape = {rgb.shape}")

# 清理
fisheye.destroy()
simulation_app.close()
```

## 调试技巧

### 1. 查看扩展日志

```python
import carb

# 在代码中添加日志
carb.log_info("This is info")
carb.log_warn("This is warning")
carb.log_error("This is error")
```

### 2. 检查扩展是否加载

```python
import omni.ext

ext_manager = omni.kit.app.get_app().get_extension_manager()
is_enabled = ext_manager.is_extension_enabled("my.custom.fisheye_camera")
print(f"Extension enabled: {is_enabled}")
```

### 3. 热重载（实验性）

在 `extension.toml` 中：
```toml
[settings]
exts."my.custom.fisheye_camera".dev_mode = true
```

然后在 Python 中：
```python
import importlib
import my.custom.fisheye_camera.fisheye_sensor

# 重新加载模块
importlib.reload(my.custom.fisheye_camera.fisheye_sensor)
```

## 优缺点对比

### ✅ 优点
- **快速迭代**：无需编译，修改即生效
- **简单直接**：不需要构建系统
- **独立开发**：不影响 Isaac Sim 源码
- **易于调试**：可以直接修改代码

### ❌ 缺点
- **性能**：纯 Python，无法使用 C++ 加速
- **非正式**：不适合发布到扩展市场
- **依赖管理**：需要手动确保依赖存在

## 最佳实践

1. **先在 extsUser 快速原型**，稳定后迁移到源码编译
2. **使用 Git 管理**：将 extsUser 目录加入版本控制
3. **模块化设计**：将核心逻辑与 UI 分离
4. **添加日志**：使用 `carb.log_*` 记录关键信息
5. **错误处理**：使用 try-except 捕获异常

## 迁移到源码编译

当扩展稳定后，可以迁移到源码：

```bash
# 1. 复制扩展到源码目录
xcopy /E /I D:\code\IsaacLab\_isaac_sim\extsUser\my.custom.fisheye_camera D:\code\IsaacSim\source\extensions\my.custom.fisheye_camera

# 2. 编译 Isaac Sim
cd D:\code\IsaacSim
.\build.bat

# 3. 删除 extsUser 中的版本（避免冲突）
rmdir /S /Q D:\code\IsaacLab\_isaac_sim\extsUser\my.custom.fisheye_camera
```

## 常见问题

### Q: 扩展没有出现在扩展管理器中？

**A:** 检查以下几点：
1. `extension.toml` 格式是否正确
2. 目录结构是否符合要求
3. 重启 Isaac Sim

### Q: 导入模块失败？

**A:** 确保：
1. `[python.module]` 配置正确
2. `__init__.py` 文件存在
3. 模块路径与目录结构匹配

### Q: 修改代码不生效？

**A:** 
1. 完全重启 Isaac Sim
2. 清除 Python 缓存：删除 `__pycache__` 目录
3. 检查是否有多个同名扩展

## 参考资料

- [Omniverse Extensions 文档](https://docs.omniverse.nvidia.com/extensions/latest/index.html)
- [Isaac Sim Python API](https://docs.omniverse.nvidia.com/py/isaacsim/index.html)
- Isaac Sim 内置扩展示例：`D:\code\IsaacLab\_isaac_sim\exts\`
