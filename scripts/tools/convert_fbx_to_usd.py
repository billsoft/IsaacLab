"""
FBX → USD 转换工具（保留骨骼动画）
=====================================
用途：将 Mixamo 下载的 FBX 文件转换为 Isaac Sim 可用的 USD 文件。

运行：
    isaaclab.bat -p scripts/tools/convert_fbx_to_usd.py --input <fbx路径> --output <usd路径>
    isaaclab.bat -p scripts/tools/convert_fbx_to_usd.py --input <fbx路径> --output <usd路径> --headless

示例：
    isaaclab.bat -p scripts/tools/convert_fbx_to_usd.py ^
        --input  D:/Downloads/mixamo_walk.fbx ^
        --output D:/code/IsaacLab/Assets/Custom/Characters/worker_walk/worker_walk.usd ^
        --headless
"""

import argparse
import asyncio
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FBX to USD converter with animation support")
parser.add_argument("--input",  required=True, help="输入 FBX 文件路径")
parser.add_argument("--output", required=True, help="输出 USD 文件路径")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb
import omni.kit.asset_converter as converter


async def convert_fbx(input_path: str, output_path: str):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    ctx = converter.AssetConverterContext()
    ctx.ignore_animation      = False   # 关键：保留动画数据
    ctx.export_animations     = True
    ctx.bake_all_blend_shapes = False
    ctx.merge_all_meshes      = False
    ctx.up_axis               = "Y"     # Y-up（与 Isaac Sim 一致）
    ctx.export_hidden_props   = False

    carb.log_info(f"[Convert] 开始转换：{input_path}")
    carb.log_info(f"[Convert] 输出到：{output_path}")

    instance = converter.get_instance()
    task = instance.create_converter_task(input_path, output_path, None, ctx)
    success = await task.wait_until_finished()

    if success:
        carb.log_info("[Convert] 转换成功！")
        print(f"\n[Convert] 成功：{output_path}")
        print("[Convert] 请用验证脚本检查 USD 结构：")
        print(f"  isaaclab.bat -p scripts/tools/inspect_usd_skel.py --usd {output_path}")
    else:
        carb.log_error(f"[Convert] 转换失败：{task.get_status_string()}")
        print(f"\n[Convert] 失败：{task.get_status_string()}")


# 运行转换
asyncio.ensure_future(convert_fbx(args.input, args.output))
for _ in range(300):
    simulation_app.update()

simulation_app.close()
