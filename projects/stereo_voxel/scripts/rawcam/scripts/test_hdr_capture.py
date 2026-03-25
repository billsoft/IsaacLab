"""测试脚本: 验证从 Isaac Sim 获取 HDR float16 和 RAW uint16 数据
====================================================================
基于官方 camera_pre_isp_pipeline.py，在简单场景中验证：
  1. HdrColor renderVar 能否输出 float16 RGBA
  2. pre-ISP pipeline 的 CFA→companding→raw sensor 是否正常
  3. 输出文件的格式和数值范围

运行：
    D:\\code\\IsaacLab\\_isaac_sim\\python.bat -u scripts/test_hdr_capture.py

输出：
    rawcam/output/test_hdr/  下的 .bin 和 .npy 文件
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Test HDR / RAW capture from Isaac Sim")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--num_warmup", type=int, default=5, help="Warmup frames before capture")
args, _ = parser.parse_known_args()

# Isaac Sim 启动
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "width": args.width, "height": args.height})

import numpy as np
import omni.graph.core as og
import omni.replicator.core as rep
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.sensors.camera import Camera
from omni.syntheticdata import SyntheticData, SyntheticDataStage
from pxr import UsdLux

# 启用 pre-ISP 扩展
enable_extension("omni.sensors.nv.camera")

# 输出目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "output", "test_hdr")
os.makedirs(OUTPUT_DIR, exist_ok=True)

W, H = args.width, args.height

print("=" * 60)
print("  HDR / RAW 采集测试")
print("=" * 60)
print(f"  分辨率: {W}x{H}")
print(f"  输出:   {OUTPUT_DIR}")

# ============================================================
# 1. 创建简单场景（与官方示例一致）
# ============================================================
stage = get_current_stage()

dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(1000)

# 彩色立方体
colors_pos = [
    ([1.0, 0.0, 0.0], [-1.5, 0.0, 0.5]),
    ([0.0, 1.0, 0.0], [1.5, 0.0, 0.5]),
    ([0.0, 0.0, 1.0], [0.0, 1.5, 0.5]),
    ([1.0, 1.0, 0.0], [0.0, -1.5, 0.5]),
]
for i, (color, pos) in enumerate(colors_pos):
    VisualCuboid(prim_path=f"/World/Cube_{i}", position=np.array(pos), color=np.array(color))

# ============================================================
# 2. 创建相机（与官方示例完全一致的方式）
# ============================================================
camera = Camera(
    prim_path="/World/Camera",
    position=np.array([0.0, 0.0, 4.5]),
    resolution=(W, H),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 90]), degrees=True),
)
camera.set_focal_length(1.5)
camera.initialize()
rp_path = camera.get_render_product_path()
print(f"  RenderProduct: {rp_path}")

# ============================================================
# 3. 启用 HdrColor 并构建 pre-ISP OmniGraph 管线
#    （完全照搬官方示例结构）
# ============================================================
print("\n[Step 1] 启用 HdrColor renderVar...")
SyntheticData.Get().enable_rendervar(render_product_path=rp_path, render_var="HdrColor")

print("[Step 2] 构建 pre-ISP OmniGraph 管线...")
wrapper_node = SyntheticData.Get().get_graph(
    stage=SyntheticDataStage.POST_RENDER, renderProductPath=rp_path
)
wrapped_graph = wrapper_node.get_wrapped_graph()

keys = og.Controller.Keys
hdr_bin = os.path.join(OUTPUT_DIR, "hdr.rgba_f16..bin")
raw_bin = os.path.join(OUTPUT_DIR, "raw.r_u16..bin")
isp_bin = os.path.join(OUTPUT_DIR, "isp.rgb_u8..bin")

(_, nodes, _, _) = og.Controller.edit(
    wrapped_graph,
    {
        keys.CREATE_NODES: [
            ("CudaEntry", "omni.graph.nodes.GpuInteropRenderProductEntry"),
            ("Tex", "omni.sensors.nv.camera.CamTextureReadTaskNode"),
            ("CC", "omni.sensors.nv.camera.ColorCorrectionTaskNode"),
            ("CFA", "omni.sensors.nv.camera.CamCfa2x2EncoderTaskNode"),
            ("noise", "omni.sensors.nv.camera.CamGeneralPurposeNoiseTask"),
            ("com", "omni.sensors.nv.camera.CamCompandingTaskNode"),
            ("dec", "omni.sensors.nv.camera.CamIspDecompandingTaskNode"),
            ("dmz", "omni.sensors.nv.camera.CamIspRGGBDemosaicingTaskNode"),
            ("rgba", "omni.sensors.nv.camera.CamRGBADatatypeConverterTaskNode"),
            ("Enc", "omni.sensors.nv.camera.CamRGBEncoderTaskNode"),
            ("writeHdr", "omni.sensors.nv.camera.CamFileWriterTaskNode"),
            ("writeRaw", "omni.sensors.nv.camera.CamFileWriterTaskNode"),
            ("writeIsp", "omni.sensors.nv.camera.CamFileWriterTaskNode"),
        ],
        keys.CONNECT: [
            # 完整管线连接（与官方示例一致）
            ("CudaEntry.outputs:gpu", "Tex.inputs:gpu"),
            ("CudaEntry.outputs:rp", "Tex.inputs:rp"),
            ("CudaEntry.outputs:simTime", "Tex.inputs:simTime"),
            ("CudaEntry.outputs:hydraTime", "Tex.inputs:hydraTime"),
            ("Tex.outputs:gpu", "CC.inputs:gpu"),
            ("Tex.outputs:rp", "CC.inputs:rp"),
            ("Tex.outputs:simTimeOut", "CC.inputs:simTimeIn"),
            ("Tex.outputs:hydraTimeOut", "CC.inputs:hydraTimeIn"),
            ("Tex.outputs:dest", "CC.inputs:src"),
            ("CC.outputs:gpu", "CFA.inputs:gpu"),
            ("CC.outputs:rp", "CFA.inputs:rp"),
            ("CC.outputs:simTimeOut", "CFA.inputs:simTimeIn"),
            ("CC.outputs:hydraTimeOut", "CFA.inputs:hydraTimeIn"),
            ("CC.outputs:dest", "CFA.inputs:src"),
            ("CFA.outputs:gpu", "noise.inputs:gpu"),
            ("CFA.outputs:rp", "noise.inputs:rp"),
            ("CFA.outputs:simTimeOut", "noise.inputs:simTimeIn"),
            ("CFA.outputs:hydraTimeOut", "noise.inputs:hydraTimeIn"),
            ("CFA.outputs:dest", "noise.inputs:src"),
            ("noise.outputs:gpu", "com.inputs:gpu"),
            ("noise.outputs:rp", "com.inputs:rp"),
            ("noise.outputs:simTimeOut", "com.inputs:simTimeIn"),
            ("noise.outputs:hydraTimeOut", "com.inputs:hydraTimeIn"),
            ("noise.outputs:dest", "com.inputs:src"),
            ("com.outputs:gpu", "dec.inputs:gpu"),
            ("com.outputs:rp", "dec.inputs:rp"),
            ("com.outputs:simTimeOut", "dec.inputs:simTimeIn"),
            ("com.outputs:hydraTimeOut", "dec.inputs:hydraTimeIn"),
            ("com.outputs:dest", "dec.inputs:src"),
            ("dec.outputs:gpu", "dmz.inputs:gpu"),
            ("dec.outputs:rp", "dmz.inputs:rp"),
            ("dec.outputs:simTimeOut", "dmz.inputs:simTimeIn"),
            ("dec.outputs:hydraTimeOut", "dmz.inputs:hydraTimeIn"),
            ("dec.outputs:dest", "dmz.inputs:src"),
            ("dmz.outputs:gpu", "rgba.inputs:gpu"),
            ("dmz.outputs:rp", "rgba.inputs:rp"),
            ("dmz.outputs:simTimeOut", "rgba.inputs:simTimeIn"),
            ("dmz.outputs:hydraTimeOut", "rgba.inputs:hydraTimeIn"),
            ("dmz.outputs:dest", "rgba.inputs:src"),
            ("rgba.outputs:gpu", "Enc.inputs:gpu"),
            ("rgba.outputs:rp", "Enc.inputs:rp"),
            ("rgba.outputs:simTimeOut", "Enc.inputs:simTimeIn"),
            ("rgba.outputs:hydraTimeOut", "Enc.inputs:hydraTimeIn"),
            ("rgba.outputs:dest", "Enc.inputs:src"),
            # 文件写入连接
            ("Tex.outputs:gpu", "writeHdr.inputs:gpu"),
            ("Tex.outputs:rp", "writeHdr.inputs:rp"),
            ("Tex.outputs:simTimeOut", "writeHdr.inputs:simTimeIn"),
            ("Tex.outputs:hydraTimeOut", "writeHdr.inputs:hydraTimeIn"),
            ("Tex.outputs:dest", "writeHdr.inputs:src"),
            ("com.outputs:gpu", "writeRaw.inputs:gpu"),
            ("com.outputs:rp", "writeRaw.inputs:rp"),
            ("com.outputs:simTimeOut", "writeRaw.inputs:simTimeIn"),
            ("com.outputs:hydraTimeOut", "writeRaw.inputs:hydraTimeIn"),
            ("com.outputs:dest", "writeRaw.inputs:src"),
            ("Enc.outputs:gpu", "writeIsp.inputs:gpu"),
            ("Enc.outputs:rp", "writeIsp.inputs:rp"),
            ("Enc.outputs:simTimeOut", "writeIsp.inputs:simTimeIn"),
            ("Enc.outputs:hydraTimeOut", "writeIsp.inputs:hydraTimeIn"),
            ("Enc.outputs:dest", "writeIsp.inputs:src"),
        ],
        keys.SET_VALUES: [
            ("Tex.inputs:aov", "HDR"),
            ("CC.inputs:output_float16", True),
            ("CC.inputs:Rr", 1.0), ("CC.inputs:Rg", 0.0), ("CC.inputs:Rb", 0.0),
            ("CC.inputs:Gr", 0.0), ("CC.inputs:Gg", 1.0), ("CC.inputs:Gb", 0.0),
            ("CC.inputs:Br", 0.0), ("CC.inputs:Bg", 0.0), ("CC.inputs:Bb", 1.0),
            ("CC.inputs:whiteBalance", [0.05, 0.05, 0.05]),
            ("CFA.inputs:CFA_CF00", [1, 0, 0]),
            ("CFA.inputs:CFA_CF01", [0, 1, 0]),
            ("CFA.inputs:CFA_CF10", [0, 1, 0]),
            ("CFA.inputs:CFA_CF11", [0, 0, 1]),
            ("CFA.inputs:cfaSemantic", "RGGB"),
            ("CFA.inputs:maximalValue", 16777215),
            ("CFA.inputs:flipHorizontal", 0),
            ("CFA.inputs:flipVertical", 0),
            ("noise.inputs:darkShotNoiseGain", 10.0),
            ("noise.inputs:darkShotNoiseSigma", 0.5),
            ("noise.inputs:hdrCombinationData", [(5.8, 4000), (58, 8000), (70, 16000)]),
            ("com.inputs:LinearCompandCoeff", [
                [0, 0], [244, 240], [512, 430], [768, 584],
                [1024, 724], [2048, 883], [4096, 1150], [8192, 1600],
                [16384, 1768], [32768, 2050], [65536, 2354], [131072, 2865],
                [262144, 3195], [524288, 3750], [1048576, 3768],
                [4194304, 3850], [8388608, 3942], [16777215, 4095],
            ]),
            ("dec.inputs:LinearCompandCoeff", [
                [0, 0], [244, 240], [512, 430], [768, 584],
                [1024, 724], [2048, 883], [4096, 1150], [8192, 1600],
                [16384, 1768], [32768, 2050], [65536, 2354], [131072, 2865],
                [262144, 3195], [524288, 3750], [1048576, 3768],
                [4194304, 3850], [8388608, 3942], [16777215, 4095],
            ]),
            ("dmz.inputs:bayerGrid", "RGGB"),
            ("dmz.inputs:outputFormat", "UINT16"),
            # 文件写入配置
            ("writeHdr.inputs:filename", hdr_bin),
            ("writeHdr.inputs:eachFrameOneFile", True),
            ("writeHdr.inputs:onlyLastFrame", True),
            ("writeRaw.inputs:filename", raw_bin),
            ("writeRaw.inputs:eachFrameOneFile", True),
            ("writeRaw.inputs:onlyLastFrame", True),
            ("writeIsp.inputs:filename", isp_bin),
            ("writeIsp.inputs:eachFrameOneFile", True),
            ("writeIsp.inputs:onlyLastFrame", True),
        ],
    },
)
print(f"  OmniGraph 管线创建完成，{len(nodes)} 个节点")

# ============================================================
# 4. 渲染（与官方示例一致：只需 update 2 帧）
# ============================================================
print(f"\n[Step 3] 渲染 {args.num_warmup + 2} 帧...")
for _ in range(args.num_warmup + 2):
    simulation_app.update()

print("[Step 4] 检查输出文件...")

# ============================================================
# 5. 读取并分析 pre-ISP 输出
# ============================================================
results = {}

# HDR float16 RGBA
hdr_file = os.path.join(OUTPUT_DIR, "hdr.rgba_f16.0.bin")
if os.path.exists(hdr_file):
    with open(hdr_file, "rb") as f:
        hdr_data = np.frombuffer(f.read(), dtype=np.float16)
    expected = H * W * 4
    print(f"\n  === HDR float16 RGBA ===")
    print(f"  文件: {os.path.getsize(hdr_file):,} bytes, {len(hdr_data):,} elements (expect {expected:,})")
    if len(hdr_data) == expected:
        hdr_img = hdr_data.reshape(H, W, 4)
        print(f"  Shape: {hdr_img.shape}, dtype: {hdr_img.dtype}")
        print(f"  Range: [{float(hdr_img.min()):.4f}, {float(hdr_img.max()):.4f}]")
        print(f"  Mean:  R={float(hdr_img[:,:,0].mean()):.4f} G={float(hdr_img[:,:,1].mean()):.4f} "
              f"B={float(hdr_img[:,:,2].mean()):.4f} A={float(hdr_img[:,:,3].mean()):.4f}")
        np.save(os.path.join(OUTPUT_DIR, "hdr_rgba_f16.npy"), hdr_img)
        results["hdr"] = True
    else:
        print(f"  [!] Size mismatch: {len(hdr_data)} != {expected}")
        results["hdr"] = False
else:
    print(f"\n  [X] HDR file not generated: {hdr_file}")
    results["hdr"] = False

# RAW uint16 单通道
raw_file = os.path.join(OUTPUT_DIR, "raw.r_u16.0.bin")
if os.path.exists(raw_file):
    with open(raw_file, "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint16)
    expected = H * W
    print(f"\n  === RAW uint16 Bayer (RGGB) ===")
    print(f"  文件: {os.path.getsize(raw_file):,} bytes, {len(raw_data):,} elements (expect {expected:,})")
    if len(raw_data) == expected:
        raw_img = raw_data.reshape(H, W)
        print(f"  Shape: {raw_img.shape}, dtype: {raw_img.dtype}")
        print(f"  Range: [{raw_img.min()}, {raw_img.max()}]")
        print(f"  Mean:  {raw_img.mean():.1f}")
        bits = int(np.ceil(np.log2(max(raw_img.max(), 1) + 1)))
        print(f"  有效位数: {bits} bit (max={raw_img.max()})")
        print(f"  Bayer 子通道:")
        print(f"    R  [0::2,0::2]: mean={raw_img[0::2,0::2].mean():.1f}")
        print(f"    Gr [0::2,1::2]: mean={raw_img[0::2,1::2].mean():.1f}")
        print(f"    Gb [1::2,0::2]: mean={raw_img[1::2,0::2].mean():.1f}")
        print(f"    B  [1::2,1::2]: mean={raw_img[1::2,1::2].mean():.1f}")
        np.save(os.path.join(OUTPUT_DIR, "raw_bayer_u16.npy"), raw_img)
        results["raw"] = True
    else:
        print(f"  [!] Size mismatch: {len(raw_data)} != {expected}")
        results["raw"] = False
else:
    print(f"\n  [X] RAW file not generated: {raw_file}")
    results["raw"] = False

# ISP uint8 RGB
isp_file = os.path.join(OUTPUT_DIR, "isp.rgb_u8.0.bin")
if os.path.exists(isp_file):
    with open(isp_file, "rb") as f:
        isp_data = np.frombuffer(f.read(), dtype=np.uint8)
    expected = H * W * 3
    print(f"\n  === ISP uint8 RGB ===")
    print(f"  文件: {os.path.getsize(isp_file):,} bytes, {len(isp_data):,} elements (expect {expected:,})")
    if len(isp_data) == expected:
        isp_img = isp_data.reshape(H, W, 3)
        print(f"  Shape: {isp_img.shape}, dtype: {isp_img.dtype}")
        print(f"  Range: [{isp_img.min()}, {isp_img.max()}]")
        print(f"  Mean:  {isp_img.mean():.1f}")
        np.save(os.path.join(OUTPUT_DIR, "isp_rgb_u8.npy"), isp_img)
        results["isp"] = True
    else:
        print(f"  [!] Size mismatch: {len(isp_data)} != {expected}")
        results["isp"] = False
else:
    print(f"\n  [X] ISP file not generated: {isp_file}")
    results["isp"] = False

# ============================================================
# 6. 总结
# ============================================================
print("\n" + "=" * 60)
print("  测试结果")
print("=" * 60)
for key, ok in results.items():
    status = "OK" if ok else "FAIL"
    print(f"  [{status:4s}] {key}")

if all(results.values()):
    print("\n  所有管线数据获取成功！")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("\n  下一步: 将 RAW uint16 量化截断到 12-bit [0,4095] 并封装为 DNG")
elif results.get("hdr"):
    print("\n  HDR 数据获取成功，可以用 HDR→12bit 量化方案")
else:
    print("\n  管线数据获取失败，请检查 omni.sensors.nv.camera 扩展")

print("\n[Done] 退出...")
simulation_app.close()
