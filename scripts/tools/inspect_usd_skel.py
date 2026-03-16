"""
USD 骨骼动画结构检查工具
===========================
用途：检查转换后的 USD 文件是否包含正确的 UsdSkel 结构和动画数据。

运行：
    isaaclab.bat -p scripts/tools/inspect_usd_skel.py --usd <usd路径>
    isaaclab.bat -p scripts/tools/inspect_usd_skel.py --usd <usd路径> --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect UsdSkel structure of a USD file")
parser.add_argument("--usd", required=True, help="要检查的 USD 文件路径")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdSkel


def inspect(usd_path: str):
    print(f"\n{'='*60}")
    print(f"检查 USD：{usd_path}")
    print('='*60)

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print("[错误] 无法打开 USD 文件！")
        return

    start = stage.GetStartTimeCode()
    end   = stage.GetEndTimeCode()
    fps   = stage.GetFramesPerSecond()
    print(f"\n时间范围：{start:.0f} ~ {end:.0f} 帧  FPS={fps:.0f}")
    if fps > 0 and end > start:
        print(f"动画时长：{(end - start) / fps:.2f} 秒")

    skel_roots, skeletons, skel_anims, meshes = [], [], [], []
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        if   t == "SkelRoot":      skel_roots.append(prim)
        elif t == "Skeleton":      skeletons.append(prim)
        elif t == "SkelAnimation": skel_anims.append(prim)
        elif t == "Mesh":          meshes.append(prim)

    print(f"\nSkelRoot     : {len(skel_roots)}")
    for p in skel_roots:
        print(f"  {p.GetPath()}")

    print(f"Skeleton     : {len(skeletons)}")
    for p in skeletons:
        j = UsdSkel.Skeleton(p).GetJointsAttr().Get()
        print(f"  {p.GetPath()}  joints={len(j) if j else 0}")

    print(f"SkelAnimation: {len(skel_anims)}")
    for p in skel_anims:
        anim   = UsdSkel.Animation(p)
        joints = anim.GetJointsAttr().Get()
        times  = anim.GetRotationsAttr().GetTimeSamples()
        print(f"  {p.GetPath()}  joints={len(joints) if joints else 0}  samples={len(times)}")

    print(f"Mesh         : {len(meshes)}")

    print(f"\n{'='*60}")
    ok = True
    if not skel_roots:
        print("[✗] 无 SkelRoot — USD 无骨骼"); ok = False
    if not skel_anims:
        print("[✗] 无 SkelAnimation — 无动画数据（FBX 转换时未保留动画）"); ok = False
    else:
        total = sum(len(UsdSkel.Animation(p).GetRotationsAttr().GetTimeSamples())
                    for p in skel_anims)
        if total == 0:
            print("[✗] SkelAnimation 无时间采样 — 动画为空"); ok = False
        else:
            print(f"[✓] 动画数据正常：共 {total} 个旋转关键帧")
    if ok:
        print("[✓] USD 结构完整，可用于 Isaac Sim 骨骼动画播放")
    print('='*60)


inspect(args.usd)
simulation_app.close()
